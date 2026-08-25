# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometry metrics comparing rendered depth and normals against reference maps.

Depth is compared in absolute world units, so a run is only comparable against another
run in the same normalized world; no per-image scale or shift is fitted. Pixels without
reference geometry (sky, non-finite, the sentinel written by the OB3D readers) are
excluded, since a rendered depth there has nothing to be right or wrong about.

Both metrics deliberately keep pixels the model reconstructs badly. Dropping
low-confidence predictions would let a model improve its score by becoming less certain
about exactly the regions it handles worst.
"""

from __future__ import annotations

import torch

from threedgrut.datasets.gt_geometry import DEPTH_SENTINEL_THRESHOLD
from threedgrut.utils.depth_geometry import depth_gradient_magnitude, normals_from_points, unproject_to_world

# Below this accumulated opacity a ray has passed through almost nothing, so the
# expected depth is dominated by the transparent remainder and carries no surface.
MIN_ACCUMULATED_OPACITY = 0.5

# A reference normal must be a direction; anything shorter is a gap in the reference map.
MIN_REFERENCE_NORMAL_NORM = 0.5

# Standard depth accuracy thresholds: the fraction of pixels within a ratio of 1.25^k.
DELTA_THRESHOLDS = (1.25, 1.25**2, 1.25**3)

# A surface rendered closer than this fraction of its reference depth is counted as a
# floater. Half is well outside plausible depth noise, so the count reflects geometry
# placed in the wrong place rather than an imprecise estimate of the right one.
FLOATER_RATIO = 0.5

# Standard normal accuracy thresholds in degrees.
ANGLE_THRESHOLDS_DEG = (11.25, 22.5, 30.0)

# Pixels at or above this quantile of relative depth gradient are the "high gradient"
# tail: occlusion boundaries, where an expected depth is least likely to sit on a surface.
HIGH_GRADIENT_PERCENTILE = 0.9


def reference_depth_validity(depth_gt: torch.Tensor) -> torch.Tensor:
    """Pixels whose reference depth describes a real surface.

    Mirrors `gt_geometry.depth_validity` for tensors: the sky is stored as a large
    sentinel rather than a NaN, so it has to be excluded by magnitude.
    """
    return torch.isfinite(depth_gt) & (depth_gt.abs() < DEPTH_SENTINEL_THRESHOLD) & (depth_gt > 0.0)


def expected_depth(
    pred_dist: torch.Tensor,
    pred_opacity: torch.Tensor,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the raw accumulated ray distance into an expected depth.

    The tracer accumulates `sum(alpha_i * T_i * d_i)` without dividing by the accumulated
    opacity `sum(alpha_i * T_i)`, so a ray that is not fully opaque reports a distance
    biased toward zero by its own transparency. Dividing recovers the expectation over
    the ray's own weight distribution; on a 1500-iteration sponza model this halves
    absolute relative error (0.062 -> 0.032) and halves the negative depth bias.

    Returns the depth and a mask of rays opaque enough for it to mean anything.
    """
    opacity = pred_opacity.clamp_min(0.0)
    confident = opacity >= min_opacity
    depth = torch.where(confident, pred_dist / opacity.clamp_min(1e-6), torch.zeros_like(pred_dist))
    return depth, confident


def world_view_dirs(rays_dir: torch.Tensor, T_to_world: torch.Tensor) -> torch.Tensor:
    """Unit ray directions rotated from ray space into the world frame.

    Rendered normals live in world space, so the view direction has to be brought into
    the same frame before it can be used as a control.
    """
    rotation = T_to_world[:, :3, :3]
    dirs = torch.einsum("bij,bhwj->bhwi", rotation, rays_dir)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def depth_metrics(
    pred_depth: torch.Tensor,
    depth_gt: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    """Absolute depth error over `valid` pixels, in world units.

    `valid` should already exclude pixels lacking reference geometry. Predicted pixels
    are not filtered by confidence: a ray that renders nothing where the reference has a
    surface is a genuine failure and is counted as such.
    """
    mask = valid & reference_depth_validity(depth_gt)
    count = int(mask.sum())
    if count == 0:
        return {}

    pred = pred_depth[mask].detach().double()
    gt = depth_gt[mask].detach().double()
    error = pred - gt

    # A prediction of zero (nothing rendered) has no finite log or ratio, so the
    # scale-sensitive terms use only the pixels where the model committed to a surface.
    positive = pred > 0
    ratio = torch.maximum(pred[positive] / gt[positive], gt[positive] / pred[positive])

    metrics = {
        "depth_abs_rel": float((error.abs() / gt).mean()),
        "depth_sq_rel": float((error**2 / gt).mean()),
        "depth_mae": float(error.abs().mean()),
        "depth_rmse": float(torch.sqrt((error**2).mean())),
        "depth_bias": float(error.mean()),
        "depth_valid_px": float(count),
        # Fraction of reference surfaces the model rendered anything at all for.
        "depth_covered_frac": float(positive.double().mean()),
        # Surfaces placed far in front of the reference: semi-transparent ghost layers
        # that a symmetric average hides, since they are a small fraction of pixels with
        # a large one-sided error. Separating them matters because they behave quite
        # differently from the mild, symmetric noise that dominates abs-rel.
        "depth_floater_frac": float((positive & (pred < FLOATER_RATIO * gt)).double().mean()),
    }
    for index, threshold in enumerate(DELTA_THRESHOLDS, start=1):
        # Pixels with no prediction cannot be within a ratio, so they count as misses
        # rather than being quietly dropped from the denominator.
        metrics[f"depth_delta{index}"] = float((ratio < threshold).sum()) / count
    return metrics


def normal_metrics(
    pred_normals: torch.Tensor,
    normal_gt: torch.Tensor,
    valid: torch.Tensor,
    view_dirs: torch.Tensor | None = None,
) -> dict[str, float]:
    """Angular error between rendered and reference normals, in degrees.

    Both are expected to be unit length and in the same world frame, oriented toward the
    camera. The error is signed, not folded into [0, 90]: a normal pointing away from the
    reference is wrong, and taking the absolute cosine would hide exactly the
    inside-out surfaces worth catching.

    A raw angular error is close to uninterpretable on its own. The renderer flips every
    normal to face the camera and the reference normals of visible surfaces face the
    camera too, so both vectors are confined to the same hemisphere and a prediction
    carrying no geometry at all still scores far better than the 90 degrees that
    "random" suggests. When `view_dirs` is supplied this reports the control directly:
    the error of pointing every normal straight back along the view ray, which uses no
    geometry whatsoever. A normal buffer that does not beat that control has not been
    shown to know anything about the surface.
    """
    reference_norm = normal_gt.detach().norm(dim=-1)
    predicted_norm = pred_normals.detach().norm(dim=-1)
    # A zero prediction marks a ray that hit nothing; it has no direction to score.
    mask = valid & (reference_norm > MIN_REFERENCE_NORMAL_NORM) & (predicted_norm > 1e-6)
    count = int(mask.sum())
    if count == 0:
        return {}

    pred = (pred_normals[mask] / predicted_norm[mask].unsqueeze(-1)).detach().double()
    gt = (normal_gt[mask] / reference_norm[mask].unsqueeze(-1)).detach().double()
    angles = torch.rad2deg(torch.acos((pred * gt).sum(-1).clamp(-1.0, 1.0)))

    metrics = {
        "normal_mean_deg": float(angles.mean()),
        "normal_median_deg": float(angles.median()),
        "normal_rmse_deg": float(torch.sqrt((angles**2).mean())),
        "normal_valid_px": float(count),
    }
    for threshold in ANGLE_THRESHOLDS_DEG:
        key = f"normal_pct_{str(threshold).replace('.', '_')}"
        metrics[key] = float((angles < threshold).double().mean())

    if view_dirs is not None:
        view = view_dirs.detach()[mask].double()
        view = view / view.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        control = torch.rad2deg(torch.acos((-view * gt).sum(-1).clamp(-1.0, 1.0)))
        metrics["normal_viewdir_control_deg"] = float(control.mean())
        # Positive means the buffer beats the no-geometry control. Negative means the
        # reported error is coming from the camera-facing convention, not the surface.
        metrics["normal_gain_vs_viewdir_deg"] = float(control.mean() - angles.mean())
    return metrics


def depth_derived_normal_metrics(
    pred_depth: torch.Tensor,
    reference_depth: torch.Tensor | None,
    normal_gt: torch.Tensor,
    pred_normals: torch.Tensor,
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    T_to_world: torch.Tensor,
    valid: torch.Tensor,
    high_gradient_percentile: float = HIGH_GRADIENT_PERCENTILE,
) -> dict[str, float]:
    """How good a supervision target the depth-implied normal actually is.

    The depth-normal consistency loss pulls the rendered normal towards the normal implied
    by the rendered depth. That is only worth doing if the target is better than what it
    replaces, which is a measurable claim rather than an assumption, so this reports:

    * `depth_normal_mean_deg` -- the target's own error against the reference normals.
      This is what the loss converges towards, so it bounds what the loss can achieve.
    * `depth_normal_vs_rendered_deg` -- disagreement between target and rendered normal,
      which is the quantity the loss actually minimises. Near zero means the loss has
      nothing to say whatever its weight.
    * `depth_normal_reference_mean_deg` -- the same target computed from the *reference*
      depth. This is the control that separates two very different failures: if the
      reference depth also produces a poor normal then the finite-difference operator is
      the limit and no improvement in the rendered depth will help, whereas a large gap
      between the two means the rendered depth is what is wrong.
    * `depth_normal_high_grad_deg` / `depth_normal_low_grad_deg` -- the target's error
      restricted to the pixels with the largest and smallest relative depth gradient. Expected depth is a
      weighted mean along the ray, so it lands between surfaces at an occlusion boundary
      and describes no surface there; if that mechanism matters in practice the damage is
      concentrated in the high-gradient tail and can be masked away, whereas a flat
      profile means the problem is diffuse and masking will not fix it.
    """
    world_dirs = world_view_dirs(rays_dir, T_to_world)
    points = unproject_to_world(pred_depth, rays_ori, rays_dir, T_to_world)
    derived, usable = normals_from_points(points, world_dirs, valid)

    reference_norm = normal_gt.norm(dim=-1)
    mask = usable & valid & (reference_norm > MIN_REFERENCE_NORMAL_NORM)
    if not bool(mask.any()):
        return {}

    gt = (normal_gt[mask] / reference_norm[mask].unsqueeze(-1)).double()
    target = derived[mask].double()
    metrics = {
        "depth_normal_mean_deg": float(_angles_deg(target, gt).mean()),
        "depth_normal_valid_px": float(int(mask.sum())),
    }

    rendered_norm = pred_normals.norm(dim=-1)
    both = mask & (rendered_norm > 1e-6)
    if bool(both.any()):
        rendered = (pred_normals[both] / rendered_norm[both].unsqueeze(-1)).double()
        metrics["depth_normal_vs_rendered_deg"] = float(_angles_deg(derived[both].double(), rendered).mean())

    # Split on the rendered depth's own gradient: the mask a loss could apply is one it
    # can compute, and it has no access to the reference.
    gradient = depth_gradient_magnitude(pred_depth)[mask]
    if gradient.numel() > 1:
        cutoff = torch.quantile(gradient.float(), high_gradient_percentile)
        steep = gradient >= cutoff
        angles = _angles_deg(target, gt)
        if bool(steep.any()) and not bool(steep.all()):
            metrics["depth_normal_high_grad_deg"] = float(angles[steep].mean())
            metrics["depth_normal_low_grad_deg"] = float(angles[~steep].mean())
            metrics["depth_normal_high_grad_frac"] = float(steep.double().mean())

    if reference_depth is not None:
        reference_points = unproject_to_world(reference_depth, rays_ori, rays_dir, T_to_world)
        reference_derived, reference_usable = normals_from_points(reference_points, world_dirs, valid)
        control = reference_usable & valid & (reference_norm > MIN_REFERENCE_NORMAL_NORM)
        if bool(control.any()):
            control_gt = (normal_gt[control] / reference_norm[control].unsqueeze(-1)).double()
            metrics["depth_normal_reference_mean_deg"] = float(
                _angles_deg(reference_derived[control].double(), control_gt).mean()
            )

    return metrics


def _angles_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))


def geometry_metrics(
    outputs: dict[str, torch.Tensor],
    depth_gt: torch.Tensor | None,
    normal_gt: torch.Tensor | None,
    view_dirs: torch.Tensor | None = None,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    batch: object | None = None,
) -> dict[str, float]:
    """Depth and normal metrics for one frame, skipping whichever reference is absent.

    `outputs` is a tracer render dict. `view_dirs` are world-space ray directions, used
    only to report the no-geometry control described in `normal_metrics`. `batch` carries
    the rays and pose needed to unproject depth; supplying it adds the depth-derived
    normal diagnostics. Returns an empty dict when neither reference is available, so
    callers can merge unconditionally.
    """
    metrics: dict[str, float] = {}
    has_depth = depth_gt is not None and depth_gt.numel() > 0
    has_normals = normal_gt is not None and normal_gt.numel() > 0

    pred_depth = None
    if has_depth:
        pred_depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], min_opacity)
        valid = reference_depth_validity(depth_gt)
        metrics.update(depth_metrics(pred_depth.squeeze(-1), depth_gt.squeeze(-1), valid.squeeze(-1)))

    if has_normals:
        # Normals are only scored where a surface exists, which the reference depth
        # defines when it is available; otherwise the reference normal's own length does.
        valid = torch.ones(normal_gt.shape[:-1], dtype=torch.bool, device=normal_gt.device)
        if has_depth:
            valid = valid & reference_depth_validity(depth_gt.squeeze(-1))
        metrics.update(normal_metrics(outputs["pred_normals"], normal_gt, valid, view_dirs))

        if pred_depth is not None and batch is not None:
            # The rendered depth is only a surface where the ray is opaque enough to have
            # one, so the diagnostic inherits that restriction rather than differencing
            # zeros left behind by transparent rays.
            metrics.update(
                depth_derived_normal_metrics(
                    pred_depth,
                    depth_gt,
                    normal_gt,
                    outputs["pred_normals"],
                    batch.rays_ori,
                    batch.rays_dir,
                    batch.T_to_world,
                    valid & confident.squeeze(-1),
                )
            )

    return metrics
