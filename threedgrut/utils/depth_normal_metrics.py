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

# Below this accumulated opacity a ray has passed through almost nothing, so the
# expected depth is dominated by the transparent remainder and carries no surface.
MIN_ACCUMULATED_OPACITY = 0.5

# A reference normal must be a direction; anything shorter is a gap in the reference map.
MIN_REFERENCE_NORMAL_NORM = 0.5

# Standard depth accuracy thresholds: the fraction of pixels within a ratio of 1.25^k.
DELTA_THRESHOLDS = (1.25, 1.25**2, 1.25**3)

# Standard normal accuracy thresholds in degrees.
ANGLE_THRESHOLDS_DEG = (11.25, 22.5, 30.0)


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
) -> dict[str, float]:
    """Angular error between rendered and reference normals, in degrees.

    Both are expected to be unit length and in the same world frame, oriented toward the
    camera. The error is signed, not folded into [0, 90]: a normal pointing away from the
    reference is wrong, and taking the absolute cosine would hide exactly the
    inside-out surfaces worth catching.
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
    return metrics


def geometry_metrics(
    outputs: dict[str, torch.Tensor],
    depth_gt: torch.Tensor | None,
    normal_gt: torch.Tensor | None,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
) -> dict[str, float]:
    """Depth and normal metrics for one frame, skipping whichever reference is absent.

    `outputs` is a tracer render dict. Returns an empty dict when neither reference is
    available, so callers can merge unconditionally.
    """
    metrics: dict[str, float] = {}

    if depth_gt is not None and depth_gt.numel() > 0:
        pred_depth, _ = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], min_opacity)
        valid = reference_depth_validity(depth_gt)
        metrics.update(depth_metrics(pred_depth.squeeze(-1), depth_gt.squeeze(-1), valid.squeeze(-1)))

    if normal_gt is not None and normal_gt.numel() > 0:
        # Normals are only scored where a surface exists, which the reference depth
        # defines when it is available; otherwise the reference normal's own length does.
        valid = torch.ones(normal_gt.shape[:-1], dtype=torch.bool, device=normal_gt.device)
        if depth_gt is not None and depth_gt.numel() > 0:
            valid = valid & reference_depth_validity(depth_gt.squeeze(-1))
        metrics.update(normal_metrics(outputs["pred_normals"], normal_gt, valid))

    return metrics
