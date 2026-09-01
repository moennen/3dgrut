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

"""Ordinal (relative) depth supervision from a monocular pseudo-depth prior.

The prior is trustworthy in *ordering* and unreliable in scale, so this loss reads nothing but
the sign of its differences: for a pair of pixels it asks "does the prior agree with the render
about which one is nearer?", and penalises only disagreement. That makes it invariant to any
monotonically increasing transform of the prior -- not merely to the affine ambiguity -- so no
alignment against COLMAP points or anything else is needed, and the disparity-vs-depth
distinction reduces to a single sign flip.

That sign flip is real, not hypothetical: Depth Anything V2 emits disparity (larger is nearer)
while Depth Anything 3's monocular model emits depth (larger is farther), so `quantity` is a
required argument rather than an assumption. Because this loss reads only the ordering, it is
also the term least able to *benefit* from a better prior's global structure -- see
`docs/normal-supervision.md`.

The loss is one-sided by construction. A pair the render already orders correctly contributes
exactly zero and no gradient, so the term cannot fight geometry it agrees with; it only pushes
where the render contradicts the prior. Measured on sponza, the prior beats the model on 91% of
the pixels where the model is outside `delta1`, which is the population this is meant to reach.

Two departures from the reference implementation in `/mnt/oss/blob-to-spoke`. Only the second
survived measurement:

* **`gate` drops pairs whose prior disparity is nearly tied, and defaults to 0 (off).** It was
  added on the theory that ties are where the prior's sign is noise: offline, the prior's
  ordering agrees with ground truth on 84% of pairs ungated and 97% once pairs below 0.05 of the
  disparity IQR are dropped, so ungated roughly one pair in six looked like it was pushing the
  wrong way. Trained, that reasoning is wrong. Over three seeds the gate is neutral on sponza
  and lone-monk (within 0.0003 `abs_rel`) and clearly harmful on emerald-square, where gating
  gives +0.7% depth against the baseline while ungated gives -10.3%.
  The offline metric counted pairs instead of asking what they teach. A large `|dDisp|` is a
  *long-range* comparison, which is precisely the regime where this prior drifts -- one global
  affine scores `abs_rel` 0.068 against 0.011 per 16x16 patch -- so the gate selects for the
  prior's weakest structure and discards the local structure that is its strongest. It is also
  self-defeating: agreement of 97% means only 3% of the surviving pairs disagree with the render
  at all, and a one-sided loss learns nothing from the rest. Kept as a knob because it is one
  comparison and the sweep needs it, not because it helps.
* **Pairs are formed by cropping, not by wrapping.** Rolling the image pairs opposite edges,
  which are unrelated in 3D, manufacturing disagreements the prior never claimed.

The comparison happens in *rendered* depth, which is Euclidean ray distance here rather than the
z-depth the reference uses. Ordering is unaffected by that distinction along a fixed ray, but the
two are not interchangeable in general, so nothing else from that reference should be ported
without conversion.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY, expected_depth

# How a prior's values order distance: `+1` when a larger value means *farther* and `-1` when it
# means *nearer*. A monocular prior is affine either to depth or to inverse depth, and this sign
# is the entire difference between the two as far as an ordinal loss is concerned. It is named
# per model rather than guessed, because guessing wrong trains against a mirrored scene while
# leaving the loss curve smooth and plausible.
FARTHER_SIGN = {"depth": 1.0, "disparity": -1.0}

# Quantiles defining the robust spread of a prior map. The gate is expressed relative to
# this so that it is invariant to the prior's arbitrary scale, exactly as the loss itself is --
# which is what makes gate=0 vs gate>0 a comparison of the gate rather than of the prior's units.
_IQR_QUANTILES = (0.25, 0.75)

# `torch.quantile` refuses inputs beyond this size, so large maps are strided down first. A
# fixed stride keeps this free of any host synchronisation.
_MAX_QUANTILE_ELEMENTS = 1 << 23


def prior_scale(prior: torch.Tensor) -> torch.Tensor:
    """Robust spread (inter-quartile range) of a prior map, as a 0-dim tensor."""
    flat = prior.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return prior.new_zeros(())
    if finite.numel() > _MAX_QUANTILE_ELEMENTS:
        finite = finite[:: (finite.numel() // _MAX_QUANTILE_ELEMENTS) + 1]
    quantiles = torch.quantile(finite.float(), torch.tensor(_IQR_QUANTILES, device=finite.device))
    return (quantiles[1] - quantiles[0]).to(prior.dtype)


def sample_pair_offset(height: int, width: int, shift_fraction: float, rng: Optional[random.Random] = None):
    """A random non-zero pixel offset of up to `shift_fraction` of the longer image side.

    Uses the host RNG deliberately: this is a control decision, not data, and drawing it on the
    device would force a synchronisation every iteration to read it back.
    """
    draw = rng if rng is not None else random
    limit = max(int(round(shift_fraction * max(height, width))), 1)
    while True:
        dy = draw.randint(-limit, limit)
        dx = draw.randint(-limit, limit)
        if dy != 0 or dx != 0:
            return dy, dx


def _shifted_views(tensor: torch.Tensor, dy: int, dx: int):
    """The overlapping regions of `tensor` and `tensor` translated by (dy, dx).

    Cropping rather than wrapping: `torch.roll` would pair the top of the image with the bottom.
    """
    height, width = tensor.shape[-3:-1] if tensor.dim() >= 3 else tensor.shape[-2:]
    top, bottom = max(dy, 0), height + min(dy, 0)
    left, right = max(dx, 0), width + min(dx, 0)
    base = tensor[..., top:bottom, left:right, :]
    shifted = tensor[..., top - dy : bottom - dy, left - dx : right - dx, :]
    return base, shifted


def compute_pseudo_depth_order_loss(
    pred_dist: torch.Tensor,
    pred_opacity: torch.Tensor,
    prior: torch.Tensor,
    scene_extent: float,
    *,
    quantity: str,
    shift_fraction: float = 0.05,
    gate: float = 0.0,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    rng: Optional[random.Random] = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise pixel pairs the render orders opposite to the prior.

    `pred_dist` and `pred_opacity` are the tracer's [B, H, W, 1] buffers; `prior` is the cached
    prior as [B, H, W, 1], in whichever quantity the model that produced it emits.

    `quantity` is `"disparity"` (larger means nearer) or `"depth"` (larger means farther) and has
    no default: the two priors in use here disagree on it, and reading one as the other is a
    silent failure rather than a loud one. `PseudoDepthCache.quantity` reports what was stored.

    ``confidence`` is an optional detached per-pixel reliability map of the same shape. A pair
    uses the geometric mean of its endpoint weights, so one ambiguous endpoint downweights the
    pair without completely discarding the other. Returns a 0-dim weighted mean over surviving
    pairs in units of `scene_extent`, or zero when no pair survives.
    """
    if quantity not in FARTHER_SIGN:
        raise ValueError(f"unknown prior quantity {quantity!r}; expected one of {sorted(FARTHER_SIGN)}")
    if prior.shape[:-1] != pred_dist.shape[:-1]:
        raise ValueError(
            f"pseudo-depth prior {tuple(prior.shape)} does not match the rendered "
            f"buffers {tuple(pred_dist.shape)}; it must be resampled to the training resolution"
        )
    if scene_extent <= 0:
        raise ValueError(f"scene_extent must be positive, got {scene_extent}")

    depth, confident = expected_depth(pred_dist, pred_opacity, min_opacity)
    if confidence is not None and confidence.shape != pred_dist.shape:
        raise ValueError("pseudo-depth confidence must match pred_dist [B, H, W, 1]")
    if confidence is not None:
        confidence = torch.nan_to_num(confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    height, width = depth.shape[-3:-1]
    dy, dx = sample_pair_offset(height, width, shift_fraction, rng)

    depth_base, depth_shifted = _shifted_views(depth, dy, dx)
    prior_base, prior_shifted = _shifted_views(prior, dy, dx)
    confident_base, confident_shifted = _shifted_views(confident, dy, dx)
    if confidence is None:
        pair_confidence = torch.ones_like(depth_base)
    else:
        confidence_base, confidence_shifted = _shifted_views(confidence, dy, dx)
        pair_confidence = (confidence_base * confidence_shifted).clamp_min(0.0).sqrt()

    # A disparity prior is affine to *inverse* depth and so decreases with distance, where a
    # depth prior increases with it; `FARTHER_SIGN` turns both into the same depth ordering.
    prior_delta = prior_base - prior_shifted
    prior_order = FARTHER_SIGN[quantity] * torch.sign(prior_delta)

    # A pair whose prior difference is within the noise carries no ordering information; see the
    # module docstring for what including them costs.
    threshold = gate * prior_scale(prior)
    keep = (
        confident_base
        & confident_shifted
        & (prior_delta.abs() >= threshold)
        & (prior_order != 0)
        & torch.isfinite(prior_delta)
    )

    # Negative exactly where the render contradicts the prior, with a magnitude equal to the
    # rendered gap; `relu` keeps the disagreements and discards the rest, so agreeing pairs
    # receive no gradient at all.
    disagreement = torch.relu(-(depth_base - depth_shifted) * prior_order)
    weights = torch.where(keep, pair_confidence, torch.zeros_like(pair_confidence))
    masked = disagreement * weights

    # Clamped division rather than a Python branch on the mask: this runs every iteration, and
    # reading the count on the host would stall the training loop.
    return masked.sum() / weights.sum().clamp_min(1.0) / scene_extent


def aligned_prior_distance(prior: torch.Tensor, affine: torch.Tensor, rays_dir: torch.Tensor) -> torch.Tensor:
    """The prior as Euclidean ray distance, given its per-frame affine into scene z.

    `prior` is [B, H, W, 1] in the model's own units, `affine` is [B, 2] of (scale, offset) as
    fitted by `threedgrut.datasets.sparse_depth_alignment`, and `rays_dir` is [B, H, W, 3] in
    camera space (normalised or not -- only its direction is used).

    The two steps are not interchangeable and the order matters. The affine belongs in z, because
    that is the space the prior is ambiguous in; ray distance is z divided by the per-pixel cosine
    between the ray and the optical axis, which is a different quantity for every pixel. Applying
    the affine to a distance map, or comparing z against the renderer, would both look entirely
    reasonable and be wrong by up to the cosine at the frame corner.

    Non-finite where the frame has no alignment, or where the aligned z is not positive -- the
    offset can push the prior's nearest values behind the camera, and those pixels describe no
    surface. Callers must mask on `torch.isfinite`.
    """
    if prior.shape[-1] != 1:
        raise ValueError(f"prior must be [B, H, W, 1], got {tuple(prior.shape)}")
    if affine.dim() != 2 or affine.shape[-1] != 2:
        raise ValueError(f"affine must be [B, 2] of (scale, offset), got {tuple(affine.shape)}")
    if affine.shape[0] != prior.shape[0]:
        raise ValueError(f"affine batch {affine.shape[0]} does not match prior batch {prior.shape[0]}")

    scale = affine[:, 0].view(-1, 1, 1, 1)
    offset = affine[:, 1].view(-1, 1, 1, 1)
    prior_z = scale * prior + offset

    # cos(theta) between the ray and the camera's forward axis. Guarding the denominator rather
    # than the numerator: a ray exactly perpendicular to the axis has no z to speak of, and the
    # clamp keeps it finite so the mask below can reject it on the depth instead.
    to_z = (rays_dir[..., 2:3] / rays_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)).abs().clamp_min(1e-6)
    distance = prior_z / to_z
    return torch.where(distance > 0, distance, torch.full_like(distance, float("nan")))


def compute_pseudo_depth_l1_loss(
    pred_dist: torch.Tensor,
    pred_opacity: torch.Tensor,
    prior: torch.Tensor,
    affine: torch.Tensor,
    rays_dir: torch.Tensor,
    scene_extent: float,
    *,
    quantity: str,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 between the rendered ray distance and the sparse-aligned prior, in units of `scene_extent`.

    Unlike the ordinal term this reads the prior's *values*, so it can only be as good as the
    alignment: see `threedgrut.datasets.sparse_depth_alignment` for why a per-frame affine fitted
    to COLMAP points is the form used, and `docs/normal-supervision.md` for the measurement that
    reopened this loss after it had been abandoned.

    That measurement also bounds what to expect. The aligned prior beats the 7k model by 2.2x on
    lone-monk and 2.4x on emerald-square but *loses* to it on sponza (0.057 against 0.037), and
    even where it wins it is closer to ground truth on only 65-77% of pixels. So a quarter to a
    third of pixels are pulled the wrong way by construction, and this term is a floor rather
    than a teacher of detail. It is not expected to help every scene and is off by default.

    `quantity` must be `"depth"`. A disparity prior is affine to *inverse* z, so the alignment
    this consumes would have had to be fitted differently; rather than silently apply a z-space
    affine to it, this refuses.

    Returns a 0-dim tensor, zero when no pixel survives the mask.
    """
    if quantity != "depth":
        raise ValueError(
            f"pseudo-depth L1 needs a depth prior, got {quantity!r}. The alignment it consumes is "
            "an affine in z, which a disparity prior is not; use the ordinal loss instead, which "
            "reads only the prior's ordering."
        )
    if prior.shape[:-1] != pred_dist.shape[:-1]:
        raise ValueError(
            f"pseudo-depth prior {tuple(prior.shape)} does not match the rendered "
            f"buffers {tuple(pred_dist.shape)}; it must be resampled to the training resolution"
        )
    if scene_extent <= 0:
        raise ValueError(f"scene_extent must be positive, got {scene_extent}")

    depth, confident = expected_depth(pred_dist, pred_opacity, min_opacity)
    if confidence is not None and confidence.shape != pred_dist.shape:
        raise ValueError("pseudo-depth confidence must match pred_dist [B, H, W, 1]")
    if confidence is not None:
        confidence = torch.nan_to_num(confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    target = aligned_prior_distance(prior, affine, rays_dir)

    # `confident` excludes pixels the render places no surface at; there is nothing there for a
    # depth target to correct, and pulling on them would fight the opacity that put them there.
    keep = confident & torch.isfinite(target)
    # The target is detached only in the sense that it is data, but it can contain NaN, and NaN
    # survives multiplication by zero -- so the residual has to be built from a sanitised target
    # rather than masked after the fact, or the gradient is NaN everywhere.
    safe_target = torch.where(keep, target, depth.detach())
    residual = (depth - safe_target).abs()
    weights = torch.where(
        keep, confidence if confidence is not None else torch.ones_like(residual), torch.zeros_like(residual)
    )
    return (residual * weights).sum() / weights.sum().clamp_min(1.0) / scene_extent
