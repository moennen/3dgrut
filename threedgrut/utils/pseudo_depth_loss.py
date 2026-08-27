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

# Quantiles defining the robust spread of a disparity map. The gate is expressed relative to
# this so that it is invariant to the prior's arbitrary scale, exactly as the loss itself is --
# which is what makes gate=0 vs gate>0 a comparison of the gate rather than of the prior's units.
_IQR_QUANTILES = (0.25, 0.75)

# `torch.quantile` refuses inputs beyond this size, so large maps are strided down first. A
# fixed stride keeps this free of any host synchronisation.
_MAX_QUANTILE_ELEMENTS = 1 << 23


def disparity_scale(prior_disparity: torch.Tensor) -> torch.Tensor:
    """Robust spread (inter-quartile range) of a disparity map, as a 0-dim tensor."""
    flat = prior_disparity.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return prior_disparity.new_zeros(())
    if finite.numel() > _MAX_QUANTILE_ELEMENTS:
        finite = finite[:: (finite.numel() // _MAX_QUANTILE_ELEMENTS) + 1]
    quantiles = torch.quantile(finite.float(), torch.tensor(_IQR_QUANTILES, device=finite.device))
    return (quantiles[1] - quantiles[0]).to(prior_disparity.dtype)


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
    prior_disparity: torch.Tensor,
    scene_extent: float,
    shift_fraction: float = 0.05,
    gate: float = 0.0,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Penalise pixel pairs the render orders opposite to the prior.

    `pred_dist` and `pred_opacity` are the tracer's [B, H, W, 1] buffers; `prior_disparity` is
    the cached prior as [B, H, W, 1], **disparity** (larger means nearer), which is what
    `PseudoDepthCache` stores.

    Returns a 0-dim tensor: the mean over surviving pairs of the rendered depth gap, in units of
    `scene_extent`, on pairs whose order disagrees with the prior. Zero when no pair survives.
    """
    if prior_disparity.shape[:-1] != pred_dist.shape[:-1]:
        raise ValueError(
            f"pseudo-depth prior {tuple(prior_disparity.shape)} does not match the rendered "
            f"buffers {tuple(pred_dist.shape)}; it must be resampled to the training resolution"
        )
    if scene_extent <= 0:
        raise ValueError(f"scene_extent must be positive, got {scene_extent}")

    depth, confident = expected_depth(pred_dist, pred_opacity, min_opacity)
    height, width = depth.shape[-3:-1]
    dy, dx = sample_pair_offset(height, width, shift_fraction, rng)

    depth_base, depth_shifted = _shifted_views(depth, dy, dx)
    prior_base, prior_shifted = _shifted_views(prior_disparity, dy, dx)
    confident_base, confident_shifted = _shifted_views(confident, dy, dx)

    # Disparity is affine to *inverse* depth, so it decreases with distance: the prior's depth
    # ordering is the negated disparity ordering. Getting this backwards trains against a
    # mirrored scene while still producing a smooth, plausible-looking loss curve.
    prior_delta = prior_base - prior_shifted
    prior_order = -torch.sign(prior_delta)

    # A pair whose prior difference is within the noise carries no ordering information; see the
    # module docstring for what including them costs.
    threshold = gate * disparity_scale(prior_disparity)
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
    masked = torch.where(keep, disagreement, torch.zeros_like(disagreement))

    # Clamped division rather than a Python branch on the mask: this runs every iteration, and
    # reading the count on the host would stall the training loop.
    return masked.sum() / keep.sum().clamp_min(1) / scene_extent
