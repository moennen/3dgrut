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

"""Depth-variance: penalise rays whose weight is spread along their own length.

A ray's compositing weights `w_i = alpha_i * T_i` define a distribution over distance. Its
variance measures how far the rendered depth is from describing a *surface*: zero means all
the weight sits at one distance, large means the ray is averaging across a soft cloud and the
expected depth it reports names a place where nothing is. Penalising it asks the model to
resolve such rays into surfaces. This is the distortion-style regulariser of the NeRF
literature, expressed in the second moment the tracer now accumulates.

Two properties of the formulation are deliberate, and both are places this could go wrong.

**It penalises an un-normalized accumulator, so transparency is an escape hatch.** The term
falls if the model concentrates a ray's weight -- the intent -- but it *also* falls if the
model simply makes the scene less opaque, since every `w_i` shrinks. Only the photometric loss
argues against that, and the same is true of the distortion losses this follows, which are
used successfully on that basis. It is a genuine risk rather than a theoretical one, so the
ablation must report accumulated opacity alongside the geometry metrics: a variance
improvement bought by fading the scene out is not the effect we are looking for.

**Within a scene it is scale-dependent, by choice.** Variance carries squared distance units,
so a far surface with the same *relative* spread is penalised more than a near one. The
alternative -- dividing by depth to get a scale-free spread -- needs guarding where the depth
is small and changes what the term means. Across scenes the dependence is removed, dividing by
the squared scene extent so one lambda transfers between worlds whose units differ by orders of
magnitude, as `use_scale_flatten` already does with its linear scale.
"""

from __future__ import annotations

import torch

from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY


def depth_variance_loss(
    pred_dist: torch.Tensor,
    pred_dist_sq: torch.Tensor,
    pred_opacity: torch.Tensor,
    scene_extent: float,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opacity-weighted variance of the per-ray distance distribution.

    `pred_dist`, `pred_dist_sq` and `pred_opacity` are the tracer's [B, H, W, 1] buffers,
    holding `sum(w*t)`, `sum(w*t^2)` and `sum(w)`. The per-ray variance of the normalized
    weights is `M2/acc - (D/acc)^2`; weighting it back by `acc` so that rays carrying little
    surface count for little reduces to `M2 - D^2/acc`, which is the form evaluated here --
    one division instead of three, and no `1/acc^2` to cancel.

    `scene_extent` normalizes the squared units. Rays below `min_opacity` are excluded: their
    accumulated weight is dominated by the transparent remainder, so their "variance" is a
    statement about the background rather than about geometry.

    Returns the loss and the pixel count it was averaged over, as device tensors. Runs without
    host synchronisation -- called every iteration -- so the empty-mask case is a clamped
    division rather than a Python branch.
    """
    if pred_dist_sq.numel() == 0:
        raise ValueError("pred_dist_sq is empty: the depth-variance loss requires render.enable_depth_variance=true")

    accumulated = pred_opacity
    valid = (accumulated >= min_opacity) & torch.isfinite(pred_dist_sq) & torch.isfinite(pred_dist)

    # The denominator is detached. Left attached, d/d(acc) of `D^2/acc` contributes a
    # `1/acc^2` term that grows without bound as a ray approaches the mask threshold from
    # above, so rays that barely qualify would dominate the gradient. Detaching keeps the
    # weight distribution's *shape* as what the term acts on; the opacity still receives
    # gradient through `M2` and `D`, which is where the concentration signal lives.
    safe = accumulated.detach().clamp_min(min_opacity)

    # Non-negative in exact arithmetic (Jensen), but the difference of two accumulators
    # cancels to a small negative value on a nearly-resolved ray, which would otherwise
    # reward further sharpening with an unboundedly negative loss.
    variance = (pred_dist_sq - pred_dist * pred_dist / safe).clamp_min(0.0)

    per_pixel = torch.where(valid, variance, torch.zeros_like(variance))
    count = valid.sum()
    normalizer = max(scene_extent, 1e-8) ** 2
    return per_pixel.sum() / (count.clamp_min(1) * normalizer), count
