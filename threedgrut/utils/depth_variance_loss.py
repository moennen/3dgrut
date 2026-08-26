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

Equivalently, and more legibly, the term is pairwise:

    M2 - D^2/acc  ==  (1/(2*acc)) * sum_ij w_i w_j (t_i - t_j)^2

so every hit is pulled towards every other with strength `w_i w_j / acc`. That is the mip-NeRF
360 distortion loss with a squared distance. The identity is pinned by
`test_the_term_is_the_pairwise_distortion_loss`, and it is the form to reason in: it makes the
two-hit case visibly a double well in the near opacity, since `w_near * w_far` carries a
factor `a(1-a)` that peaks at one half. See `docs/normal-supervision.md` for why that sinks
the term as an unsupervised regulariser.

The gradient is the whole point, so it is worth writing down. With `mu = D/acc` the expected
depth, differentiating `M2 - D^2/acc` gives

    dL/dt_i = 2 * w_i * (t_i - mu)      each hit is pulled towards the expected depth
    dL/dw_i = (t_i - mu)^2              each hit's weight is pushed down, by how far it is

Both are translation-invariant, and the second is non-negative: the term never asks for more
weight anywhere, and it discounts a hit in proportion to its *squared* distance from the
surface the ray is describing. Those are the two mechanisms the term is supposed to have, and
they only exist if the `acc` in the denominator carries gradient -- see the note at the
division, which is where an earlier version of this got it wrong and produced exactly the
scene collapse it was supposed to prevent.

The absolute form above has two defects, and `depth_variance_relative` fixes both by dividing
by the squared expected depth. `Var/mu^2` is computable from the same three buffers --

    Var/mu^2 = acc*M2/D^2 - 1

-- so it costs no accumulator, no backward path and no kernel change; it is the same render.

**Uniform fading no longer reduces it.** The absolute form is homogeneous of degree one in the
weights, so halving every `w_i` halves it without resolving anything, and only the photometric
loss argues against that. `Var/mu^2` is homogeneous of degree *zero*: scaling every weight
leaves it exactly unchanged, which removes the escape hatch rather than merely monitoring it.
The ablation still reads accumulated opacity, but it is no longer load-bearing.

**And it is scale-free rather than merely extent-normalised.** Variance carries squared
distance units, so under the absolute form a far surface with the same *relative* spread
contributes `mu^2` times more: measured 0.01 / 0.25 / 4.0 for one ray at `mu` = 2 / 10 / 40.
Dividing by the squared scene extent only fixes this *between* scenes, and imperfectly -- it is
why no lambda transferred between sponza and emerald-square. `Var/mu^2` is exactly flat across
all three. Note the mip-NeRF/2DGS `|t_i - t_j|` kernel is only a partial fix here, linear in
`mu` rather than flat, and it needs a new accumulator to compute.

The reason to prefer it is larger than either, though, and was not anticipated. Dividing by
`mu^2` penalises the near collapse specifically, because committing to a near floater is what
makes `mu` small. That breaks the symmetry of the double well in the right direction: the
barrier in `a_near` moves from 0.487 to 0.818, shrinking the basin that locks a ray onto a
floater from 51% of the axis to 18%. It does *not* remove the degeneracy -- both wells are
still exactly zero, and the near well is still absorbing once `T` reaches zero -- so this is a
quantitative improvement to a term that remains unanchored, not a repair of it.

Its one cost is that `dL/dt` scales as `1/mu`, so rays with a small expected depth are
amplified. `min_distance` floors them; the opacity mask alone does not, since a ray can be
fully opaque and very close.
"""

from __future__ import annotations

import torch

from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY

MIN_EXPECTED_DISTANCE = 1e-3
"""Floor on the expected depth for the relative form, whose gradient scales as `1/mu`."""


def depth_variance_loss(
    pred_dist: torch.Tensor,
    pred_dist_sq: torch.Tensor,
    pred_opacity: torch.Tensor,
    scene_extent: float,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    relative: bool = False,
    min_distance: float = MIN_EXPECTED_DISTANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opacity-weighted variance of the per-ray distance distribution.

    `pred_dist`, `pred_dist_sq` and `pred_opacity` are the tracer's [B, H, W, 1] buffers,
    holding `sum(w*t)`, `sum(w*t^2)` and `sum(w)`. The per-ray variance of the normalized
    weights is `M2/acc - (D/acc)^2`; weighting it back by `acc` so that rays carrying little
    surface count for little reduces to `M2 - D^2/acc`, which is the form evaluated here --
    one division instead of three, and no `1/acc^2` to cancel.

    With `relative`, the term is instead `Var/mu^2 == acc*M2/D^2 - 1`, the *squared coefficient
    of variation*: same buffers, same render, but dimensionless. It is scale-free rather than
    extent-normalised, invariant to uniform fading, and biased against the near collapse -- see
    the module docstring. `scene_extent` is then unused, since there are no units left to
    normalise, and rays whose expected depth falls below `min_distance` are dropped because the
    `1/mu` in the gradient would otherwise amplify them without bound.

    `scene_extent` normalizes the squared units of the absolute form. Rays below `min_opacity`
    are excluded in both: their accumulated weight is dominated by the transparent remainder,
    so their "variance" is a statement about the background rather than about geometry.

    Returns the loss and the pixel count it was averaged over, as device tensors. Runs without
    host synchronisation -- called every iteration -- so the empty-mask case is a clamped
    division rather than a Python branch.
    """
    if pred_dist_sq.numel() == 0:
        raise ValueError("pred_dist_sq is empty: the depth-variance loss requires render.enable_depth_variance=true")

    accumulated = pred_opacity
    valid = (accumulated >= min_opacity) & torch.isfinite(pred_dist_sq) & torch.isfinite(pred_dist)

    # The denominator must stay attached to the graph. Detaching it drops the `+D^2/acc^2`
    # term from the gradient, which is what completes the square: with it,
    # `dL/dw_i = (t_i - mu)^2`, non-negative and translation-invariant, so a hit's weight is
    # pushed down in proportion to how far it sits from the expected depth. Without it the
    # gradient is `(t_i - mu)^2 - mu^2`, whose spurious offset is negative and grows with the
    # *squared depth*, so it pushes opacity up hardest on the most distant geometry. Measured:
    # that drove accumulated opacity to 1.0, floaters from 0.018 to 0.23 and depth bias from
    # -4.2 to -12.5 on emerald-square, while barely touching the much nearer sponza. The
    # `1/acc^2` this looks like is not a divergence -- `D` scales with `acc`, so the term is
    # just `mu^2` -- and `clamp_min` only guards rays the mask already drops.
    safe = accumulated.clamp_min(min_opacity)

    if relative:
        # Var/mu^2 = acc*M2/D^2 - 1. The same three buffers, so the `acc` here carries gradient
        # for the same reason as above -- it is what makes the term degree-zero in the weights
        # and so immune to uniform fading, rather than merely monitored for it.
        valid = valid & (pred_dist >= min_distance * accumulated)
        safe_dist_sq = (pred_dist * pred_dist).clamp_min((min_distance * min_opacity) ** 2)
        variance = (safe * pred_dist_sq / safe_dist_sq - 1.0).clamp_min(0.0)
        normalizer = 1.0
    else:
        # Non-negative in exact arithmetic (Jensen), but the difference of two accumulators
        # cancels to a small negative value on a nearly-resolved ray, which would otherwise
        # reward further sharpening with an unboundedly negative loss.
        variance = (pred_dist_sq - pred_dist * pred_dist / safe).clamp_min(0.0)
        normalizer = max(scene_extent, 1e-8) ** 2

    per_pixel = torch.where(valid, variance, torch.zeros_like(variance))
    count = valid.sum()
    return per_pixel.sum() / (count.clamp_min(1) * normalizer), count
