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

"""Unit tests for the depth-variance loss.

The loss is evaluated in the algebraically simplified form `M2 - D^2/acc`, which is not
obviously the variance it claims to be, so the tests build rays from explicit weight
distributions and compare against the variance computed directly from those weights. A
formulation that dropped the `acc` weighting, or normalized once too often, would still look
plausible on real buffers and would still fall during training.
"""

from __future__ import annotations

import pytest
import torch

from threedgrut.utils.depth_variance_loss import depth_variance_loss


def _ray(weights: list[float], distances: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate one ray's buffers from explicit compositing weights."""
    w = torch.tensor(weights, dtype=torch.float64)
    t = torch.tensor(distances, dtype=torch.float64)
    shape = (1, 1, 1, 1)
    return (
        (w * t).sum().reshape(shape),
        (w * t * t).sum().reshape(shape),
        w.sum().reshape(shape),
    )


def _expected(weights: list[float], distances: list[float]) -> float:
    """Opacity-weighted variance, computed the long way from the weights themselves."""
    w = torch.tensor(weights, dtype=torch.float64)
    t = torch.tensor(distances, dtype=torch.float64)
    acc = w.sum()
    mean = (w * t).sum() / acc
    second = (w * t * t).sum() / acc
    return float(acc * (second - mean * mean))


def test_matches_the_variance_of_the_weight_distribution():
    """The simplified form must equal acc * Var[t] under the normalized weights."""
    weights = [0.5, 0.3, 0.2]
    distances = [2.0, 5.0, 9.0]
    dist, dist_sq, opacity = _ray(weights, distances)

    loss, count = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)

    assert int(count) == 1
    assert loss.item() == pytest.approx(_expected(weights, distances), rel=1e-9)


def test_a_ray_concentrated_at_one_distance_costs_nothing():
    """Zero spread is the term's optimum, and it must be reachable rather than merely small."""
    dist, dist_sq, opacity = _ray([0.9], [4.0])

    loss, count = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)

    assert int(count) == 1
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_spreading_the_same_weight_further_costs_more():
    """The ordering the term exists to impose."""
    tight, _ = depth_variance_loss(*_ray([0.5, 0.5], [4.0, 5.0]), scene_extent=1.0)
    wide, _ = depth_variance_loss(*_ray([0.5, 0.5], [1.0, 8.0]), scene_extent=1.0)

    assert wide.item() > tight.item()


def test_cancellation_cannot_drive_the_loss_negative():
    """A nearly-resolved ray must bottom out at zero, not reward further sharpening.

    In float32 the two accumulators of a sharp ray at distance cancel to a small negative
    value. Unclamped that is a *negative* loss the optimizer can pursue without bound, so
    this pins the clamp rather than the arithmetic.
    """
    # Chosen so M2 - D^2/acc is negative in float32: a single hit at a large distance, where
    # the squared accumulator loses the low bits the division needs to cancel exactly.
    distance, weight = 8.0e3, 0.75
    dist = torch.tensor([[[[distance * weight]]]], dtype=torch.float32)
    dist_sq = torch.tensor([[[[distance * distance * weight]]]], dtype=torch.float32)
    opacity = torch.tensor([[[[weight]]]], dtype=torch.float32)
    # Confirm the premise: without the clamp this case is negative.
    raw = dist_sq - dist * dist / opacity
    assert raw.item() <= 0.0, "premise stale: pick a distance where float32 cancellation undershoots"

    loss, _ = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)

    assert loss.item() >= 0.0


def test_transparent_rays_are_excluded():
    """A ray below the opacity floor describes the background, not geometry."""
    spread = _ray([0.02, 0.02], [1.0, 9.0])
    loss, count = depth_variance_loss(*spread, scene_extent=1.0, min_opacity=0.5)

    assert int(count) == 0
    assert loss.item() == pytest.approx(0.0)


def test_averages_over_valid_pixels_only():
    """Adding an excluded ray must not dilute the term the way a full mean would."""
    dist, dist_sq, opacity = _ray([0.5, 0.5], [2.0, 6.0])
    alone, _ = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)

    # Same ray, padded with a transparent one along the width axis.
    pad = torch.zeros_like(dist)
    padded, count = depth_variance_loss(
        torch.cat([dist, pad], dim=2),
        torch.cat([dist_sq, pad], dim=2),
        torch.cat([opacity, pad], dim=2),
        scene_extent=1.0,
    )

    assert int(count) == 1
    assert padded.item() == pytest.approx(alone.item(), rel=1e-9)


def test_scene_extent_normalizes_squared_units():
    """Variance carries squared distance, so the normalizer must be squared too.

    Scaling a scene by `s` scales every distance by `s` and the variance by `s^2`. A term
    normalized by `extent` rather than `extent^2` would leave a factor of `s` behind, and one
    lambda would not transfer between scenes.
    """
    weights, distances = [0.4, 0.6], [3.0, 7.0]
    scale = 4.0
    base, _ = depth_variance_loss(*_ray(weights, distances), scene_extent=1.0)
    scaled, _ = depth_variance_loss(
        *_ray(weights, [d * scale for d in distances]),
        scene_extent=scale,
    )

    assert scaled.item() == pytest.approx(base.item(), rel=1e-9)


def test_gradient_reaches_both_accumulators():
    """Both buffers must carry gradient, since the tracer differentiates both."""
    dist, dist_sq, opacity = _ray([0.5, 0.5], [2.0, 6.0])
    dist = dist.clone().requires_grad_(True)
    dist_sq = dist_sq.clone().requires_grad_(True)

    loss, _ = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)
    loss.backward()

    assert dist.grad is not None and dist.grad.abs().sum() > 0
    assert dist_sq.grad is not None and dist_sq.grad.abs().sum() > 0


def _per_hit_gradients(weights: list[float], distances: list[float]) -> tuple[list[float], list[float]]:
    """Backpropagate through the accumulation itself, to per-hit weight and distance.

    The loss only sees the three accumulated buffers, so a test on those alone cannot say what
    the term does to a *hit*. Re-accumulating them from leaves recovers the quantities the
    tracer's own backward pass will chain into.
    """
    w = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(distances, dtype=torch.float64, requires_grad=True)
    shape = (1, 1, 1, 1)
    loss, _ = depth_variance_loss(
        (w * t).sum().reshape(shape),
        (w * t * t).sum().reshape(shape),
        w.sum().reshape(shape),
        scene_extent=1.0,
    )
    loss.backward()
    return [float(g) for g in w.grad], [float(g) for g in t.grad]


@pytest.mark.parametrize(
    "weights,distances",
    [([0.6, 0.4], [3.0, 8.0]), ([0.3, 0.4, 0.3], [1.0, 2.0, 4.0]), ([0.5, 0.2, 0.1, 0.7], [2.0, 5.0, 9.0, 11.0])],
)
def test_the_term_is_the_pairwise_distortion_loss(weights, distances):
    """`M2 - D^2/acc == (1/2acc) * sum_ij w_i w_j (t_i - t_j)^2`, an identity.

    Not a reformulation for its own sake: the pairwise view is what makes the term's behaviour
    legible. Every hit is pulled towards every other with strength `w_i w_j / acc`, which is
    the mip-NeRF 360 distortion loss with a squared distance. It is also what shows the
    two-hit case to be a double well in the near opacity -- `w_near * w_far` carries a factor
    `a(1-a)`, peaking at one half -- so the term is equally satisfied by deleting a floater or
    by promoting it to fully opaque.
    """
    w = torch.tensor(weights, dtype=torch.float64)
    t = torch.tensor(distances, dtype=torch.float64)
    shape = (1, 1, 1, 1)

    loss, _ = depth_variance_loss(
        (w * t).sum().reshape(shape), (w * t * t).sum().reshape(shape), w.sum().reshape(shape), scene_extent=1.0
    )

    pairwise = sum(w[i] * w[j] * (t[i] - t[j]) ** 2 for i in range(len(weights)) for j in range(len(weights))) / (
        2 * w.sum()
    )
    assert float(loss) == pytest.approx(float(pairwise), rel=1e-12)


def test_weight_gradient_is_the_squared_distance_from_the_expected_depth():
    """`dL/dw_i = (t_i - mu)^2`: non-negative, so weight is only ever pushed down.

    This is the test that catches detaching the `acc` denominator, which drops the `+D^2/acc^2`
    term and leaves `(t_i - mu)^2 - mu^2` -- an offset that is negative and grows with the
    squared depth, so it pushes opacity *up* hardest on the most distant geometry. That
    version trained, reported a falling loss, and collapsed the scene towards the camera.
    """
    weights, distances = [0.3, 0.4, 0.3], [1.0, 2.0, 4.0]
    grad_w, _ = _per_hit_gradients(weights, distances)

    mu = sum(x * d for x, d in zip(weights, distances)) / sum(weights)
    assert grad_w == pytest.approx([(d - mu) ** 2 for d in distances], rel=1e-9)
    assert all(g >= 0.0 for g in grad_w)


def test_distance_gradient_pulls_each_hit_towards_the_expected_depth():
    """`dL/dt_i = 2*w_i*(t_i - mu)`: hits in front are pushed back, hits behind pulled in."""
    weights, distances = [0.3, 0.4, 0.3], [1.0, 2.0, 4.0]
    _, grad_t = _per_hit_gradients(weights, distances)

    mu = sum(x * d for x, d in zip(weights, distances)) / sum(weights)
    assert grad_t == pytest.approx([2 * x * (d - mu) for x, d in zip(weights, distances)], rel=1e-9)
    # Descent moves each hit towards mu, so the sign must oppose the offset.
    for weight, distance, gradient in zip(weights, distances, grad_t):
        assert gradient * (distance - mu) >= 0.0


def test_gradients_are_translation_invariant():
    """Sliding a whole ray down its own direction changes nothing about its spread.

    The detached-denominator bug failed exactly here, and visibly: shifting a ray by +100 took
    its weight gradient from about -4 to about -10500.
    """
    weights, distances = [0.3, 0.4, 0.3], [1.0, 2.0, 4.0]
    near_w, near_t = _per_hit_gradients(weights, distances)
    far_w, far_t = _per_hit_gradients(weights, [d + 100.0 for d in distances])

    assert far_w == pytest.approx(near_w, rel=1e-6)
    assert far_t == pytest.approx(near_t, rel=1e-6)


def test_a_resolved_ray_at_any_distance_has_a_negligible_gradient():
    """A sharp ray is already optimal, near or far, so nothing should push on it hard.

    Under the bug the same sharp ray gave a weight gradient of -4 at t=2 and -1600 at t=40,
    which is the scale dependence that destroyed the distant scene.
    """
    near_w, _ = _per_hit_gradients([0.3, 0.4, 0.3], [1.9, 2.0, 2.1])
    far_w, _ = _per_hit_gradients([0.3, 0.4, 0.3], [39.9, 40.0, 40.1])

    assert max(abs(g) for g in near_w) < 1e-2
    assert far_w == pytest.approx(near_w, rel=1e-6)


def test_opacity_receives_gradient():
    """The accumulated opacity is part of the term, not a constant normalizer."""
    dist, dist_sq, opacity = _ray([0.5, 0.5], [2.0, 6.0])
    opacity = opacity.clone().requires_grad_(True)

    loss, _ = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)
    loss.backward()

    # dL/dacc = +D^2/acc^2 = mu^2, the term that completes the square.
    assert float(opacity.grad) == pytest.approx(4.0**2, rel=1e-9)


def test_empty_buffer_is_rejected_rather_than_averaged():
    """The config-typo case: variance loss on, variance buffer off. Must raise, not return 0."""
    empty = torch.zeros((0,), dtype=torch.float32)
    with pytest.raises(ValueError, match="enable_depth_variance"):
        depth_variance_loss(empty, empty, empty, scene_extent=1.0)


def _barrier(relative: bool, a_far: float = 0.9, t_near: float = 2.0, t_far: float = 10.0) -> float:
    """Locate the double well's barrier in `a_near`, by bisection on the gradient's sign.

    A ray with a floater at `t_near` in front of the true surface at `t_far`. Compositing gives
    `w_near = a_near` and `w_far = a_far * (1 - a_near)`, so the term carries a factor
    `a_near * (1 - a_near)` and vanishes at both ends: deleting the floater and promoting it to
    fully opaque are equally optimal. The barrier is where descent stops doing the former and
    starts doing the latter, and it is the number that decides how much damage the term does.
    """

    def gradient(a_near: float) -> float:
        a = torch.tensor(a_near, dtype=torch.float64, requires_grad=True)
        w = torch.stack([a, a_far * (1 - a)])
        t = torch.tensor([t_near, t_far], dtype=torch.float64)
        shape = (1, 1, 1, 1)
        loss, _ = depth_variance_loss(
            (w * t).sum().reshape(shape),
            (w * t * t).sum().reshape(shape),
            w.sum().reshape(shape),
            scene_extent=1.0,
            relative=relative,
        )
        return float(torch.autograd.grad(loss, a)[0])

    lo, hi = 0.01, 0.999
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gradient(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def test_relative_form_is_invariant_to_scene_scale():
    """`Var/mu^2` is dimensionless, so scaling every distance leaves it untouched.

    This is the property the absolute form lacks: variance carries squared distance units, so
    the same *relative* spread costs `mu^2` more on a far ray than a near one. That is what
    made lambda scene-dependent, and dividing by the scene extent only fixes it between scenes.
    """
    weights = [0.5, 0.5]
    base = [9.5, 10.5]
    reference, _ = depth_variance_loss(*_ray(weights, base), scene_extent=1.0, relative=True)

    for scale in (0.2, 5.0, 100.0):
        scaled, _ = depth_variance_loss(*_ray(weights, [d * scale for d in base]), scene_extent=1.0, relative=True)
        assert float(scaled) == pytest.approx(float(reference), rel=1e-9)

    # And the absolute form is *not* invariant, growing as the square of the scale.
    absolute_1x, _ = depth_variance_loss(*_ray(weights, base), scene_extent=1.0)
    absolute_5x, _ = depth_variance_loss(*_ray(weights, [d * 5.0 for d in base]), scene_extent=1.0)
    assert float(absolute_5x) == pytest.approx(25.0 * float(absolute_1x), rel=1e-9)


def test_relative_form_closes_the_transparency_escape():
    """Degree zero in the weights: fading the scene out no longer reduces the term.

    The absolute form is degree one, so halving every weight halves it without resolving any
    ray -- a descent direction that only the photometric loss argues against. The relative form
    removes it outright rather than leaving it to be monitored.

    The claim is about the kernel, so it is asserted over the range where the kernel applies.
    Fading a ray past `MIN_ACCUMULATED_OPACITY` drops it from the term altogether, in both
    forms; that is the mask's decision, and it is not a gradient the term supplies.
    """
    weights = [0.3, 0.4, 0.3]
    distances = [1.0, 2.0, 4.0]
    reference, _ = depth_variance_loss(*_ray(weights, distances), scene_extent=1.0, relative=True)

    for fade in (0.9, 0.7, 0.5):
        faded, count = depth_variance_loss(
            *_ray([w * fade for w in weights], distances), scene_extent=1.0, relative=True
        )
        assert int(count) == 1, "the ray must stay above the opacity floor for the claim to be about fading"
        assert float(faded) == pytest.approx(float(reference), rel=1e-9)

    half, _ = depth_variance_loss(*_ray([w * 0.5 for w in weights], distances), scene_extent=1.0)
    full, _ = depth_variance_loss(*_ray(weights, distances), scene_extent=1.0)
    assert float(half) == pytest.approx(0.5 * float(full), rel=1e-9)


def test_relative_form_shrinks_the_basin_that_locks_rays_onto_floaters():
    """Dividing by `mu^2` biases the double well against the near collapse.

    Committing to a near floater is precisely what makes `mu` small, so the relative form
    charges for it. The barrier moves from roughly half the axis to four fifths of it, which
    is the mechanism by which this is expected to do less damage than the absolute form. Note
    what it does *not* do: both wells are still exactly zero, so the degeneracy survives and
    the term still needs a depth anchor to be more than a sharpener.
    """
    absolute = _barrier(relative=False)
    relative = _barrier(relative=True)

    assert absolute == pytest.approx(0.487, abs=0.02)
    assert relative == pytest.approx(0.818, abs=0.02)
    assert relative > absolute + 0.25


def test_relative_form_drops_rays_whose_expected_depth_is_below_the_floor():
    """`dL/dt` scales as `1/mu`, so a ray at almost zero depth would dominate the batch.

    The opacity mask does not catch these: a ray can be entirely opaque and still very close.
    """
    dist, dist_sq, opacity = _ray([0.5, 0.5], [1e-6, 3e-6])

    loss, count = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0, relative=True)

    assert int(count) == 0
    assert float(loss) == 0.0
    assert torch.isfinite(loss)
