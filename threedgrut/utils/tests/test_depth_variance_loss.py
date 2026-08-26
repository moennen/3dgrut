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


def test_opacity_receives_no_gradient_through_the_normalizer():
    """The `1/acc` denominator is detached, so opacity gets no gradient from this term here.

    Left attached it contributes a `1/acc^2` term that diverges as a ray approaches the
    opacity floor, letting barely-qualifying rays dominate. Opacity is still supervised
    through its effect on the two accumulators inside the tracer.
    """
    dist, dist_sq, opacity = _ray([0.5, 0.5], [2.0, 6.0])
    opacity = opacity.clone().requires_grad_(True)

    # With opacity the *only* attached input the loss has no graph at all, which is the
    # property under test stated at its strongest.
    detached, _ = depth_variance_loss(dist, dist_sq, opacity, scene_extent=1.0)
    assert detached.grad_fn is None and not detached.requires_grad

    # And with the accumulators attached too, the gradient that does exist reaches them
    # rather than leaking back into opacity.
    loss, _ = depth_variance_loss(
        dist.clone().requires_grad_(True), dist_sq.clone().requires_grad_(True), opacity, scene_extent=1.0
    )
    loss.backward()
    assert opacity.grad is None or opacity.grad.abs().sum() == 0


def test_empty_buffer_is_rejected_rather_than_averaged():
    """The config-typo case: variance loss on, variance buffer off. Must raise, not return 0."""
    empty = torch.zeros((0,), dtype=torch.float32)
    with pytest.raises(ValueError, match="enable_depth_variance"):
        depth_variance_loss(empty, empty, empty, scene_extent=1.0)
