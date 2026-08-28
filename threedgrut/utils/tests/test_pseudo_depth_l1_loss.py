# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the regression (L1) pseudo-depth loss.

Unlike the ordinal term, this one reads the prior's values, which puts three silent failures
within reach: applying the affine in the wrong space, comparing z against a renderer that emits
ray distance, and letting an unalignable frame contribute a NaN that poisons every gradient in
the batch. Each has a test here, because none of them changes the shape of a loss curve.
"""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.utils.pseudo_depth_loss import aligned_prior_distance, compute_pseudo_depth_l1_loss


def _buffers(depth: torch.Tensor, opacity: float | torch.Tensor = 1.0):
    """Tracer-shaped [1, H, W, 1] buffers, with `pred_dist` opacity-premultiplied."""
    depth = depth[None, ..., None].float()
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.full_like(depth, float(opacity))
    else:
        opacity = opacity[None, ..., None].float()
    return depth * opacity, opacity


def _rays(height: int, width: int, focal: float = 4.0) -> torch.Tensor:
    """Camera-space pinhole ray directions, so off-axis pixels genuinely have z < |d|."""
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - (height - 1) / 2,
        torch.arange(width, dtype=torch.float32) - (width - 1) / 2,
        indexing="ij",
    )
    directions = torch.stack([xs / focal, ys / focal, torch.ones_like(xs)], dim=-1)
    return (directions / directions.norm(dim=-1, keepdim=True))[None]


def _affine(scale: float, offset: float) -> torch.Tensor:
    return torch.tensor([[scale, offset]], dtype=torch.float32)


# -- the z-to-distance conversion ----------------------------------------------------------------


def test_aligned_prior_is_ray_distance_not_z():
    """An off-axis pixel's target must exceed the aligned z by exactly 1/cos(theta)."""
    rays = _rays(3, 3, focal=1.0)  # 45 degrees at the edge, so the distinction is large
    prior = torch.full((1, 3, 3, 1), 2.0)
    distance = aligned_prior_distance(prior, _affine(1.0, 0.0), rays)

    centre = distance[0, 1, 1, 0]
    corner = distance[0, 0, 0, 0]
    assert centre == pytest.approx(2.0)  # on-axis, distance == z
    # The corner ray is (-1, -1, 1)/sqrt(3), so cos(theta) = 1/sqrt(3).
    assert corner == pytest.approx(2.0 * math.sqrt(3.0), rel=1e-5)
    assert corner > centre, "a z-depth target would make these equal, which is the bug"


def test_affine_is_applied_before_the_conversion():
    """Scaling in z then converting is not the same as converting then scaling, and z is right.

    The offset is what separates the two: it must be added to z, where it is a constant, rather
    than to the distance, where the per-pixel cosine would have already scaled it.
    """
    rays = _rays(3, 3, focal=1.0)
    prior = torch.full((1, 3, 3, 1), 2.0)
    scale, offset = 3.0, 5.0
    got = aligned_prior_distance(prior, _affine(scale, offset), rays)

    to_z = rays[..., 2:3].abs()
    fitted_in_z = (scale * prior + offset) / to_z
    fitted_in_distance = scale * (prior / to_z) + offset
    torch.testing.assert_close(got, fitted_in_z)
    assert not torch.allclose(got, fitted_in_distance), "the two spaces must not be conflated"


def test_non_positive_aligned_depth_is_rejected():
    """A negative offset can push the prior's nearest values behind the camera."""
    rays = _rays(2, 2)
    prior = torch.tensor([[1.0, 5.0], [10.0, 20.0]])[None, ..., None]
    distance = aligned_prior_distance(prior, _affine(1.0, -6.0), rays)
    assert torch.isnan(distance[0, 0, 0, 0]) and torch.isnan(distance[0, 0, 1, 0])
    assert torch.isfinite(distance[0, 1, 0, 0]) and torch.isfinite(distance[0, 1, 1, 0])


# -- the loss ------------------------------------------------------------------------------------


def test_perfectly_aligned_prior_costs_nothing():
    rays = _rays(4, 4)
    prior = torch.linspace(1.0, 4.0, 16).reshape(4, 4)[None, ..., None]
    target = aligned_prior_distance(prior, _affine(2.0, 1.0), rays)
    pred_dist, pred_opacity = _buffers(target[0, ..., 0])

    loss = compute_pseudo_depth_l1_loss(
        pred_dist, pred_opacity, prior, _affine(2.0, 1.0), rays, scene_extent=1.0, quantity="depth"
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_the_mean_absolute_gap_over_scene_extent():
    rays = _rays(4, 4, focal=1e6)  # effectively orthographic, so distance == z and the gap is exact
    prior = torch.full((1, 4, 4, 1), 3.0)
    pred_dist, pred_opacity = _buffers(torch.full((4, 4), 5.0))

    extent = 4.0
    loss = compute_pseudo_depth_l1_loss(
        pred_dist, pred_opacity, prior, _affine(1.0, 0.0), rays, scene_extent=extent, quantity="depth"
    )
    assert loss.item() == pytest.approx((5.0 - 3.0) / extent, rel=1e-4)


def test_gradient_pulls_the_render_towards_the_prior():
    rays = _rays(4, 4, focal=1e6)
    prior = torch.full((1, 4, 4, 1), 3.0)
    pred_dist, pred_opacity = _buffers(torch.full((4, 4), 5.0))
    pred_dist.requires_grad_(True)

    compute_pseudo_depth_l1_loss(
        pred_dist, pred_opacity, prior, _affine(1.0, 0.0), rays, scene_extent=1.0, quantity="depth"
    ).backward()
    # Render is too far, so the loss must decrease when the rendered distance decreases.
    assert (pred_dist.grad > 0).all()


def test_unaligned_frame_contributes_nothing_and_no_nan_gradient():
    """A frame with no alignment carries NaN coefficients, which must not reach the gradient."""
    rays = _rays(4, 4)
    prior = torch.full((1, 4, 4, 1), 3.0)
    pred_dist, pred_opacity = _buffers(torch.full((4, 4), 5.0))
    pred_dist.requires_grad_(True)

    loss = compute_pseudo_depth_l1_loss(
        pred_dist,
        pred_opacity,
        prior,
        _affine(float("nan"), float("nan")),
        rays,
        scene_extent=1.0,
        quantity="depth",
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred_dist.grad).all() and (pred_dist.grad == 0).all()


def test_a_mix_of_aligned_and_unaligned_frames_keeps_the_aligned_one():
    """The batch dimension must not let one unalignable frame silence the others."""
    rays = _rays(4, 4, focal=1e6).expand(2, -1, -1, -1).contiguous()
    prior = torch.full((2, 4, 4, 1), 3.0)
    pred_dist = torch.full((2, 4, 4, 1), 5.0, requires_grad=True)
    pred_opacity = torch.ones(2, 4, 4, 1)
    affine = torch.tensor([[1.0, 0.0], [float("nan"), float("nan")]])

    loss = compute_pseudo_depth_l1_loss(
        pred_dist, pred_opacity, prior, affine, rays, scene_extent=1.0, quantity="depth"
    )
    assert loss.item() == pytest.approx(2.0, rel=1e-4)
    loss.backward()
    assert (pred_dist.grad[0] > 0).all(), "the aligned frame must still be supervised"
    assert (pred_dist.grad[1] == 0).all(), "the unaligned frame must contribute no gradient"


def test_transparent_pixels_are_not_supervised():
    rays = _rays(4, 4, focal=1e6)
    prior = torch.full((1, 4, 4, 1), 3.0)
    opacity = torch.ones(4, 4)
    opacity[:, :2] = 0.0  # half the frame renders no surface
    pred_dist, pred_opacity = _buffers(torch.full((4, 4), 5.0), opacity)

    loss = compute_pseudo_depth_l1_loss(
        pred_dist, pred_opacity, prior, _affine(1.0, 0.0), rays, scene_extent=1.0, quantity="depth"
    )
    # Averaged over the confident half only, so the empty pixels neither dilute nor contribute.
    assert loss.item() == pytest.approx(2.0, rel=1e-4)


def test_disparity_prior_is_refused():
    rays = _rays(2, 2)
    prior = torch.ones(1, 2, 2, 1)
    pred_dist, pred_opacity = _buffers(torch.ones(2, 2))
    with pytest.raises(ValueError, match="depth prior"):
        compute_pseudo_depth_l1_loss(
            pred_dist, pred_opacity, prior, _affine(1.0, 0.0), rays, scene_extent=1.0, quantity="disparity"
        )


def test_mismatched_prior_resolution_is_refused():
    rays = _rays(4, 4)
    prior = torch.ones(1, 2, 2, 1)
    pred_dist, pred_opacity = _buffers(torch.ones(4, 4))
    with pytest.raises(ValueError, match="does not match the rendered"):
        compute_pseudo_depth_l1_loss(
            pred_dist, pred_opacity, prior, _affine(1.0, 0.0), rays, scene_extent=1.0, quantity="depth"
        )
