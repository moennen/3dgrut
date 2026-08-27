# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ordinal pseudo-depth loss.

The loss is one-sided and reads only signs, so its value is a weak witness: the tests below
pin the *sign convention* (disparity decreases with depth) and the *gradient direction*, which
are the two things that can be inverted while leaving a plausible-looking loss curve.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.utils.pseudo_depth_loss import compute_pseudo_depth_order_loss, prior_scale, sample_pair_offset

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _ScriptedRng:
    """Yields a fixed sequence from `randint`, so the pair offset under test is deterministic."""

    def __init__(self, *values: int):
        self._values = list(values)

    def randint(self, low: int, high: int) -> int:
        return self._values.pop(0)


def _buffers(depth: torch.Tensor, opacity: float | torch.Tensor = 1.0):
    """Tracer-shaped [1, H, W, 1] buffers for a given per-pixel depth.

    `pred_dist` is opacity-premultiplied, which is what `expected_depth` undoes.
    """
    depth = depth[None, ..., None].float()
    if not isinstance(opacity, torch.Tensor):
        opacity = torch.full_like(depth, float(opacity))
    else:
        opacity = opacity[None, ..., None].float()
    return depth * opacity, opacity


def _ramp(height: int = 8, width: int = 8) -> torch.Tensor:
    """Depth increasing along x, so nearer pixels sit at small x."""
    return torch.arange(width, dtype=torch.float32).expand(height, width) + 1.0


def _horizontal_offset_rng() -> _ScriptedRng:
    return _ScriptedRng(0, 1)  # dy = 0, dx = 1


def test_agreeing_order_costs_nothing():
    depth = _ramp()
    # Disparity is inverse depth, so it must *decrease* where depth increases.
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth)
    loss = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    assert loss.item() == pytest.approx(0.0)


def test_contradicting_order_is_penalised():
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    # Render orders the scene backwards relative to the prior.
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    loss = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    assert loss.item() > 0.0


def test_reading_the_prior_as_depth_inverts_the_loss():
    """The disparity/depth confusion must not be silent: it exactly swaps the two outcomes."""
    depth = _ramp()
    consistent = (10.0 - depth)[None, ..., None]  # correct: disparity
    mistaken = depth[None, ..., None]  # wrong: the prior read as a distance
    pred_dist, pred_opacity = _buffers(depth)
    good = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, consistent, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    bad = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, mistaken, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    assert good.item() == pytest.approx(0.0)
    assert bad.item() > 0.0


def test_a_depth_prior_agrees_where_a_disparity_prior_would_not():
    """`quantity` is the whole difference between the two prior families.

    The same map read as depth and as disparity must give exactly opposite verdicts, which is
    what makes declaring it per backend load-bearing rather than decorative.
    """
    depth = _ramp()
    prior = depth[None, ..., None]  # increases with distance, i.e. a depth prior
    pred_dist, pred_opacity = _buffers(depth)
    as_depth = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="depth", rng=_horizontal_offset_rng()
    )
    as_disparity = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    assert as_depth.item() == pytest.approx(0.0)
    assert as_disparity.item() > 0.0


def test_quantity_must_be_stated():
    """No default, so a caller that has not thought about the convention cannot get one wrong."""
    parameter = inspect.signature(compute_pseudo_depth_order_loss).parameters["quantity"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_unknown_quantity_is_rejected():
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth)
    with pytest.raises(ValueError, match="unknown prior quantity"):
        compute_pseudo_depth_order_loss(pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="inverse_depth")


def test_invariant_to_monotone_transforms_of_the_prior():
    """Only the ordering is read, so any increasing map of the prior gives the same loss."""
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    baseline = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", gate=0.0, rng=_horizontal_offset_rng()
    )
    for transformed in (3.0 * prior + 7.0, prior.exp(), prior**3):
        loss = compute_pseudo_depth_order_loss(
            pred_dist,
            pred_opacity,
            transformed,
            scene_extent=1.0,
            quantity="disparity",
            gate=0.0,
            rng=_horizontal_offset_rng(),
        )
        assert loss.item() == pytest.approx(baseline.item(), rel=1e-5)


def test_gradient_pushes_the_contradicted_pixel_towards_agreement():
    """A pixel the prior says is farther than its neighbour must be pushed outwards."""
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    pred_dist.requires_grad_(True)
    loss = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    loss.backward()
    grad = pred_dist.grad[0, ..., 0]
    # Descent moves depth against the gradient. The prior's disparity is largest at column 0, so
    # it calls that column the *nearest*; the reversed render puts it farthest, and agreement
    # means pulling it in -- which requires a positive gradient. The far end mirrors this.
    assert torch.all(grad[:, 0] > 0)
    assert torch.all(grad[:, -1] < 0)


def test_gate_discards_pairs_tied_relative_to_the_maps_spread():
    """The gate is relative to the prior's own IQR, so it survives the affine ambiguity.

    A linear ramp over 8 columns has an IQR of ~3.5 and a neighbour gap of 1, i.e. 0.29 of the
    spread: a gate below that keeps the pair and a gate above it drops the pair.
    """
    depth = _ramp()
    prior = -torch.arange(8, dtype=torch.float32).expand(8, 8)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    kept = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", gate=0.2, rng=_horizontal_offset_rng()
    )
    dropped = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", gate=0.5, rng=_horizontal_offset_rng()
    )
    assert kept.item() > 0.0
    assert dropped.item() == pytest.approx(0.0)


def test_gate_defaults_to_off():
    """Gating measured worse, so the default is a result and not a placeholder.

    Offline it looked compelling -- the prior's ordinal agreement with ground truth rises from
    84% to 97% -- but trained over 3 seeds it costs emerald-square its whole 10% depth gain,
    because a large prior gap selects the long-range pairs a monocular prior is worst at. Pinned
    so that turning it back on has to be a deliberate, re-measured decision.
    """
    assert inspect.signature(compute_pseudo_depth_order_loss).parameters["gate"].default == 0.0
    assert OmegaConf.load(_REPO_ROOT / "configs" / "base_gs.yaml").loss.pseudo_depth_gate == 0.0


def test_gate_is_invariant_to_rescaling_the_prior():
    """Scaling the whole prior must not change which pairs are gated; only ratios matter."""
    depth = _ramp()
    prior = -torch.arange(8, dtype=torch.float32).expand(8, 8)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    baseline = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", gate=0.2, rng=_horizontal_offset_rng()
    )
    rescaled = compute_pseudo_depth_order_loss(
        pred_dist,
        pred_opacity,
        1e-4 * prior + 5.0,
        scene_extent=1.0,
        quantity="disparity",
        gate=0.2,
        rng=_horizontal_offset_rng(),
    )
    assert rescaled.item() == pytest.approx(baseline.item(), rel=1e-5)


def test_unconfident_pixels_are_excluded():
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1), opacity=torch.zeros(8, 8))
    loss = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    # No pair survives, so the clamped division must yield a clean zero rather than a NaN.
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)


def test_scene_extent_scales_the_loss():
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth.flip(-1))
    unit = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    scaled = compute_pseudo_depth_order_loss(
        pred_dist, pred_opacity, prior, scene_extent=4.0, quantity="disparity", rng=_horizontal_offset_rng()
    )
    assert scaled.item() == pytest.approx(unit.item() / 4.0, rel=1e-6)


def test_shape_mismatch_is_rejected():
    depth = _ramp()
    pred_dist, pred_opacity = _buffers(depth)
    with pytest.raises(ValueError, match="resampled to the training resolution"):
        compute_pseudo_depth_order_loss(
            pred_dist,
            pred_opacity,
            torch.zeros(1, 4, 4, 1),
            scene_extent=1.0,
            quantity="disparity",
            rng=_horizontal_offset_rng(),
        )


def test_non_positive_scene_extent_is_rejected():
    depth = _ramp()
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth)
    with pytest.raises(ValueError, match="scene_extent must be positive"):
        compute_pseudo_depth_order_loss(pred_dist, pred_opacity, prior, scene_extent=0.0, quantity="disparity")


def test_pairs_do_not_wrap_around_the_image():
    """Wrapping would pair opposite edges, inventing a disagreement the prior never claimed."""
    height = width = 6
    # Monotone in x for both, so every *cropped* pair agrees and the only way to see a non-zero
    # loss is to compare the last column against the first.
    depth = _ramp(height, width)
    prior = (10.0 - depth)[None, ..., None]
    pred_dist, pred_opacity = _buffers(depth)
    for dy, dx in ((0, 5), (0, -5), (5, 0), (3, 3)):
        loss = compute_pseudo_depth_order_loss(
            pred_dist, pred_opacity, prior, scene_extent=1.0, quantity="disparity", gate=0.0, rng=_ScriptedRng(dy, dx)
        )
        assert loss.item() == pytest.approx(0.0), f"offset {(dy, dx)} wrapped around"


def test_prior_scale_is_the_interquartile_range():
    values = torch.arange(101, dtype=torch.float32)
    assert prior_scale(values).item() == pytest.approx(50.0, abs=1e-4)


def test_prior_scale_ignores_non_finite_entries():
    values = torch.tensor([float("nan"), 0.0, 1.0, 2.0, 3.0, 4.0, float("inf")])
    assert torch.isfinite(prior_scale(values))


def test_prior_scale_of_empty_input_is_zero():
    assert prior_scale(torch.tensor([float("nan")])).item() == pytest.approx(0.0)


def test_sample_pair_offset_is_never_degenerate_and_stays_in_range():
    import random

    rng = random.Random(0)
    for _ in range(200):
        dy, dx = sample_pair_offset(100, 200, 0.05, rng)
        assert (dy, dx) != (0, 0)
        assert abs(dy) <= 10 and abs(dx) <= 10
