# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from threedgrut.utils.depth_normal_metrics import (
    depth_metrics,
    expected_depth,
    geometry_metrics,
    normal_metrics,
    reference_depth_validity,
    world_view_dirs,
)

SENTINEL = 1e10


def test_perfect_depth_scores_zero_error() -> None:
    gt = torch.tensor([[1.0, 2.0, 3.0]])
    metrics = depth_metrics(gt.clone(), gt, torch.ones_like(gt, dtype=torch.bool))

    assert metrics["depth_abs_rel"] == pytest.approx(0.0)
    assert metrics["depth_rmse"] == pytest.approx(0.0)
    assert metrics["depth_delta1"] == pytest.approx(1.0)
    assert metrics["depth_valid_px"] == 3


def test_sky_sentinel_is_excluded_from_depth() -> None:
    """The sky is stored as ~1e10; scoring it would swamp every other pixel."""
    gt = torch.tensor([[2.0, SENTINEL]])
    pred = torch.tensor([[2.0, 5.0]])
    metrics = depth_metrics(pred, gt, torch.ones_like(gt, dtype=torch.bool))

    assert metrics["depth_valid_px"] == 1
    assert metrics["depth_abs_rel"] == pytest.approx(0.0)


def test_non_finite_and_negative_reference_depth_is_excluded() -> None:
    gt = torch.tensor([[4.0, float("nan"), float("inf"), -1.0, 0.0]])
    assert reference_depth_validity(gt).sum() == 1


def test_depth_bias_reports_signed_direction() -> None:
    """Under- and over-estimation must be distinguishable, not folded into a magnitude."""
    gt = torch.tensor([[10.0, 10.0]])
    under = depth_metrics(torch.tensor([[8.0, 8.0]]), gt, torch.ones_like(gt, dtype=torch.bool))
    over = depth_metrics(torch.tensor([[12.0, 12.0]]), gt, torch.ones_like(gt, dtype=torch.bool))

    assert under["depth_bias"] == pytest.approx(-2.0)
    assert over["depth_bias"] == pytest.approx(2.0)
    assert under["depth_mae"] == pytest.approx(over["depth_mae"])


def test_depth_metrics_are_absolute_not_scale_invariant() -> None:
    """A globally rescaled prediction must be penalized, or world drift goes unnoticed."""
    gt = torch.tensor([[1.0, 2.0, 4.0]])
    metrics = depth_metrics(gt * 1.5, gt, torch.ones_like(gt, dtype=torch.bool))
    assert metrics["depth_abs_rel"] == pytest.approx(0.5)


def test_delta_thresholds_are_ordered_and_bounded() -> None:
    gt = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    pred = torch.tensor([[1.0, 1.2, 1.5, 3.0]])
    metrics = depth_metrics(pred, gt, torch.ones_like(gt, dtype=torch.bool))

    assert metrics["depth_delta1"] <= metrics["depth_delta2"] <= metrics["depth_delta3"]
    assert metrics["depth_delta1"] == pytest.approx(0.5)  # 1.0 and 1.2 within 1.25


def test_unrendered_pixels_count_as_misses_not_omissions() -> None:
    """A ray that renders nothing where a surface exists is a failure, not a free pass.

    Excluding it from the denominator would let a model raise delta1 by rendering less.
    """
    gt = torch.tensor([[1.0, 1.0]])
    pred = torch.tensor([[1.0, 0.0]])
    metrics = depth_metrics(pred, gt, torch.ones_like(gt, dtype=torch.bool))

    assert metrics["depth_valid_px"] == 2
    assert metrics["depth_delta1"] == pytest.approx(0.5)
    assert metrics["depth_covered_frac"] == pytest.approx(0.5)


def test_empty_mask_returns_no_metrics() -> None:
    gt = torch.tensor([[SENTINEL, SENTINEL]])
    assert depth_metrics(torch.ones_like(gt), gt, torch.ones_like(gt, dtype=torch.bool)) == {}


def test_expected_depth_divides_out_accumulated_opacity() -> None:
    """The tracer accumulates alpha-weighted depth without normalizing by the weight."""
    surface_depth = 4.0
    opacity = torch.tensor([[0.8]])
    raw = torch.tensor([[surface_depth * 0.8]])

    depth, confident = expected_depth(raw, opacity)

    assert bool(confident.all())
    assert float(depth) == pytest.approx(surface_depth)


def test_expected_depth_flags_transparent_rays() -> None:
    raw = torch.tensor([[1.0, 1.0]])
    opacity = torch.tensor([[0.9, 0.01]])
    depth, confident = expected_depth(raw, opacity, min_opacity=0.5)

    assert confident.tolist() == [[True, False]]
    assert float(depth[0, 1]) == 0.0  # no surface, so no invented depth


def test_zero_opacity_does_not_produce_infinity() -> None:
    depth, confident = expected_depth(torch.zeros(1, 1), torch.zeros(1, 1))
    assert torch.isfinite(depth).all()
    assert not bool(confident.any())


def test_perfect_normals_score_zero_degrees() -> None:
    gt = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    metrics = normal_metrics(gt.clone(), gt, torch.ones(1, 2, dtype=torch.bool))

    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["normal_pct_11_25"] == pytest.approx(1.0)
    assert metrics["normal_valid_px"] == 2


def test_normal_angle_is_measured_correctly() -> None:
    gt = torch.tensor([[[0.0, 0.0, 1.0]]])
    pred = torch.tensor([[[0.0, math.sin(math.radians(30)), math.cos(math.radians(30))]]])
    metrics = normal_metrics(pred, gt, torch.ones(1, 1, dtype=torch.bool))
    assert metrics["normal_mean_deg"] == pytest.approx(30.0, abs=1e-4)


def test_inverted_normals_are_penalized_not_forgiven() -> None:
    """An unsigned (absolute-cosine) error would score a flipped normal as perfect."""
    gt = torch.tensor([[[0.0, 0.0, 1.0]]])
    metrics = normal_metrics(-gt, gt, torch.ones(1, 1, dtype=torch.bool))
    assert metrics["normal_mean_deg"] == pytest.approx(180.0, abs=1e-4)


def test_normals_are_normalized_before_comparison() -> None:
    """The buffer is alpha-premultiplied, so magnitude must not affect the angle."""
    gt = torch.tensor([[[0.0, 0.0, 1.0]]])
    metrics = normal_metrics(gt * 0.03, gt, torch.ones(1, 1, dtype=torch.bool))
    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-5)


def test_rays_without_a_normal_are_skipped() -> None:
    """Zero-length predictions mean "no surface hit" and have no direction to score."""
    gt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    pred = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    metrics = normal_metrics(pred, gt, torch.ones(1, 2, dtype=torch.bool))
    assert metrics["normal_valid_px"] == 1


def test_gaps_in_the_reference_normals_are_skipped() -> None:
    gt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    pred = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    metrics = normal_metrics(pred, gt, torch.ones(1, 2, dtype=torch.bool))
    assert metrics["normal_valid_px"] == 1
    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-6)


def test_median_resists_a_single_outlier_that_moves_the_mean() -> None:
    gt = torch.zeros(1, 5, 3)
    gt[..., 2] = 1.0
    pred = gt.clone()
    pred[0, 4] = torch.tensor([0.0, 0.0, -1.0])  # one fully inverted pixel
    metrics = normal_metrics(pred, gt, torch.ones(1, 5, dtype=torch.bool))

    assert metrics["normal_median_deg"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["normal_mean_deg"] == pytest.approx(36.0, abs=1e-3)


def test_geometry_metrics_combines_both_and_tolerates_missing_reference() -> None:
    depth_gt = torch.full((1, 2, 2, 1), 3.0)
    normal_gt = torch.zeros(1, 2, 2, 3)
    normal_gt[..., 2] = 1.0
    outputs = {
        "pred_dist": torch.full((1, 2, 2, 1), 3.0),
        "pred_opacity": torch.ones(1, 2, 2, 1),
        "pred_normals": normal_gt.clone(),
    }

    both = geometry_metrics(outputs, depth_gt, normal_gt)
    assert both["depth_abs_rel"] == pytest.approx(0.0)
    assert both["normal_mean_deg"] == pytest.approx(0.0, abs=1e-6)

    assert "normal_mean_deg" not in geometry_metrics(outputs, depth_gt, None)
    assert "depth_abs_rel" not in geometry_metrics(outputs, None, normal_gt)
    assert geometry_metrics(outputs, None, None) == {}


def test_geometry_metrics_excludes_sky_from_normals_via_reference_depth() -> None:
    """Reference normals in the sky are meaningless even when they are unit length."""
    depth_gt = torch.tensor([[[[3.0], [SENTINEL]]]])
    normal_gt = torch.zeros(1, 1, 2, 3)
    normal_gt[..., 2] = 1.0
    outputs = {
        "pred_dist": torch.tensor([[[[3.0], [3.0]]]]),
        "pred_opacity": torch.ones(1, 1, 2, 1),
        # Correct on the surface, wrong in the sky.
        "pred_normals": torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]]),
    }
    metrics = geometry_metrics(outputs, depth_gt, normal_gt)

    assert metrics["normal_valid_px"] == 1
    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-6)


def test_viewdir_control_is_zero_gain_for_the_no_geometry_cheat() -> None:
    """Pointing every normal back down the view ray must show exactly no gain.

    This is the degenerate predictor the control exists to expose: it uses no geometry,
    yet scores well simply because both vectors are forced into the same hemisphere.
    """
    view = torch.tensor([[[[0.0, 0.3, 0.95], [0.5, 0.0, 0.87]]]])
    view = view / view.norm(dim=-1, keepdim=True)
    gt = torch.tensor([[[[0.0, 0.0, -1.0], [0.1, 0.2, -0.97]]]])
    gt = gt / gt.norm(dim=-1, keepdim=True)

    metrics = normal_metrics(-view.squeeze(0), gt.squeeze(0), torch.ones(1, 2, dtype=torch.bool), view.squeeze(0))

    # acos is steep near zero, so float32 inputs leave a few micro-degrees of slack.
    assert metrics["normal_gain_vs_viewdir_deg"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["normal_viewdir_control_deg"] == pytest.approx(metrics["normal_mean_deg"], abs=1e-4)


def test_viewdir_control_reports_negative_gain_when_the_buffer_is_worse() -> None:
    """A buffer that loses to the cheat must be reported as losing, not merely as ~48 degrees."""
    view = torch.tensor([[[0.0, 0.0, 1.0]]])
    gt = torch.tensor([[[0.0, 0.2, -0.98]]])
    gt = gt / gt.norm(dim=-1, keepdim=True)
    worse = torch.tensor([[[0.0, 0.8, -0.6]]])

    metrics = normal_metrics(worse, gt, torch.ones(1, 1, dtype=torch.bool), view)
    assert metrics["normal_gain_vs_viewdir_deg"] < 0.0


def test_viewdir_control_reports_positive_gain_for_a_real_normal() -> None:
    view = torch.tensor([[[0.0, 0.0, 1.0]]])
    gt = torch.tensor([[[0.0, 0.6, -0.8]]])
    metrics = normal_metrics(gt.clone(), gt, torch.ones(1, 1, dtype=torch.bool), view)

    assert metrics["normal_mean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["normal_gain_vs_viewdir_deg"] > 30.0


def test_control_is_absent_when_no_view_direction_is_given() -> None:
    gt = torch.tensor([[[0.0, 0.0, 1.0]]])
    metrics = normal_metrics(gt.clone(), gt, torch.ones(1, 1, dtype=torch.bool))
    assert "normal_viewdir_control_deg" not in metrics


def test_world_view_dirs_rotates_into_the_world_frame() -> None:
    """A 90 degree yaw must move a forward ray onto the world x axis, not leave it alone."""
    rays = torch.tensor([[[[0.0, 0.0, 2.0]]]])  # deliberately not unit length
    yaw = torch.eye(4).unsqueeze(0)
    yaw[0, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    dirs = world_view_dirs(rays, yaw)
    assert torch.allclose(dirs[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(1, 1, 1))
