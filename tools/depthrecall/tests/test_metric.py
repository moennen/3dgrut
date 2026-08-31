# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the recall operator.

The metric's whole value is that it has no free parameters beyond tau, so what needs
pinning is the *definitions*: which pairs land in the denominator, which sign means what,
and that a depth map agreeing exactly with the scan scores 1.0. Each of these has a
plausible-looking wrong answer that would still produce a publishable-looking curve.
"""

from pathlib import Path

import numpy as np
import pytest

from depthrecall.io_cameras import View
from depthrecall.metric import MetricConfig, evaluate

FX = FY = 100.0
CX = CY = 50.0
WIDTH = HEIGHT = 100
PLANE_Z = 3.0


def pixel_rays(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Unit-z ray directions through every pixel centre, shape (H, W, 3)."""
    i = np.arange(width) + 0.5
    j = np.arange(height) + 0.5
    u, v = np.meshgrid(i, j)
    return np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)


def plane_depth_map(convention: str, plane_z: float = PLANE_Z) -> np.ndarray:
    """Analytic depth of the plane z = plane_z, in the requested convention."""
    rays = pixel_rays()
    if convention == "z":
        return np.full((HEIGHT, WIDTH), plane_z, dtype=np.float64)
    return np.linalg.norm(rays, axis=-1) * plane_z


def plane_points(plane_z: float = PLANE_Z, stride: int = 4) -> np.ndarray:
    """GT points lying exactly on pixel-centre rays, so sampling is exact."""
    rays = pixel_rays()[::stride, ::stride].reshape(-1, 3)
    return rays * plane_z


def make_view(tmp_path: Path, depth: np.ndarray, name: str = "0000") -> View:
    path = tmp_path / f"{name}.npy"
    np.save(path, depth)
    return View(
        name=name,
        width=WIDTH,
        height=HEIGHT,
        K=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64),
        R=np.eye(3),
        t=np.zeros(3),
        depth_path=path,
    )


def config(taus, convention: str = "ray", **kwargs) -> MetricConfig:
    return MetricConfig(taus=np.asarray(taus, dtype=np.float64), depth_convention=convention, **kwargs)


def test_exact_depth_scores_full_recall(tmp_path: Path) -> None:
    """The scan against a depth map that reproduces it must score 1.0 at every tau.

    This is the sanity rung the whole metric rests on: if it does not hold, every number
    the tool reports is offset by an unknown constant.
    """
    view = make_view(tmp_path, plane_depth_map("ray"))
    result = evaluate([view], plane_points(), config([1e-9, 1e-6, 1e-3]))
    assert result.in_frustum_pairs == plane_points().shape[0]
    assert result.no_surface_pairs == 0
    np.testing.assert_allclose(result.recall, 1.0)


def test_wrong_depth_convention_is_not_silently_tolerated(tmp_path: Path) -> None:
    """Scoring a z-depth map as ray distance must fail loudly in the numbers.

    A plane is constant in z but not in ray distance, so the error grows towards the image
    corners: exactly the shape of mistake that yields a plausible curve rather than a crash.
    """
    view = make_view(tmp_path, plane_depth_map("z"))
    points = plane_points()
    as_z = evaluate([view], points, config([1e-9], convention="z"))
    as_ray = evaluate([view], points, config([1e-9], convention="ray"))
    np.testing.assert_allclose(as_z.recall, 1.0)
    assert as_ray.recall[0] == 0.0
    # The corner-most error is the plane depth times the ray-length excess, ~0.35 here.
    assert as_ray.median_signed_delta < -0.05


def test_missing_surface_counts_as_failure_not_exclusion(tmp_path: Path) -> None:
    """Holes must stay in the denominator, or a method that renders nothing scores 1.0."""
    depth = plane_depth_map("ray")
    depth[: HEIGHT // 2, :] = 0.0  # invalid: no surface rendered
    view = make_view(tmp_path, depth)
    result = evaluate([view], plane_points(), config([1e-9, 1.0]))

    total = plane_points().shape[0]
    assert result.in_frustum_pairs == total
    assert result.no_surface_pairs == pytest.approx(total / 2, rel=0.05)
    # Even at a tau far larger than any residual, recall is capped by the hole fraction.
    assert result.recall[-1] == pytest.approx(0.5, rel=0.05)


def test_visibility_depth_excludes_a_scan_point_behind_the_visible_surface(tmp_path: Path) -> None:
    rendered = make_view(tmp_path, plane_depth_map("ray"))
    visibility = tmp_path / "visibility.npy"
    np.save(visibility, plane_depth_map("ray"))
    front = plane_points(stride=20)
    back = front.copy()
    back[:, 2] *= 2
    result = evaluate(
        [rendered],
        np.concatenate([front, back]),
        config([1e-6], visibility_depths={rendered.name: visibility}),
    )
    assert result.in_frustum_pairs == len(front)
    np.testing.assert_allclose(result.recall, 1.0)


def test_constant_offset_pins_the_sign_of_delta(tmp_path: Path) -> None:
    """A surface rendered *behind* the scan must read as too_far, and vice versa.

    The sign is the only diagnostic the metric offers about *how* a reconstruction is wrong,
    so an inverted convention would invert every conclusion drawn from it.
    """
    offset = 0.05
    behind = make_view(tmp_path, plane_depth_map("ray") + offset, name="behind")
    in_front = make_view(tmp_path, plane_depth_map("ray") - offset, name="front")
    taus = config([offset / 2, offset * 2])

    far = evaluate([behind], plane_points(), taus)
    near = evaluate([in_front], plane_points(), taus)

    # Below the offset nothing passes; above it everything does.
    assert far.recall[0] == 0.0 and far.recall[1] == pytest.approx(1.0)
    assert near.recall[0] == 0.0 and near.recall[1] == pytest.approx(1.0)
    # And the sign separates the two cases.
    assert far.too_far[0] == pytest.approx(1.0) and far.too_near[0] == 0.0
    assert near.too_near[0] == pytest.approx(1.0) and near.too_far[0] == 0.0
    assert far.median_signed_delta > 0 and near.median_signed_delta < 0


def test_recall_is_monotone_in_tau(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    depth = plane_depth_map("ray") + rng.normal(0, 0.02, size=(HEIGHT, WIDTH))
    view = make_view(tmp_path, depth)
    result = evaluate([view], plane_points(), config([0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))
    assert np.all(np.diff(result.recall) >= 0)
    assert result.recall[0] < result.recall[-1]


def test_points_outside_the_frustum_are_excluded_from_the_denominator(tmp_path: Path) -> None:
    """Unobservable points carry no information and must not dilute recall."""
    view = make_view(tmp_path, plane_depth_map("ray"))
    points = plane_points()
    behind_camera = points * np.array([1.0, 1.0, -1.0])
    far_off_axis = points + np.array([100.0, 0.0, 0.0])
    augmented = np.concatenate([points, behind_camera, far_off_axis], axis=0)

    result = evaluate([view], augmented, config([1e-9]))
    assert result.in_frustum_pairs == points.shape[0]
    np.testing.assert_allclose(result.recall, 1.0)
    assert result.per_view[0].coverage == pytest.approx(1 / 3, rel=0.02)


def test_alignment_is_applied_to_ground_truth(tmp_path: Path) -> None:
    """A similarity applied to the GT must be undone exactly by passing its inverse.

    DTU's GT lives in millimetres and the model in COLMAP units; this is the step that
    reconciles them, and getting it backwards is a silent 300x scale error.
    """
    view = make_view(tmp_path, plane_depth_map("ray"))
    points = plane_points()

    scale = 324.66
    similarity = np.eye(4)
    similarity[:3, :3] *= scale
    similarity[:3, 3] = [10.0, -5.0, 2.0]
    moved = (np.concatenate([points, np.ones((len(points), 1))], axis=1) @ similarity.T)[:, :3]

    aligned = evaluate([view], moved, config([1e-6]), alignment=np.linalg.inv(similarity))
    np.testing.assert_allclose(aligned.recall, 1.0)

    unaligned = evaluate([view], moved, config([1e-6]))
    assert unaligned.recall[0] == 0.0


def test_pooling_weights_views_by_pair_count_not_equally(tmp_path: Path) -> None:
    """Recall is over all (point, view) pairs, so a view seeing more points counts more.

    Averaging per-view recalls instead would let a view observing a handful of points swing
    the total as much as one observing thousands.
    """
    good = make_view(tmp_path, plane_depth_map("ray"), name="good")
    bad = make_view(tmp_path, plane_depth_map("ray") + 10.0, name="bad")

    dense = plane_points(stride=4)
    result = evaluate([good, bad], dense, config([1e-6]))

    per_view = np.array([v.recall[0] for v in result.per_view])
    weights = np.array([v.in_frustum for v in result.per_view], dtype=float)
    expected = float((per_view * weights).sum() / weights.sum())
    assert result.recall[0] == pytest.approx(expected)
    assert result.recall[0] == pytest.approx(0.5, rel=0.01)


def test_downsampling_reduces_the_population_reproducibly(tmp_path: Path) -> None:
    view = make_view(tmp_path, plane_depth_map("ray"))
    points = plane_points(stride=1)
    full = evaluate([view], points, config([1e-6]))
    coarse = evaluate([view], points, config([1e-6]), downsample_voxel=0.05)
    assert coarse.gt_points < full.gt_points
    assert coarse.gt_voxel_size == 0.05
    # The same request twice must give the same population, or two runs are incomparable.
    again = evaluate([view], points, config([1e-6]), downsample_voxel=0.05)
    assert again.gt_hash == coarse.gt_hash
    assert coarse.gt_hash != full.gt_hash


def test_identity_hashes_change_with_the_view_set(tmp_path: Path) -> None:
    """Two runs are only comparable over the same views and points; the hash makes a
    mismatch detectable instead of silent."""
    depth = plane_depth_map("ray")
    a = make_view(tmp_path, depth, name="a")
    b = make_view(tmp_path, depth, name="b")
    points = plane_points()
    one = evaluate([a], points, config([1e-6]))
    two = evaluate([a, b], points, config([1e-6]))
    assert one.viewset_hash != two.viewset_hash
    assert one.gt_hash == two.gt_hash


def test_max_valid_depth_and_invalid_value_mark_pixels_missing(tmp_path: Path) -> None:
    depth = plane_depth_map("ray")
    depth[0, :] = 999.0
    view = make_view(tmp_path, depth)
    points = plane_points()
    clipped = evaluate([view], points, config([1e-6], max_valid_depth=100.0))
    assert clipped.no_surface_pairs > 0
    sentinel = evaluate([view], points, config([1e-6], invalid_value=999.0))
    assert sentinel.no_surface_pairs == clipped.no_surface_pairs


def test_rejects_malformed_configuration() -> None:
    with pytest.raises(ValueError):
        MetricConfig(taus=np.array([1.0, 0.5]), depth_convention="ray")
    with pytest.raises(ValueError):
        MetricConfig(taus=np.array([1.0]), depth_convention="euclidean")
    with pytest.raises(ValueError):
        MetricConfig(taus=np.array([]), depth_convention="ray")
    with pytest.raises(ValueError):
        MetricConfig(taus=np.array([-1.0]), depth_convention="ray")


def test_pixel_centre_offset_matters_on_a_depth_gradient(tmp_path: Path) -> None:
    """The half-pixel convention is not cosmetic where depth varies across the image.

    A ramp makes the offset observable; on the flat plane used elsewhere it would cancel,
    which is how such an error survives casual testing.
    """
    ramp = np.tile(np.linspace(2.0, 4.0, WIDTH), (HEIGHT, 1))
    view = make_view(tmp_path, ramp)
    # Points placed on the pixel-centre rays of the ramp, in z convention for simplicity.
    rays = pixel_rays()
    points = (rays * ramp[..., None])[::4, ::4].reshape(-1, 3)

    correct = evaluate([view], points, config([1e-9], convention="z"))
    shifted = evaluate([view], points, config([1e-9], convention="z", pixel_center_offset=0.0))
    np.testing.assert_allclose(correct.recall, 1.0)
    assert shifted.recall[0] < 1.0
