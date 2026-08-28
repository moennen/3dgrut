# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aligning a monocular prior to COLMAP's sparse points.

Everything here is built forwards from a known camera and a known prior: points are placed in
the world, projected by hand, and the affine that generated the prior is then recovered. That
matters because the failure modes of this code are all *plausible* -- a missing downscale factor
or a missing world scale produces a perfectly smooth depth map that is uniformly wrong, and only
a comparison against an independently known depth reveals it. `observation_consistency` is that
comparison, so the tests below check it fires on each of those mistakes rather than just
checking it is small when everything is right.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from threedgrut.datasets.sparse_depth_alignment import (
    MIN_OBSERVATIONS,
    fit_frame_alignment,
    fit_trimmed_affine,
    observation_consistency,
    read_points3d_with_ids,
    sparse_observations,
)
from threedgrut.datasets.utils import Image

# A camera at the origin looking down +z with no rotation, so a world point's z *is* its camera z
# and hand-computing the expected values needs no matrix algebra.
IDENTITY_QVEC = np.array([1.0, 0.0, 0.0, 0.0])
ZERO_TVEC = np.array([0.0, 0.0, 0.0])

FOCAL = 100.0
FULL_SHAPE = (64, 64)  # COLMAP's resolution


def _world_points(count: int = 40) -> dict[int, np.ndarray]:
    """Points spread across the image at a range of depths, one per grid cell.

    Depth varies with the point index so that a fit against them is determined rather than
    degenerate: every point at one depth would leave the affine's scale unconstrained.
    """
    points = {}
    for index in range(count):
        col, row = 8 + (index % 8) * 6, 8 + (index // 8) * 6
        depth = 2.0 + index * 0.5
        # Invert the pinhole projection for a principal point at the image centre.
        x = (col - FULL_SHAPE[1] / 2) * depth / FOCAL
        y = (row - FULL_SHAPE[0] / 2) * depth / FOCAL
        points[index + 1] = np.array([x, y, depth])
    return points


def _image(points: dict[int, np.ndarray], shape=FULL_SHAPE) -> Image:
    """A COLMAP image record whose `xys` are the exact projections of `points`, full-resolution."""
    ids, xys = [], []
    for point_id, xyz in sorted(points.items()):
        xys.append([xyz[0] * FOCAL / xyz[2] + shape[1] / 2, xyz[1] * FOCAL / xyz[2] + shape[0] / 2])
        ids.append(point_id)
    return Image(
        id=1,
        qvec=IDENTITY_QVEC,
        tvec=ZERO_TVEC,
        camera_id=1,
        name="frame.png",
        xys=np.asarray(xys, dtype=np.float64),
        point3D_ids=np.asarray(ids, dtype=np.int64),
    )


PLANE_Z = 5.0


def _plane_points(count: int = 40) -> dict[int, np.ndarray]:
    """Points spread across the image, all on the plane z = `PLANE_Z`.

    A plane gives a *dense* reference depth map, which is what `depth_gt` is, so a projection
    error lands on a neighbouring pixel that still has a value to disagree with. Points at
    scattered depths would give a sparse reference, where a mislocated point usually falls on a
    hole and gets dropped -- which hides the error instead of revealing it.
    """
    points = {}
    for index in range(count):
        col, row = 8 + (index % 8) * 6, 8 + (index // 8) * 6
        x = (col - FULL_SHAPE[1] / 2) * PLANE_Z / FOCAL
        y = (row - FULL_SHAPE[0] / 2) * PLANE_Z / FOCAL
        points[index + 1] = np.array([x, y, PLANE_Z])
    return points


def _plane_distance(shape: tuple[int, int], world_scale: float = 1.0) -> np.ndarray:
    """Ray distance to that plane at every pixel of a frame of `shape`.

    The focal length scales with the frame, so this is resolution-independent by construction --
    which is what lets it stand in for a downsampled `depth_gt`.
    """
    focal = FOCAL * shape[0] / FULL_SHAPE[0]
    rows, cols = np.indices(shape, dtype=np.float64)
    dx = (cols - shape[1] / 2) / focal
    dy = (rows - shape[0] / 2) / focal
    return PLANE_Z * world_scale * np.sqrt(1.0 + dx**2 + dy**2)


# -- reading points --------------------------------------------------------------------------


def test_text_and_binary_readers_agree(tmp_path):
    points = {7: np.array([1.0, 2.0, 3.0]), 9: np.array([-4.0, 5.0, 6.0])}

    text_dir = tmp_path / "text"
    text_dir.mkdir()
    lines = ["# a comment line COLMAP writes", ""]
    for point_id, xyz in points.items():
        lines.append(f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} 128 128 128 0.5 1 0 2 0")
    (text_dir / "points3D.txt").write_text("\n".join(lines))

    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    with open(binary_dir / "points3D.bin", "wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point_id, xyz in points.items():
            handle.write(struct.pack("<QdddBBBd", point_id, *xyz, 128, 128, 128, 0.5))
            track = [(1, 0), (2, 0)]
            handle.write(struct.pack("<Q", len(track)))
            for image_id, index in track:
                handle.write(struct.pack("<ii", image_id, index))

    from_text = read_points3d_with_ids(text_dir)
    from_binary = read_points3d_with_ids(binary_dir)
    assert sorted(from_text) == sorted(from_binary) == sorted(points)
    for point_id in points:
        np.testing.assert_allclose(from_text[point_id], points[point_id])
        np.testing.assert_allclose(from_binary[point_id], points[point_id])


def test_missing_points_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="points3D"):
        read_points3d_with_ids(tmp_path)


# -- projecting into a frame -----------------------------------------------------------------


def test_observations_carry_both_z_and_ray_distance():
    """The two must differ off-axis, since the prior is fitted in z and compared in distance."""
    points = _world_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    assert len(observations) == len(points)

    for col, row, z, distance in observations:
        # Reconstruct the camera-space point from its pixel and z, then check the norm.
        x = (col - FULL_SHAPE[1] / 2) * z / FOCAL
        y = (row - FULL_SHAPE[0] / 2) * z / FOCAL
        assert distance == pytest.approx(float(np.linalg.norm([x, y, z])))
        assert distance >= z
    off_axis = observations[np.argmax(np.abs(observations[:, 0] - FULL_SHAPE[1] / 2))]
    assert off_axis[3] > off_axis[2] * 1.001, "an off-axis point must have distance strictly above z"


def test_downscaled_frame_moves_the_pixels_and_leaves_depth_alone():
    points = _world_points()
    image = _image(points)
    half_shape = (FULL_SHAPE[0] // 2, FULL_SHAPE[1] // 2)

    full = sparse_observations(image, points, 1.0, 1.0, FULL_SHAPE)
    half = sparse_observations(image, points, 2.0, 1.0, half_shape)
    assert len(half) == len(full)
    np.testing.assert_allclose(half[:, :2], full[:, :2] / 2.0)
    np.testing.assert_allclose(half[:, 2:], full[:, 2:])


def test_world_scale_scales_depth_and_not_pixels():
    points = _world_points()
    image = _image(points)
    unscaled = sparse_observations(image, points, 1.0, 1.0, FULL_SHAPE)
    scaled = sparse_observations(image, points, 1.0, 0.25, FULL_SHAPE)
    np.testing.assert_allclose(scaled[:, :2], unscaled[:, :2])
    np.testing.assert_allclose(scaled[:, 2:], unscaled[:, 2:] * 0.25)


def test_points_behind_the_camera_and_off_frame_are_dropped():
    points = _world_points(count=16)
    points[900] = np.array([0.0, 0.0, -5.0])  # behind
    points[901] = np.array([50.0, 50.0, 1.0])  # projects far outside the frame
    image = _image(points)
    # Untracked keypoints (id -1) are what COLMAP writes for unmatched detections.
    image = image._replace(
        xys=np.vstack([image.xys, [[10.0, 10.0]]]),
        point3D_ids=np.append(image.point3D_ids, -1),
    )
    observations = sparse_observations(image, points, 1.0, 1.0, FULL_SHAPE)
    assert len(observations) == 16
    assert (observations[:, 2] > 0).all()


# -- fitting ---------------------------------------------------------------------------------


def test_trimmed_fit_ignores_a_gross_outlier():
    prior = np.arange(50, dtype=np.float64)
    target = 3.0 * prior + 7.0
    target[0] = 10_000.0
    scale, offset = fit_trimmed_affine(prior, target)
    assert scale == pytest.approx(3.0, rel=1e-6)
    assert offset == pytest.approx(7.0, rel=1e-6)


def test_untrimmed_fit_does_not():
    prior = np.arange(50, dtype=np.float64)
    target = 3.0 * prior + 7.0
    target[0] = 10_000.0
    scale, _ = fit_trimmed_affine(prior, target, trim=0.0)
    assert abs(scale - 3.0) > 0.5, "this is what the trimming is for"


def test_frame_alignment_recovers_the_affine_that_generated_the_prior():
    points = _world_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)

    # A prior that is a known affine function of true z, which is what a perfect prior would be.
    truth_z = np.full(FULL_SHAPE, np.nan)
    for col, row, z, _distance in observations:
        truth_z[int(round(row)), int(round(col))] = z
    prior = np.nan_to_num((truth_z - 4.0) / 2.5, nan=0.0)

    coefficients = fit_frame_alignment(prior, observations)
    assert coefficients is not None
    scale, offset = coefficients
    assert scale == pytest.approx(2.5, rel=1e-4)
    assert offset == pytest.approx(4.0, rel=1e-4)


def test_too_few_points_is_unfittable():
    points = _world_points(count=MIN_OBSERVATIONS - 1)
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    prior = np.random.default_rng(0).random(FULL_SHAPE)
    assert fit_frame_alignment(prior, observations) is None


def test_inverted_prior_is_refused_rather_than_fitted_with_a_negative_scale():
    """A disparity prior read as depth fits a negative scale, which would teach the scene inside out."""
    points = _world_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    prior = np.zeros(FULL_SHAPE)
    for col, row, z, _distance in observations:
        prior[int(round(row)), int(round(col))] = 1.0 / z  # inverse depth
    assert fit_frame_alignment(prior, observations) is None


# -- the self-check --------------------------------------------------------------------------


HALF_SHAPE = (FULL_SHAPE[0] // 2, FULL_SHAPE[1] // 2)


def test_consistency_is_near_zero_for_a_correct_projection():
    points = _plane_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    # Not exactly zero: the check samples the reference at the rounded pixel, as the real one
    # does. On the three OB3D scenes measured this sits at 0.004-0.007 for the same reason.
    assert observation_consistency(observations, _plane_distance(FULL_SHAPE)) < 1e-3


def test_consistency_catches_a_missing_downscale_factor():
    """The mistake this exists for: pixels at COLMAP's resolution used against a halved frame."""
    points = _plane_points()
    image = _image(points)
    reference = _plane_distance(HALF_SHAPE)

    correct = sparse_observations(image, points, 2.0, 1.0, HALF_SHAPE)
    wrong = sparse_observations(image, points, 1.0, 1.0, HALF_SHAPE)
    assert observation_consistency(correct, reference) < 1e-3
    # Most points land outside the halved frame and are dropped outright; the survivors sit at
    # twice their true angle from the axis, where the plane is measurably farther away.
    assert len(wrong) < len(correct)
    assert observation_consistency(wrong, reference) > 1e-2


def test_consistency_catches_a_missing_world_scale():
    points = _plane_points()
    image = _image(points)
    reference = _plane_distance(FULL_SHAPE, world_scale=0.25)

    correct = sparse_observations(image, points, 1.0, 0.25, FULL_SHAPE)
    wrong = sparse_observations(image, points, 1.0, 1.0, FULL_SHAPE)
    assert observation_consistency(correct, reference) < 1e-3
    assert observation_consistency(wrong, reference) == pytest.approx(3.0, rel=0.01)


def test_consistency_catches_z_used_where_ray_distance_was_wanted():
    """A constant-z reference is what a z-depth renderer would give for this plane."""
    points = _plane_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    z_reference = np.full(FULL_SHAPE, PLANE_Z)
    consistency = observation_consistency(observations, z_reference)
    # A few percent, not a factor: which is exactly why this error survives a glance at a depth
    # map and has to be read off a number.
    assert consistency > 1e-2
    assert observation_consistency(observations, _plane_distance(FULL_SHAPE)) < consistency / 10


def test_consistency_is_none_without_overlap():
    points = _plane_points()
    observations = sparse_observations(_image(points), points, 1.0, 1.0, FULL_SHAPE)
    assert observation_consistency(observations, np.full(FULL_SHAPE, np.nan)) is None
    assert observation_consistency(np.zeros((0, 4)), np.ones(FULL_SHAPE)) is None
