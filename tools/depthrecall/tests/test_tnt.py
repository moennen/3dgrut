# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Tanks and Temples adapter.

Unlike DTU, TnT's alignment is *estimated*, so the thing to pin is not just that the algebra is
right but that the estimate reports its own error -- a recall quoted at a tau below the
registration residual is measuring the registration.
"""

import json

import numpy as np
import pytest

from depthrecall.tnt import (
    OFFICIAL_TAU_METRES,
    apply_transform,
    crop_volume_mask,
    fit_similarity,
    gt_to_render_alignment,
    read_crop_volume,
    read_sfm_log,
)


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def similarity(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = t
    return M


def test_fit_similarity_recovers_a_known_transform() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3)) * 3.0
    truth = similarity(2.5, rotation(np.array([0.2, 0.5, -0.8]), 1.1), np.array([4.0, -2.0, 7.0]))
    target = apply_transform(truth, source)

    fitted = fit_similarity(source, target)
    np.testing.assert_allclose(fitted, truth, atol=1e-9)


def test_fit_similarity_does_not_return_a_reflection() -> None:
    """A mirrored fit halves the residual on degenerate data while inverting the geometry.

    Points near-coplanar make the reflected solution numerically competitive, so the sign of
    the determinant has to be forced rather than left to the SVD.
    """
    rng = np.random.default_rng(1)
    source = np.column_stack([rng.normal(size=40), rng.normal(size=40), rng.normal(size=40) * 1e-6])
    target = apply_transform(similarity(1.0, np.eye(3), np.zeros(3)), source)
    target[:, 2] *= -1  # a reflection, which no similarity can express

    fitted = fit_similarity(source, target)
    assert np.linalg.det(fitted[:3, :3]) > 0


def test_fit_similarity_rejects_too_few_or_mismatched_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_similarity(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="matching"):
        fit_similarity(np.zeros((5, 3)), np.zeros((4, 3)))


def write_log(path, poses) -> None:
    lines = []
    for i, pose in enumerate(poses):
        lines.append(f"{i} {i} 0")
        lines.extend(" ".join(f"{v:.12f}" for v in row) for row in pose)
    path.write_text("\n".join(lines) + "\n")


def test_read_sfm_log_returns_camera_to_world_poses(tmp_path) -> None:
    poses = [np.eye(4), similarity(1.0, rotation(np.array([0.0, 0.0, 1.0]), 0.5), np.array([1.0, 2.0, 3.0]))]
    path = tmp_path / "s.log"
    write_log(path, poses)

    indices, read = read_sfm_log(path)
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(read, np.stack(poses), atol=1e-12)
    # The translation column is the camera centre, not a world-to-camera offset.
    np.testing.assert_allclose(read[1][:3, 3], [1.0, 2.0, 3.0])


def test_read_sfm_log_rejects_a_truncated_entry(tmp_path) -> None:
    path = tmp_path / "bad.log"
    path.write_text("0 0 0\n1 0 0 0\n0 1 0 0\n")
    with pytest.raises(ValueError, match="Truncated"):
        read_sfm_log(path)


def test_gt_to_render_alignment_composes_both_legs(tmp_path) -> None:
    """With cameras that match exactly, the composition must invert the whole chain.

    The scan is built by pushing known points through `trans`, so a correct alignment maps them
    back onto the render frame to within float noise, and the residual it reports is ~0.
    """
    rng = np.random.default_rng(2)
    # Official SfM frame: camera centres and a few scene points.
    log_centres = rng.normal(size=(12, 3))
    poses = []
    for centre in log_centres:
        pose = np.eye(4)
        pose[:3, 3] = centre
        poses.append(pose)
    log_path = tmp_path / "scene_COLMAP_SfM.log"
    write_log(log_path, poses)

    # trans: official SfM -> scan, with the ~4x scale TnT actually carries.
    trans = similarity(4.1385, rotation(np.array([0.3, 0.2, 0.9]), 0.7), np.array([-8.0, 0.7, -21.9]))
    trans_path = tmp_path / "scene_trans.txt"
    np.savetxt(trans_path, trans)

    # The render reconstruction differs from the official SfM by another similarity.
    log_to_render = similarity(1.0036, rotation(np.array([0.1, -0.4, 0.3]), 0.05), np.array([0.2, -0.1, 0.05]))
    render_centres = apply_transform(log_to_render, log_centres)

    alignment = gt_to_render_alignment(trans_path, log_path, render_centres)
    assert alignment.n_correspondences == 12
    assert alignment.residual_median < 1e-9
    np.testing.assert_allclose(alignment.matrix, log_to_render @ np.linalg.inv(trans), atol=1e-9)

    # A scan point maps back to the render frame it came from.
    point_in_render = np.array([[0.4, -0.2, 0.9]])
    point_in_scan = apply_transform(trans @ np.linalg.inv(log_to_render), point_in_render)
    np.testing.assert_allclose(apply_transform(alignment.matrix, point_in_scan), point_in_render, atol=1e-9)


def test_gt_to_render_alignment_reports_residual_in_scan_units(tmp_path) -> None:
    """The residual has to be comparable with a tau in metres, not in render units.

    On real Barn these differ by 4.1x, which is the difference between a residual that looks
    comfortably below the official 0.01 m threshold and one that exceeds it.
    """
    rng = np.random.default_rng(3)
    log_centres = rng.normal(size=(30, 3)) * 2.0
    poses = []
    for centre in log_centres:
        pose = np.eye(4)
        pose[:3, 3] = centre
        poses.append(pose)
    log_path = tmp_path / "s.log"
    write_log(log_path, poses)

    scan_scale = 4.1385
    trans = similarity(scan_scale, np.eye(3), np.zeros(3))
    trans_path = tmp_path / "t.txt"
    np.savetxt(trans_path, trans)

    # Perturb a single camera: a perturbation applied to all of them is a translation, which the
    # fit absorbs exactly and which would report a residual of zero.
    offset = 0.01
    render_centres = log_centres.copy()
    render_centres[0, 0] += offset

    alignment = gt_to_render_alignment(trans_path, log_path, render_centres)
    # The largest residual is ~the offset expressed in scan units, i.e. multiplied by the scale.
    assert alignment.residual_max > offset * scan_scale * 0.5
    assert alignment.residual_max < offset * scan_scale * 1.5


def test_alignment_warns_when_tau_is_near_the_registration_residual() -> None:
    from depthrecall.tnt import GtAlignment

    alignment = GtAlignment(
        matrix=np.eye(4), scale=1.0, residual_median=0.012, residual_max=0.024, n_correspondences=410
    )
    # Barn's official threshold sits *below* the residual measured on the real scene, which is
    # the case the warning exists for.
    assert alignment.warn_if_coarse(np.array([OFFICIAL_TAU_METRES["Barn"]])) is not None
    assert alignment.warn_if_coarse(np.array([0.2])) is None


def test_gt_to_render_alignment_rejects_a_camera_count_mismatch(tmp_path) -> None:
    poses = [np.eye(4) for _ in range(4)]
    log_path = tmp_path / "s.log"
    write_log(log_path, poses)
    trans_path = tmp_path / "t.txt"
    np.savetxt(trans_path, np.eye(4))
    with pytest.raises(ValueError, match="against 4"):
        gt_to_render_alignment(trans_path, log_path, np.zeros((3, 3)))


def write_crop(path, polygon, axis="Z", axis_min=-1.0, axis_max=1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "class_name": "SelectionPolygonVolume",
                "orthogonal_axis": axis,
                "axis_min": axis_min,
                "axis_max": axis_max,
                "bounding_polygon": polygon,
                "version_major": 1,
                "version_minor": 0,
            }
        )
    )


def test_crop_volume_keeps_points_inside_the_polygon_and_band(tmp_path) -> None:
    square = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    path = tmp_path / "c.json"
    write_crop(path, square, axis="Z", axis_min=-0.5, axis_max=0.5)

    points = np.array(
        [
            [0.0, 0.0, 0.0],  # inside
            [0.9, -0.9, 0.4],  # inside, near a corner
            [1.5, 0.0, 0.0],  # outside the polygon
            [0.0, 0.0, 2.0],  # inside the polygon, outside the band
        ]
    )
    np.testing.assert_array_equal(crop_volume_mask(points, path), [True, True, False, False])


def test_crop_volume_handles_a_concave_polygon(tmp_path) -> None:
    """The real crops are concave (Barn's has 20 vertices), so a convex-hull test is not enough."""
    # An L shape covering (0..2, 0..1) and (0..1, 1..2), so (1.5, 1.5) is outside.
    l_shape = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
    path = tmp_path / "l.json"
    write_crop(path, l_shape, axis_min=-1.0, axis_max=1.0)

    points = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0]])
    np.testing.assert_array_equal(crop_volume_mask(points, path), [True, True, True, False])


def test_crop_volume_respects_the_orthogonal_axis(tmp_path) -> None:
    """The polygon lies in the plane orthogonal to the named axis; assuming Z silently crops
    the wrong two dimensions, which on a real scene still keeps a plausible-looking subset."""
    # Open3D ignores the orthogonal axis's own column in each polygon vertex, so with
    # orthogonal_axis X the polygon is carried in the Y and Z columns, not the first two.
    square = [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]]
    path = tmp_path / "x.json"
    write_crop(path, square, axis="X", axis_min=-10.0, axis_max=10.0)

    # Both points are inside the X band; only the first is inside the polygon in (Y, Z).
    points = np.array([[5.0, 0.0, 0.0], [5.0, 3.0, 0.0]])
    np.testing.assert_array_equal(crop_volume_mask(points, path), [True, False])


def test_read_crop_volume_rejects_a_foreign_json(tmp_path) -> None:
    path = tmp_path / "n.json"
    path.write_text(json.dumps({"class_name": "PointCloud"}))
    with pytest.raises(ValueError, match="SelectionPolygonVolume"):
        read_crop_volume(path)


def test_official_taus_cover_the_training_set() -> None:
    """These are quoted from the benchmark toolbox; a typo here silently changes every number."""
    assert OFFICIAL_TAU_METRES["Barn"] == 0.01
    assert OFFICIAL_TAU_METRES["Ignatius"] == 0.003
    assert set(OFFICIAL_TAU_METRES) == {
        "Barn",
        "Caterpillar",
        "Church",
        "Courthouse",
        "Ignatius",
        "Meetingroom",
        "Truck",
    }
