# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DTU adapter.

The risk here is not that the code crashes but that it produces a self-consistent camera in
the wrong frame or at the wrong scale: DTU's normalized-to-scan transform is ~325x, and a
projection matrix is only defined up to scale, so a decomposition that ignores the homogeneous
scale yields a plausible K with a translation that is 325x wrong. These tests build a
projection from known parts and check the parts come back.
"""

import numpy as np
import pytest

from depthrecall.dtu import (
    above_ground_plane_mask,
    decompose_projection,
    read_dtu_views,
    read_ground_plane,
    rq_decompose,
    scan_to_normalized,
)
from depthrecall.metric import MetricConfig, evaluate


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def make_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    P = np.eye(4)
    P[:3, :4] = scale * K @ np.concatenate([R, t[:, None]], axis=1)
    return P


def test_rq_decompose_factors_into_upper_triangular_times_orthogonal() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(3, 3))
    R, Q = rq_decompose(matrix)
    np.testing.assert_allclose(R @ Q, matrix, atol=1e-12)
    np.testing.assert_allclose(np.tril(R, -1), 0.0, atol=1e-12)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("scale", [1.0, 324.65518, -17.0, 1e-3])
def test_decomposition_recovers_K_R_t_regardless_of_homogeneous_scale(scale: float) -> None:
    """A projection matrix scaled by any non-zero factor is the same camera.

    DTU's world_mat composed with a scale_mat comes out scaled by ~325, and a decomposition
    that merely normalizes K[2, 2] reproduces K correctly while leaving t off by that factor.
    """
    K = np.array([[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]])
    R = rotation(np.array([0.3, -0.8, 0.5]), 0.9)
    t = np.array([0.4, -1.2, 3.5])

    K_out, R_out, t_out = decompose_projection(make_projection(K, R, t, scale))
    np.testing.assert_allclose(K_out, K, rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(R_out, R, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(t_out, t, rtol=1e-9, atol=1e-9)


def test_decomposition_rejects_a_degenerate_projection() -> None:
    P = np.zeros((3, 4))
    P[0, 0] = P[1, 1] = 1.0
    with pytest.raises(ValueError):
        decompose_projection(P)


def write_cameras_npz(path, world_mats, scale_mat) -> None:
    arrays = {}
    for index, world_mat in enumerate(world_mats):
        arrays[f"world_mat_{index}"] = world_mat
        arrays[f"scale_mat_{index}"] = scale_mat
    np.savez(path, **arrays)


def test_scan_to_normalized_is_the_inverse_of_the_scale_matrix(tmp_path) -> None:
    scale_mat = np.diag([325.0, 325.0, 325.0, 1.0])
    scale_mat[:3, 3] = [-50.0, -37.0, 660.0]
    write_cameras_npz(tmp_path / "cameras.npz", [np.eye(4)], scale_mat)

    transform = scan_to_normalized(tmp_path / "cameras.npz")
    np.testing.assert_allclose(transform @ scale_mat, np.eye(4), atol=1e-9)
    # A point on the unit sphere maps to scan coordinates and back.
    point = np.array([0.3, -0.5, 0.2, 1.0])
    scan_point = scale_mat @ point
    np.testing.assert_allclose((transform @ scan_point)[:3], point[:3], atol=1e-9)


def test_scan_to_normalized_rejects_disagreeing_scale_matrices(tmp_path) -> None:
    """One scan has one alignment; if the file says otherwise, taking the first is a guess."""
    arrays = {
        "world_mat_0": np.eye(4),
        "scale_mat_0": np.diag([325.0, 325.0, 325.0, 1.0]),
        "world_mat_1": np.eye(4),
        "scale_mat_1": np.diag([400.0, 400.0, 400.0, 1.0]),
    }
    np.savez(tmp_path / "cameras.npz", **arrays)
    with pytest.raises(ValueError, match="no single alignment"):
        scan_to_normalized(tmp_path / "cameras.npz")


def test_normalized_and_scan_frames_give_the_same_recall(tmp_path) -> None:
    """The whole point of the alignment plumbing: the frame must not change the measurement.

    A plane is placed in scan coordinates, depth is rendered in each frame, and the same recall
    must come out when the thresholds are scaled by the same factor as the geometry. This is
    what catches an alignment applied in the wrong direction or composed on the wrong side --
    both of which leave every individual step looking reasonable.
    """
    size = 32
    fx = fy = 60.0
    K = np.array([[fx, 0.0, size / 2], [0.0, fy, size / 2], [0.0, 0.0, 1.0]])
    scan_scale = 325.0
    scale_mat = np.diag([scan_scale, scan_scale, scan_scale, 1.0])
    scale_mat[:3, 3] = [10.0, -20.0, 500.0]

    # One camera, expressed in the normalized frame, then pushed into the scan frame so the
    # npz has the DTU layout: world_mat projects *scan* coordinates.
    R = rotation(np.array([0.1, 1.0, 0.2]), 0.4)
    t = np.array([0.05, -0.1, 3.0])
    P_normalized = make_projection(K, R, t)
    world_mat = P_normalized @ np.linalg.inv(scale_mat)
    write_cameras_npz(tmp_path / "cameras.npz", [world_mat], scale_mat)

    # A frontoparallel plane at z = 3 in the normalized camera, as a point cloud plus the
    # matching ray-distance depth map.
    u = (np.arange(size) + 0.5 - size / 2) / fx
    v = (np.arange(size) + 0.5 - size / 2) / fy
    uu, vv = np.meshgrid(u, v)
    plane_z = 3.0
    cam = np.stack([uu * plane_z, vv * plane_z, np.full_like(uu, plane_z)], axis=-1)
    depth_normalized = np.linalg.norm(cam, axis=-1).astype(np.float32)
    # X_cam = R X_w + t  =>  X_w = R^T (X_cam - t), i.e. (X_cam - t) @ R in row-vector form.
    points_normalized = (cam.reshape(-1, 3) - t) @ R
    points_scan = (np.concatenate([points_normalized, np.ones((points_normalized.shape[0], 1))], axis=1) @ scale_mat.T)[
        :, :3
    ]

    depth_dir = tmp_path / "depths"
    depth_dir.mkdir()
    np.save(depth_dir / "0000.npy", depth_normalized)
    scan_dir = tmp_path / "depths_scan"
    scan_dir.mkdir()
    np.save(scan_dir / "0000.npy", (depth_normalized * scan_scale).astype(np.float32))

    tau = 1e-4
    normalized_views = read_dtu_views(tmp_path / "cameras.npz", depth_dir, space="normalized", extension=".npy")
    normalized_result = evaluate(
        normalized_views,
        points_scan,
        MetricConfig(taus=np.array([tau]), depth_convention="ray"),
        alignment=scan_to_normalized(tmp_path / "cameras.npz"),
    )

    scan_views = read_dtu_views(tmp_path / "cameras.npz", scan_dir, space="scan", extension=".npy")
    scan_result = evaluate(
        scan_views,
        points_scan,
        MetricConfig(taus=np.array([tau * scan_scale]), depth_convention="ray"),
    )

    assert normalized_result.in_frustum_pairs == size * size
    np.testing.assert_allclose(normalized_result.recall, 1.0)
    assert scan_result.in_frustum_pairs == normalized_result.in_frustum_pairs
    np.testing.assert_allclose(scan_result.recall, normalized_result.recall)


def test_wrong_alignment_direction_destroys_recall(tmp_path) -> None:
    """The failure mode the adapter exists to prevent should be visible when forced.

    Using scale_mat where its inverse belongs leaves the points ~325x away, so nothing is in
    frustum or nothing matches. Asserting this keeps the direction from being "fixed" later by
    someone who finds the inverse surprising.
    """
    size = 16
    fx = fy = 30.0
    K = np.array([[fx, 0.0, size / 2], [0.0, fy, size / 2], [0.0, 0.0, 1.0]])
    scale_mat = np.diag([325.0, 325.0, 325.0, 1.0])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 3.0])
    world_mat = make_projection(K, R, t) @ np.linalg.inv(scale_mat)
    write_cameras_npz(tmp_path / "cameras.npz", [world_mat], scale_mat)

    u = (np.arange(size) + 0.5 - size / 2) / fx
    uu, vv = np.meshgrid(u, u)
    cam = np.stack([uu * 3.0, vv * 3.0, np.full_like(uu, 3.0)], axis=-1)
    depth = np.linalg.norm(cam, axis=-1).astype(np.float32)
    depth_dir = tmp_path / "depths"
    depth_dir.mkdir()
    np.save(depth_dir / "0000.npy", depth)

    points_normalized = (cam.reshape(-1, 3) - t) @ R
    points_scan = (np.concatenate([points_normalized, np.ones((points_normalized.shape[0], 1))], axis=1) @ scale_mat.T)[
        :, :3
    ]
    views = read_dtu_views(tmp_path / "cameras.npz", depth_dir, space="normalized", extension=".npy")
    config = MetricConfig(taus=np.array([1e-3]), depth_convention="ray")

    correct = evaluate(views, points_scan, config, alignment=scan_to_normalized(tmp_path / "cameras.npz"))
    np.testing.assert_allclose(correct.recall, 1.0)

    backwards = evaluate(views, points_scan, config, alignment=scale_mat)
    assert backwards.recall[0] < 0.01


def test_read_dtu_views_requires_a_depth_map_per_camera(tmp_path) -> None:
    """A missing depth map must not silently shrink the view set, which would change the
    denominator of the metric rather than raising."""
    scale_mat = np.diag([325.0, 325.0, 325.0, 1.0])
    K = np.array([[50.0, 0.0, 8.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]])
    mats = [make_projection(K, np.eye(3), np.array([0.0, 0.0, 3.0])) @ np.linalg.inv(scale_mat) for _ in range(2)]
    write_cameras_npz(tmp_path / "cameras.npz", mats, scale_mat)

    depth_dir = tmp_path / "depths"
    depth_dir.mkdir()
    np.save(depth_dir / "0000.npy", np.ones((16, 16), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="0001"):
        read_dtu_views(tmp_path / "cameras.npz", depth_dir, extension=".npy")


def test_read_ground_plane_accepts_text_and_npy(tmp_path) -> None:
    coefficients = np.array([0.1, -0.9, 0.2, 30.0])
    txt = tmp_path / "p.txt"
    np.savetxt(txt, coefficients)
    npy = tmp_path / "p.npy"
    np.save(npy, coefficients.reshape(1, 4))  # DTU's .mat stores it as a row, so allow that

    np.testing.assert_allclose(read_ground_plane(txt), coefficients)
    np.testing.assert_allclose(read_ground_plane(npy), coefficients)


def test_read_ground_plane_rejects_the_wrong_shape(tmp_path) -> None:
    bad = tmp_path / "p.txt"
    np.savetxt(bad, np.zeros(3))
    with pytest.raises(ValueError, match="4 plane coefficients"):
        read_ground_plane(bad)


def test_above_ground_plane_keeps_the_object_side() -> None:
    """The sign convention follows the official evaluation: `P . [x, y, z, 1] > 0` is kept."""
    plane = np.array([0.0, 0.0, -1.0, 10.0])  # keeps z < 10
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 15.0], [0.0, 0.0, 10.0]])
    np.testing.assert_array_equal(above_ground_plane_mask(points, plane), [True, False, False])


def test_ground_plane_cull_is_a_gt_side_mask_only() -> None:
    """The plane culls ground truth; it must not be confused with ObsMask.

    ObsMask filters the *reconstruction*, guarding the precision direction against being
    penalised where the scanner never observed. A recall over ground-truth points has no such
    direction, so only GT-side masks can change this metric -- which is why this package
    implements the plane and not ObsMask.
    """
    plane = np.array([0.0, 0.0, -1.0, 10.0])
    gt = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 15.0]])
    mask = above_ground_plane_mask(gt, plane)
    # The mask is per ground-truth point, so it can only ever shrink the denominator.
    assert mask.shape == (len(gt),)
    assert mask.dtype == bool
