# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinhole fit and pose conversion in `scripts/export_depth_maps.py`.

These are the two steps that decide whether `depthrecall` samples the pixel a point actually
projects to. Both have failure modes that look fine: a transposed rotation renders a mirrored
but plausible scene, and a fitted K silently absorbs distortion into a slightly wrong focal
length. The residual check and the round-trip below are what make those visible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_depth_maps import DEFAULT_MAX_PINHOLE_RESIDUAL, fit_pinhole, world_to_camera  # noqa: E402


def pinhole_rays(fx=80.0, fy=81.0, cx=None, cy=None, width=64, height=48):
    """Camera-space rays of a pinhole camera, one per pixel centre.

    The principal point defaults to the image centre, so the rays actually span the image
    rather than a small off-axis patch -- a fit over a narrow patch is nearly degenerate and
    would understate any deviation from the model.
    """
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    u = np.arange(width) + 0.5
    v = np.arange(height) + 0.5
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)


def test_fit_recovers_the_intrinsics_that_generated_the_rays() -> None:
    fx, fy, cx, cy = 800.0, 810.0, 30.0, 20.0
    K, residual = fit_pinhole(pinhole_rays(fx, fy, cx, cy))
    assert residual < 1e-8
    np.testing.assert_allclose(K[0, 0], fx)
    np.testing.assert_allclose(K[1, 1], fy)
    np.testing.assert_allclose(K[0, 2], cx)
    np.testing.assert_allclose(K[1, 2], cy)


def test_fit_is_insensitive_to_ray_normalization() -> None:
    """The renderer may hand over unit-length rays; a pinhole K is a property of the
    direction, so the fit must not depend on the scaling."""
    rays = pinhole_rays()
    unit = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    K_raw, _ = fit_pinhole(rays)
    K_unit, residual = fit_pinhole(unit)
    np.testing.assert_allclose(K_raw, K_unit, rtol=1e-9)
    assert residual < 1e-8


def test_distortion_shows_up_as_a_large_residual() -> None:
    """A distorted camera must not be quietly absorbed into a wrong focal length.

    This is the guard that keeps the export honest: the fit always succeeds, so the residual
    is the only thing distinguishing a camera depthrecall can project from one it cannot.
    """
    rays = pinhole_rays(fx=50.0, fy=50.0)
    r2 = rays[..., 0] ** 2 + rays[..., 1] ** 2
    distorted = rays.copy()
    distorted[..., :2] *= (1.0 + 0.1 * r2)[..., None]

    _, residual = fit_pinhole(distorted)
    # What matters is the margin over the gate the exporter enforces, not an absolute pixel
    # count: this distortion leaves a residual almost an order of magnitude above it.
    assert residual > 5 * DEFAULT_MAX_PINHOLE_RESIDUAL


def test_fit_rejects_rays_behind_the_camera() -> None:
    rays = pinhole_rays()
    rays[0, 0, 2] = -1.0
    with pytest.raises(ValueError):
        fit_pinhole(rays)


def test_world_to_camera_matches_the_colmap_convention() -> None:
    """R, t must satisfy X_cam = R @ X_world + t for a camera-to-world pose."""
    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 0.7
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_c2w = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    center = np.array([1.5, -2.0, 3.0])

    pose = np.eye(4)
    pose[:3, :3] = R_c2w
    pose[:3, 3] = center

    R, t = world_to_camera(pose)
    points = rng.normal(size=(20, 3))
    # Ground truth: the inverse of the pose applied to each point.
    expected = (points - center) @ R_c2w
    np.testing.assert_allclose(points @ R.T + t, expected, atol=1e-12)
    # The camera centre maps to the origin, which a transposed R would not satisfy.
    np.testing.assert_allclose(R @ center + t, np.zeros(3), atol=1e-12)


def test_export_output_is_consumable_by_depthrecall(tmp_path: Path) -> None:
    """The manifest this script writes must round-trip through depthrecall's reader.

    The two sides are separate packages by design, so nothing but a test keeps the schema
    (and the recorded 'ray' convention) in agreement.
    """
    depthrecall_root = Path(__file__).resolve().parents[2] / "tools" / "depthrecall"
    sys.path.insert(0, str(depthrecall_root))
    from depthrecall.io_cameras import read_manifest_views
    from depthrecall.metric import MetricConfig, evaluate

    fx = fy = 100.0
    cx = cy = 32.0
    size = 64
    rays = pinhole_rays(fx, fy, cx, cy, width=size, height=size)
    K, residual = fit_pinhole(rays)
    assert residual < 1e-8

    plane_z = 4.0
    depth = np.linalg.norm(rays, axis=-1) * plane_z
    depths = tmp_path / "depths"
    depths.mkdir()
    np.save(depths / "00000.npy", depth.astype(np.float32))

    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0.0, -1.0]
    R, t = world_to_camera(pose)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "views": [
                    {
                        "name": "00000",
                        "width": size,
                        "height": size,
                        "K": K.tolist(),
                        "R": R.tolist(),
                        "t": t.tolist(),
                        "depth_path": "depths/00000.npy",
                        "depth_convention": "ray",
                    }
                ]
            }
        )
    )

    views = read_manifest_views(manifest)
    assert views[0].depth_path.exists()
    assert views[0].depth_convention == "ray"

    # Points on the plane, expressed in world coordinates via the same pose.
    points_cam = (rays * plane_z).reshape(-1, 3)
    points_world = points_cam @ pose[:3, :3].T + pose[:3, 3]
    result = evaluate(views, points_world, MetricConfig(taus=np.array([1e-6]), depth_convention="ray"))
    assert result.in_frustum_pairs == points_world.shape[0]
    np.testing.assert_allclose(result.recall, 1.0)
