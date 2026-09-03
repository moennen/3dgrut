# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from threedgrut.geometry.tsdf import (
    TSDFConfig,
    adaptive_voxel_size,
    ray_distance_to_z_depth,
    rgb_to_uint8,
    world_bounds_mask,
    z_depth_to_ray_distance,
)


def test_ray_distance_to_z_depth_is_exact_on_the_optical_axis():
    ray = np.full((3, 3), 4.0, dtype=np.float32)
    K = np.array([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])
    z = ray_distance_to_z_depth(ray, K)
    np.testing.assert_allclose(z[1, 1], 4.0)


def test_ray_distance_to_z_depth_shortens_oblique_rays():
    ray = np.full((3, 3), 5.0, dtype=np.float32)
    K = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]])
    z = ray_distance_to_z_depth(ray, K)
    assert z[0, 0] < z[1, 1]
    np.testing.assert_allclose(z[0, 0], 5.0 / np.sqrt(3.0))


def test_depth_convention_conversions_round_trip():
    K = np.array([[4.0, 0.0, 1.5], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]])
    ray = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    np.testing.assert_allclose(z_depth_to_ray_distance(ray_distance_to_z_depth(ray, K), K), ray)


def test_rgb_to_uint8_preserves_uint8_colors():
    rgb = np.array([[[0, 12, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(rgb_to_uint8(rgb, (1, 1)), rgb)


def test_rgb_to_uint8_uses_black_for_depth_only_frames():
    np.testing.assert_array_equal(rgb_to_uint8(None, (1, 2)), np.zeros((1, 2, 3), dtype=np.uint8))


def test_rgb_to_uint8_converts_renderer_float_colors():
    rgb = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    np.testing.assert_array_equal(rgb_to_uint8(rgb, (1, 1)), np.array([[[0, 128, 255]]], dtype=np.uint8))


@pytest.mark.parametrize(
    "rgb, message",
    [
        (np.zeros((1, 1), dtype=np.uint8), "must match depth"),
        (np.array([[[1.1, 0.0, 0.0]]], dtype=np.float32), "must be in"),
    ],
)
def test_rgb_to_uint8_rejects_invalid_input(rgb, message):
    with pytest.raises(ValueError, match=message):
        rgb_to_uint8(rgb, (1, 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size": 0.0, "truncation": 1.0, "max_depth": 1.0},
        {"voxel_size": 1.0, "truncation": 0.5, "max_depth": 1.0},
        {"voxel_size": 1.0, "truncation": 1.0, "max_depth": 0.0},
    ],
)
def test_tsdf_config_rejects_invalid_units(kwargs):
    with pytest.raises(ValueError):
        TSDFConfig(**kwargs)


def test_adaptive_voxel_size_only_coarsens_bounded_grid():
    bounds = np.array([[0.0, 0.0, 0.0], [20.0, 10.0, 5.0]])
    assert adaptive_voxel_size(0.01, bounds, 1000) == pytest.approx(0.02)
    assert adaptive_voxel_size(0.05, bounds, 1000) == pytest.approx(0.05)
    assert adaptive_voxel_size(0.01, None, 1000) == pytest.approx(0.01)


def test_world_bounds_mask_unprojects_z_depth_in_world_coordinates():
    K = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    depth = np.full((1, 3), 2.0, dtype=np.float32)
    # Pixel centres project to x = -2, 0, 2 at z=2.  Keep only the centre point.
    bounds = np.array([[-0.1, -1.0, 1.9], [0.1, 1.0, 2.1]])
    np.testing.assert_array_equal(world_bounds_mask(depth, K, np.eye(4), bounds), [[False, True, False]])
