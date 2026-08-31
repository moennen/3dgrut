# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from threedgrut.geometry.tsdf import TSDFConfig, ray_distance_to_z_depth


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
