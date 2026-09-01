# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from threedgrut.utils.normal_variance_loss import normal_direction_variance_loss


def _ray(weights: list[float], normals: list[list[float]]):
    weight = torch.tensor(weights, dtype=torch.float64)
    normal = torch.tensor(normals, dtype=torch.float64)
    return (weight[:, None] * normal).sum(0).reshape(1, 1, 1, 3), weight.sum().reshape(1, 1, 1, 1)


def test_one_coherent_surface_has_zero_directional_variance():
    accum, opacity = _ray([0.4, 0.5], [[0, 0, 1], [0, 0, 1]])
    loss, count = normal_direction_variance_loss(accum, opacity)
    assert int(count) == 1
    assert loss.item() == pytest.approx(0.0)


def test_opposite_normals_maximize_directional_variance():
    accum, opacity = _ray([0.5, 0.5], [[0, 0, 1], [0, 0, -1]])
    loss, _ = normal_direction_variance_loss(accum, opacity)
    assert loss.item() == pytest.approx(1.0)


def test_uniformly_fading_a_ray_does_not_evade_the_loss():
    normal = [[0, 0, 1], [1, 0, 0]]
    full, _ = normal_direction_variance_loss(*_ray([0.45, 0.45], normal))
    faded, _ = normal_direction_variance_loss(*_ray([0.30, 0.30], normal))
    assert faded.item() == pytest.approx(full.item())


def test_transparent_rays_are_excluded():
    loss, count = normal_direction_variance_loss(*_ray([0.1, 0.1], [[0, 0, 1], [0, 0, -1]]))
    assert int(count) == 0
    assert loss.item() == pytest.approx(0.0)


def test_gradient_reaches_the_raw_normal_accumulator():
    accum, opacity = _ray([0.5, 0.5], [[0, 0, 1], [1, 0, 0]])
    accum.requires_grad_()
    loss, _ = normal_direction_variance_loss(accum, opacity)
    loss.backward()
    assert accum.grad is not None and accum.grad.abs().sum() > 0
