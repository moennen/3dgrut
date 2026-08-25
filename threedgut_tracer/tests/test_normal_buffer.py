# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rendered-normal buffer plumbing that do not require a CUDA build.

The kernel side is covered by rendering; what is checked here is the Python contract
around it: how the raw accumulated buffer is turned into unit normals, and the guards
that stop the feature being combined with a kernel that cannot produce it.
"""

import pytest
import torch
from omegaconf import OmegaConf

from threedgut_tracer.setup_3dgut import setup_3dgut
from threedgut_tracer.tracer import Tracer

# The helper is name-mangled; bind it once so the tests read clearly.
_resolve_normals = Tracer._Tracer__resolve_normals


def make_conf(enable_normals: bool):
    return OmegaConf.create({"render": {"enable_normals": enable_normals}})


def fake_tracer(enable_normals: bool):
    """A Tracer stand-in: __resolve_normals only reads `conf`, so skip CUDA init."""
    return type("FakeTracer", (), {"conf": make_conf(enable_normals)})()


def test_disabled_build_returns_the_legacy_placeholder() -> None:
    """`enable_normals=false` returns an empty buffer; callers must see the old constant."""
    pred_features = torch.zeros(1, 4, 5, 3)
    normals = _resolve_normals(fake_tracer(False), torch.empty(0), pred_features)

    assert normals.shape == pred_features.shape
    # The historical placeholder: a single constant direction at every pixel.
    assert torch.allclose(normals, torch.full_like(normals, 1.0 / 3**0.5))
    assert len(torch.unique(normals.reshape(-1, 3), dim=0)) == 1


def test_empty_buffer_falls_back_even_when_enabled() -> None:
    """Guards against a config/build mismatch turning into a shape error at runtime."""
    pred_features = torch.zeros(1, 2, 2, 3)
    normals = _resolve_normals(fake_tracer(True), torch.empty(0), pred_features)
    assert normals.shape == pred_features.shape


def test_accumulated_normals_are_normalized_to_unit_length() -> None:
    """The kernel accumulates alpha-premultiplied normals, so only direction survives."""
    raw = torch.tensor([[[[0.0, 0.0, 0.25], [3.0, 4.0, 0.0]]]]).reshape(1, 2, 3)
    normals = _resolve_normals(fake_tracer(True), raw, torch.zeros(1, 1, 2, 3))

    assert torch.allclose(normals[0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(normals[0, 0, 1], torch.tensor([0.6, 0.8, 0.0]))
    assert torch.allclose(normals.norm(dim=3), torch.ones(1, 1, 2))


def test_direction_is_preserved_regardless_of_accumulated_magnitude() -> None:
    """Two rays covering the same surface differently must report the same normal."""
    direction = torch.tensor([1.0, -2.0, 0.5])
    raw = torch.stack([direction * 0.01, direction * 7.0]).reshape(1, 2, 3)
    normals = _resolve_normals(fake_tracer(True), raw, torch.zeros(1, 1, 2, 3))
    assert torch.allclose(normals[0, 0, 0], normals[0, 0, 1], atol=1e-6)


def test_rays_that_hit_nothing_stay_zero() -> None:
    """Background must stay (0,0,0) so consumers can tell it from a real surface.

    Normalizing a zero accumulator would invent an arbitrary unit direction and make
    empty pixels indistinguishable from geometry, silently biasing any normal metric.
    """
    raw = torch.zeros(1, 3, 3)
    raw[0, 1] = torch.tensor([0.0, 0.0, 2.0])
    normals = _resolve_normals(fake_tracer(True), raw, torch.zeros(1, 1, 3, 3))

    assert torch.equal(normals[0, 0, 0], torch.zeros(3))
    assert torch.equal(normals[0, 0, 2], torch.zeros(3))
    assert torch.allclose(normals[0, 0, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_tiny_but_nonzero_accumulation_is_not_amplified_into_noise() -> None:
    """Below the epsilon the direction is meaningless, so it must be dropped, not scaled up."""
    raw = torch.full((1, 1, 3), 1e-9)
    normals = _resolve_normals(fake_tracer(True), raw, torch.zeros(1, 1, 1, 3))
    assert torch.equal(normals[0, 0, 0], torch.zeros(3))


def test_normals_with_load_balancing_is_rejected() -> None:
    """The load-balanced kernel never accumulates normals; it would return zeros."""
    conf = OmegaConf.create({"render": {"enable_normals": True, "splat": {"fine_grained_load_balancing": True}}})
    with pytest.raises(ValueError, match="fine_grained_load_balancing"):
        setup_3dgut(conf)
