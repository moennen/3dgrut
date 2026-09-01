# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both checks here reject a configuration that would otherwise train at full cost while
supervising nothing, so each test's job is to pin that the rejection is unconditional.

With `enable_normals=false` the tracers substitute a *constant* placeholder normal rather
than returning nothing, so a normal loss on such a build supervises against that constant.
With `primitive_type=trisurfel` the kernel forces `scale.z` and drops its gradient, so a
flatness penalty on the smallest axis shrinks dead storage. Both produce entirely
healthy-looking loss curves.
"""

import pytest
from omegaconf import OmegaConf

from threedgrut.utils.geometry_supervision import (
    check_appearance_variance_is_rendered,
    check_depth_variance_is_rendered,
    check_flatness_applies,
    check_normals_are_rendered,
    normal_supervision_requested,
)


def _conf(use_depth_normal: bool, enable_normals: bool):
    return OmegaConf.create(
        {"render": {"enable_normals": enable_normals}, "loss": {"use_depth_normal": use_depth_normal}}
    )


def _flatten_conf(use_scale_flatten: bool, primitive_type: str):
    return OmegaConf.create(
        {"render": {"primitive_type": primitive_type}, "loss": {"use_scale_flatten": use_scale_flatten}}
    )


def test_a_normal_loss_without_rendered_normals_is_rejected() -> None:
    with pytest.raises(ValueError, match="render.enable_normals is false"):
        check_normals_are_rendered(_conf(use_depth_normal=True, enable_normals=False))


def test_a_normal_loss_with_rendered_normals_is_accepted() -> None:
    check_normals_are_rendered(_conf(use_depth_normal=True, enable_normals=True))


@pytest.mark.parametrize("enable_normals", [True, False])
def test_runs_without_a_normal_loss_are_untouched(enable_normals: bool) -> None:
    """Every existing config has no normal loss, and must keep training either way."""
    check_normals_are_rendered(_conf(use_depth_normal=False, enable_normals=enable_normals))


def test_a_config_without_the_sections_is_allowed() -> None:
    """Render and playground entry points build from configs carrying neither section."""
    check_normals_are_rendered(OmegaConf.create({}))
    check_flatness_applies(OmegaConf.create({}))
    check_depth_variance_is_rendered(OmegaConf.create({}))
    assert normal_supervision_requested(OmegaConf.create({})) is False


def test_flatness_on_an_already_flat_primitive_is_rejected() -> None:
    with pytest.raises(ValueError, match="already flattens"):
        check_flatness_applies(_flatten_conf(use_scale_flatten=True, primitive_type="trisurfel"))


def test_flatness_on_ellipsoids_is_accepted() -> None:
    """The ellipsoid primitives are the ones the term exists for."""
    check_flatness_applies(_flatten_conf(use_scale_flatten=True, primitive_type="instances"))


@pytest.mark.parametrize("primitive_type", ["instances", "trisurfel"])
def test_runs_without_the_flatness_term_are_untouched(primitive_type: str) -> None:
    check_flatness_applies(_flatten_conf(use_scale_flatten=False, primitive_type=primitive_type))


def _variance_conf(use_depth_variance: bool, enable_depth_variance: bool, method: str = "3dgut"):
    return OmegaConf.create(
        {
            "render": {"method": method, "enable_depth_variance": enable_depth_variance},
            "loss": {"use_depth_variance": use_depth_variance},
        }
    )


def test_a_variance_loss_without_the_moment_buffer_is_rejected() -> None:
    """The buffer is empty rather than absent, so the loss would be a silent zero."""
    with pytest.raises(ValueError, match="render.enable_depth_variance is false"):
        check_depth_variance_is_rendered(_variance_conf(use_depth_variance=True, enable_depth_variance=False))


def test_a_variance_loss_under_3dgrt_is_rejected() -> None:
    """3DGRT never renders the second moment, at any setting."""
    with pytest.raises(ValueError, match="3DGUT-only"):
        check_depth_variance_is_rendered(
            _variance_conf(use_depth_variance=True, enable_depth_variance=True, method="3dgrt")
        )


def test_a_variance_loss_with_the_moment_buffer_is_accepted() -> None:
    check_depth_variance_is_rendered(_variance_conf(use_depth_variance=True, enable_depth_variance=True))


@pytest.mark.parametrize("enable_depth_variance", [True, False])
@pytest.mark.parametrize("method", ["3dgut", "3dgrt"])
def test_runs_without_the_variance_term_are_untouched(method: str, enable_depth_variance: bool) -> None:
    """Including 3DGRT runs, which must not be disturbed by a term they cannot use."""
    check_depth_variance_is_rendered(
        _variance_conf(use_depth_variance=False, enable_depth_variance=enable_depth_variance, method=method)
    )


@pytest.mark.parametrize(
    "method,enabled,message",
    [
        ("3dgut", False, "render.enable_appearance_variance is false"),
        ("3dgrt", True, "3DGUT-only"),
    ],
)
def test_appearance_variance_requires_its_3dgut_second_moment(method: str, enabled: bool, message: str) -> None:
    conf = OmegaConf.create(
        {
            "render": {"method": method, "enable_appearance_variance": enabled},
            "loss": {"use_appearance_variance": True},
        }
    )
    with pytest.raises(ValueError, match=message):
        check_appearance_variance_is_rendered(conf)


def test_appearance_variance_accepts_enabled_3dgut():
    conf = OmegaConf.create(
        {
            "render": {"method": "3dgut", "enable_appearance_variance": True},
            "loss": {"use_appearance_variance": True},
        }
    )
    check_appearance_variance_is_rendered(conf)
