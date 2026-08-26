# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""With `enable_normals=false` the tracers substitute a *constant* placeholder normal
rather than returning nothing, so a normal loss on such a build supervises against that
constant while producing an entirely healthy-looking loss curve. Both backends reject the
combination, and these tests pin that the rejection is not conditional on anything else.
"""

import pytest
from omegaconf import OmegaConf

from threedgrut.utils.normal_supervision import check_normals_are_rendered, normal_supervision_requested


def _conf(use_depth_normal: bool, enable_normals: bool):
    return OmegaConf.create(
        {"render": {"enable_normals": enable_normals}, "loss": {"use_depth_normal": use_depth_normal}}
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
    assert normal_supervision_requested(OmegaConf.create({})) is False
