# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject 3DGRT configurations that ask for normal supervision the backward cannot deliver.

Only `referenceSlangBwd` differentiates the rendered normal buffer. `referenceBwd` -- the
default -- calls a hand-derived `processHitBwd` with no normal term, and
`referenceB2FSlangBwd` is an upstream stub with its `processHitBwd` commented out entirely.
Both render normals perfectly well in the forward pass, so a normal-supervised run on either
looks healthy while the normal term contributes nothing at all.

The check keys on the loss flag rather than on `render.enable_normals`, because rendering
normals for evaluation metrics or visualization is a legitimate forward-only use that has to
keep working on every pipeline.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from threedgrt_tracer import tracer as grt_tracer


def _conf(backward_pipeline_type: str, use_depth_normal: bool = True, enable_normals: bool = True):
    return OmegaConf.create(
        {
            "render": {"backward_pipeline_type": backward_pipeline_type, "enable_normals": enable_normals},
            "loss": {"use_depth_normal": use_depth_normal},
        }
    )


@pytest.mark.parametrize(
    ("backward_pipeline_type", "supported"),
    [
        ("referenceSlangBwd", True),
        ("referenceBwd", False),
        ("referenceB2FSlangBwd", False),
    ],
)
def test_supports_normal_gradients_matches_kernel_reality(backward_pipeline_type: str, supported: bool) -> None:
    assert grt_tracer.supports_normal_gradients(_conf(backward_pipeline_type)) is supported


@pytest.mark.parametrize("backward_pipeline_type", ["referenceBwd", "referenceB2FSlangBwd"])
def test_normal_supervision_is_rejected_on_an_undifferentiated_pipeline(backward_pipeline_type: str) -> None:
    with pytest.raises(ValueError, match="does not differentiate the rendered normal buffer"):
        grt_tracer.check_normal_supervision_supported(_conf(backward_pipeline_type))


def test_normal_supervision_is_accepted_on_the_slang_pipeline() -> None:
    grt_tracer.check_normal_supervision_supported(_conf("referenceSlangBwd"))


def test_forward_only_normals_are_allowed_on_any_pipeline() -> None:
    """`enable_normals` for evaluation metrics must keep working without a normal loss.

    This is the regression that matters: existing runs render normals purely to score them,
    and the default pipeline has to keep training.
    """
    grt_tracer.check_normal_supervision_supported(_conf("referenceBwd", use_depth_normal=False))


def test_a_config_without_a_loss_section_is_allowed() -> None:
    """Rendering and playground entry points build a tracer from a config carrying no loss."""
    grt_tracer.check_normal_supervision_supported(
        OmegaConf.create({"render": {"backward_pipeline_type": "referenceBwd", "enable_normals": True}})
    )
