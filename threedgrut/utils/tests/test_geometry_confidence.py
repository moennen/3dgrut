import math

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.utils.geometry_confidence import rendered_geometry_confidence, validate_confidence_config
from threedgrut.utils.geometry_supervision import check_confidence_inputs_are_rendered


def _outputs(opacity: torch.Tensor) -> dict[str, torch.Tensor]:
    shape = opacity.shape
    return {
        "pred_opacity": opacity,
        "pred_dist": torch.full(shape, 2.0),
        "pred_dist_sq": torch.full(shape, 4.0),
        "pred_normal_accum": torch.tensor([0.0, 0.0, -1.0]).view(1, 1, 1, 3).expand(*shape[:-1], 3).clone(),
        "pred_features": torch.ones(*shape[:-1], 3),
        "pred_feature_sq": torch.ones(*shape[:-1], 3),
    }


def test_opacity_confidence_gates_background_and_ramps_to_one():
    outputs = _outputs(torch.tensor([[[[0.2], [0.5], [0.95]]]]))
    confidence = rendered_geometry_confidence(outputs, min_opacity=0.5, full_opacity=0.95, min_weight=0.05)
    torch.testing.assert_close(confidence[..., 0], torch.tensor([[[0.0, 0.05, 1.0]]]))


def test_dispersion_sources_multiply_detached_confidence():
    outputs = _outputs(torch.ones(1, 1, 1, 1, requires_grad=True))
    outputs["pred_dist_sq"] = torch.full((1, 1, 1, 1), 8.0)
    outputs["pred_normal_accum"] = torch.zeros(1, 1, 1, 3)
    outputs["pred_feature_sq"] = torch.full((1, 1, 1, 3), 2.0)
    confidence = rendered_geometry_confidence(
        outputs,
        min_opacity=0.5,
        full_opacity=1.0,
        min_weight=0.0,
        depth_variance_weight=1.0,
        normal_variance_weight=1.0,
        appearance_variance_weight=1.0,
    )
    # Depth and normal variance are one; the chosen feature moments give relative variance 0.5.
    torch.testing.assert_close(confidence, torch.full_like(confidence, math.exp(-2.5)))
    assert not confidence.requires_grad


def test_confidence_configuration_and_buffers_fail_loudly_when_incompatible():
    config = OmegaConf.create(
        {
            "render": {"method": "3dgrt", "enable_depth_variance": False},
            "loss": {
                "confidence": {
                    "enabled": True,
                    "min_opacity": 0.5,
                    "full_opacity": 0.95,
                    "min_weight": 0.05,
                    "depth_variance_weight": 1.0,
                    "normal_variance_weight": 0.0,
                    "appearance_variance_weight": 0.0,
                    "multiview_agreement_weight": 1.0,
                }
            },
        }
    )
    validate_confidence_config(config)
    with pytest.raises(ValueError, match="depth confidence"):
        check_confidence_inputs_are_rendered(config)


def test_invalid_confidence_bounds_are_rejected():
    config = OmegaConf.create(
        {
            "loss": {
                "confidence": {
                    "enabled": True,
                    "min_opacity": 0.9,
                    "full_opacity": 0.5,
                    "min_weight": 0.0,
                    "depth_variance_weight": 0.0,
                    "normal_variance_weight": 0.0,
                    "appearance_variance_weight": 0.0,
                    "multiview_agreement_weight": 0.0,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="min_opacity"):
        validate_confidence_config(config)
