"""Detached reliability maps for geometry supervision.

The confidence here is deliberately an *estimate*, not a learned output: learning a weight for
the same residual it suppresses has a trivial zero-confidence solution.  It is therefore
detached before it reaches a loss.  The estimate combines the evidence already rendered on a
ray -- accumulated opacity and optional depth, normal, and appearance dispersion -- and can be
combined with independent cross-view depth agreement by the reprojection loss.
"""

from __future__ import annotations

import torch


def _relative_depth_variance(
    pred_dist: torch.Tensor, pred_dist_sq: torch.Tensor, opacity: torch.Tensor
) -> torch.Tensor:
    """Squared coefficient of variation from the tracer's raw depth moments."""
    safe_dist_sq = pred_dist.square().clamp_min(1e-12)
    return (opacity * pred_dist_sq / safe_dist_sq - 1.0).clamp_min(0.0)


def _normal_variance(normal_accum: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    """Directional variance of the alpha-composited unit normal distribution."""
    mean_norm_sq = normal_accum.square().sum(dim=-1, keepdim=True) / opacity.clamp_min(1e-8).square()
    return (1.0 - mean_norm_sq).clamp(0.0, 1.0)


def _appearance_variance(
    feature_accum: torch.Tensor, feature_sq_accum: torch.Tensor, opacity: torch.Tensor
) -> torch.Tensor:
    """Relative feature variance shared with the appearance-variance regularizer."""
    safe_opacity = opacity.clamp_min(1e-8)
    mean_sq_norm = (feature_accum / safe_opacity).square().sum(dim=-1, keepdim=True)
    second_energy = (feature_sq_accum / safe_opacity).sum(dim=-1, keepdim=True)
    return (second_energy - mean_sq_norm).clamp_min(0.0) / second_energy.clamp_min(1e-8)


def rendered_geometry_confidence(
    outputs: dict[str, torch.Tensor],
    *,
    min_opacity: float,
    full_opacity: float,
    min_weight: float,
    depth_variance_weight: float = 0.0,
    normal_variance_weight: float = 0.0,
    appearance_variance_weight: float = 0.0,
) -> torch.Tensor:
    """Return a detached ``[B,H,W,1]`` reliability weight for rendered geometry.

    Opacity ramps from zero at ``min_opacity`` to one at ``full_opacity``. Each enabled
    dispersion multiplies it by ``exp(-weight * variance)``.  This makes the components
    interpretable: a weight of one maps a unit normal/appearance variance to ``e^-1``.
    ``min_weight`` is applied only to rays that passed the opacity gate, preserving a small
    learning signal without letting low-opacity background rays become supervision targets.
    """
    if not 0.0 <= min_opacity <= full_opacity <= 1.0:
        raise ValueError("confidence opacity bounds must satisfy 0 <= min_opacity <= full_opacity <= 1")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("confidence min_weight must be in [0, 1]")
    weights = (depth_variance_weight, normal_variance_weight, appearance_variance_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("confidence variance weights must be non-negative")

    opacity = outputs["pred_opacity"]
    valid = torch.isfinite(opacity) & (opacity >= min_opacity)
    opacity_confidence = (opacity - min_opacity) / max(full_opacity - min_opacity, 1e-8)
    confidence = opacity_confidence.clamp(0.0, 1.0)

    if depth_variance_weight:
        pred_dist_sq = outputs.get("pred_dist_sq")
        if pred_dist_sq is None or pred_dist_sq.numel() == 0:
            raise ValueError("depth confidence requires render.enable_depth_variance=true")
        confidence = confidence * torch.exp(
            -depth_variance_weight * _relative_depth_variance(outputs["pred_dist"], pred_dist_sq, opacity)
        )
    if normal_variance_weight:
        normal_accum = outputs.get("pred_normal_accum")
        if normal_accum is None or normal_accum.numel() == 0:
            raise ValueError("normal confidence requires render.enable_normals=true")
        confidence = confidence * torch.exp(-normal_variance_weight * _normal_variance(normal_accum, opacity))
    if appearance_variance_weight:
        feature_sq = outputs.get("pred_feature_sq")
        if feature_sq is None or feature_sq.numel() == 0:
            raise ValueError("appearance confidence requires render.enable_appearance_variance=true")
        confidence = confidence * torch.exp(
            -appearance_variance_weight
            * _appearance_variance(outputs.get("pred_latent", outputs["pred_features"]), feature_sq, opacity)
        )

    confidence = torch.where(valid, confidence.clamp_min(min_weight), torch.zeros_like(confidence))
    return confidence.detach()


def validate_confidence_config(conf) -> None:
    """Validate scalar confidence options before the training renderer is constructed."""
    options = conf.loss.confidence
    if not options.enabled:
        return
    if not 0.0 <= options.min_opacity <= options.full_opacity <= 1.0:
        raise ValueError("loss.confidence requires 0 <= min_opacity <= full_opacity <= 1")
    if not 0.0 <= options.min_weight <= 1.0:
        raise ValueError("loss.confidence.min_weight must be in [0, 1]")
    weights = (options.depth_variance_weight, options.normal_variance_weight, options.appearance_variance_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("loss.confidence variance weights must be non-negative")
    if options.multiview_agreement_weight < 0.0:
        raise ValueError("loss.confidence.multiview_agreement_weight must be non-negative")
