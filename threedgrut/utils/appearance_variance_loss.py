"""Primitive-agnostic alpha-composited appearance dispersion loss."""

from __future__ import annotations

import torch


def appearance_feature_variance_loss(
    feature_accum: torch.Tensor,
    feature_sq_accum: torch.Tensor,
    opacity: torch.Tensor,
    *,
    min_opacity: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalise rays carrying inconsistent per-hit appearance vectors.

    Inputs are raw alpha-weighted accumulators: ``sum(w f)``, ``sum(w f²)``, and
    ``sum(w)``.  The returned per-pixel term is ``Var(f) / E[||f||²]``; it is bounded,
    invariant to uniform opacity fading and feature scale, and applies equally to RGB
    (SH) and NHT's pre-decoder latent feature vector.  Fully transparent or zero-energy
    rays are masked out rather than manufacturing a gradient from numerical epsilon.
    """
    if feature_sq_accum.numel() == 0:
        raise ValueError(
            "pred_feature_sq is empty: appearance variance requires render.enable_appearance_variance=true"
        )
    if feature_accum.shape != feature_sq_accum.shape:
        raise ValueError(
            f"feature accumulators must have the same shape, got {feature_accum.shape} and {feature_sq_accum.shape}"
        )
    if opacity.shape[:-1] != feature_accum.shape[:-1] or opacity.shape[-1] != 1:
        raise ValueError("opacity must have a singleton channel and match the feature image dimensions")

    safe_opacity = opacity.clamp_min(eps)
    mean_sq_norm = (feature_accum / safe_opacity).square().sum(dim=-1, keepdim=True)
    second_energy = (feature_sq_accum / safe_opacity).sum(dim=-1, keepdim=True)
    relative_variance = (second_energy - mean_sq_norm).clamp_min(0.0) / second_energy.clamp_min(eps)

    valid = (opacity >= min_opacity) & (second_energy > eps) & torch.isfinite(relative_variance)
    valid_float = valid.to(relative_variance.dtype)
    loss = (relative_variance * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    return loss, valid
