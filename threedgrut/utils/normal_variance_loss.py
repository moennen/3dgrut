# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regularize disagreement between surface normals contributing to one ray."""

from __future__ import annotations

import torch

from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY


def normal_direction_variance_loss(
    normal_accum: torch.Tensor,
    opacity: torch.Tensor,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalized directional variance of the ray's alpha-weighted normals.

    The tracer accumulates ``N = sum_i w_i n_i`` where each camera-facing primitive normal
    ``n_i`` has unit norm and ``opacity = sum_i w_i``.  Therefore

    ``1 - ||N / opacity||^2``

    is the trace of the weighted directional covariance. It is zero for any one coherent
    surface and approaches one when the ray combines cancelling directions. Dividing by opacity
    makes the loss invariant to uniformly fading every hit; the opacity gate excludes rays that
    have no meaningful surface distribution. This applies identically to ellipsoids and
    trisurfels, because it depends only on their composited unit normals.
    """
    if normal_accum.numel() == 0:
        raise ValueError("normal_accum is empty: normal variance requires render.enable_normals=true")
    if normal_accum.shape[:-1] != opacity.shape[:-1] or opacity.shape[-1] != 1:
        raise ValueError("normal_accum must be [..., 3] and opacity must be [..., 1]")

    valid = (opacity >= min_opacity) & torch.isfinite(opacity) & torch.isfinite(normal_accum).all(dim=-1, keepdim=True)
    direction_mean_sq = normal_accum.square().sum(dim=-1, keepdim=True) / opacity.clamp_min(min_opacity).square()
    variance = (1.0 - direction_mean_sq).clamp(0.0, 1.0)
    per_pixel = torch.where(valid, variance, torch.zeros_like(variance))
    count = valid.sum()
    return per_pixel.sum() / count.clamp_min(1), count
