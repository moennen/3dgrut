# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Depth-normal consistency: the rendered depth and the rendered normal must agree.

The term is `1 - cos` between the rendered normal and the normal implied by
differencing the unprojected rendered depth, over the pixels where both are defined.

It is deliberately a *mutual* constraint rather than a normal target. Measurement at 7k on
OB3D (recorded in `docs/normal-supervision.md`) found the depth-implied normal to be worse
than the rendered normal for trisurfel by up to 15 degrees, so treating it as a target
would degrade the primitive whose normals are already best. What the depth-implied normal
does have is a systematic relationship to the depth, and the rendered normal -- an explicit
primitive orientation, not a derivative -- is the smoother of the two. Leaving gradient on
both sides lets the normal act as the local smoothness prior the depth lacks, which is
where the term earns its keep. Detaching either side would throw that away, so neither is
detached.
"""

from __future__ import annotations

import torch

from threedgrut.utils.depth_geometry import (
    depth_gradient_magnitude,
    normals_from_points,
    unproject_to_world,
    world_ray_dirs,
)
from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY, expected_depth

# A blended normal shorter than this came from rays that accumulated almost no oriented
# surface, so normalizing it would amplify noise into a confident-looking direction.
MIN_RENDERED_NORMAL_NORM = 1e-6


def depth_normal_consistency_loss(
    pred_dist: torch.Tensor,
    pred_opacity: torch.Tensor,
    pred_normals: torch.Tensor,
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    T_to_world: torch.Tensor,
    min_opacity: float = MIN_ACCUMULATED_OPACITY,
    grad_percentile: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`1 - cos` between the rendered normal and the depth-implied normal.

    `pred_dist`, `pred_opacity` and `pred_normals` are the tracer's [B, H, W, C] buffers;
    the rays and pose come from the batch. `min_opacity` drops rays too transparent to
    carry a surface. `grad_percentile`, if given, additionally drops the steepest tail of
    the relative depth gradient -- occlusion boundaries, where an expected depth lands
    between two surfaces and describes neither; measured worth 1.5-2 degrees of target
    quality, so a trim rather than a fix.

    Returns the loss and the number of pixels it was averaged over, both as device
    tensors. The average is over *valid* pixels rather than all of them, so that the
    effective weight does not drift with the valid fraction the way it does in the
    reference implementation, which fills invalid pixels with zero and takes a full mean.

    Runs entirely on device with no host synchronisation: this is called every training
    iteration, and reading a mask's population back to the host to branch on it would
    stall the pipeline on each one. The empty-mask case is handled by a clamped division
    rather than an early return for the same reason.
    """
    depth, confident = expected_depth(pred_dist, pred_opacity, min_opacity)
    valid = confident.squeeze(-1)

    if grad_percentile is not None:
        with torch.no_grad():
            gradient = depth_gradient_magnitude(depth)
            # Quantile of the gradient over the valid pixels only, computed without
            # gathering them: masked-out pixels are pushed to +inf so they land above any
            # cutoff, which keeps the operation a fixed-shape device kernel.
            candidates = torch.where(valid, gradient, torch.full_like(gradient, float("inf")))
            fraction = valid.float().mean().clamp_min(1e-6)
            # Rescale the percentile so it applies to the valid subset rather than to the
            # padded whole, then cap it below 1 so the +inf padding is never selected.
            adjusted = (grad_percentile * fraction).clamp(0.0, 1.0 - 1e-6)
            cutoff = torch.quantile(candidates.flatten().float(), adjusted)
            valid = valid & (gradient < cutoff)

    world_dirs = world_ray_dirs(rays_dir, T_to_world)
    points = unproject_to_world(depth, rays_ori, rays_dir, T_to_world)
    derived, usable = normals_from_points(points, world_dirs, valid)

    rendered_norm = pred_normals.norm(dim=-1)
    mask = usable & (rendered_norm > MIN_RENDERED_NORMAL_NORM)

    rendered = pred_normals / rendered_norm.unsqueeze(-1).clamp_min(MIN_RENDERED_NORMAL_NORM)
    cosine = (rendered * derived).sum(-1)
    per_pixel = torch.where(mask, 1.0 - cosine, torch.zeros_like(cosine))

    count = mask.sum()
    # With no valid pixel this is 0/1 = 0, still attached to the graph, so the term's
    # presence in the total loss does not depend on the data.
    return per_pixel.sum() / count.clamp_min(1), count
