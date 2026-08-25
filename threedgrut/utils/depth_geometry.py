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

"""Turning a rendered depth map into world-space points and the normals they imply.

Shared by the depth-normal consistency loss and by the diagnostics that decide whether
that loss is well-posed, so both operate on exactly the same definition of
"depth-implied normal". Everything here is differentiable with respect to depth.

Two conventions are worth stating because getting either wrong produces a plausible
result that is silently wrong:

  * Depth is the Euclidean distance along the ray, matching the tracer and the OB3D
    reference, so a point is `origin + distance * unit_direction` rather than anything
    involving a division by a z component.
  * The orientation of a cross product depends on the camera's handedness and on the
    pixel ordering, so it is never relied upon. The normal is oriented against the view
    ray instead, which is what the renderer does to its own normals and therefore the
    only convention under which the two are comparable.
"""

from __future__ import annotations

import torch

# Below this the cross product of two central differences is numerically meaningless:
# the two differences are parallel, or the neighbourhood is degenerate.
MIN_CROSS_NORM = 1e-8


def unproject_to_world(
    depth: torch.Tensor,
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    T_to_world: torch.Tensor | None = None,
) -> torch.Tensor:
    """World-space points at `depth` along each ray.

    `depth` is [B, H, W, 1] Euclidean ray distance; `rays_ori` and `rays_dir` are
    [B, H, W, 3] in ray space, and `T_to_world` is the [B, 4, 4] pose that carries them
    into the world (pass None when the rays are already in world space).

    Directions are renormalized rather than assumed unit: a non-unit direction would
    rescale the depth and make the resulting geometry wrong by a per-pixel factor that
    is invisible in the output.
    """
    dirs = rays_dir / rays_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    points = rays_ori + dirs * depth
    if T_to_world is None:
        return points
    rotation = T_to_world[:, :3, :3]
    translation = T_to_world[:, :3, 3]
    return torch.einsum("bij,bhwj->bhwi", rotation, points) + translation[:, None, None, :]


def world_ray_dirs(rays_dir: torch.Tensor, T_to_world: torch.Tensor | None = None) -> torch.Tensor:
    """Unit ray directions in the world frame."""
    dirs = rays_dir
    if T_to_world is not None:
        dirs = torch.einsum("bij,bhwj->bhwi", T_to_world[:, :3, :3], dirs)
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normals_from_points(
    points: torch.Tensor,
    view_dirs: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Surface normals implied by a map of world-space points.

    The normal at a pixel is the cross product of the central differences of its four
    neighbours, which is the normal of the plane through them. `points` is [B, H, W, 3],
    `view_dirs` the matching unit world-space ray directions, and `valid` an optional
    [B, H, W] mask of pixels whose depth is trustworthy.

    Returns the unit normals and the mask of pixels where one could be computed. The
    one-pixel border has no central difference and is invalid by construction, as is any
    pixel with an untrustworthy neighbour: a normal computed across a hole is a normal of
    the hole.
    """
    dx = points[:, 1:-1, 2:, :] - points[:, 1:-1, :-2, :]
    dy = points[:, 2:, 1:-1, :] - points[:, :-2, 1:-1, :]
    cross = torch.linalg.cross(dx, dy, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True)

    interior = torch.zeros_like(points)
    interior[:, 1:-1, 1:-1, :] = cross / norm.clamp_min(MIN_CROSS_NORM)

    usable = torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
    usable[:, 1:-1, 1:-1] = norm.squeeze(-1) > MIN_CROSS_NORM
    if valid is not None:
        neighbours = (
            valid[:, 1:-1, 1:-1] & valid[:, 1:-1, 2:] & valid[:, 1:-1, :-2] & valid[:, 2:, 1:-1] & valid[:, :-2, 1:-1]
        )
        usable[:, 1:-1, 1:-1] &= neighbours

    # Orient toward the camera, matching the renderer's own convention. Without this the
    # comparison against a rendered normal would be dominated by an arbitrary global sign.
    facing = (interior * view_dirs).sum(-1, keepdim=True) > 0.0
    normals = torch.where(facing, -interior, interior)
    return normals * usable.unsqueeze(-1), usable


def depth_gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    """Central-difference gradient magnitude of a [B, H, W, 1] depth map, relative to depth.

    Scale-free, so a threshold on it means the same thing in a scene measured in metres
    and one measured in normalized units. Used to identify the occlusion boundaries where
    an expected depth is least likely to describe a real surface.
    """
    squeezed = depth.squeeze(-1)
    gradient = torch.zeros_like(squeezed)
    dx = squeezed[:, 1:-1, 2:] - squeezed[:, 1:-1, :-2]
    dy = squeezed[:, 2:, 1:-1] - squeezed[:, :-2, 1:-1]
    scale = squeezed[:, 1:-1, 1:-1].abs().clamp_min(1e-6)
    gradient[:, 1:-1, 1:-1] = torch.sqrt(dx**2 + dy**2) / scale
    return gradient
