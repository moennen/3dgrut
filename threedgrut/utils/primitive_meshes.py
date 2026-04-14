# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compute primitive meshes from the model in torch (no clamping, model codepath only)."""

from __future__ import annotations

import torch

from threedgrut.utils.misc import quaternion_to_so3

# Tetrahedron geometry matching particlePrimitives.cu (enclosing regular tetrahedra).
# r = inradius = 1, s = edge = sqrt(24), h = height = 4, face inradius = sqrt(2).
TETRAHEDRON_NUM_VRT = 4
TETRAHEDRON_NUM_TRI = 4
TETRAHEDRON_EDGE = 4.898979485566356  # sqrt(24)
TETRAHEDRON_FACE_IN_RADIUS = 1.4142135623730951  # sqrt(2)
TETRAHEDRON_FACE_HEIGHT = 4.242640687119285  # edge * sqrt(3) / 2
TETRAHEDRON_HEIGHT = 4.0
TETRAHEDRON_IN_RADIUS = 1.0

# Local vertex positions (same order as particlePrimitives).
TETRAHEDRON_LOCAL_VRT = torch.tensor(
    [
        [-0.5 * TETRAHEDRON_EDGE, -TETRAHEDRON_FACE_IN_RADIUS, -1.0],
        [0.0, TETRAHEDRON_FACE_HEIGHT - TETRAHEDRON_FACE_IN_RADIUS, -1.0],
        [0.0, 0.0, TETRAHEDRON_HEIGHT - TETRAHEDRON_IN_RADIUS],
        [0.5 * TETRAHEDRON_EDGE, -TETRAHEDRON_FACE_IN_RADIUS, -1.0],
    ],
    dtype=torch.float32,
)

# Triangle indices per tetrahedron (same as particlePrimitives).
TETRAHEDRON_TRI = torch.tensor(
    [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]],
    dtype=torch.int64,
)


def _kernel_scale_torch(
    density: torch.Tensor,
    min_response: float = 0.5,
    degree: float = 2.0,
) -> torch.Tensor:
    """Scalar kernel scale per particle. No clamping (responseModulation=1)."""
    min_response = min(min_response, 0.97)
    dev = density.device
    dty = density.dtype
    n = density.shape[0]
    if degree == -1:
        return torch.full((n,), 3.0, device=dev, dtype=dty)
    if degree < -1:
        k = abs(degree)
        s = 1.0 / (3.0**k)
        log_mr = torch.log(torch.tensor(min_response, device=dev, dtype=dty))
        term = ((1.0 / (log_mr - 1.0)) + 1.0) / s
        return torch.pow(term, 1.0 / k).expand(n)
    if degree == 0:
        scale = ((1.0 - min_response) / 3.0) / -0.329630334487
        return torch.full((n,), scale, device=dev, dtype=dty)
    b = degree
    a = -4.5 / (3.0**b)
    log_mr = torch.log(torch.tensor(min_response, device=dev, dtype=dty))
    return torch.pow(log_mr / a, 1.0 / b).expand(n)


def compute_primitive_mesh(
    primitive_name: str,
    model: object,
    *,
    min_response: float = 0.5,
    degree: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute primitive mesh from model in torch.

    Returns device tensors (vertices, faces) suitable for polyscope. Only tetrahedra
    implemented; follows particlePrimitives tetrahedron definition without clamping.

    Args:
        primitive_name: Currently only "tetrahedra" is supported.
        model: ExportableModel-like with get_positions(), get_rotation(), get_scale(), get_density().
        min_response: Kernel min response for scale (no clamping).
        degree: Kernel degree for scale.

    Returns:
        vertices: (V, 3) float tensor on model device.
        faces: (F, 3) int64 tensor on model device (0-based indices into vertices).
    """
    if primitive_name != "tetrahedra":
        raise NotImplementedError(f"primitive_mesh only supports 'tetrahedra', got '{primitive_name}'")

    device = getattr(model, "device", "cuda")
    pos = model.get_positions()
    rot = model.get_rotation()
    scl = model.get_scale()
    dns = model.get_density()

    n = pos.shape[0]
    if n == 0:
        return (
            torch.empty(0, 3, device=device, dtype=torch.float32),
            torch.empty(0, 3, device=device, dtype=torch.int64),
        )

    # Ensure local template on same device/dtype as model data
    local_vrt = TETRAHEDRON_LOCAL_VRT.to(device=device, dtype=pos.dtype)
    local_tri = TETRAHEDRON_TRI.to(device=device)

    # Per-particle kernel scale (scalar), then broadcast to (N, 3) with per-axis scale
    dns_1d = dns.squeeze(-1) if dns.dim() > 1 else dns
    k = _kernel_scale_torch(dns_1d, min_response=min_response, degree=degree)
    kscl = (k.unsqueeze(-1) * scl).to(pos.dtype)

    # R: (N, 3, 3). Local vertices (4, 3) -> (N, 4, 3) with scale then rotate and translate
    R = quaternion_to_so3(rot)
    # vertices_local: (N, 4, 3) = (4, 3) * (N, 1, 3)
    vertices_local = local_vrt.unsqueeze(0) * kscl.unsqueeze(1)
    # rotated: (N, 4, 3) = (N, 4, 3) @ (N, 3, 3).T
    vertices_world = torch.bmm(vertices_local, R.transpose(1, 2)) + pos.unsqueeze(1)

    V = n * TETRAHEDRON_NUM_VRT
    F = n * TETRAHEDRON_NUM_TRI
    vertices = vertices_world.reshape(V, 3)
    # Face indices: per-tetra offset by vertex start index
    base = torch.arange(n, device=device, dtype=torch.int64).unsqueeze(1) * TETRAHEDRON_NUM_VRT
    faces = (local_tri.unsqueeze(0) + base.unsqueeze(2)).reshape(F, 3)

    return vertices, faces
