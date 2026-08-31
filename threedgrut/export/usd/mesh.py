# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author extracted triangle meshes into a standard USD scene."""

from __future__ import annotations

import numpy as np
from pxr import Sdf, UsdGeom, Vt

from threedgrut.export.usd.stage_utils import create_gaussian_model_root


def add_mesh_to_usd_stage(
    stage,
    mesh,
    *,
    root_path: str = "/World/Mesh",
    normalizing_transform: np.ndarray | None = None,
    coordinate_transform: np.ndarray | None = None,
) -> str:
    """Add an Open3D-style colored triangle mesh to ``stage``.

    The root transform follows the Gaussian export convention exactly, so the mesh and
    ParticleField occupy the same coordinate frame. Vertex RGB is authored as the standard USD
    ``primvars:displayColor`` vertex primvar.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices must have shape [N, 3], got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"mesh triangles must have shape [M, 3], got {triangles.shape}")
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("cannot author an empty mesh into USD")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("mesh triangle indices must refer to existing vertices")

    create_gaussian_model_root(
        stage,
        root_path=root_path,
        normalizing_transform=normalizing_transform,
        coordinate_transform=coordinate_transform,
    )
    usd_mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/Surface")
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1)))

    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if len(colors):
        if colors.shape != vertices.shape:
            raise ValueError(f"mesh vertex colors must have shape {vertices.shape}, got {colors.shape}")
        if not np.all(np.isfinite(colors)):
            raise ValueError("mesh vertex colors must be finite")
        display_color = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        display_color.Set(Vt.Vec3fArray.FromNumpy(np.clip(colors, 0.0, 1.0)))

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(normals):
        if normals.shape != vertices.shape:
            raise ValueError(f"mesh vertex normals must have shape {vertices.shape}, got {normals.shape}")
        usd_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return str(usd_mesh.GetPath())
