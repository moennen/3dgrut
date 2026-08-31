# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from threedgrut.export.usd.mesh import add_mesh_to_usd_stage
from threedgrut.export.usd.stage_utils import initialize_usd_stage


def test_add_mesh_to_usd_stage_authors_colored_triangle_mesh():
    mesh = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        vertex_colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        vertex_normals=np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32),
    )
    stage = initialize_usd_stage()

    assert add_mesh_to_usd_stage(stage, mesh) == "/World/Mesh/Surface"

    usd_mesh = stage.GetPrimAtPath("/World/Mesh/Surface")
    assert len(usd_mesh.GetAttribute("points").Get()) == 3
    assert list(usd_mesh.GetAttribute("faceVertexCounts").Get()) == [3]
    assert list(usd_mesh.GetAttribute("faceVertexIndices").Get()) == [0, 1, 2]
    colors = usd_mesh.GetAttribute("primvars:displayColor")
    assert len(colors.Get()) == 3
    assert colors.GetMetadata("interpolation") == "vertex"
    assert len(usd_mesh.GetAttribute("normals").Get()) == 3


def test_add_mesh_to_usd_stage_rejects_empty_mesh():
    empty = SimpleNamespace(
        vertices=np.empty((0, 3), dtype=np.float32),
        triangles=np.empty((0, 3), dtype=np.int32),
        vertex_colors=np.empty((0, 3), dtype=np.float32),
        vertex_normals=np.empty((0, 3), dtype=np.float32),
    )

    try:
        add_mesh_to_usd_stage(initialize_usd_stage(), empty)
    except ValueError as error:
        assert str(error) == "cannot author an empty mesh into USD"
    else:  # pragma: no cover - keeps the failure message focused
        raise AssertionError("expected an empty mesh to be rejected")
