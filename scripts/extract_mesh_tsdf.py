# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a colored mesh from a 3dgrut checkpoint by fusing rendered training RGB-D views.

This is the AmbiSuR-style TSDF baseline: render every training view, integrate calibrated depth
maps together with their source RGB, extract the TSDF zero crossing, then remove small
disconnected components. The exported PLY has source-image vertex colors. 3dgrut depth is
Euclidean ray distance, so the integrator converts it to camera z-depth before handing it to
Open3D.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/extract_mesh_tsdf.py \
      --checkpoint runs/example/ckpt_last.pt --out mesh.ply --voxel-size 0.002
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import open_dict

from threedgrut.geometry.checkpoint_mesh import extract_colored_tsdf_mesh
from threedgrut.geometry.tsdf import TSDFConfig


def extract(args: argparse.Namespace) -> dict:
    import threedgrut.datasets as datasets
    from threedgrut.model.model import MixtureOfGaussians

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    with open_dict(conf):
        if args.scene_path:
            conf.path = args.scene_path

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    train_dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    config = TSDFConfig(
        voxel_size=args.voxel_size,
        truncation=args.truncation or args.voxel_size * args.truncation_voxels,
        max_depth=args.max_depth or model.scene_extent * args.max_depth_extent,
        min_component_triangles=args.min_component_triangles,
        keep_largest_components=args.keep_largest_components,
    )
    result = extract_colored_tsdf_mesh(
        model,
        train_dataset,
        config,
        min_opacity=args.min_opacity,
        max_pinhole_residual=args.max_pinhole_residual,
        num_workers=args.num_workers,
    )
    mesh = result.mesh
    import open3d as o3d

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_normals=True, write_vertex_colors=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "mesh": str(output),
        "views": result.views,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "vertex_colors": len(mesh.vertex_colors),
        "color_source": "training RGB images",
        "max_pinhole_residual_px": result.max_pinhole_residual_px,
        "depth_convention": "ray converted to z for TSDF integration",
        "tsdf": config.__dict__,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Output mesh PLY path")
    parser.add_argument("--scene-path", help="Override checkpoint dataset path")
    parser.add_argument("--voxel-size", type=float, required=True, help="TSDF voxel size in scene units")
    parser.add_argument("--truncation", type=float, default=None, help="TSDF truncation in scene units")
    parser.add_argument("--truncation-voxels", type=float, default=5.0)
    parser.add_argument("--max-depth", type=float, default=None, help="Ignore z-depth beyond this distance")
    parser.add_argument("--max-depth-extent", type=float, default=2.0)
    parser.add_argument("--min-opacity", type=float, default=0.5)
    parser.add_argument("--min-component-triangles", type=int, default=50)
    parser.add_argument("--keep-largest-components", type=int, default=1)
    parser.add_argument("--max-pinhole-residual", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(extract(args), indent=2))


if __name__ == "__main__":
    main()
