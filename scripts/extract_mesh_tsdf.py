# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a mesh from a 3dgrut checkpoint by fusing rendered training depths.

This is the AmbiSuR-style TSDF baseline: render every training view, integrate calibrated depth
maps, extract the TSDF zero crossing, then remove small disconnected components. 3dgrut depth is
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

import numpy as np
import torch
from omegaconf import open_dict

from threedgrut.geometry.tsdf import TSDFConfig, create_volume, extract_mesh, integrate_ray_depth


def fit_pinhole(rays_dir: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit camera intrinsics to rendered camera-space ray directions."""
    height, width = rays_dir.shape[:2]
    rays = rays_dir.astype(np.float64)
    if np.any(rays[..., 2] <= 0):
        raise ValueError("Mesh extraction supports only pinhole cameras with positive camera-space z rays")
    x_over_z, y_over_z = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
    u = np.broadcast_to(np.arange(width, dtype=np.float64) + 0.5, (height, width))
    v = np.broadcast_to((np.arange(height, dtype=np.float64) + 0.5)[:, None], (height, width))

    def solve(ratio: np.ndarray, pixel: np.ndarray) -> tuple[float, float, np.ndarray]:
        design = np.stack([ratio.ravel(), np.ones(ratio.size)], axis=1)
        (focal, centre), *_ = np.linalg.lstsq(design, pixel.ravel(), rcond=None)
        return float(focal), float(centre), design @ np.array([focal, centre]) - pixel.ravel()

    fx, cx, residual_u = solve(x_over_z, u)
    fy, cy, residual_v = solve(y_over_z, v)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, float(max(np.abs(residual_u).max(), np.abs(residual_v).max()))


def world_to_camera(pose: np.ndarray) -> np.ndarray:
    """Return Open3D's world-to-camera extrinsic from a camera-to-world pose."""
    rotation = pose[:3, :3]
    centre = pose[:3, 3]
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = -rotation.T @ centre
    return extrinsic


def extract(args: argparse.Namespace) -> dict:
    import threedgrut.datasets as datasets
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY, expected_depth

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    with open_dict(conf):
        if args.scene_path:
            conf.path = args.scene_path

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    train_dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    config = TSDFConfig(
        voxel_size=args.voxel_size,
        truncation=args.truncation or args.voxel_size * args.truncation_voxels,
        max_depth=args.max_depth or model.scene_extent * args.max_depth_extent,
        min_component_triangles=args.min_component_triangles,
        keep_largest_components=args.keep_largest_components,
    )
    volume = create_volume(config)
    residuals: list[float] = []

    with torch.no_grad():
        for index, batch in enumerate(loader):
            gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = model(gpu_batch)
            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], args.min_opacity)
            depth_np = depth.squeeze().float().cpu().numpy()
            valid = confident.squeeze().cpu().numpy()
            rays = gpu_batch.rays_dir[0].float().cpu().numpy()
            K, residual = fit_pinhole(rays)
            if residual > args.max_pinhole_residual:
                raise ValueError(
                    f"View {index} cannot be represented as a pinhole camera within "
                    f"{args.max_pinhole_residual} px (residual {residual:.3g}px)"
                )
            residuals.append(residual)
            pose = gpu_batch.T_to_world[0].float().cpu().numpy()
            integrate_ray_depth(volume, depth_np, K, world_to_camera(pose), config, valid=valid)

    mesh = extract_mesh(volume, config)
    import open3d as o3d

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_normals=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "mesh": str(output),
        "views": len(residuals),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "max_pinhole_residual_px": max(residuals, default=0.0),
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
