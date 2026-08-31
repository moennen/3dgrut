# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared colored TSDF extraction from a trained 3dgrut model and its training views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from threedgrut.geometry.tsdf import DepthFrame, TSDFConfig, fuse_depth_frames


@dataclass(frozen=True)
class CheckpointMeshResult:
    """Mesh and rendering diagnostics from training-view TSDF fusion."""

    mesh: object
    views: int
    max_pinhole_residual_px: float


def fit_pinhole(rays_dir: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit pinhole intrinsics to rendered camera-space ray directions."""
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


def extract_colored_tsdf_mesh(
    model,
    train_dataset,
    config: TSDFConfig,
    *,
    min_opacity: float = 0.5,
    max_pinhole_residual: float = 0.1,
    num_workers: int = 2,
) -> CheckpointMeshResult:
    """Render training RGB-D frames and fuse a colored mesh with the common TSDF implementation."""
    from threedgrut.utils.depth_normal_metrics import expected_depth

    model.build_acc()
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    frames: list[DepthFrame] = []
    residuals: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = model(gpu_batch)
            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], min_opacity)
            K, residual = fit_pinhole(gpu_batch.rays_dir[0].float().cpu().numpy())
            if residual > max_pinhole_residual:
                raise ValueError(
                    f"View {index} cannot be represented as a pinhole camera within "
                    f"{max_pinhole_residual} px (residual {residual:.3g}px)"
                )
            residuals.append(residual)
            frames.append(
                DepthFrame(
                    depth=depth.squeeze().float().cpu().numpy(),
                    K=K,
                    world_to_camera=world_to_camera(gpu_batch.T_to_world[0].float().cpu().numpy()),
                    convention="ray",
                    valid=confident.squeeze().cpu().numpy(),
                    rgb=gpu_batch.rgb_gt[0].float().cpu().numpy(),
                )
            )
    return CheckpointMeshResult(fuse_depth_frames(frames, config), len(frames), max(residuals, default=0.0))
