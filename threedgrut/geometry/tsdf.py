# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TSDF mesh extraction from calibrated depth maps and optional source RGB.

This is the practical RGB-D fusion baseline used by AmbiSuR and many Gaussian reconstruction
projects. It deliberately lives outside the renderer: the only inputs are calibrated depth maps
and poses, so a checkpoint from either tracer can be meshed the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class TSDFConfig:
    """Parameters that materially define a TSDF mesh and must be recorded with it."""

    voxel_size: float
    truncation: float
    max_depth: float
    min_component_triangles: int = 50
    keep_largest_components: int = 1

    def __post_init__(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
        if self.truncation < self.voxel_size:
            raise ValueError("truncation must be at least one voxel")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {self.max_depth}")
        if self.min_component_triangles < 0:
            raise ValueError("min_component_triangles must be non-negative")
        if self.keep_largest_components < 1:
            raise ValueError("keep_largest_components must be at least one")


@dataclass(frozen=True)
class DepthFrame:
    """One calibrated depth image to fuse into a TSDF.

    ``world_to_camera`` follows Open3D/COLMAP's convention.  Keeping the depth convention on
    the frame makes the conversion at the one Open3D boundary explicit: callers which already
    have z-depth (monocular priors) and callers which render ray distance (3dgrut) now use the
    exact same fusion code. ``rgb`` is optional for backward compatibility, but when supplied it
    is fused into the TSDF and becomes per-vertex mesh color. It must be aligned ``[H, W, 3]``
    RGB in uint8 or floating-point ``[0, 1]`` form.
    """

    depth: np.ndarray
    K: np.ndarray
    world_to_camera: np.ndarray
    convention: Literal["ray", "z"]
    valid: np.ndarray | None = None
    rgb: np.ndarray | None = None


def _ray_length_grid(K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Per-pixel ratio between Euclidean ray distance and camera z-depth."""
    u = np.arange(width, dtype=np.float32) + 0.5
    v = np.arange(height, dtype=np.float32)[:, None] + 0.5
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.sqrt(1.0 + x[None, :] ** 2 + y**2).astype(np.float32)


def ray_distance_to_z_depth(ray_distance: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert Euclidean ray distance to a pinhole camera's z-depth.

    3dgrut renders Euclidean distance along a ray. Open3D's RGB-D TSDF integrator expects
    camera z-depth. Supplying the former makes an oblique plane bow outward, which can look
    plausible in a mesh while being geometrically wrong.
    """
    ray_distance = np.asarray(ray_distance, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if ray_distance.ndim != 2:
        raise ValueError(f"ray_distance must be [H, W], got {ray_distance.shape}")
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"K must be a pinhole 3x3 matrix with positive focal lengths, got {K}")

    ray_length = _ray_length_grid(K, *ray_distance.shape)
    return ray_distance / ray_length


def z_depth_to_ray_distance(z_depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert a pinhole camera's z-depth to Euclidean distance along each ray."""
    z_depth = np.asarray(z_depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if z_depth.ndim != 2:
        raise ValueError(f"z_depth must be [H, W], got {z_depth.shape}")
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"K must be a pinhole 3x3 matrix with positive focal lengths, got {K}")
    return z_depth * _ray_length_grid(K, *z_depth.shape)


def rgb_to_uint8(rgb: np.ndarray | None, depth_shape: tuple[int, int]) -> np.ndarray:
    """Validate an aligned RGB image and convert it to Open3D's RGB8 format.

    A missing image intentionally produces black colors so depth-only callers retain their
    previous behaviour. Floating point source images follow the usual renderer convention of
    RGB values in ``[0, 1]``; integer images must already be in the ``[0, 255]`` range.
    """
    height, width = depth_shape
    if rgb is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    rgb = np.asarray(rgb)
    if rgb.shape != (height, width, 3):
        raise ValueError(f"rgb image {rgb.shape} must match depth as [H, W, 3] ({height}, {width}, 3)")
    if np.issubdtype(rgb.dtype, np.floating):
        if not np.all(np.isfinite(rgb)):
            raise ValueError("rgb image must contain only finite values")
        if np.any((rgb < 0.0) | (rgb > 1.0)):
            raise ValueError("floating-point rgb image must be in [0, 1]")
        return np.rint(rgb * 255.0).astype(np.uint8)
    if not np.issubdtype(rgb.dtype, np.integer):
        raise ValueError(f"rgb image must be uint8 or floating point, got {rgb.dtype}")
    if np.any((rgb < 0) | (rgb > 255)):
        raise ValueError("integer rgb image must be in [0, 255]")
    return rgb.astype(np.uint8, copy=False)


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TSDF mesh extraction requires the optional mesh dependency. Install with "
            "`pip install '.[mesh]'` (or install open3d)."
        ) from exc
    return o3d


def create_volume(config: TSDFConfig):
    """Create Open3D's scalable TSDF volume with the configured world-unit parameters."""
    o3d = _open3d()
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_size,
        sdf_trunc=config.truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def integrate_depth(
    volume,
    depth: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    config: TSDFConfig,
    convention: Literal["ray", "z"],
    valid: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> None:
    """Integrate one calibrated RGB-D map into an Open3D TSDF volume.

    Open3D requires z-depth.  This is the only place that interprets a depth convention; all
    extraction paths should pass through it instead of duplicating a conversion.
    """
    o3d = _open3d()
    K = np.asarray(K, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if world_to_camera.shape != (4, 4):
        raise ValueError(f"world_to_camera must be 4x4, got {world_to_camera.shape}")

    if convention == "ray":
        z_depth = ray_distance_to_z_depth(depth, K)
    elif convention == "z":
        z_depth = np.asarray(depth, dtype=np.float32)
    else:
        raise ValueError(f"depth convention must be 'ray' or 'z', got {convention!r}")
    usable = np.isfinite(z_depth) & (z_depth > 0) & (z_depth <= config.max_depth)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != z_depth.shape:
            raise ValueError(f"valid mask {valid.shape} does not match depth {z_depth.shape}")
        usable &= valid
    z_depth = np.where(usable, z_depth, 0.0).astype(np.float32)

    height, width = z_depth.shape
    color = o3d.geometry.Image(rgb_to_uint8(rgb, (height, width)))
    # Float32 depth plus scale 1 preserves normalized scene units; the common uint16-mm path
    # silently quantizes small normalized scenes.
    depth = o3d.geometry.Image(z_depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=config.max_depth, convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    volume.integrate(rgbd, intrinsic, world_to_camera)


def integrate_ray_depth(
    volume,
    ray_distance: np.ndarray,
    K: np.ndarray,
    world_to_camera: np.ndarray,
    config: TSDFConfig,
    valid: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> None:
    """Backward-compatible ray-distance wrapper around :func:`integrate_depth`."""
    integrate_depth(volume, ray_distance, K, world_to_camera, config, "ray", valid, rgb)


def filter_components(mesh, config: TSDFConfig):
    """Remove tiny floaters while keeping a configurable number of major components."""
    o3d = _open3d()
    if len(mesh.triangles) == 0:
        return mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        clusters, triangle_counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    triangle_counts = np.asarray(triangle_counts)
    ranked = np.argsort(triangle_counts)[::-1]
    keep_clusters = ranked[: config.keep_largest_components]
    keep = np.isin(clusters, keep_clusters) & (triangle_counts[clusters] >= config.min_component_triangles)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def extract_mesh(volume, config: TSDFConfig):
    """Extract, clean, and compute normals for the fused surface."""
    mesh = volume.extract_triangle_mesh()
    mesh = filter_components(mesh, config)
    if len(mesh.triangles):
        mesh.compute_vertex_normals()
    return mesh


def fuse_depth_frames(frames: Iterable[DepthFrame], config: TSDFConfig):
    """Fuse posed ray- or z-depth frames and return the cleaned, optionally colored TSDF mesh.

    This is intentionally the common entry point for checkpoint renders and external depth
    models.  It consumes an iterator, so a benchmark need not keep every prediction in memory.
    """
    volume = create_volume(config)
    count = 0
    for frame in frames:
        integrate_depth(
            volume,
            frame.depth,
            frame.K,
            frame.world_to_camera,
            config,
            frame.convention,
            valid=frame.valid,
            rgb=frame.rgb,
        )
        count += 1
    if count == 0:
        raise ValueError("Cannot fuse an empty depth-frame sequence")
    return extract_mesh(volume, config)
