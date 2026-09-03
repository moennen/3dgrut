# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark monocular depth models on OB3D, DTU and Tanks and Temples.

The same three conditions are evaluated for every model: raw output, one scale per frame, and
one affine per frame.  Scale/affine use the benchmark scan z-buffer as an explicit *oracle*
alignment source; this exposes the geometry of a relative prior, but is not a deployable depth
pipeline.  Raw metrics remain useful only for metric models such as MoGe-3.

The benchmark writes raw maps, aligned z/ray maps, recall JSON, TSDF meshes and one JSONL record
per cell.  TSDF fusion is delegated to :mod:`threedgrut.geometry.tsdf`, the exact same posed-depth
path used by ``scripts/extract_mesh_tsdf.py``.

Run with the optional upstream packages available::

  PYTHONPATH=/mnt/oss/meshdeps:/mnt/oss/MoGe:/mnt/oss/moge3deps:/mnt/oss/Depth-Anything-3/src:/mnt/oss/da3deps \\
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/evaluate_depth_models.py \\
  --out-dir /tmp/depth-benchmark --max-frames 8 --max-image-side 640
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEPTHRECALL_ROOT = ROOT / "tools" / "depthrecall"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DEPTHRECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTHRECALL_ROOT))

from depthrecall.dtu import above_ground_plane_mask, observed_volume_mask, read_dtu_views, read_ground_plane
from depthrecall.io_cameras import View, read_colmap_views
from depthrecall.io_points import apply_alignment, read_ply, voxel_downsample
from depthrecall.metric import MetricConfig
from depthrecall.metric import evaluate as evaluate_recall
from depthrecall.surface import DEFAULT_QUERY_CHUNK_SIZE, evaluate_surface
from depthrecall.tnt import OFFICIAL_TAU_METRES, crop_volume_mask, gt_to_render_alignment, read_crop_volume
from threedgrut.datasets.gt_geometry import depth_validity, find_gt_paths, read_gt_map, resize_gt_map
from threedgrut.datasets.pseudo_depth import BACKENDS
from threedgrut.geometry.tsdf import (
    DepthFrame,
    TSDFConfig,
    adaptive_voxel_size,
    expand_world_bounds,
    fuse_depth_frames,
    z_depth_to_ray_distance,
)

MODELS = {
    "dav2": ("transformers", "depth-anything/Depth-Anything-V2-Base-hf"),
    "dav3": ("depth_anything_3", "depth-anything/DA3MONO-LARGE"),
    "moge3": ("moge3", "/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt"),
}
ALIGNMENTS: tuple[Literal["raw", "scale", "affine"], ...] = ("raw", "scale", "affine")

# Keep these suite definitions in the runner rather than deriving them from directory iteration:
# a benchmark invocation must mean the same experiment on every machine.  TnT Church is present
# in the official point-cloud tree but does not have the matching GOF/COLMAP reconstruction, so it
# cannot be evaluated by this posed-depth protocol.
FULL_SCENES = {
    "ob3d": (
        "archiviz-flat",
        "barbershop",
        "bistro",
        "classroom",
        "emerald-square",
        "fisher-hut",
        "lone-monk",
        "pavillion",
        "restroom",
        "san-miguel",
        "sponza",
        "sun-temple",
    ),
    "dtu": (
        "scan105",
        "scan106",
        "scan110",
        "scan114",
        "scan118",
        "scan122",
        "scan24",
        "scan37",
        "scan40",
        "scan55",
        "scan63",
        "scan65",
        "scan69",
        "scan83",
        "scan97",
    ),
    "tnt": ("Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"),
}
# Fixed every-third scene selections from FULL_SCENES.  Do not randomise this set: it
# is the comparable reduced protocol used for development and regression checks.
REDUCED_SCENES = {suite: scenes[::3] for suite, scenes in FULL_SCENES.items()}


def selected_scenes(
    dataset_scale: Literal["full", "reduced"], overrides: dict[str, str | None]
) -> dict[str, tuple[str, ...]]:
    """Resolve a reproducible dataset preset, allowing an explicit suite-level override."""
    preset = FULL_SCENES if dataset_scale == "full" else REDUCED_SCENES
    selected = {}
    for suite, fallback in preset.items():
        override = overrides[suite]
        if override is None:
            selected[suite] = fallback
            continue
        scenes = tuple(scene.strip() for scene in override.split(",") if scene.strip())
        if not scenes:
            raise ValueError(f"--{suite}-scenes must name at least one scene")
        selected[suite] = scenes
    return selected


@dataclass(frozen=True)
class Frame:
    name: str
    image_path: Path
    view: View
    gt_z: np.ndarray | None = None


@dataclass(frozen=True)
class BenchmarkScene:
    family: Literal["dtu", "tnt"]
    name: str
    frames: list[Frame]
    gt_points: np.ndarray
    gt_to_world: np.ndarray | None
    recall_taus: np.ndarray
    mesh_taus: np.ndarray
    masks: list[str]
    visibility: dict[str, Path]
    dtu_obsmask: Path | None = None
    tnt_crop: Path | None = None


def transform_aabb(bounds: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform an AABB and return its conservative axis-aligned enclosure."""
    bounds = np.asarray(bounds, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if bounds.shape != (2, 3) or transform.shape != (4, 4):
        raise ValueError("bounds must be (2, 3) and transform must be (4, 4)")
    corners = np.stack(np.meshgrid(*zip(bounds[0], bounds[1]), indexing="ij"), axis=-1).reshape(-1, 3)
    homogeneous = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
    transformed = (homogeneous @ transform.T)[:, :3]
    return np.stack([transformed.min(axis=0), transformed.max(axis=0)])


def tnt_crop_aabb(path: Path) -> np.ndarray:
    """Conservative AABB of the official TnT selection-polygon volume, in scan coordinates."""
    axis, axis_min, axis_max, polygon = read_crop_volume(path)
    bounds = np.stack([polygon.min(axis=0), polygon.max(axis=0)])
    bounds[0, axis], bounds[1, axis] = axis_min, axis_max
    return bounds


def dtu_obsmask_aabb(path: Path) -> np.ndarray:
    """AABB of DTU's official reconstruction-side observation volume, in scan millimetres."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("DTU ObsMask files need scipy") from exc
    data = loadmat(str(path))
    lower = np.asarray(data["BB"], dtype=np.float64)[0]
    resolution = float(np.asarray(data["Res"]).reshape(-1)[0])
    shape = np.asarray(data["ObsMask"]).shape
    if lower.shape != (3,) or len(shape) != 3 or resolution <= 0:
        raise ValueError(f"Malformed DTU observation mask: {path}")
    return np.stack([lower, lower + resolution * (np.asarray(shape) - 1)])


def benchmark_source_bounds(scene: BenchmarkScene) -> tuple[np.ndarray, str]:
    """Return the official, explicitly GT-assisted reconstruction bound in scan coordinates."""
    if scene.dtu_obsmask is not None:
        bounds, source = dtu_obsmask_aabb(scene.dtu_obsmask), "DTU official ObsMask volume"
    elif scene.tnt_crop is not None:
        bounds, source = tnt_crop_aabb(scene.tnt_crop), "TnT official SelectionPolygonVolume"
    else:
        raise ValueError(f"{scene.family}/{scene.name} has no official reconstruction bound")
    return bounds, source


def benchmark_fusion_bounds(scene: BenchmarkScene) -> tuple[np.ndarray, str]:
    """Return the official, explicitly GT-assisted reconstruction bound in render coordinates."""
    bounds, source = benchmark_source_bounds(scene)
    return (transform_aabb(bounds, scene.gt_to_world), source) if scene.gt_to_world is not None else (bounds, source)


def tsdf_config_for_bounds(
    voxel_size: float,
    max_depth: float,
    bounds: np.ndarray | None,
    source: str | None,
    bound_mode: Literal["none", "benchmark"],
    max_voxels_per_axis: int | None,
) -> tuple[TSDFConfig, dict]:
    """Choose a recorded TSDF configuration from an optional already-render-space bound."""
    effective_voxel = adaptive_voxel_size(voxel_size, bounds, max_voxels_per_axis)
    padded_bounds = expand_world_bounds(bounds, effective_voxel * 5) if bounds is not None else None
    config = TSDFConfig(effective_voxel, effective_voxel * 5, max_depth, world_bounds=padded_bounds)
    metadata = {
        "requested_voxel_size": voxel_size,
        "voxel_size": config.voxel_size,
        "truncation": config.truncation,
        "max_depth": config.max_depth,
        "bounds_mode": bound_mode,
        "bounds_source": source,
        "bounds_render": padded_bounds.tolist() if padded_bounds is not None else None,
        "max_voxels_per_axis": max_voxels_per_axis,
    }
    if bounds is not None:
        grid_shape = np.ceil((bounds[1] - bounds[0]) / config.voxel_size).astype(int)
        metadata["grid_shape"] = grid_shape.tolist()
        metadata["dense_grid_voxels"] = int(np.prod(grid_shape, dtype=np.int64))
    return config, metadata


def tsdf_config_for_scene(
    scene: BenchmarkScene,
    voxel_size: float,
    max_depth: float,
    bound_mode: Literal["none", "benchmark"],
    max_voxels_per_axis: int | None,
) -> tuple[TSDFConfig, dict]:
    """Choose a recorded TSDF configuration without hiding benchmark-assisted culling."""
    bounds, source = benchmark_fusion_bounds(scene) if bound_mode == "benchmark" else (None, None)
    return tsdf_config_for_bounds(voxel_size, max_depth, bounds, source, bound_mode, max_voxels_per_axis)


def _scaled_shape(height: int, width: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None or max(height, width) <= max_side:
        return height, width
    scale = max_side / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize(array: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    if array.shape[:2] == shape:
        return np.asarray(array)
    # PIL keeps float maps in mode F.  Ground truth uses nearest; predictions use bilinear,
    # matching the pseudo-depth cache rather than quantising the model's smooth output.
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(np.asarray(array, dtype=np.float32))
    return np.asarray(image.resize((shape[1], shape[0]), method), dtype=np.float32)


def _resize_rgb(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image)


def _scale_view(view: View, shape: tuple[int, int]) -> View:
    height, width = shape
    if (view.height, view.width) == shape:
        return view
    K = view.K.copy()
    K[0] *= width / view.width
    K[1] *= height / view.height
    return replace(view, width=width, height=height, K=K)


def _z_from_ray(ray: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Reuse the shared, tested convention conversion rather than restating the pinhole formula.
    from threedgrut.geometry.tsdf import ray_distance_to_z_depth

    return ray_distance_to_z_depth(ray, K)


def _fit_scale(source: np.ndarray, target: np.ndarray) -> float:
    denominator = float(source @ source)
    return float(source @ target / denominator) if denominator > 0 else float("nan")


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.stack([source, np.ones_like(source)], axis=1)
    coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coefficient[0]), float(coefficient[1])


def align_prediction(
    prediction: np.ndarray, gt_z: np.ndarray, quantity: str, alignment: Literal["raw", "scale", "affine"]
) -> np.ndarray:
    """Return z-depth after an alignment in the model's native depth quantity.

    DAv2 is affine in disparity, not depth; fitting it directly to z turns the requested ladder
    into a different and substantially weaker model.  The conversion happens only after fitting.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    gt_z = np.asarray(gt_z, dtype=np.float64)
    valid = np.isfinite(prediction) & (prediction > 0) & np.isfinite(gt_z) & (gt_z > 0)
    target = gt_z.copy()
    if quantity == "disparity":
        target[valid] = 1.0 / gt_z[valid]
    fitted = prediction.copy()
    if alignment == "scale":
        scale = _fit_scale(prediction[valid], target[valid])
        fitted *= scale
    elif alignment == "affine":
        scale, shift = _fit_affine(prediction[valid], target[valid])
        fitted = scale * fitted + shift
    elif alignment != "raw":
        raise ValueError(f"Unknown alignment {alignment}")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / fitted if quantity == "disparity" else fitted
    return np.where(np.isfinite(z) & (z > 0), z, np.nan).astype(np.float32)


def depth_metrics(pred_z: np.ndarray, gt_z: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred_z) & (pred_z > 0) & np.isfinite(gt_z) & (gt_z > 0)
    if not valid.any():
        return {"valid_pixels": 0, "abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    pred, gt = pred_z[valid].astype(np.float64), gt_z[valid].astype(np.float64)
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "valid_pixels": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "mae": float(np.mean(np.abs(pred - gt))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def _camera_json_view(path: Path, image_shape: tuple[int, int]) -> View:
    camera = json.loads(path.read_text())[0]
    intrinsics = camera["intrinsics"]
    extrinsics = camera["extrinsics"]
    K = np.array(
        [[intrinsics["focal"], 0.0, intrinsics["cx"]], [0.0, intrinsics["focal"], intrinsics["cy"]], [0.0, 0.0, 1.0]]
    )
    return View(
        name=path.stem.replace("_cam", ""),
        width=int(camera["width"]),
        height=int(camera["height"]),
        K=K,
        R=np.asarray(extrinsics["rotation"], dtype=np.float64),
        t=np.asarray(extrinsics["translation"], dtype=np.float64),
        depth_path=Path(),
    )


def ob3d_frames(scene_root: Path, max_frames: int, max_side: int | None) -> list[Frame]:
    images = sorted(scene_root.joinpath("images").glob("*"))[:max_frames]
    gt_paths = find_gt_paths([str(path) for path in images], str(scene_root), "depths", "depth")
    frames = []
    for image, gt_path in zip(images, gt_paths):
        if gt_path is None:
            raise FileNotFoundError(f"No OB3D depth for {image}")
        original = np.asarray(Image.open(image))
        view = _camera_json_view(
            scene_root / "cameras" / f"{image.stem.replace('_rgb', '')}_cam.json", original.shape[:2]
        )
        shape = _scaled_shape(*original.shape[:2], max_side)
        view = _scale_view(view, shape)
        ray = resize_gt_map(read_gt_map(gt_path, 1)[..., 0], *shape)
        ray = np.where(depth_validity(ray), ray, np.nan)
        frames.append(Frame(image.stem.replace("_rgb", ""), image, view, _z_from_ray(ray, view.K)))
    return frames


def dtu_scene(
    root: Path, eval_root: Path, name: str, max_frames: int, max_side: int | None, out: Path
) -> BenchmarkScene:
    scan = name.removeprefix("scan")
    scene = root / name
    first = np.asarray(Image.open(next(scene.joinpath("images").glob("*.png"))))
    shape = _scaled_shape(*first.shape[:2], max_side)
    views = [
        _scale_view(view, shape)
        for view in read_dtu_views(scene / "cameras.npz", image_size=(first.shape[1], first.shape[0]), space="scan")
    ]
    points = read_ply(eval_root / "Points" / "stl" / f"stl{int(scan):03d}_total.ply").xyz
    plane_path = eval_root / "ObsMask" / f"Plane{int(scan)}.mat"
    plane = read_ground_plane(plane_path)
    points = points[above_ground_plane_mask(points, plane)]
    frames = [Frame(view.name, scene / "images" / f"{view.name}.png", view) for view in views[:max_frames]]
    visibility = visibility_maps(frames, points, out / "visibility")
    return BenchmarkScene(
        family="dtu",
        name=name,
        frames=frames,
        gt_points=points,
        gt_to_world=None,
        recall_taus=np.asarray([0.5, 1, 2, 5, 10, 20, 50], dtype=np.float64),
        mesh_taus=np.asarray([0.2], dtype=np.float64),
        masks=["DTU ground plane"],
        visibility=visibility,
        dtu_obsmask=eval_root / "ObsMask" / f"ObsMask{int(scan)}_10.mat",
    )


def tnt_scene(
    root: Path, reconstruction_root: Path, name: str, max_frames: int, max_side: int | None, out: Path
) -> BenchmarkScene:
    source = root / name
    reconstruction = reconstruction_root / "TrainingSet" / name
    all_views = read_colmap_views(reconstruction / "sparse" / "0", None)
    alignment = gt_to_render_alignment(
        source / f"{name}_trans.txt",
        source / f"{name}_COLMAP_SfM.log",
        np.stack([view.cam_center() for view in all_views]),
    )
    frames = []
    for view in all_views[:max_frames]:
        shape = _scaled_shape(view.height, view.width, max_side)
        frames.append(Frame(view.name, reconstruction / "images" / view.name, _scale_view(view, shape)))
    points = read_ply(source / f"{name}.ply").xyz
    points = points[crop_volume_mask(points, source / f"{name}.json")]
    visibility = visibility_maps(frames, apply_alignment(points, alignment.matrix), out / "visibility")
    tau = OFFICIAL_TAU_METRES[name]
    return BenchmarkScene(
        family="tnt",
        name=name,
        frames=frames,
        gt_points=points,
        gt_to_world=alignment.matrix,
        recall_taus=np.asarray([tau, 2 * tau, 5 * tau], dtype=np.float64),
        mesh_taus=np.asarray([tau], dtype=np.float64),
        masks=["TnT official crop volume"],
        visibility=visibility,
        tnt_crop=source / f"{name}.json",
    )


def visibility_maps(frames: list[Frame], points_world: np.ndarray, out_dir: Path) -> dict[str, Path]:
    """Z-buffer GT into ray maps for the self-occlusion-corrected recall metric."""
    from scripts.rasterize_depth import rasterize

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for frame in frames:
        path = out_dir / f"{frame.name}.npy"
        np.save(path, rasterize(frame.view, points_world, frame.view.width, frame.view.height, convention="ray"))
        paths[frame.name] = path
    return paths


def _view_with_depth(frame: Frame, path: Path) -> View:
    return replace(frame.view, depth_path=path, depth_convention="ray")


def _world_to_camera(view: View) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], transform[:3, 3] = view.R, view.t
    return transform


def _sample_mesh(mesh, count: int) -> np.ndarray:
    return np.asarray(mesh.sample_points_uniformly(number_of_points=count).points, dtype=np.float64)


def memory_snapshot() -> dict[str, int | None]:
    """Return process RSS and, when available, the enclosing cgroup's current memory in bytes."""
    rss_bytes = None
    for line in Path("/proc/self/status").read_text().splitlines() if Path("/proc/self/status").exists() else []:
        if line.startswith("VmRSS:"):
            rss_bytes = int(line.split()[1]) * 1024
            break
    cgroup_current = Path("/sys/fs/cgroup/memory.current")
    return {
        "rss_bytes": rss_bytes,
        "cgroup_current_bytes": int(cgroup_current.read_text()) if cgroup_current.exists() else None,
    }


def write_memory_snapshot(path: Path, stage: str) -> None:
    """Append and flush a sample so the last TSDF frame survives an OOM kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"stage": stage, **memory_snapshot()}) + "\n")
        handle.flush()


def predict_frames(predictor, frames: list[Frame]) -> list[np.ndarray]:
    """Infer model maps in memory; they are small beside TSDF/mesh allocations."""
    predictions = []
    for frame in frames:
        prediction = predictor.predict(_resize_rgb(frame.image_path, (frame.view.height, frame.view.width)))
        predictions.append(_resize(prediction, (frame.view.height, frame.view.width)))
    return predictions


def camera_fusion_max_depth(frames: list[Frame], radius_multiplier: float = 2.0) -> float:
    """AmbiSuR-style fusion cap: twice the nearest camera's distance to the camera focus.

    Monocular depths may have near-zero disparity values which turn into arbitrarily distant
    points after alignment.  A TSDF's integration range must come from the camera/scene geometry,
    rather than those predictions.  This mirrors AmbiSuR's adaptive extractor: it finds the point
    nearest all optical axes and uses twice the closest camera radius as ``depth_trunc``.
    """
    if not frames:
        raise ValueError("Cannot estimate a fusion range without camera frames")
    if radius_multiplier <= 0:
        raise ValueError(f"radius_multiplier must be positive, got {radius_multiplier}")
    origins = np.stack([frame.view.cam_center() for frame in frames])
    directions = np.stack([frame.view.R.T[:, 2] for frame in frames])
    projection = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    normal = np.transpose(projection, (0, 2, 1)) @ projection
    focus = np.linalg.pinv(normal.mean(axis=0)) @ (normal @ origins[..., None]).mean(axis=0)[:, 0]
    radius = float(np.linalg.norm(origins - focus, axis=1).min())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Could not derive a positive camera-focus radius, got {radius}")
    return radius_multiplier * radius


def prepare_aligned_depths(
    frames: list[Frame],
    predictions: list[np.ndarray],
    visibility: dict[str, Path],
    quantity: str,
    alignment: Literal["raw", "scale", "affine"],
    out_dir: Path,
) -> list[View]:
    """Write aligned ray-depth maps one frame at a time and return their views."""
    if len(frames) != len(predictions):
        raise ValueError("Each benchmark frame needs exactly one prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for frame, prediction in zip(frames, predictions):
        gt_z = _z_from_ray(np.load(visibility[frame.name]), frame.view.K)
        z = align_prediction(prediction, gt_z, quantity, alignment)
        path = out_dir / f"{frame.name}.npy"
        np.save(path, z_depth_to_ray_distance(z, frame.view.K))
        views.append(_view_with_depth(frame, path))
    return views


def saved_depth_frames(frames: list[Frame], views: list[View]) -> Iterable[DepthFrame]:
    """Lazily load one aligned RGB-D frame at a time for TSDF integration."""
    for frame, view in zip(frames, views):
        yield DepthFrame(
            np.load(view.depth_path),
            frame.view.K,
            _world_to_camera(frame.view),
            "ray",
            rgb=_resize_rgb(frame.image_path, (frame.view.height, frame.view.width)),
        )


def _empty_surface_metrics(taus: np.ndarray) -> dict:
    """Report a failed mesh explicitly instead of silently dropping its benchmark cell."""
    return {
        "accuracy": None,
        "completeness": None,
        "overall": None,
        "taus": taus.tolist(),
        "precision": [0.0] * len(taus),
        "recall": [0.0] * len(taus),
        "fscore": [0.0] * len(taus),
        "empty_prediction": True,
    }


def evaluate_benchmark_scene(
    scene: BenchmarkScene,
    predictions: list[np.ndarray],
    quantity: str,
    out: Path,
    voxel_size: float,
    mesh_samples: int,
    gt_voxel: float | None,
    surface_query_chunk_size: int,
    fusion_max_depth: float | None,
    fusion_depth_radius_multiplier: float,
    tsdf_bound_mode: Literal["none", "benchmark"],
    tsdf_max_voxels_per_axis: int | None,
    memory_profile: bool = False,
) -> list[dict]:
    # The oracle alignment is fitted to the same scan z-buffer used for visibility, in the
    # model's native quantity. A zero is an empty pixel and is excluded by ``align_prediction``.
    rows = []
    inferred_fusion_max_depth = camera_fusion_max_depth(scene.frames, fusion_depth_radius_multiplier)
    max_depth = fusion_max_depth if fusion_max_depth is not None else inferred_fusion_max_depth
    fusion_range_source = "explicit" if fusion_max_depth is not None else "camera_focus_radius"
    if max_depth <= 0:
        raise ValueError(f"fusion_max_depth must be positive, got {max_depth}")
    for alignment in ALIGNMENTS:
        condition = out / alignment
        memory_path = condition / "memory.jsonl"
        snapshot = lambda stage: write_memory_snapshot(memory_path, stage) if memory_profile else None
        depth_dir = condition / "depths"
        views = prepare_aligned_depths(scene.frames, predictions, scene.visibility, quantity, alignment, depth_dir)
        recall = evaluate_recall(
            views,
            scene.gt_points,
            MetricConfig(scene.recall_taus, "ray", visibility_depths=scene.visibility),
            alignment=scene.gt_to_world,
            gt_masks=scene.masks,
        ).asdict()
        config, tsdf_metadata = tsdf_config_for_scene(
            scene, voxel_size, max_depth, tsdf_bound_mode, tsdf_max_voxels_per_axis
        )
        snapshot("before_tsdf")
        mesh = fuse_depth_frames(
            saved_depth_frames(scene.frames, views),
            config,
            on_integrated_frame=lambda count: snapshot(f"tsdf_frame_{count}"),
        )
        snapshot("after_tsdf_mesh_extraction")
        import open3d as o3d

        mesh_path = condition / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_normals=True, write_vertex_colors=True)
        snapshot("after_mesh_write")
        predicted = _sample_mesh(mesh, mesh_samples) if len(mesh.triangles) else np.empty((0, 3))
        snapshot("after_mesh_sample")
        # The sampled points, rather than the potentially much larger Open3D mesh, are all the
        # surface evaluator needs.  Drop the C++ mesh before building either KD-tree.
        del mesh
        gc.collect()
        snapshot("after_mesh_release")
        if scene.dtu_obsmask is not None and len(predicted):
            predicted = predicted[observed_volume_mask(predicted, scene.dtu_obsmask)]
        if scene.tnt_crop is not None and len(predicted):
            if scene.gt_to_world is None:  # pragma: no cover - construction enforces this
                raise ValueError("A TnT crop requires the scan-to-world alignment")
            prediction_in_scan = apply_alignment(predicted, np.linalg.inv(scene.gt_to_world))
            predicted = predicted[crop_volume_mask(prediction_in_scan, scene.tnt_crop)]
        reference = (
            apply_alignment(scene.gt_points, scene.gt_to_world) if scene.gt_to_world is not None else scene.gt_points
        )
        if gt_voxel is not None:
            reference = voxel_downsample(reference, gt_voxel)
        surface = (
            evaluate_surface(
                predicted,
                reference,
                scene.mesh_taus,
                query_chunk_size=surface_query_chunk_size,
            ).asdict(scene.mesh_taus)
            if len(predicted)
            else _empty_surface_metrics(scene.mesh_taus)
        )
        snapshot("after_surface_metrics")
        rows.append(
            {
                "suite": scene.family,
                "scene": scene.name,
                "alignment": alignment,
                "recall": recall,
                "surface": surface,
                "mesh": str(mesh_path),
                "views": len(views),
                "oracle_alignment": alignment != "raw",
                "tsdf": {
                    **tsdf_metadata,
                    "max_depth_source": fusion_range_source,
                    "camera_focus_radius_multiplier": (
                        fusion_depth_radius_multiplier if fusion_max_depth is None else None
                    ),
                },
            }
        )
    return rows


def evaluate_ob3d(model: str, predictor, frames: list[Frame], out: Path) -> list[dict]:
    quantity = predictor.QUANTITY
    maps = []
    for frame in frames:
        prediction = predictor.predict(_resize_rgb(frame.image_path, (frame.view.height, frame.view.width)))
        maps.append(_resize(prediction, (frame.view.height, frame.view.width)))
    rows = []
    for alignment in ALIGNMENTS:
        maps_dir = out / alignment / "depths"
        maps_dir.mkdir(parents=True, exist_ok=True)
        aligned_maps = [align_prediction(pred, frame.gt_z, quantity, alignment) for pred, frame in zip(maps, frames)]
        for frame, prediction in zip(frames, aligned_maps):
            np.save(maps_dir / f"{frame.name}.npy", prediction)
        metrics = [depth_metrics(prediction, frame.gt_z) for prediction, frame in zip(aligned_maps, frames)]
        weights = np.asarray([metric["valid_pixels"] for metric in metrics], dtype=np.float64)
        rows.append(
            {
                "suite": "ob3d",
                "scene": out.parent.name,
                "model": model,
                "alignment": alignment,
                "depth": {
                    key: float(np.average([m[key] for m in metrics], weights=weights))
                    for key in ("abs_rel", "rmse", "mae", "delta1")
                },
                "frames": len(frames),
                "oracle_alignment": alignment != "raw",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", default="dav2,dav3,moge3", help="Comma-separated: dav2,dav3,moge3")
    parser.add_argument(
        "--moge3-model",
        default=MODELS["moge3"][1],
        help="Local MoGe-3 checkpoint path (default is this workstation's ViT-L checkpoint)",
    )
    parser.add_argument(
        "--dataset-scale",
        choices=("full", "reduced"),
        default="full",
        help="Scene preset: full uses every supported scene; reduced uses the fixed one-third subset.",
    )
    parser.add_argument("--ob3d-root", type=Path, default=Path("/mnt/data/nerf_datasets/ob3d/OB3D_colmap"))
    parser.add_argument("--ob3d-scenes", default=None, help="Comma-separated override for the OB3D preset scenes")
    parser.add_argument("--dtu-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu"))
    parser.add_argument("--dtu-eval-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu_eval"))
    parser.add_argument("--dtu-scenes", default=None, help="Comma-separated override for the DTU preset scenes")
    parser.add_argument("--tnt-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt"))
    parser.add_argument(
        "--tnt-reconstruction-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt_gof")
    )
    parser.add_argument("--tnt-scenes", default=None, help="Comma-separated override for the TnT preset scenes")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--max-image-side", type=int, default=None, help="Downscale before inference/evaluation")
    parser.add_argument("--voxel-size-dtu", type=float, default=2.0, help="TSDF voxel size in DTU millimetres")
    parser.add_argument("--voxel-size-tnt", type=float, default=0.01, help="TSDF voxel size in TnT metres")
    parser.add_argument(
        "--fusion-max-depth-dtu",
        type=float,
        default=None,
        help="Explicit DTU TSDF depth cap in millimetres; default is AmbiSuR-style camera-focus range.",
    )
    parser.add_argument(
        "--fusion-max-depth-tnt",
        type=float,
        default=None,
        help="Explicit TnT TSDF depth cap in metres; default is AmbiSuR-style camera-focus range.",
    )
    parser.add_argument(
        "--fusion-depth-radius-multiplier",
        type=float,
        default=2.0,
        help="Camera-focus radius multiplier for automatic TSDF depth caps (AmbiSuR adaptive default: 2).",
    )
    parser.add_argument(
        "--tsdf-bound-mode",
        choices=("none", "benchmark"),
        default="benchmark",
        help=(
            "Pre-integrate only the official DTU ObsMask/TnT crop volume, expanded by one truncation. "
            "This mirrors PGSR/AmbiSuR but is GT-assisted benchmark extraction; use none for deployable extraction."
        ),
    )
    parser.add_argument(
        "--tsdf-max-voxels-per-axis",
        type=int,
        default=2048,
        help=(
            "Coarsen a bounded TSDF only when its longest axis would exceed this resolution; "
            "0 disables this resolution guard. Open3D scalable TSDF allocation remains sparse."
        ),
    )
    parser.add_argument(
        "--mesh-samples",
        type=int,
        default=2_000_000,
        help="Uniform mesh samples for DTU/TnT surface metrics (official evaluator-scale default)",
    )
    parser.add_argument("--gt-voxel", type=float, default=None, help="Optional GT downsample in evaluation units")
    parser.add_argument(
        "--surface-query-chunk-size",
        type=int,
        default=DEFAULT_QUERY_CHUNK_SIZE,
        help=(
            "Maximum source samples in one exact cKDTree query during mesh scoring. "
            "Does not change the surface-sample count or metric; lower it to reduce RAM."
        ),
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Write RSS/cgroup memory samples around TSDF and mesh scoring to each condition's memory.jsonl.",
    )
    args = parser.parse_args()
    if args.tsdf_max_voxels_per_axis is not None and args.tsdf_max_voxels_per_axis < 0:
        parser.error("--tsdf-max-voxels-per-axis must be non-negative")
    if args.tsdf_max_voxels_per_axis == 0:
        args.tsdf_max_voxels_per_axis = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; expected {sorted(MODELS)}")
    model_specs = dict(MODELS)
    model_specs["moge3"] = ("moge3", args.moge3_model)
    max_frames = args.max_frames or 1_000_000
    scenes = selected_scenes(
        args.dataset_scale,
        {"ob3d": args.ob3d_scenes, "dtu": args.dtu_scenes, "tnt": args.tnt_scenes},
    )
    records: list[dict] = []
    for model in models:
        backend, model_id = model_specs[model]
        predictor = BACKENDS[backend](model_id=model_id)
        for scene_name in scenes["ob3d"]:
            frames = ob3d_frames(args.ob3d_root / scene_name, max_frames, args.max_image_side)
            records.extend(evaluate_ob3d(model, predictor, frames, args.out_dir / "ob3d" / scene_name / model))
        for scene_name in scenes["dtu"]:
            destination = args.out_dir / "dtu" / scene_name / model
            scene = dtu_scene(
                args.dtu_root, args.dtu_eval_root, scene_name, max_frames, args.max_image_side, destination
            )
            predictions = predict_frames(predictor, scene.frames)
            for row in evaluate_benchmark_scene(
                scene,
                predictions,
                predictor.QUANTITY,
                destination,
                args.voxel_size_dtu,
                args.mesh_samples,
                args.gt_voxel,
                args.surface_query_chunk_size,
                args.fusion_max_depth_dtu,
                args.fusion_depth_radius_multiplier,
                args.tsdf_bound_mode,
                args.tsdf_max_voxels_per_axis,
                args.memory_profile,
            ):
                row["model"] = model
                records.append(row)
        for scene_name in scenes["tnt"]:
            destination = args.out_dir / "tnt" / scene_name / model
            scene = tnt_scene(
                args.tnt_root, args.tnt_reconstruction_root, scene_name, max_frames, args.max_image_side, destination
            )
            predictions = predict_frames(predictor, scene.frames)
            for row in evaluate_benchmark_scene(
                scene,
                predictions,
                predictor.QUANTITY,
                destination,
                args.voxel_size_tnt,
                args.mesh_samples,
                args.gt_voxel,
                args.surface_query_chunk_size,
                args.fusion_max_depth_tnt,
                args.fusion_depth_radius_multiplier,
                args.tsdf_bound_mode,
                args.tsdf_max_voxels_per_axis,
                args.memory_profile,
            ):
                row["model"] = model
                records.append(row)
    output = args.out_dir / "results.jsonl"
    output.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    (args.out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "models": model_specs,
                "dataset_scale": args.dataset_scale,
                "scenes": scenes,
                "alignment": list(ALIGNMENTS),
                "oracle_alignment": "GT scan z-buffer per frame",
                "mesh_samples": args.mesh_samples,
                "surface_query_chunk_size": args.surface_query_chunk_size,
                "fusion_depth": {
                    "method": "explicit cap or AmbiSuR-style camera-focus radius",
                    "radius_multiplier": args.fusion_depth_radius_multiplier,
                    "dtu_explicit_max_depth": args.fusion_max_depth_dtu,
                    "tnt_explicit_max_depth": args.fusion_max_depth_tnt,
                },
                "tsdf_bounds": {
                    "mode": args.tsdf_bound_mode,
                    "max_voxels_per_axis": args.tsdf_max_voxels_per_axis,
                    "note": "benchmark mode is GT-assisted, matching PGSR/AmbiSuR TnT extraction",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {len(records)} cells to {output}")


if __name__ == "__main__":
    main()
