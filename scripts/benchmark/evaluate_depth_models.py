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
from depthrecall.tnt import OFFICIAL_TAU_METRES, crop_volume_mask, gt_to_render_alignment
from threedgrut.datasets.gt_geometry import depth_validity, find_gt_paths, read_gt_map, resize_gt_map
from threedgrut.datasets.pseudo_depth import BACKENDS
from threedgrut.geometry.tsdf import DepthFrame, TSDFConfig, fuse_depth_frames, z_depth_to_ray_distance

MODELS = {
    "dav2": ("transformers", "depth-anything/Depth-Anything-V2-Base-hf"),
    "dav3": ("depth_anything_3", "depth-anything/DA3MONO-LARGE"),
    "moge3": ("moge3", "/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt"),
}
ALIGNMENTS: tuple[Literal["raw", "scale", "affine"], ...] = ("raw", "scale", "affine")


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
    predictions: dict[str, list[np.ndarray]],
    out: Path,
    voxel_size: float,
    mesh_samples: int,
    gt_voxel: float | None,
    surface_query_chunk_size: int,
) -> list[dict]:
    # The oracle alignment is fitted to the same scan z-buffer used for visibility, in the
    # model's native quantity. A zero is an empty pixel and is excluded by ``align_prediction``.
    frames = [
        replace(frame, gt_z=_z_from_ray(np.load(scene.visibility[frame.name]), frame.view.K)) for frame in scene.frames
    ]
    rgb_images = [_resize_rgb(frame.image_path, (frame.view.height, frame.view.width)) for frame in frames]
    rows = []
    for alignment in ALIGNMENTS:
        aligned_z = [
            align_prediction(pred, frame.gt_z, predictions["quantity"], alignment)
            for pred, frame in zip(predictions["maps"], frames)
        ]
        condition = out / alignment
        depth_dir = condition / "depths"
        depth_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for frame, z in zip(frames, aligned_z):
            ray = z_depth_to_ray_distance(z, frame.view.K)
            path = depth_dir / f"{frame.name}.npy"
            np.save(path, ray)
            views.append(_view_with_depth(frame, path))
        recall = evaluate_recall(
            views,
            scene.gt_points,
            MetricConfig(scene.recall_taus, "ray", visibility_depths=scene.visibility),
            alignment=scene.gt_to_world,
            gt_masks=scene.masks,
        ).asdict()
        config = TSDFConfig(voxel_size, voxel_size * 5, max_depth=float(np.nanmax(np.stack(aligned_z))) * 1.05)
        mesh = fuse_depth_frames(
            [
                DepthFrame(z, frame.view.K, _world_to_camera(frame.view), "z", rgb=rgb)
                for frame, z, rgb in zip(frames, aligned_z, rgb_images)
            ],
            config,
        )
        import open3d as o3d

        mesh_path = condition / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_normals=True, write_vertex_colors=True)
        predicted = _sample_mesh(mesh, mesh_samples) if len(mesh.triangles) else np.empty((0, 3))
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
    parser.add_argument("--ob3d-root", type=Path, default=Path("/mnt/data/nerf_datasets/ob3d/OB3D_colmap"))
    parser.add_argument("--ob3d-scenes", default="emerald-square")
    parser.add_argument("--dtu-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu"))
    parser.add_argument("--dtu-eval-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu_eval"))
    parser.add_argument("--dtu-scenes", default="scan24")
    parser.add_argument("--tnt-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt"))
    parser.add_argument(
        "--tnt-reconstruction-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt_gof")
    )
    parser.add_argument("--tnt-scenes", default="Barn")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--max-image-side", type=int, default=None, help="Downscale before inference/evaluation")
    parser.add_argument("--voxel-size-dtu", type=float, default=2.0, help="TSDF voxel size in DTU millimetres")
    parser.add_argument("--voxel-size-tnt", type=float, default=0.01, help="TSDF voxel size in TnT metres")
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
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; expected {sorted(MODELS)}")
    model_specs = dict(MODELS)
    model_specs["moge3"] = ("moge3", args.moge3_model)
    max_frames = args.max_frames or 1_000_000
    records: list[dict] = []
    for model in models:
        backend, model_id = model_specs[model]
        predictor = BACKENDS[backend](model_id=model_id)
        for scene_name in args.ob3d_scenes.split(","):
            frames = ob3d_frames(args.ob3d_root / scene_name, max_frames, args.max_image_side)
            records.extend(evaluate_ob3d(model, predictor, frames, args.out_dir / "ob3d" / scene_name / model))
        for scene_name in args.dtu_scenes.split(","):
            destination = args.out_dir / "dtu" / scene_name / model
            scene = dtu_scene(
                args.dtu_root, args.dtu_eval_root, scene_name, max_frames, args.max_image_side, destination
            )
            maps = [
                _resize(
                    predictor.predict(_resize_rgb(frame.image_path, (frame.view.height, frame.view.width))),
                    (frame.view.height, frame.view.width),
                )
                for frame in scene.frames
            ]
            for row in evaluate_benchmark_scene(
                scene,
                {"maps": maps, "quantity": predictor.QUANTITY},
                destination,
                args.voxel_size_dtu,
                args.mesh_samples,
                args.gt_voxel,
                args.surface_query_chunk_size,
            ):
                row["model"] = model
                records.append(row)
        for scene_name in args.tnt_scenes.split(","):
            destination = args.out_dir / "tnt" / scene_name / model
            scene = tnt_scene(
                args.tnt_root, args.tnt_reconstruction_root, scene_name, max_frames, args.max_image_side, destination
            )
            maps = [
                _resize(
                    predictor.predict(_resize_rgb(frame.image_path, (frame.view.height, frame.view.width))),
                    (frame.view.height, frame.view.width),
                )
                for frame in scene.frames
            ]
            for row in evaluate_benchmark_scene(
                scene,
                {"maps": maps, "quantity": predictor.QUANTITY},
                destination,
                args.voxel_size_tnt,
                args.mesh_samples,
                args.gt_voxel,
                args.surface_query_chunk_size,
            ):
                row["model"] = model
                records.append(row)
    output = args.out_dir / "results.jsonl"
    output.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    (args.out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "models": model_specs,
                "alignment": list(ALIGNMENTS),
                "oracle_alignment": "GT scan z-buffer per frame",
                "mesh_samples": args.mesh_samples,
                "surface_query_chunk_size": args.surface_query_chunk_size,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(records)} cells to {output}")


if __name__ == "__main__":
    main()
