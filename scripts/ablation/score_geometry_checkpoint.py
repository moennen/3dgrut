#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score one trained geometry-ablation checkpoint with the shared posed-depth protocol.

This stays in a separate process from training: Open3D TSDF fusion and the exact surface query
can have a high host-memory peak, so an evaluation OOM must not discard the completed checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEPTHRECALL = ROOT / "tools" / "depthrecall"
for path in (ROOT, DEPTHRECALL, ROOT / "scripts" / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_depth_models import (  # noqa: E402
    Frame,
    _empty_surface_metrics,
    _sample_mesh,
    camera_fusion_max_depth,
    dtu_scene,
    observed_volume_mask,
    tnt_scene,
    visibility_maps,
    voxel_downsample,
)

from depthrecall.io_cameras import read_manifest_views
from depthrecall.io_points import apply_alignment
from depthrecall.metric import MetricConfig, evaluate
from depthrecall.surface import evaluate_surface
from threedgrut.geometry.tsdf import DepthFrame, TSDFConfig, fuse_depth_frames


def world_to_camera(view) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3], matrix[:3, 3] = view.R, view.t
    return matrix


def source_to_render_scale(matrix: np.ndarray) -> float:
    """Scale of a similarity transform; rejects an accidental non-similarity matrix."""
    lengths = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.allclose(lengths, lengths.mean(), rtol=1e-4, atol=1e-9):
        raise ValueError("The exported alignment is not a similarity transform; a voxel unit is ambiguous.")
    return float(lengths.mean())


def manifest_frames(export_dir: Path, images: Path) -> tuple[list[Frame], list[DepthFrame]]:
    """Load exactly the exported renderer cameras, retaining source RGB for colored meshes."""
    manifest = json.loads((export_dir / "manifest.json").read_text())
    views = read_manifest_views(export_dir / "manifest.json")
    frames, depths = [], []
    for entry, view in zip(manifest["views"], views):
        image = images / Path(entry["source_image"]).name
        if not image.is_file():
            raise FileNotFoundError(f"Exported source image is absent: {image}")
        frames.append(Frame(view.name, image, view))
        rgb = np.asarray(Image.open(image).convert("RGB"))
        if rgb.shape[:2] != (view.height, view.width):
            rgb = np.asarray(
                Image.fromarray(rgb).resize((view.width, view.height), Image.Resampling.BILINEAR), dtype=np.uint8
            )
        depths.append(DepthFrame(np.load(view.depth_path), view.K, world_to_camera(view), "ray", rgb=rgb))
    return frames, depths


def recall_in_source_units(result: dict, source_taus: np.ndarray, scale: float) -> dict:
    """Keep output tolerances/distances in official scan units, not normalized render units."""
    result["taus"] = source_taus.tolist()
    if result["median_signed_delta"] is not None:
        result["median_signed_delta"] /= scale
    for view in result["per_view"]:
        if view["median_signed_delta"] is not None:
            view["median_signed_delta"] /= scale
    return result


def surface_in_source_units(result: dict, source_taus: np.ndarray, scale: float) -> dict:
    """Distances scale with coordinates; precision, recall and F1 do not."""
    for key in ("accuracy", "completeness", "overall"):
        if result[key] is not None:
            result[key] /= scale
    result["taus"] = source_taus.tolist()
    return result


def evaluate_scene(args, export_dir: Path) -> dict:
    """Evaluate recall and a colored TSDF mesh in the checkpoint's render coordinate system."""
    source_to_render = np.load(export_dir / "alignment.npy")
    if args.suite == "dtu":
        benchmark = dtu_scene(args.dtu_root, args.dtu_eval_root, args.scene, 1_000_000, None, export_dir / "bootstrap")
        image_root = args.dtu_root / args.scene / "images"
        # DTU GT and its official masks live in the source/scan coordinate frame.
        scan_to_render = source_to_render
        voxel_source = args.voxel_size_dtu
    elif args.suite == "tnt":
        benchmark = tnt_scene(
            args.tnt_root, args.tnt_reconstruction_root, args.scene, 1_000_000, None, export_dir / "bootstrap"
        )
        image_root = args.tnt_reconstruction_root / "TrainingSet" / args.scene / "images"
        # tnt_scene maps official scan coordinates to its COLMAP source frame.  The checkpoint
        # may additionally normalize that source frame, so compose in this order.
        scan_to_render = source_to_render @ benchmark.gt_to_world
        voxel_source = args.voxel_size_tnt
    else:
        return {"suite": "ob3d", "export": json.loads((export_dir / "export_summary.json").read_text())}

    frames, depth_frames = manifest_frames(export_dir, image_root)
    reference = apply_alignment(benchmark.gt_points, scan_to_render)
    visibility = visibility_maps(frames, reference, export_dir / "visibility")
    scale = source_to_render_scale(scan_to_render)
    recall = recall_in_source_units(
        evaluate(
            [frame.view for frame in frames],
            benchmark.gt_points,
            MetricConfig(benchmark.recall_taus * scale, "ray", visibility_depths=visibility),
            alignment=scan_to_render,
            gt_masks=benchmark.masks,
        ).asdict(),
        benchmark.recall_taus,
        scale,
    )
    config = TSDFConfig(voxel_source * scale, voxel_source * scale * 5, camera_fusion_max_depth(frames))
    mesh = fuse_depth_frames(depth_frames, config)
    import open3d as o3d

    mesh_path = export_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_normals=True, write_vertex_colors=True)
    predicted = _sample_mesh(mesh, args.mesh_samples) if len(mesh.triangles) else np.empty((0, 3))
    if args.suite == "dtu" and len(predicted):
        predicted_scan = apply_alignment(predicted, np.linalg.inv(scan_to_render))
        predicted = predicted[observed_volume_mask(predicted_scan, benchmark.dtu_obsmask)]
    elif args.suite == "tnt" and len(predicted):
        from depthrecall.tnt import crop_volume_mask

        predicted_scan = apply_alignment(predicted, np.linalg.inv(scan_to_render))
        predicted = predicted[crop_volume_mask(predicted_scan, benchmark.tnt_crop)]
    if args.gt_voxel is not None:
        reference = voxel_downsample(reference, args.gt_voxel * scale)
    surface = (
        surface_in_source_units(
            evaluate_surface(
                predicted, reference, benchmark.mesh_taus * scale, query_chunk_size=args.surface_query_chunk_size
            ).asdict(benchmark.mesh_taus * scale),
            benchmark.mesh_taus,
            scale,
        )
        if len(predicted)
        else _empty_surface_metrics(benchmark.mesh_taus)
    )
    return {
        "suite": args.suite,
        "scene": args.scene,
        "views": len(frames),
        "recall": recall,
        "surface": surface,
        "mesh": str(mesh_path),
        "tsdf": {
            "voxel_size_render": config.voxel_size,
            "truncation_render": config.truncation,
            "max_depth_render": config.max_depth,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("ob3d", "dtu", "tnt"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ob3d-root", type=Path, required=True)
    parser.add_argument("--dtu-root", type=Path, required=True)
    parser.add_argument("--dtu-eval-root", type=Path, required=True)
    parser.add_argument("--tnt-root", type=Path, required=True)
    parser.add_argument("--tnt-reconstruction-root", type=Path, required=True)
    parser.add_argument("--mesh-samples", type=int, default=2_000_000)
    parser.add_argument("--surface-query-chunk-size", type=int, default=25_000)
    parser.add_argument("--voxel-size-dtu", type=float, default=2.0, help="DTU millimetres in scan space")
    parser.add_argument("--voxel-size-tnt", type=float, default=0.01, help="TnT metres in scan space")
    parser.add_argument("--gt-voxel", type=float, default=None, help="Optional GT downsample in scan-space units")
    args = parser.parse_args()
    # Keep every cell's depths, visibility maps and colored mesh.  A shared ``export`` folder
    # would make a later scene silently overwrite the artifacts linked by an earlier JSON row.
    export_dir = args.out.with_suffix("") / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_depth_maps.py"),
        "--checkpoint",
        args.checkpoint,
        "--out-dir",
        str(export_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    result = evaluate_scene(args, export_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
