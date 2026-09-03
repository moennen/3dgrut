#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed 30k geometry-improvement ablation on OB3D, DTU and TnT.

The reduced preset is deliberately stable (the same one-third split as the depth benchmark),
not a random sample.  Every cell is a fresh process and its JSONL record is flushed after the
training and metric stages, so the job can be sharded with ``--suites``/``--variants`` and resumed.

The matrix is one-factor-at-a-time over the trisurfel control plus a compatible full stack.  It
is not a Cartesian product: that would make it impossible to attribute an improvement and would
turn the reduced protocol into hundreds of GPU-days.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "scripts" / "benchmark"
sys.path.insert(0, str(BENCHMARK))
from evaluate_depth_models import selected_scenes  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: tuple[str, ...]


COMMON = (
    "render.method=3dgut",
    "render.primitive_type=trisurfel",
    "render.enable_normals=true",
)
DEPTH_NORMAL = ("loss.use_depth_normal=true", "loss.lambda_depth_normal=0.05", "loss.depth_normal_from_iter=7000")
DEPTH_VARIANCE = (
    "render.enable_depth_variance=true",
    "loss.use_depth_variance=true",
    "loss.depth_variance_relative=true",
    "loss.lambda_depth_variance=0.01",
    "loss.depth_variance_from_iter=7000",
)
NORMAL_VARIANCE = (
    "loss.use_normal_variance=true",
    "loss.lambda_normal_variance=0.01",
    "loss.normal_variance_from_iter=7000",
)
APPEARANCE_VARIANCE = (
    "render.enable_appearance_variance=true",
    "loss.use_appearance_variance=true",
    "loss.lambda_appearance_variance=0.01",
    "loss.appearance_variance_from_iter=7000",
)
MOGE3 = (
    "loss.use_pseudo_depth_l1=true",
    "loss.lambda_pseudo_depth_l1=0.1",
    "dataset.pseudo_depth.backend=moge3",
)
MULTIVIEW = ("loss.multiview.enabled=true", "loss.multiview.from_iter=7000")
CONFIDENCE = (
    "loss.confidence.enabled=true",
    "loss.confidence.depth_variance_weight=1.0",
    "loss.confidence.normal_variance_weight=1.0",
    "loss.confidence.appearance_variance_weight=1.0",
)
NHT_RADIOC4_PCA48 = (
    "model.feature_type=nht",
    "model.nht_features.dim=48",
    "model.nht_features.activation.type=none",
    "model.nht_features.interpolation_type=none",
    "model.nht_decoder.dir_encoding=Identity",
    "model.nht_decoder.dir_encoding_degree=0",
    "model.nht_decoder.color_refine_steps=0",
    "model.nht_decoder.scheduler.max_steps=30000",
    "loss.use_image_features=true",
    "loss.lambda_image_features=0.1",
    "model.nht_decoder.image_feature_dim=48",
    "dataset.image_features.backend=nvradio4",
    "dataset.image_features.model=c-radio_v4-h",
    "dataset.image_features.projector=pca",
    "dataset.image_features.output_dim=48",
    "dataset.image_features.feature_stride=16",
)

VARIANTS = (
    Variant("gaussian", "3DGUT ellipsoid reference.", ("render.primitive_type=instances",)),
    Variant("trisurfel", "Flat primitive control for every geometry loss.", ()),
    Variant("depth_normal", "Depth-implied normal consistency.", DEPTH_NORMAL),
    Variant("depth_variance", "Relative ray-depth dispersion.", DEPTH_VARIANCE),
    Variant("normal_variance", "Ray normal-direction dispersion.", NORMAL_VARIANCE),
    Variant("appearance_variance", "RGB/NHT-latent ray appearance dispersion.", APPEARANCE_VARIANCE),
    Variant("moge3_l1", "Sparse-aligned metric MoGe-3 depth regression.", MOGE3),
    Variant(
        "multiview_point",
        "Affinity-sampled, visibility-gated point consistency.",
        MULTIVIEW + ("loss.multiview.geometric.lambda_point=0.05",),
    ),
    Variant(
        "multiview_l2",
        "Affinity-sampled raw RGB L2 consistency.",
        MULTIVIEW + ("loss.multiview.raw_feature_l2.lambda=0.05",),
    ),
    Variant(
        "multiview_zncc", "Affinity-sampled RGB ZNCC consistency.", MULTIVIEW + ("loss.multiview.zncc.lambda=0.05",)
    ),
    Variant(
        "confidence",
        "Detached opacity and ray-dispersion reliability weighting.",
        DEPTH_VARIANCE + NORMAL_VARIANCE + APPEARANCE_VARIANCE + MOGE3 + CONFIDENCE,
    ),
    Variant("radioc4_pca48", "Direct NHT with frozen C-RADIOv4 PCA-48 features.", NHT_RADIOC4_PCA48),
    Variant(
        "full_geometry",
        "Compatible full stack: trisurfels, ray terms, MoGe-3, cross-view, confidence and C-RADIOv4.",
        DEPTH_NORMAL
        + DEPTH_VARIANCE
        + NORMAL_VARIANCE
        + APPEARANCE_VARIANCE
        + MOGE3
        + MULTIVIEW
        + (
            "loss.multiview.geometric.lambda_point=0.05",
            "loss.multiview.raw_feature_l2.lambda=0.05",
            "loss.multiview.zncc.lambda=0.05",
        )
        + CONFIDENCE
        + NHT_RADIOC4_PCA48,
    ),
)


def scene_path(args, suite: str, scene: str) -> Path:
    if suite == "ob3d":
        return args.ob3d_root / scene
    if suite == "dtu":
        return args.dtu_root / scene
    return args.tnt_reconstruction_root / "TrainingSet" / scene


def completed(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and row.get("score_status") == "ok":
            done.add((row["suite"], row["variant"], row["scene"]))
    return done


def newest_run(out_dir: Path, experiment: str) -> Path | None:
    runs = out_dir / "runs" / experiment
    candidates = sorted(runs.glob("*"), key=lambda path: path.stat().st_mtime) if runs.exists() else []
    return candidates[-1] if candidates else None


def prune_duplicate_checkpoints(run_dir: Path) -> list[str]:
    """Remove only intermediate snapshots superseded by ``ckpt_last.pt``.

    The score stage consumes ``ckpt_last.pt`` and produces colored meshes, exported depths and
    JSON metrics that are expensive to reproduce.  None of those are disposable here.  Training
    writes periodic snapshots under ``ours_<step>/ckpt_<step>.pt``; once a score completed, they
    are the only redundant artifacts for this ablation protocol.
    """
    removed = []
    for directory in run_dir.glob("ours_*"):
        if not directory.is_dir():
            continue
        for checkpoint in directory.glob("ckpt_*.pt"):
            checkpoint.unlink()
            removed.append(str(checkpoint))
    return removed


def train_command(args, suite: str, scene: str, variant: Variant) -> tuple[list[str], str]:
    experiment = f"geom30k_{suite}_{variant.name}_{scene}"
    overrides = list(COMMON + variant.overrides)
    # OB3D alone has per-frame reference maps.  DTU/TnT reference geometry is evaluated after
    # training through the scan protocol; asking their COLMAP loaders for absent image maps
    # would turn a valid surface benchmark cell into a dataset-loading failure.
    if suite == "ob3d":
        overrides += ["dataset.load_depth_gt=true", "dataset.load_normal_gt=true"]
    if args.cache_root is not None:
        # Dataset cache keys intentionally contain only the image folder/stem.  Namespace them
        # per benchmark scene here: DTU scans otherwise all have an ``images/000000.png`` and
        # would corrupt one another through a seemingly shared cache hit.
        cache = args.cache_root / suite / scene
        overrides += [
            f"dataset.pseudo_depth.cache_dir={cache / 'pseudo_depth'}",
            f"dataset.image_features.cache_dir={cache / 'image_features'}",
        ]
    if any(value.startswith("dataset.pseudo_depth.backend=moge3") for value in overrides):
        overrides.append(f"dataset.pseudo_depth.model={args.moge3_model}")
    return (
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--config-name",
            args.config_name,
            f"path={scene_path(args, suite, scene)}",
            f"out_dir={args.out_dir / 'runs'}",
            f"experiment_name={experiment}",
            f"n_iterations={args.n_iterations}",
            f"num_workers={args.num_workers}",
            *overrides,
            *args.override,
        ],
        experiment,
    )


def run_command(command: list[str], log: Path, timeout_s: int) -> tuple[str, int | None]:
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    with log.open("w") as handle:
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return "timeout", None
    return ("ok" if result.returncode == 0 else "failed"), result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-scale", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--ob3d-root", type=Path, default=Path("/mnt/data/nerf_datasets/ob3d/OB3D_colmap"))
    parser.add_argument("--dtu-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu"))
    parser.add_argument("--dtu-eval-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu_eval"))
    parser.add_argument("--tnt-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt"))
    parser.add_argument(
        "--tnt-reconstruction-root", type=Path, default=Path("/mnt/data/nerf_datasets/tnt_dataset/tnt_gof")
    )
    parser.add_argument("--ob3d-scenes", default=None)
    parser.add_argument("--dtu-scenes", default=None)
    parser.add_argument("--tnt-scenes", default=None)
    parser.add_argument("--suites", default="ob3d,dtu,tnt", help="Comma-separated suite shard")
    parser.add_argument("--variants", default=None, help="Comma-separated variant shard")
    parser.add_argument("--n-iterations", type=int, default=30000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Training and final render DataLoader workers per cell; default 0 avoids /dev/shm exhaustion on TnT.",
    )
    parser.add_argument("--moge3-model", default="/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Persistent root for per-scene MoGe-3 and C-RADIO caches; safe to share after prewarming.",
    )
    parser.add_argument("--config-name", default="apps/colmap_3dgut.yaml")
    parser.add_argument("--timeout-s", type=int, default=43200)
    parser.add_argument("--mesh-samples", type=int, default=2_000_000)
    parser.add_argument("--surface-query-chunk-size", type=int, default=25_000)
    parser.add_argument(
        "--tsdf-bound-mode",
        choices=("none", "benchmark"),
        default="benchmark",
        help=(
            "Use the official DTU/TnT reconstruction volume while fusing score meshes. "
            "This is GT-assisted benchmark extraction, matching PGSR/AmbiSuR."
        ),
    )
    parser.add_argument(
        "--tsdf-max-voxels-per-axis",
        type=int,
        default=2048,
        help="Coarsen a bounded scoring TSDF only above this axis resolution; 0 disables the guard.",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prune-artifacts",
        action="store_true",
        help="After successful scoring, remove only redundant ours_*/ckpt_*.pt snapshots; preserve ckpt_last.pt and all score artifacts.",
    )
    parser.add_argument(
        "--skip-score", action="store_true", help="Train only; report will show unavailable DTU/TnT metrics."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tsdf_max_voxels_per_axis < 0:
        parser.error("--tsdf-max-voxels-per-axis must be non-negative")

    selected = selected_scenes(
        args.dataset_scale, {"ob3d": args.ob3d_scenes, "dtu": args.dtu_scenes, "tnt": args.tnt_scenes}
    )
    suites = tuple(value for value in args.suites.split(",") if value)
    if set(suites) - set(selected):
        raise SystemExit(f"Unknown suite(s): {sorted(set(suites) - set(selected))}")
    by_name = {variant.name: variant for variant in VARIANTS}
    names = tuple(value for value in args.variants.split(",") if value) if args.variants else tuple(by_name)
    if set(names) - set(by_name):
        raise SystemExit(f"Unknown variant(s): {sorted(set(names) - set(by_name))}")
    variants = [by_name[name] for name in names]
    for suite in suites:
        for scene in selected[suite]:
            if not scene_path(args, suite, scene).is_dir():
                raise SystemExit(f"Missing {suite} scene: {scene_path(args, suite, scene)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset_scale": args.dataset_scale,
                "scenes": selected,
                "n_iterations": args.n_iterations,
                "moge3_model": args.moge3_model,
                "tsdf": {
                    "bound_mode": args.tsdf_bound_mode,
                    "max_voxels_per_axis": args.tsdf_max_voxels_per_axis or None,
                    "note": "benchmark mode is GT-assisted, matching PGSR/AmbiSuR extraction",
                },
                "variants": {v.name: {"description": v.description, "overrides": list(v.overrides)} for v in variants},
            },
            indent=2,
        )
        + "\n"
    )
    results = args.out_dir / "results.jsonl"
    done = completed(results) if args.resume else set()
    cells = [
        (suite, scene, variant)
        for variant in variants
        for suite in suites
        for scene in selected[suite]
        if (suite, variant.name, scene) not in done
    ]
    if args.dry_run:
        for suite, scene, variant in cells:
            print(" ".join(train_command(args, suite, scene, variant)[0]))
        print(f"{len(cells)} cell(s); default is {args.n_iterations} iterations.")
        return 0

    logs = args.out_dir / "logs"
    logs.mkdir(exist_ok=True)
    failures = 0
    for index, (suite, scene, variant) in enumerate(cells, 1):
        command, experiment = train_command(args, suite, scene, variant)
        row = {
            "suite": suite,
            "scene": scene,
            "variant": variant.name,
            "description": variant.description,
            "n_iterations": args.n_iterations,
            # The executable, script, config selector/name and four run identifiers precede
            # the reproducibility overrides. Keep num_workers too: it changes the TnT render
            # execution path and must not disappear from a JSONL record.
            "overrides": command[8:],
            "log": str(logs / f"{experiment}.log"),
        }
        print(f"[{index}/{len(cells)}] {suite}/{scene} {variant.name}", flush=True)
        started = time.perf_counter()
        status, returncode = run_command(command, Path(row["log"]), args.timeout_s)
        row.update(status=status, returncode=returncode, wall_time_s=time.perf_counter() - started)
        run_dir = newest_run(args.out_dir, experiment)
        if status == "ok" and run_dir is not None:
            row["run_dir"] = str(run_dir)
            for name in ("metrics.json", "train_stats.json"):
                path = run_dir / name
                if path.exists():
                    row.update(json.loads(path.read_text()))
            checkpoint = run_dir / "ckpt_last.pt"
            if not args.skip_score and checkpoint.exists():
                score_path = args.out_dir / "scores" / suite / variant.name / f"{scene}.json"
                score_path.parent.mkdir(parents=True, exist_ok=True)
                score = [
                    sys.executable,
                    str(Path(__file__).with_name("score_geometry_checkpoint.py")),
                    "--suite",
                    suite,
                    "--scene",
                    scene,
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(score_path),
                    "--ob3d-root",
                    str(args.ob3d_root),
                    "--dtu-root",
                    str(args.dtu_root),
                    "--dtu-eval-root",
                    str(args.dtu_eval_root),
                    "--tnt-root",
                    str(args.tnt_root),
                    "--tnt-reconstruction-root",
                    str(args.tnt_reconstruction_root),
                    "--mesh-samples",
                    str(args.mesh_samples),
                    "--surface-query-chunk-size",
                    str(args.surface_query_chunk_size),
                    "--tsdf-bound-mode",
                    args.tsdf_bound_mode,
                    "--tsdf-max-voxels-per-axis",
                    str(args.tsdf_max_voxels_per_axis),
                ]
                score_status, score_returncode = run_command(score, logs / f"{experiment}_score.log", args.timeout_s)
                row["score_status"] = score_status
                row["score_returncode"] = score_returncode
                if score_status == "ok":
                    row["evaluation"] = json.loads(score_path.read_text())
                    if args.prune_artifacts:
                        row["pruned_checkpoints"] = prune_duplicate_checkpoints(run_dir)
            else:
                row["score_status"] = "skipped"
        else:
            failures += 1
        with results.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    report = [
        sys.executable,
        str(Path(__file__).with_name("report_geometry_ablation.py")),
        str(results),
        "--markdown",
        str(args.out_dir / "geometry-ablation.md"),
        "--pdf",
        str(args.out_dir / "geometry-ablation.pdf"),
    ]
    subprocess.run(report, cwd=ROOT, check=False)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
