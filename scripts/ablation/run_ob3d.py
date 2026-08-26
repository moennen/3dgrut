#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run an ablation grid of configurations across OB3D scenes.

Each (variant, scene) pair is a separate `train.py` process, so a crash or a CUDA OOM
in one cell cannot take down the sweep or silently corrupt another cell's state. Results
are appended to a JSONL file as they complete, which makes the sweep resumable and means
a sweep interrupted after six hours is still worth exactly what it produced.

Ranking is on depth error and PSNR. Normal error is recorded but not ranked on: with no
supervision on the normal buffer it loses to a control that points every normal back
along the view ray, so it cannot currently separate a good model from a degenerate one
(see utils/depth_normal_metrics.normal_metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Variant:
    """One configuration under test, as a name plus Hydra overrides."""

    name: str
    overrides: tuple[str, ...] = ()
    description: str = ""


# Reference geometry is required for the depth metrics, and the rendered normal buffer
# is enabled everywhere so that the normal diagnostic is comparable across variants.
COMMON_OVERRIDES = (
    "dataset.load_depth_gt=true",
    "dataset.load_normal_gt=true",
    "render.enable_normals=true",
)

BASELINE_VARIANTS: tuple[Variant, ...] = (
    Variant(
        "gaussian",
        ("render.primitive_type=instances",),
        "Stock 3DGUT ellipsoids: the reference point every change has to beat.",
    ),
    Variant(
        "trisurfel",
        ("render.primitive_type=trisurfel",),
        "Flat primitives, whose local +z is an actual surface normal.",
    ),
)

# Depth-normal consistency, swept over weight on both primitives. Measurement found the
# depth-implied normal to be a worse target than trisurfel's rendered normal, so a large
# weight is expected to degrade trisurfel's normals while helping its depth; the point of
# sweeping the weight rather than picking one is to locate that trade-off instead of
# asserting it. See docs/normal-supervision.md.
DEPTH_NORMAL_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"dn{weight_name}_{primitive_name}",
        (
            f"render.primitive_type={primitive}",
            "loss.use_depth_normal=true",
            f"loss.lambda_depth_normal={weight}",
            # The config default holds the term off until 7000, which is sized for a full
            # 30k run and would switch it on exactly as a 7k sweep ends -- a no-op sweep
            # that looks like a null result. Kept at a comparable fraction of training.
            "loss.depth_normal_from_iter=3000",
        ),
        f"Depth-normal consistency at lambda={weight} on {primitive_name}.",
    )
    for primitive_name, primitive in (("gaussian", "instances"), ("trisurfel", "trisurfel"))
    for weight_name, weight in (("005", 0.005), ("05", 0.05), ("2", 0.2))
)

DEPTH_NORMAL_OVERRIDES = (
    "loss.use_depth_normal=true",
    "loss.lambda_depth_normal=0.05",
    "loss.depth_normal_from_iter=3000",
)

# Flatness regularisation, gaussians only: the surfel kernel forces scale.z and drops its
# gradient, so the term is rejected there rather than silently shrinking dead storage.
# Compare against the `trisurfel` baseline rather than only `gaussian` -- the term drives an
# ellipsoid towards what a surfel already is, so the surfel result is the value it should
# approach, and that comparison is what caught the term penalising the wrong axis. The
# weights span to ~300, the setting comparable to the PGSR reference's 100 on an
# unnormalised scale once the scene extent (~4 here) is divided out. `sf*_dn` stacks it on
# depth-normal consistency: flattening makes z the genuinely thin axis, and depth-normal
# consistency is what rotates it to face the surface.
FLATNESS_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"sf{weight_name}_gaussian" + ("_dn" if with_dn else ""),
        (
            "render.primitive_type=instances",
            "loss.use_scale_flatten=true",
            f"loss.lambda_scale_flatten={weight}",
        )
        + (DEPTH_NORMAL_OVERRIDES if with_dn else ()),
        f"Flatness at lambda={weight} on gaussians" + (" with depth-normal." if with_dn else "."),
    )
    for with_dn in (False, True)
    # 0.1 and 1 are here because 3 already collapses z/xy to 0.01, so without them the sweep
    # could not separate the effect of flattening from that of over-flattening. 1 is the knee.
    for weight_name, weight in (("01", 0.1), ("1", 1.0), ("3", 3.0), ("30", 30.0), ("300", 300.0))
)

ALL_VARIANTS: tuple[Variant, ...] = BASELINE_VARIANTS + DEPTH_NORMAL_VARIANTS + FLATNESS_VARIANTS


def scene_dirs(dataset_root: Path, scenes: list[str] | None) -> list[Path]:
    """Resolve scene directories, rejecting names that do not exist.

    A typo in a scene name would otherwise show up hours later as a missing row.
    """
    if scenes:
        resolved = [dataset_root / name for name in scenes]
        missing = [str(path) for path in resolved if not path.is_dir()]
        if missing:
            raise SystemExit(f"No such scene directory: {', '.join(missing)}")
        return resolved
    return sorted(path for path in dataset_root.iterdir() if (path / "sparse").is_dir())


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path) as handle:
        return json.load(handle)


def find_run_dir(out_dir: Path, experiment: str) -> Path | None:
    """The newest timestamped run directory `train.py` created for this experiment."""
    parent = out_dir / experiment
    if not parent.is_dir():
        return None
    runs = sorted((path for path in parent.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return runs[-1] if runs else None


@dataclass
class Cell:
    variant: Variant
    scene: Path
    n_iterations: int
    out_dir: Path
    config_name: str
    extra_overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experiment(self) -> str:
        return f"abl_{self.variant.name}_{self.scene.name}"

    def command(self) -> list[str]:
        return [
            sys.executable,
            str(REPO_ROOT / "train.py"),
            "--config-name",
            self.config_name,
            f"path={self.scene}",
            f"out_dir={self.out_dir}",
            f"experiment_name={self.experiment}",
            f"n_iterations={self.n_iterations}",
            *COMMON_OVERRIDES,
            *self.variant.overrides,
            *self.extra_overrides,
        ]


def run_cell(cell: Cell, log_dir: Path, timeout_s: int) -> dict:
    """Train one (variant, scene) pair and collect its metrics and cost.

    Failures are recorded as rows rather than raised: one variant that OOMs should not
    discard the rest of the sweep, and a missing row is much easier to misread than an
    explicit failure with its log path attached.
    """
    log_path = log_dir / f"{cell.experiment}.log"
    started = time.perf_counter()
    row = {
        "variant": cell.variant.name,
        "scene": cell.scene.name,
        "n_iterations": cell.n_iterations,
        "overrides": list(cell.variant.overrides + cell.extra_overrides),
        "log": str(log_path),
    }

    with open(log_path, "w") as log_handle:
        try:
            completed = subprocess.run(
                cell.command(),
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            row |= {"status": "timeout", "wall_time_s": time.perf_counter() - started}
            return row

    row["wall_time_s"] = time.perf_counter() - started
    if returncode != 0:
        # The tail is worth inlining: the whole point of a row is to be readable
        # without going and opening a file.
        row |= {"status": "failed", "returncode": returncode, "error_tail": tail_of(log_path)}
        return row

    run_dir = find_run_dir(cell.out_dir, cell.experiment)
    if run_dir is None:
        row |= {"status": "no_output", "error_tail": tail_of(log_path)}
        return row

    metrics = load_json(run_dir / "metrics.json")
    if not metrics:
        row |= {"status": "no_metrics", "error_tail": tail_of(log_path)}
        return row

    row |= {"status": "ok", "run_dir": str(run_dir), **metrics, **load_json(run_dir / "train_stats.json")}
    return row


def tail_of(path: Path, lines: int = 15) -> str:
    try:
        with open(path, errors="replace") as handle:
            return "".join(handle.readlines()[-lines:]).strip()
    except OSError:
        return ""


def completed_keys(results_path: Path) -> set[tuple[str, str]]:
    """Cells already recorded as successful, so a resumed sweep does not repeat them."""
    done = set()
    if not results_path.is_file():
        return done
    with open(results_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A partially written final line from an interrupted sweep.
            if row.get("status") == "ok":
                done.add((row["variant"], row["scene"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory holding OB3D COLMAP scenes")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where runs, logs and results.jsonl are written")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names; default is every scene found")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Variant names; default is the two baselines, since the others are only "
        "interpretable next to a baseline run on the same scenes",
    )
    parser.add_argument("--n-iterations", type=int, default=7000, help="Training iterations per cell")
    parser.add_argument("--config-name", default="apps/colmap_3dgut.yaml")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override for every cell")
    parser.add_argument("--timeout-s", type=int, default=7200, help="Per-cell wall-clock limit")
    parser.add_argument("--resume", action="store_true", help="Skip cells already recorded as ok")
    parser.add_argument(
        "--kernel-timings",
        action="store_true",
        help="Measure per-frame render time. Adds CUDA synchronization, so it makes ms/frame "
        "real but iteration speed less representative; keep it consistent across a sweep.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands and exit")
    args = parser.parse_args()

    by_name = {variant.name: variant for variant in ALL_VARIANTS}
    if args.variants:
        unknown = sorted(set(args.variants) - set(by_name))
        if unknown:
            raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}. Known: {', '.join(sorted(by_name))}")
        variants = [by_name[name] for name in args.variants]
    else:
        variants = list(BASELINE_VARIANTS)

    extra = tuple(args.override)
    if args.kernel_timings:
        extra += ("render.enable_kernel_timings=true",)

    scenes = scene_dirs(args.dataset_root, args.scenes)
    if not scenes:
        raise SystemExit(f"No COLMAP scenes found under {args.dataset_root}")

    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    already_done = completed_keys(results_path) if args.resume else set()

    cells = [
        Cell(
            variant=variant,
            scene=scene,
            n_iterations=args.n_iterations,
            out_dir=args.out_dir,
            config_name=args.config_name,
            extra_overrides=extra,
        )
        # Scene is the inner loop so that a sweep stopped early has complete
        # variants to compare rather than one variant across every scene.
        for variant in variants
        for scene in scenes
        if (variant.name, scene.name) not in already_done
    ]

    if args.dry_run:
        for cell in cells:
            print(" ".join(cell.command()))
        print(f"\n{len(cells)} cell(s); {len(already_done)} already complete.")
        return 0

    print(f"Running {len(cells)} cell(s) -> {results_path}", flush=True)
    failures = 0
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.variant.name} / {cell.scene.name} ... ", end="", flush=True)
        row = run_cell(cell, log_dir, args.timeout_s)
        with open(results_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

        if row["status"] == "ok":
            print(
                f"ok  psnr {row.get('mean_psnr', float('nan')):.2f}  "
                f"depth_abs_rel {row.get('depth_abs_rel', float('nan')):.4f}  "
                f"({row['wall_time_s']:.0f}s)",
                flush=True,
            )
        else:
            failures += 1
            print(f"{row['status'].upper()} (see {row['log']})", flush=True)

    print(f"\n{len(cells) - failures}/{len(cells)} succeeded. Report with:")
    print(f"  python {Path(__file__).parent / 'report.py'} {results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
