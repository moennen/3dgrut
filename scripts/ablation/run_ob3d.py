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

# Depth variance along the ray: penalises rays whose weight is spread over distance instead of
# concentrated on a surface. The prior is weak -- the stage-0 diagnostic found the existing
# depth gradient nearly as good at spotting bad depth as the variance buffer is (0.02-0.10 AUC),
# which is why this is swept as an optimisation target rather than adopted.
#
# Two things this sweep has to separate. First, the weight: at lambda=1 the term is ~18% of L1
# late in a sponza run, so 0.1 to 10 spans negligible to dominant, and 100 is included to find
# where it breaks rather than to be used. Second, and the reason `d_cover` matters more here
# than elsewhere: the term penalises an *un-normalized* accumulator, so the model can reduce it
# by making the scene transparent instead of by resolving surfaces. A variance win that arrives
# with a fall in `d_cover` is that escape hatch, not the effect being measured.
#
# `dv*_dn` stacks it on depth-normal consistency. That is the combination the point of extending
# the Slang backward was to enable, and the diagnostic suggested it is where any remaining
# spread lives, since depth-normal already removes two thirds of it.
DEPTH_VARIANCE_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"dv{weight_name}_{primitive_name}" + ("_dn" if with_dn else ""),
        (
            f"render.primitive_type={primitive}",
            "render.enable_depth_variance=true",
            "loss.use_depth_variance=true",
            f"loss.lambda_depth_variance={weight}",
            # Same reasoning as depth_normal_from_iter: the 7000 default would switch the term
            # on exactly as a 7k sweep ends and report a null result that is really a no-op.
            "loss.depth_variance_from_iter=3000",
        )
        + (DEPTH_NORMAL_OVERRIDES if with_dn else ()),
        f"Depth variance at lambda={weight} on {primitive_name}" + (" with depth-normal." if with_dn else "."),
    )
    for primitive_name, primitive in (("gaussian", "instances"), ("trisurfel", "trisurfel"))
    for with_dn, weight_name, weight in (
        # 0.001 and 0.01 exist because 0.1 already costs emerald-square 6.5 dB: without a
        # window below it the sweep cannot distinguish "wrong form" from "wrong weight".
        (False, "0001", 0.001),
        (False, "001", 0.01),
        (False, "01", 0.1),
        (False, "1", 1.0),
        (False, "10", 10.0),
        (False, "100", 100.0),
        (True, "1", 1.0),
    )
)

# The relative form, `Var/mu^2`. Same render, same buffers, one different line in the loss, so
# these differ from `dv*` only in `depth_variance_relative`. Three properties motivate it and
# each is a thing to read in the results:
#
#   - it is dimensionless, so unlike `dv*` the *same* lambda should be usable on both sponza and
#     emerald-square. That is the primary claim, and it is a cross-scene comparison, so read the
#     per-scene tables rather than the average.
#   - it is degree zero in the weights, so `d_cover` should not move at all.
#   - it shifts the floater-locking barrier from 0.49 to 0.82, so `floater_frac` and the
#     `wrong | tight` pairing from depth_variance_mechanism.py should degrade far less steeply
#     with lambda than the absolute form's 0.0014 -> 0.24.
#
# Lambda does not carry over between the forms: the absolute one was divided by the squared
# scene extent, this one is a dimensionless ratio around 1e-2 on a typical unresolved ray, so
# the grid is re-centred rather than reused.
DEPTH_VARIANCE_RELATIVE_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"dvrel{weight_name}_gaussian" + ("_dn" if with_dn else ""),
        (
            "render.primitive_type=instances",
            "render.enable_depth_variance=true",
            "loss.use_depth_variance=true",
            "loss.depth_variance_relative=true",
            f"loss.lambda_depth_variance={weight}",
            "loss.depth_variance_from_iter=3000",
        )
        + (DEPTH_NORMAL_OVERRIDES if with_dn else ()),
        f"Relative depth variance at lambda={weight} on gaussian" + (" with depth-normal." if with_dn else "."),
    )
    for with_dn, weight_name, weight in (
        (False, "001", 0.01),
        (False, "01", 0.1),
        (False, "1", 1.0),
        (False, "10", 10.0),
        (False, "100", 100.0),
        # 2DGS pairs its distortion term with normal consistency rather than running it alone,
        # and dn05 is our strongest geometry result, so the combination is the one worth having
        # at more than a single weight.
        (True, "01", 0.1),
        (True, "1", 1.0),
    )
)

# Ordinal supervision from a monocular pseudo-depth prior (DepthAnythingV2). Unlike every term
# above, this one brings *external information* rather than an internal consistency condition,
# which is what makes it the interesting comparison: the depth-variance family can only ask a ray
# to commit to some surface, while this says which of two pixels is nearer.
#
# The prior earns its place on measurement, not on being a foundation model. Aligned globally by
# one affine per frame it reaches abs_rel 0.068 against ground truth, *worse* than the model being
# trained (0.058), which is why nothing here fits a scale; per-16x16-patch it reaches 0.011, so the
# ordering is what is worth reading. On the pixels where the trained model fails delta1 the prior
# is better 91% of the time, so there is real signal in exactly the population this targets.
#
# Three things this sweep must separate.
#
#   - The weight. Measured: 0.1 is the setting, improving depth 10-11% on all three scenes over
#     3 seeds; 1.0 trades 2.4 dB of emerald PSNR for a little more lone-monk depth, and 10 and
#     above degrade both. 100 is retained to show where it breaks, not to use.
#   - Whether the *gate* matters, which was the one substantive departure from the reference
#     implementation in /mnt/oss/blob-to-spoke. It does, in the direction opposite to the one
#     predicted: `pd01_gaussian_gated` is neutral against `pd01_gaussian` on sponza and
#     lone-monk and gives up emerald-square's entire gain (+0.7% vs -10.3%). The offline
#     argument for it -- ordinal agreement with ground truth rising from 84% to 97% -- counted
#     pairs rather than asking what they teach, and a large disparity gap turns out to select
#     for long-range comparisons, the regime where this prior drifts. The gate is off by default
#     and these variants keep the comparison runnable.
#   - Whether it supplies the anchor the depth-variance family lacks. `pd1_dvrel1_gaussian` is
#     the test of that: variance alone locks floaters at whatever wrong distance they already
#     occupy (`wrong | tight` rising 170x), because nothing tells it *where*. Measured, the
#     ordinal term does repair it -- sponza goes from +6% depth for `dvrel1_gaussian` alone to
#     -6% paired, and sponza floaters from 1.62% to 1.14% against a 0.30% baseline -- but the
#     pair is still worse than the ordinal term alone (-10%), so the variance term does not earn
#     its place even once anchored.
#
# Read this one on lone-monk especially. Its depth is wrong in an opaque, confidently-placed way
# that no ray-concentration term can reach -- 17.7% delta1 failures with 0.01% floaters -- so it
# is the scene where an external prior should win and the variance family cannot. It does: -11%
# depth where `dvrel1_gaussian` manages -1%.
PSEUDO_DEPTH_OVERRIDES = (
    "loss.use_pseudo_depth_order=true",
    "loss.lambda_pseudo_depth_order=1.0",
)

PSEUDO_DEPTH_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"pd{weight_name}_{primitive_name}" + suffix,
        (
            f"render.primitive_type={primitive}",
            "loss.use_pseudo_depth_order=true",
            f"loss.lambda_pseudo_depth_order={weight}",
            f"loss.pseudo_depth_gate={gate}",
        )
        + extra,
        f"Ordinal pseudo-depth at lambda={weight} on {primitive_name}, gate={gate}{note}",
    )
    for primitive_name, primitive in (("gaussian", "instances"), ("trisurfel", "trisurfel"))
    for weight_name, weight, gate, suffix, note, extra in (
        ("01", 0.1, 0.0, "", ".", ()),
        ("1", 1.0, 0.0, "", ".", ()),
        ("10", 10.0, 0.0, "", ".", ()),
        ("100", 100.0, 0.0, "", ".", ()),
        # The gate, swept over the same lambdas rather than compared at one, because it is not
        # weight-neutral: it keeps only the large-gap pairs, which raises the mean the loss
        # reports and so raises the effective weight. Comparing gated against ungated at a
        # single lambda would measure that shift as much as the gate itself.
        ("01", 0.1, 0.05, "_gated", ", gated (refuted; see above).", ()),
        ("1", 1.0, 0.05, "_gated", ", gated (refuted; see above).", ()),
        ("10", 10.0, 0.05, "_gated", ", gated (refuted; see above).", ()),
        # Stacked on depth-normal consistency. Both weights matter: 0.1 is where the ordinal
        # term measured best alone, so `pd01_*_dn` is the combination of the two terms at their
        # own optima, and `pd1_*_dn` shows what over-weighting the ordinal half costs once the
        # normal half is present. Compare both against `dn05_gaussian`, not only against the
        # baseline, or the depth-normal term's contribution is credited to this one.
        ("01", 0.1, 0.0, "_dn", " with depth-normal.", DEPTH_NORMAL_OVERRIDES),
        ("1", 1.0, 0.0, "_dn", " with depth-normal.", DEPTH_NORMAL_OVERRIDES),
    )
)

# The same ordinal term reading a Depth Anything 3 prior instead of Depth Anything V2.
#
# Measured over 3 seeds at 7k, this buys depth `abs_rel` -15.6% on sponza, -13.2% on lone-monk
# and -16.0% on emerald-square, against -12.9% / -13.3% / -5.3% for the DAv2 prior: a tie on two
# scenes and a 3x gain on the third, with less of the PSNR cost (-0.60 dB vs -0.96 dB on
# emerald). DA3's accuracy gain is concentrated in *global* composition -- one affine per frame,
# `abs_rel` 0.085 vs 0.123 on emerald -- and nearly gone per 16x16 patch, and emerald is the
# scene with the widest depth range, which is the best available reading of why it wins there.
#
# It is not that DA3 supplies more of what the term consumes. Its ordinal agreement barely
# differs from DAv2's on emerald (86.7% vs 86.1%) where the training gain is largest, and
# differs most on sponza (84.7% vs 82.0%) where the gain is smallest -- the wrong way round. A
# prediction to the contrary is recorded and corrected in `docs/normal-supervision.md`.
#
# Held at lambda 0.1, the weight the DAv2 prior measured best at, so that this compares priors
# and not weights; if DA3 shifts the optimum that is a separate sweep. Note the weights are
# CC BY-NC 4.0, so a win here cannot simply become the default.
PSEUDO_DEPTH_DA3_OVERRIDES = (
    "dataset.pseudo_depth.backend=depth_anything_3",
    "dataset.pseudo_depth.model=depth-anything/DA3MONO-LARGE",
)

PSEUDO_DEPTH_DA3_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"pd01da3_{primitive_name}",
        (
            f"render.primitive_type={primitive}",
            "loss.use_pseudo_depth_order=true",
            "loss.lambda_pseudo_depth_order=0.1",
        )
        + PSEUDO_DEPTH_DA3_OVERRIDES,
        f"Ordinal pseudo-depth at lambda=0.1 on {primitive_name}, from a Depth Anything 3 prior.",
    )
    for primitive_name, primitive in (("gaussian", "instances"), ("trisurfel", "trisurfel"))
)

# The pairing the ordinal term exists to test: an anchor for the relative depth-variance term.
#
# Swept at both terms' own optima and at the over-weighted pair, because the first attempt at
# this comparison was run at lambda 1 for *both* -- 10x the ordinal term's measured best and
# 100x the relative-variance term's -- and a null from that configuration says nothing about
# whether the two compose. `pd01_dvrel001_gaussian` is the honest test.
PSEUDO_DEPTH_VARIANCE_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(
        f"pd{pd_name}_dvrel{dv_name}_gaussian",
        (
            "render.primitive_type=instances",
            "render.enable_depth_variance=true",
            "loss.use_depth_variance=true",
            "loss.depth_variance_relative=true",
            f"loss.lambda_depth_variance={dv_weight}",
            "loss.depth_variance_from_iter=3000",
            "loss.use_pseudo_depth_order=true",
            f"loss.lambda_pseudo_depth_order={pd_weight}",
        ),
        f"Relative depth variance at lambda={dv_weight} anchored by ordinal pseudo-depth at lambda={pd_weight}.",
    )
    for pd_name, pd_weight, dv_name, dv_weight in (
        # Each term at the weight it measured best alone.
        ("01", 0.1, "001", 0.01),
        # Both over-weighted; retained because it is what the first pass measured.
        ("1", 1.0, "1", 1.0),
    )
)

ALL_VARIANTS: tuple[Variant, ...] = (
    BASELINE_VARIANTS
    + DEPTH_NORMAL_VARIANTS
    + FLATNESS_VARIANTS
    + DEPTH_VARIANCE_VARIANTS
    + PSEUDO_DEPTH_VARIANTS
    + PSEUDO_DEPTH_DA3_VARIANTS
    + PSEUDO_DEPTH_VARIANCE_VARIANTS
    + DEPTH_VARIANCE_RELATIVE_VARIANTS
)


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
