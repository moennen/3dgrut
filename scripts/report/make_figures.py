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

"""Plots for the geometry-supervision report, read from the ablation `results.jsonl` files.

Everything here is drawn from run records rather than retyped from the write-up, so a figure
cannot drift from the measurements. The one exception is the prior-alignment diagnostic, whose
numbers come from `pseudo_depth_diagnostic.py` and are kept in `PRIOR_ALIGNMENT` below with the
command that produced them.

Two conventions worth stating because they carry the arguments:

* Depth is plotted as *percentage change against the per-scene baseline*, not as raw `abs_rel`.
  The three scenes differ by 3x in absolute error, so a raw axis compares scenes instead of
  treatments.
* Normals are plotted as `n_gain` with an explicit line at zero. Zero is the view-direction
  control, which uses no geometry at all; a normal buffer below that line has not earned its
  name, and plotting the raw angle hides which side of it a variant sits on.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENES = ("sponza", "lone-monk", "emerald-square")

# Colour per scene, kept identical in every figure so a reader tracks a scene across slides.
SCENE_COLOUR = {"sponza": "#4C72B0", "lone-monk": "#DD8452", "emerald-square": "#55A868"}

# From `scripts/ablation/pseudo_depth_diagnostic.py --checkpoint <sponza 7k>`: `abs_rel` of
# DepthAnythingV2-Base against ground truth, affine-aligned at decreasing granularity, plus the
# model the prior would be teaching. The alignment uses ground truth, so these are upper bounds
# on what any alignment-based loss could reach.
PRIOR_ALIGNMENT = (("global\naffine", 0.0677), ("64x64\npatch", 0.0158), ("16x16\npatch", 0.0116))
PRIOR_MODEL_REFERENCE = 0.0580

# The first sweep named the gated variant `pd01_gaussian` and the ungated one `*_nogate`; the
# default flipped afterwards. Records keep the original names, so resolve them here.
UNGATED = "pd01_gaussian_nogate"
GATED = "pd01_gaussian"


def load(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """All successful records, grouped by (variant, scene). Repeats are separate seeds."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                grouped[(record["variant"], record["scene"])].append(record)
    return grouped


def _stat(grouped, variant: str, scene: str, key: str) -> tuple[float, float] | None:
    values = [r[key] for r in grouped.get((variant, scene), []) if key in r]
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values))


def _finish(fig, ax_list, out: Path) -> None:
    for ax in np.atleast_1d(ax_list).ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_prior_alignment(out: Path) -> None:
    """Why the planned regression loss was abandoned: no single affine is good enough."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = [name for name, _ in PRIOR_ALIGNMENT]
    values = [value for _, value in PRIOR_ALIGNMENT]
    colours = ["#C44E52" if v > PRIOR_MODEL_REFERENCE else "#55A868" for v in values]
    ax.bar(labels, values, color=colours, width=0.6)
    ax.axhline(
        PRIOR_MODEL_REFERENCE,
        color="k",
        ls="--",
        lw=1.2,
        label=f"the 7k model being taught ({PRIOR_MODEL_REFERENCE:.4f})",
    )
    for x, value in enumerate(values):
        ax.text(x, value + 0.002, f"{value:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("prior abs_rel vs GT")
    ax.set_title("Monocular prior, aligned with ground truth\n(sponza; lower is better)", fontsize=10)
    ax.set_ylim(0, max(values) * 1.25)
    ax.legend(fontsize=8, loc="upper right")
    _finish(fig, ax, out)


def fig_lambda_sweep(grouped, out: Path) -> None:
    """The usable band. Plotted on both axes the term trades between."""
    weights = [(0.1, UNGATED), (1.0, "pd1_gaussian"), (10.0, "pd10_gaussian_nogate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for scene in SCENES:
        base_depth = _stat(grouped, "gaussian", scene, "depth_abs_rel")
        base_psnr = _stat(grouped, "gaussian", scene, "mean_psnr")
        if not base_depth or not base_psnr:
            continue
        xs, depth, psnr = [], [], []
        for weight, variant in weights:
            depth_stat = _stat(grouped, variant, scene, "depth_abs_rel")
            psnr_stat = _stat(grouped, variant, scene, "mean_psnr")
            if not depth_stat or not psnr_stat:
                continue
            xs.append(weight)
            depth.append(100.0 * (depth_stat[0] - base_depth[0]) / base_depth[0])
            psnr.append(psnr_stat[0] - base_psnr[0])
        axes[0].plot(xs, depth, "o-", color=SCENE_COLOUR[scene], label=scene)
        axes[1].plot(xs, psnr, "o-", color=SCENE_COLOUR[scene], label=scene)

    for ax, ylabel, title in (
        (axes[0], "depth abs_rel change (%)", "Depth: better is down"),
        (axes[1], "PSNR change (dB)", "Appearance: better is up"),
    ):
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\lambda_{\mathrm{pseudo\ depth}}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    _finish(fig, axes, out)


def fig_gate(grouped, out: Path) -> None:
    """The refuted idea, per seed, because the effect is one scene rather than an average."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    width, offsets = 0.35, np.arange(len(SCENES))
    for shift, (variant, label, colour) in enumerate(
        ((UNGATED, "ungated (now default)", "#55A868"), (GATED, "gated at 0.05 IQR", "#C44E52"))
    ):
        means, errs, points = [], [], []
        for scene in SCENES:
            base = _stat(grouped, "gaussian", scene, "depth_abs_rel")
            runs = [r["depth_abs_rel"] for r in grouped.get((variant, scene), [])]
            deltas = [100.0 * (v - base[0]) / base[0] for v in runs]
            means.append(np.mean(deltas))
            errs.append(np.std(deltas))
            points.append(deltas)
        position = offsets + (shift - 0.5) * width
        ax.bar(position, means, width, yerr=errs, capsize=3, label=label, color=colour, alpha=0.85)
        for x, seeds in zip(position, points):
            ax.scatter([x] * len(seeds), seeds, color="k", s=9, zorder=3)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(offsets)
    ax.set_xticklabels(SCENES)
    ax.set_ylabel("depth abs_rel change (%)")
    ax.set_title("The gate: neutral twice, and it costs emerald its entire gain\n(3 seeds, dots)", fontsize=10)
    ax.legend(fontsize=8)
    _finish(fig, ax, out)


def _combination_rows(grouped, key: str, relative: bool):
    variants = (
        (UNGATED, "pd 0.1"),
        ("dn05_gaussian", "dn 0.05"),
        ("dvrel001_gaussian", "dvrel 0.01"),
        ("pd01_gaussian_dn", "pd + dn"),
        ("pd01_dvrel001_gaussian", "pd + dvrel"),
    )
    rows = []
    for variant, label in variants:
        values = []
        for scene in SCENES:
            stat = _stat(grouped, variant, scene, key)
            base = _stat(grouped, "gaussian", scene, key)
            if not stat or not base:
                values.append(np.nan)
            elif relative:
                values.append(100.0 * (stat[0] - base[0]) / base[0])
            else:
                values.append(stat[0])
        rows.append((label, values))
    return rows


def fig_compounding(grouped, out: Path) -> None:
    """Depth is sub-additive; normals are where the pairing pays. Same variants, both panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    depth_rows = _combination_rows(grouped, "depth_abs_rel", relative=True)
    normal_rows = _combination_rows(grouped, "normal_gain_vs_viewdir_deg", relative=False)

    for ax, rows, ylabel, title in (
        (axes[0], depth_rows, "depth abs_rel change (%)", "Depth: combination $\\approx$ max, not sum"),
        (axes[1], normal_rows, "n_gain vs viewdir control (deg)", "Normals: only the pairing clears the control"),
    ):
        positions = np.arange(len(rows))
        width = 0.26
        for index, scene in enumerate(SCENES):
            values = [row[1][index] for row in rows]
            ax.bar(positions + (index - 1) * width, values, width, label=scene, color=SCENE_COLOUR[scene])
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(positions)
        ax.set_xticklabels([row[0] for row in rows], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
    axes[1].text(
        0.02,
        0.06,
        "below 0: loses to a normal that uses no geometry",
        transform=axes[1].transAxes,
        fontsize=7,
        style="italic",
    )
    axes[0].legend(fontsize=8)
    _finish(fig, axes, out)


def fig_scene_split(grouped, out: Path) -> None:
    """The report's central claim: the two terms fix different scenes."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rows = (
        ("dn 0.05\n(internal consistency)", "dn05_gaussian"),
        ("pd 0.1\n(external prior)", UNGATED),
    )
    positions = np.arange(len(rows))
    width = 0.26
    for index, scene in enumerate(SCENES):
        values = []
        for _, variant in rows:
            stat = _stat(grouped, variant, scene, "depth_abs_rel")
            base = _stat(grouped, "gaussian", scene, "depth_abs_rel")
            values.append(100.0 * (stat[0] - base[0]) / base[0] if stat and base else np.nan)
        bars = ax.bar(positions + (index - 1) * width, values, width, label=scene, color=SCENE_COLOUR[scene])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:+.1f}", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=1.0)
    # Headroom for the value labels, which sit just above each (negative) bar's end.
    ax.set_ylim(min(-19.0, ax.get_ylim()[0]), 3.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in rows], fontsize=9)
    ax.set_ylabel("depth abs_rel change (%)")
    ax.set_title("Complementary by scene, not additive\nlone-monk is unreachable without the prior", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, ax, out)


def report_rgb_cost(fig_root: Path, scene: str, variant: str, frame: int) -> None:
    """Print where a variant's PSNR cost lands, from the rendered panels.

    PSNR is a single number and says nothing about whether a 1\u2009dB loss is a broad softening or
    a few pixels. This differences both renders against the reference and reports the split, so
    the deck's claim about *where* the cost sits is checkable. Requires Pillow, which the
    renderer already depends on.
    """
    from PIL import Image

    def load_rgb(name: str) -> np.ndarray:
        return np.asarray(Image.open(fig_root / scene / name).convert("RGB")).astype(np.float64)

    reference = load_rgb(f"gt_f{frame}_rgb.png")
    errors = {tag: np.abs(load_rgb(f"{tag}_f{frame}_rgb.png") - reference).mean(-1) for tag in ("baseline", variant)}
    advantage = errors["baseline"] - errors[variant]  # positive where `variant` is closer
    print(f"\n[{scene} f{frame}] mean |RGB error| vs reference, 0-255:")
    for tag, error in errors.items():
        print(f"  {tag:14s} {error.mean():.2f}")
    print(f"  {variant} closer on {(advantage > 0).mean():.1%} of pixels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out-dir", default="/tmp/report_fig/plots")
    parser.add_argument(
        "--fig-root",
        type=Path,
        default=None,
        help="rendered-panel root; when given, also print the RGB cost breakdown quoted in the deck",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = load(args.results)
    print(f"{len(grouped)} (variant, scene) cells")

    fig_prior_alignment(out_dir / "prior_alignment.pdf")
    fig_lambda_sweep(grouped, out_dir / "lambda_sweep.pdf")
    fig_gate(grouped, out_dir / "gate.pdf")
    fig_compounding(grouped, out_dir / "compounding.pdf")
    fig_scene_split(grouped, out_dir / "scene_split.pdf")

    if args.fig_root:
        report_rgb_cost(args.fig_root, "emerald-square", "pd01_dn", frame=4)


if __name__ == "__main__":
    main()
