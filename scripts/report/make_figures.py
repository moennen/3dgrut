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

# The prior-family diagnostic (`pseudo_depth_diagnostic.py --json`), read from disk rather than
# transcribed. `PRIOR_ALIGNMENT` above stays hardcoded because it is measured against the
# *gaussian* 7k checkpoint that Experiment 3 argued about, while this sweep uses the `trisurfel`
# baselines the DA3 comparison was trained from; the two are not interchangeable.
#
# Order is the order plotted and tabulated: the training default first, then DA3's monocular
# models, then its any-view models smallest-first.
PRIOR_FAMILY = (
    ("Depth-Anything-V2-Base-hf", "DAv2-Base", "#4C72B0", "mono"),
    ("DA3MONO-LARGE", "DA3MONO-L", "#C44E52", "mono"),
    ("DA3METRIC-LARGE", "DA3METRIC-L", "#8172B2", "mono, metric"),
    ("DA3-SMALL", "DA3-SMALL", "#CCB974", "any-view"),
    ("DA3-LARGE", "DA3-LARGE", "#55A868", "any-view"),
    ("DA3-LARGE-1.1", "DA3-LARGE-1.1", "#64B5CD", "any-view"),
)

# The rungs of the alignment ladder, as `pseudo_depth_diagnostic.ALIGNMENTS` labels them, with
# short names for axes. Kept in increasing order of freedom, which is the axis of the argument.
PRIOR_RUNGS = (
    ("raw (no fit)", "raw"),
    ("scale only", "scale\nonly"),
    ("affine, global", "affine\nglobal"),
    ("affine, 64x64", "affine\n64x64"),
    ("affine, 32x32", "affine\n32x32"),
    ("affine, 16x16", "affine\n16x16"),
)

# Column headers for the rungs when tabulated, short enough to fit nine of them on a slide.
_RUNG_HEADER = {
    "raw (no fit)": "raw",
    "scale only": "scale",
    "affine, global": "affine",
    "affine, 64x64": "64px",
    "affine, 32x32": "32px",
    "affine, 16x16": "16px",
}

# The DA3-vs-DAv2 ablation, all on the trisurfel primitive at lambda=0.1.
DA3_REFERENCES = ("trisurfel",)
DA3_VARIANTS = ("pd01_trisurfel", "pd01da3_trisurfel")

# Experiment 8: the sparse-point-aligned regression term, all trisurfel with a DA3MONO prior.
# Its own root again, for the same reason as the DA3 sweep -- it re-ran the trisurfel baseline
# with its own three seeds, and the ordinal row is re-run here rather than borrowed from
# DA3_ROOT so that every row in the table shares one baseline.
L1_REFERENCES = ("trisurfel",)
L1_VARIANTS = ("pd01da3_trisurfel", "pdl1001_trisurfel", "pdl101_trisurfel", "pdl11_trisurfel")
L1_COMBINED = "pd01_pdl101_trisurfel"

# `delta1` is the headline for this term rather than `abs_rel`: it moves twice as far, and the
# fraction of pixels within 25% of truth is the quantity a mesh would care about. Reported as the
# failure rate `1 - delta1`, because 0.982 vs 0.813 reads as a small change and 1.8% vs 18.7%
# does not.
L1_METRICS = (
    ("depth_abs_rel", r"\code{abs\_rel}", "{:.4f}", True),
    ("depth_delta1_fail_pct", r"$\delta_1$ fail", "{:.1f}\\%", True),
    ("mean_psnr", "PSNR", "{:.2f}", False),
)

# The first sweep named the gated variant `pd01_gaussian` and the ungated one `*_nogate`; the
# default flipped afterwards. Records keep the original names, so resolve them here.
UNGATED = "pd01_gaussian_nogate"
GATED = "pd01_gaussian"

# The two reference runs. Both primitives at default settings, no geometry term. Every treatment
# is a modification of `gaussian`, but `trisurfel` is the stronger geometry baseline, so a
# treatment that does not beat it has not earned anything.
REFERENCES = ("gaussian", "trisurfel")


def load(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """All successful records, grouped by (variant, scene). Repeats are separate seeds.

    A results file can contain a torn line: the sweep appends a record per cell, and an
    interrupted or retried write leaves a fragment spliced onto the next record. One of these
    was found in the DA3 sweep. Such a line is skipped -- but *loudly*, with a count, because
    the failure it causes is a cell quietly averaging over fewer seeds than the caption claims,
    and an error bar shrinking to zero on a single seed looks like a strong result.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = 0
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"WARNING: {path}:{number} is not valid JSON, skipping ({error})")
                skipped += 1
                continue
            if record.get("status") != "ok":
                continue
            # A row whose iteration count was overridden is not comparable to a 7k one, and a
            # 200-iteration smoke run merged into a seed average once already. The harness now
            # records the effective count, so drop anything that is not the sweep's length.
            if record.get("n_iterations") not in (None, 7000):
                skipped += 1
                print(f"WARNING: {path}:{number} ran {record['n_iterations']} iterations, not 7000; skipping")
                continue
            if "depth_delta1" in record:
                record["depth_delta1_fail_pct"] = 100.0 * (1.0 - record["depth_delta1"])
            grouped[(record["variant"], record["scene"])].append(record)
    if skipped:
        print(f"WARNING: skipped {skipped} record(s); check seed counts below")
    return grouped


def report_seed_counts(grouped, variants: tuple[str, ...], label: str) -> None:
    """Print the seed count behind every cell, and flag any that are uneven.

    The deck quotes standard errors, which are only comparable across a row if the row has the
    same n in every cell. This makes an uneven grid visible at build time instead of leaving it
    to be inferred from a suspiciously tight error bar.
    """
    counts = {(variant, scene): len(grouped.get((variant, scene), [])) for variant in variants for scene in SCENES}
    distinct = sorted(set(counts.values()))
    print(f"{label}: seeds per cell {distinct}")
    if len(distinct) > 1:
        for (variant, scene), n in sorted(counts.items()):
            if n != max(distinct):
                print(f"  WARNING: {variant} / {scene} has {n} seed(s), not {max(distinct)}")


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


def load_prior_diagnostics(directory: Path) -> dict[tuple[str, str], dict]:
    """Diagnostic summaries keyed by (scene, short model name).

    Read from `--json` output rather than transcribed into this file, because the ladder is 108
    numbers and the earlier hardcoded three already cost one wrong claim in the deck.
    """
    summaries: dict[tuple[str, str], dict] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        summaries[(data["scene"], data["model_id"].split("/")[-1])] = data
    return summaries


def _rung(summary: dict, label: str, field: str = "abs_rel") -> float:
    for row in summary["alignments"]:
        if row["alignment"] == label:
            return row[field]
    raise KeyError(f"{summary['model_id']}: no alignment row {label!r}")


def fig_prior_ladder(diag, out: Path) -> None:
    """How much of each prior's accuracy is the prior, and how much is the fit.

    Log scale, because the interesting span is a factor of 100 between the unfitted and the
    finely fitted rung, and on a linear axis every prior collapses onto the same flat tail.
    """
    fig, axes = plt.subplots(1, len(SCENES), figsize=(11.0, 3.3), sharey=True)
    x = np.arange(len(PRIOR_RUNGS))
    for ax, scene in zip(axes, SCENES):
        for model, label, colour, _ in PRIOR_FAMILY:
            summary = diag.get((scene, model))
            if summary is None:
                continue
            ax.plot(
                x,
                [_rung(summary, rung) for rung, _ in PRIOR_RUNGS],
                marker="o",
                ms=3.5,
                lw=1.4,
                color=colour,
                label=label,
            )
        # The model the prior would be teaching. A prior above this line has nothing to offer at
        # that alignment, whatever its ranking against the other priors. Every cell for a scene
        # scores the same checkpoint, so any one of them carries this number.
        reference = diag[(scene, "DA3-LARGE")]["model_abs_rel"]
        ax.axhline(reference, color="k", ls="--", lw=1.1)
        ax.text(len(x) - 1, reference * 1.12, "7k model taught", ha="right", va="bottom", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([short for _, short in PRIOR_RUNGS], fontsize=6.5)
        ax.set_title(scene, fontsize=9, color=SCENE_COLOUR[scene])
        ax.grid(alpha=0.3, which="both", axis="y")
    axes[0].set_ylabel("prior abs_rel vs GT")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Prior accuracy against ground truth by alignment freedom (lower is better)", fontsize=10)
    _finish(fig, list(axes), out)


def fig_prior_ordinal(diag, out: Path) -> None:
    """Ordinal agreement per prior, against the trained model's own agreement.

    Separated from the abs_rel ladder because it is alignment-free and so has no ladder, and
    because it is the quantity the shipped loss actually reads.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.13
    base = np.arange(len(SCENES))
    for index, (model, label, colour, _) in enumerate(PRIOR_FAMILY):
        values = [100 * diag[(scene, model)]["ordinal"]["prior_agreement"] for scene in SCENES]
        ax.bar(base + (index - 2.5) * width, values, width=width, color=colour, label=label)
    for position, scene in zip(base, SCENES):
        model_agreement = 100 * diag[(scene, "DA3-LARGE")]["ordinal"]["model_agreement"]
        ax.plot(
            [position - 3 * width, position + 3 * width],
            [model_agreement] * 2,
            color="k",
            ls="--",
            lw=1.2,
            label="trained model" if position == 0 else None,
        )
    ax.set_xticks(base)
    ax.set_xticklabels(SCENES)
    ax.set_ylim(60, 92)
    ax.set_ylabel("pair ordering agreement\nwith GT (%)")
    ax.set_title("What the ordinal loss reads (higher is better)", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    _finish(fig, ax, out)


def fig_prior_offline_vs_trained(diag, grouped_da3, out: Path) -> None:
    """The mechanism check: which offline number ranked the scenes the way training did.

    Both axes are *changes* from swapping DAv2 for DA3MONO, one point per scene. That is the
    comparison the question actually asks -- "would this diagnostic have told me where the swap
    pays?" -- and it is not the same as the sign agreeing: measured per scene, the sign agrees
    for both candidates, and only the ordering separates them.

    Stated as ranks rather than a correlation coefficient, because with three scenes a
    coefficient invites more confidence than three points can support.
    """
    panels = (
        ("$\\Delta$ ordinal agreement (pp)", lambda s: 100 * s["ordinal"]["prior_agreement"], "%+.2f"),
        ("$\\Delta$ abs_rel, one affine per frame", lambda s: _rung(s, "affine, global"), "%+.4f"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, (xlabel, extract, _) in zip(axes, panels):
        points = []
        for scene in SCENES:
            gains = []
            for variant in DA3_VARIANTS:
                stat = _stat(grouped_da3, variant, scene, "depth_abs_rel")
                reference = _stat(grouped_da3, "trisurfel", scene, "depth_abs_rel")
                gains.append(100 * (stat[0] - reference[0]) / reference[0] if stat and reference else np.nan)
            delta_offline = extract(diag[(scene, "DA3MONO-LARGE")]) - extract(
                diag[(scene, "Depth-Anything-V2-Base-hf")]
            )
            points.append((delta_offline, gains[1] - gains[0], scene))
            ax.plot(delta_offline, gains[1] - gains[0], "o", ms=7, color=SCENE_COLOUR[scene])
            ax.annotate(
                scene,
                (delta_offline, gains[1] - gains[0]),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=7,
                color=SCENE_COLOUR[scene],
            )
        # A predictor is useful here if ordering scenes by it orders them by the trained change.
        by_offline = [scene for _, _, scene in sorted(points)]
        by_trained = [scene for _, _, scene in sorted(points, key=lambda p: p[1])]
        ax.set_title("ranks match" if by_offline == by_trained else "ranks disagree", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.3)
        ax.margins(0.25)
    axes[0].set_ylabel("$\\Delta$ trained depth abs_rel\nvs baseline (pp; lower = DA3 better)")
    fig.suptitle("Does the offline diagnostic rank the scenes the way training did? (DA3MONO $-$ DAv2)", fontsize=9.5)
    _finish(fig, list(axes), out)


def write_prior_family_table(diag, path: Path, rungs: tuple[str, ...]) -> None:
    """Priors as rows, (scene, rung) as column groups. Best in each column bolded."""
    lines = [
        "\\begin{tabular}{l" + ("r" * len(rungs) + "@{\\hskip 1.2em}") * len(SCENES) + "}",
        "\\toprule",
        "& " + " & ".join(f"\\multicolumn{{{len(rungs)}}}{{c}}{{\\texttt{{{scene}}}}}" for scene in SCENES) + " \\\\",
        "prior & " + " & ".join(_RUNG_HEADER[rung] for _ in SCENES for rung in rungs) + r" \\",
        "\\midrule",
    ]
    columns = [(scene, rung) for scene in SCENES for rung in rungs]
    best = {}
    for scene, rung in columns:
        values = [_rung(diag[(scene, m)], rung) for m, _, _, _ in PRIOR_FAMILY if (scene, m) in diag]
        best[(scene, rung)] = min(values) if values else float("nan")
    for model, label, _, family in PRIOR_FAMILY:
        cells = []
        for scene, rung in columns:
            summary = diag.get((scene, model))
            if summary is None:
                cells.append("---")
                continue
            value = _rung(summary, rung)
            text = f"{value:.4f}"
            cells.append(rf"\textbf{{{text}}}" if value == best[(scene, rung)] else text)
        lines.append(rf"\texttt{{{label}}} {{\tiny ({family})}} & " + " & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def fig_l1_vs_teacher(grouped, out: Path) -> None:
    """The trained model against the prior it was trained on, per scene.

    The argument that killed this term was per-pixel: the prior is closer to truth on only
    65-77% of pixels, so it should act as a floor. Putting the teacher's own accuracy on the same
    axis as the student's is the whole refutation -- the student is below the teacher everywhere,
    because per-frame prior error does not survive being fitted by one shared geometry.
    """
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    rows = ("trisurfel", "pd01da3_trisurfel", "pdl101_trisurfel")
    row_labels = ("baseline", "ordinal $\\lambda$0.1", "L1 $\\lambda$0.1")
    hatches = ("", "//", "")
    width, offsets = 0.26, np.arange(len(rows)) * 0.27 - 0.27

    for index, (variant, label, hatch) in enumerate(zip(rows, row_labels, hatches)):
        values, errors = [], []
        for scene in SCENES:
            stat = _stat(grouped, variant, scene, "depth_abs_rel")
            values.append(stat[0] if stat else np.nan)
            errors.append(stat[1] if stat else 0.0)
        ax.bar(
            np.arange(len(SCENES)) + offsets[index],
            values,
            width,
            yerr=errors,
            capsize=3,
            label=label,
            color=["#B0B0B0", "#8172B2", "#C44E52"][index],
            hatch=hatch,
            edgecolor="white",
        )
    # No bar-top labels: on sponza all three runs and the prior sit within 0.009 of each other,
    # so four numbers overlap into noise. The exact values are in `tab_l1.tex` on the previous
    # slide; what this figure has to show is the red bar falling below the dashed line, and only
    # the dashed line's value is not tabulated anywhere.
    #
    # One tick per scene rather than a line across the plot: the teacher's accuracy is a
    # different number per scene and a single axhline would imply otherwise. The label sits just
    # above the line's left end, where no bar reaches.
    for x, scene in enumerate(SCENES):
        prior = ALIGNED_PRIOR_ABS_REL[scene]
        ax.plot([x - 0.44, x + 0.44], [prior, prior], color="k", ls="--", lw=1.3, zorder=5)
        ax.text(x - 0.42, prior + 0.002, f"{prior:.3f}", va="bottom", ha="left", fontsize=7.5)

    ax.plot([], [], color="k", ls="--", lw=1.3, label="the aligned prior itself")
    ax.set_xticks(range(len(SCENES)))
    ax.set_xticklabels(SCENES)
    ax.set_ylabel("depth abs_rel vs GT")
    ax.set_title(
        "The L1 term takes the model below the prior that taught it (3 seeds; lower is better)",
        fontsize=10,
    )
    ax.set_ylim(0, 0.138)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _finish(fig, ax, out)


def _reference(grouped, scene: str, key: str, lower_better: bool) -> float | None:
    """The better of the two reference runs on this scene and metric.

    Every reported change is against this rather than against `gaussian` alone. It matters:
    `trisurfel` has the better `abs_rel` on emerald-square and the better normals on all three,
    so a gaussian-relative number overstates a treatment wherever trisurfel was already ahead --
    `pd 0.1`'s emerald depth gain is 10.3% against `gaussian` and 7.4% against the best
    reference.
    """
    values = [_stat(grouped, variant, scene, key) for variant in REFERENCES]
    finite = [stat[0] for stat in values if stat]
    if not finite:
        return None
    return (min if lower_better else max)(finite)


def fig_lambda_sweep(grouped, out: Path) -> None:
    """The usable band. Plotted on both axes the term trades between."""
    weights = [(0.1, UNGATED), (1.0, "pd1_gaussian"), (10.0, "pd10_gaussian_nogate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for scene in SCENES:
        base_depth = _reference(grouped, scene, "depth_abs_rel", lower_better=True)
        base_psnr = _reference(grouped, scene, "mean_psnr", lower_better=False)
        if base_depth is None or base_psnr is None:
            continue
        xs, depth, psnr = [], [], []
        for weight, variant in weights:
            depth_stat = _stat(grouped, variant, scene, "depth_abs_rel")
            psnr_stat = _stat(grouped, variant, scene, "mean_psnr")
            if not depth_stat or not psnr_stat:
                continue
            xs.append(weight)
            depth.append(100.0 * (depth_stat[0] - base_depth) / base_depth)
            psnr.append(psnr_stat[0] - base_psnr)
        axes[0].plot(xs, depth, "o-", color=SCENE_COLOUR[scene], label=scene)
        axes[1].plot(xs, psnr, "o-", color=SCENE_COLOUR[scene], label=scene)

    for ax, ylabel, title in (
        (axes[0], "depth abs_rel vs best reference (%)", "Depth: better is down"),
        (axes[1], "PSNR vs best reference (dB)", "Appearance: better is up"),
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
            base = _reference(grouped, scene, "depth_abs_rel", lower_better=True)
            runs = [r["depth_abs_rel"] for r in grouped.get((variant, scene), [])]
            deltas = [100.0 * (v - base) / base for v in runs]
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
    ax.set_ylabel("depth abs_rel vs best reference (%)")
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
            # Only the relative panel needs a denominator, and the one metric plotted relative
            # here (`abs_rel`) is lower-is-better.
            base = _reference(grouped, scene, key, lower_better=True) if relative else None
            if not stat or (relative and base is None):
                values.append(np.nan)
            elif relative:
                values.append(100.0 * (stat[0] - base) / base)
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
        (axes[0], depth_rows, "depth abs_rel vs best reference (%)", "Depth: combination $\\approx$ max, not sum"),
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
    # Every depth bar is negative and starts at zero, so an in-axes legend sits on top of one.
    # Open headroom above the axis and put it there.
    axes[0].set_ylim(top=abs(axes[0].get_ylim()[0]) * 0.42)
    axes[0].legend(fontsize=8, ncol=3, loc="upper center", frameon=False)
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
            base = _reference(grouped, scene, "depth_abs_rel", lower_better=True)
            values.append(100.0 * (stat[0] - base) / base if stat and base is not None else np.nan)
        bars = ax.bar(positions + (index - 1) * width, values, width, label=scene, color=SCENE_COLOUR[scene])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:+.1f}", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=1.0)
    # Headroom for the value labels, which sit just above each (negative) bar's end.
    ax.set_ylim(min(-19.0, ax.get_ylim()[0]), 3.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in rows], fontsize=9)
    ax.set_ylabel("depth abs_rel vs best reference (%)")
    ax.set_title("Complementary by scene, not additive\nlone-monk is unreachable without the prior", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _finish(fig, ax, out)


# Metric key -> (column header, format, lower-is-better).
METRICS = (
    ("depth_abs_rel", r"\code{abs\_rel}", "{:.4f}", True),
    ("normal_mean_deg", r"normal$^\circ$", "{:.1f}", True),
    # Math mode so the sign is a real minus rather than a hyphen.
    ("normal_gain_vs_viewdir_deg", r"\code{n\_gain}", "${:+.1f}$", False),
    ("mean_psnr", "PSNR", "{:.2f}", False),
)

# Display names, so the deck never has to explain a run-directory name.
LABELS = {
    "gaussian": r"\code{gaussian} (ref)",
    "trisurfel": r"\code{trisurfel} (ref)",
    UNGATED: r"\code{pd 0.1}",
    GATED: r"\code{pd 0.1} gated",
    "pd1_gaussian": r"\code{pd 1}",
    "pd10_gaussian_nogate": r"\code{pd 10}",
    "dn05_gaussian": r"\code{dn 0.05}",
    "dvrel001_gaussian": r"\code{dvrel 0.01}",
    "dvrel1_gaussian": r"\code{dvrel 1}",
    "pd01_gaussian_dn": r"\code{pd 0.1 + dn 0.05}",
    "pd01_dvrel001_gaussian": r"\code{pd 0.1 + dvrel 0.01}",
    "pd1_dvrel1_gaussian": r"\code{pd 1 + dvrel 1}",
    # Experiment 7. Both are lambda=0.1 ordinal supervision on the trisurfel primitive, so the
    # label names the prior rather than the term: the prior is the only thing that differs.
    "pd01_trisurfel": r"\code{pd 0.1}, DAv2-Base",
    "pd01da3_trisurfel": r"\code{pd 0.1}, DA3MONO-L",
    # Experiment 8. `pd` is the ordinal term, `pdl1` the regression one; both read the same
    # DA3MONO prior, so the label names the loss and the weight.
    "pdl1001_trisurfel": r"\code{pdl1 0.01}",
    "pdl101_trisurfel": r"\code{pdl1 0.1}",
    "pdl11_trisurfel": r"\code{pdl1 1}",
    L1_COMBINED: r"\code{pd 0.1 + pdl1 0.1}",
}

# The aligned prior's own `abs_rel`, measured on the val split through the *training* path --
# dataset alignment plus the loss's z-to-distance conversion -- by
# `threedgrut/datasets/tests/test_sparse_depth_alignment_ob3d.py`. Plotted as the teacher's
# accuracy, because the trained model ends up below it on every scene and that is the point.
ALIGNED_PRIOR_ABS_REL = {"sponza": 0.0582, "lone-monk": 0.0422, "emerald-square": 0.0722}


def _best_cells(grouped, rows, scene, metrics) -> dict[str, list[str]]:
    """Formatted cells for one scene, with the best value in each metric bolded.

    "Best" counts the reference rows, so a treatment that does not beat `trisurfel` is visibly
    not the best cell rather than just a number in a list.
    """
    cells: dict[str, list[str]] = {variant: [] for variant in rows}
    for key, _, fmt, lower_better in metrics:
        values = [(_stat(grouped, variant, scene, key) or (float("nan"),))[0] for variant in rows]
        finite = [v for v in values if not np.isnan(v)]
        best = (min if lower_better else max)(finite) if finite else float("nan")
        for variant, value in zip(rows, values):
            text = "---" if np.isnan(value) else fmt.format(value)
            if not np.isnan(value) and value == best:
                text = rf"\textbf{{{text}}}"
            cells[variant].append(text)
    return cells


def write_experiment_table(
    grouped, path: Path, variants: tuple[str, ...], metrics=METRICS, references: tuple[str, ...] = REFERENCES
) -> None:
    """Runs as rows, scenes as column groups. Both reference runs are always the first rows.

    Generated rather than typed. Transcribing these by hand into the deck produced two wrong
    numbers and one wrong claim on the first pass, so the deck `\\input`s this instead.

    Scenes go across rather than down because a slide is wider than it is tall: stacking three
    scene blocks vertically overflowed the frame as soon as a table had more than two treatments.
    """
    rows = references + variants
    group = "r" * len(metrics)
    scene_heads = " & ".join(rf"\multicolumn{{{len(metrics)}}}{{c}}{{\itshape {scene}}}" for scene in SCENES)
    metric_heads = " & ".join(" & ".join(head for _, head, _, _ in metrics) for _ in SCENES)
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}" for i in range(len(SCENES))
    )
    lines = [
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{l" + group * len(SCENES) + "}",
        r"\toprule",
        rf"Run & {scene_heads} \\",
        rules,
        rf" & {metric_heads} \\",
        r"\midrule",
    ]
    per_scene = [_best_cells(grouped, rows, scene, metrics) for scene in SCENES]
    for variant in rows:
        cells = [text for scene_cells in per_scene for text in scene_cells[variant]]
        lines.append(LABELS.get(variant, variant) + " & " + " & ".join(cells) + r" \\")
        if variant == references[-1] and variants:
            # Rule between the reference block and the treatments, so the comparison the table
            # exists to make is visible without reading the row labels.
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_scene_table(
    grouped,
    path: Path,
    variants: tuple[str, ...],
    key: str,
    fmt: str,
    lower_better: bool,
    references: tuple[str, ...] = REFERENCES,
) -> None:
    """One metric, variants as rows and scenes as columns, with the delta against that scene's
    best reference. Compact enough to sit beside a figure."""
    lines = [
        r"\begin{tabular}{l" + "rr" * len(SCENES) + "}",
        r"\toprule",
        r"Run & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{scene}}}" for scene in SCENES) + r" \\",
        r"\midrule",
    ]
    reference_best = {}
    for scene in SCENES:
        values = [(_stat(grouped, variant, scene, key) or (float("nan"),))[0] for variant in REFERENCES]
        reference_best[scene] = (min if lower_better else max)(v for v in values if not np.isnan(v))

    for variant in references + variants:
        cells = []
        for scene in SCENES:
            stat = _stat(grouped, variant, scene, key)
            if not stat:
                cells += ["---", ""]
                continue
            delta = 100.0 * (stat[0] - reference_best[scene]) / abs(reference_best[scene])
            improved = (delta < 0) if lower_better else (delta > 0)
            colour = "good" if improved else "bad"
            marker = "" if variant in REFERENCES else rf"\textcolor{{{colour}}}{{${delta:+.1f}$\%}}"
            cells += [fmt.format(stat[0]), marker]
        lines.append(LABELS.get(variant, variant) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


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
        "--da3-results",
        nargs="*",
        type=Path,
        default=(),
        help="results.jsonl for the DA3-vs-DAv2 prior sweep; kept separate from `results` so "
        "that its extra baseline seeds cannot change any number already published in the deck",
    )
    parser.add_argument(
        "--l1-results",
        nargs="*",
        type=Path,
        default=(),
        help="results.jsonl for the sparse-aligned regression sweep (Experiment 8); separate "
        "from `results` for the same reason as --da3-results",
    )
    parser.add_argument(
        "--prior-diagnostic-dir",
        type=Path,
        default=None,
        help="directory of `pseudo_depth_diagnostic.py --json` summaries, one per (scene, prior)",
    )
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

    # One table per experiment, each carrying both reference rows so the comparison is built in.
    write_experiment_table(grouped, out_dir / "tab_reference.tex", ())
    write_experiment_table(grouped, out_dir / "tab_dn.tex", ("dn05_gaussian",))
    write_experiment_table(grouped, out_dir / "tab_dv.tex", ("dvrel1_gaussian", "dvrel001_gaussian"))
    write_experiment_table(grouped, out_dir / "tab_pd.tex", (UNGATED, "pd1_gaussian", "pd10_gaussian_nogate"))
    write_experiment_table(grouped, out_dir / "tab_gate.tex", (UNGATED, GATED))
    write_experiment_table(
        grouped,
        out_dir / "tab_combined.tex",
        (UNGATED, "dn05_gaussian", "pd01_gaussian_dn", "dvrel001_gaussian", "pd01_dvrel001_gaussian"),
    )
    # The weighting trap: the same pair at its measured optima and at the first pass's lambda=1.
    write_scene_table(
        grouped,
        out_dir / "tab_trap.tex",
        (UNGATED, "dvrel001_gaussian", "pd01_dvrel001_gaussian", "pd1_dvrel1_gaussian"),
        "depth_abs_rel",
        "{:.4f}",
        lower_better=True,
    )
    write_scene_table(
        grouped,
        out_dir / "tab_depth_summary.tex",
        (UNGATED, "dn05_gaussian", "dvrel001_gaussian", "pd01_gaussian_dn", "pd01_dvrel001_gaussian"),
        "depth_abs_rel",
        "{:.4f}",
        lower_better=True,
    )
    write_scene_table(
        grouped,
        out_dir / "tab_normal_summary.tex",
        (UNGATED, "dn05_gaussian", "dvrel001_gaussian", "pd01_gaussian_dn", "pd01_dvrel001_gaussian"),
        "normal_mean_deg",
        "{:.1f}",
        lower_better=True,
    )

    # Experiment 7 is a separate data source on both counts: its own sweep for the trained
    # numbers and the diagnostic JSON for the offline ones. Skipped rather than faked when
    # either is absent, so the deck still builds from the original artifacts alone.
    if args.prior_diagnostic_dir:
        diag = load_prior_diagnostics(args.prior_diagnostic_dir)
        print(f"{len(diag)} (scene, prior) diagnostic cells")
        fig_prior_ladder(diag, out_dir / "prior_ladder.pdf")
        fig_prior_ordinal(diag, out_dir / "prior_ordinal.pdf")
        write_prior_family_table(
            diag, out_dir / "tab_prior_family.tex", ("raw (no fit)", "scale only", "affine, global", "affine, 16x16")
        )
    if args.da3_results:
        grouped_da3 = load(args.da3_results)
        report_seed_counts(grouped_da3, DA3_REFERENCES + DA3_VARIANTS, "DA3 sweep")
        write_experiment_table(grouped_da3, out_dir / "tab_da3.tex", DA3_VARIANTS, references=DA3_REFERENCES)
        if args.prior_diagnostic_dir:
            fig_prior_offline_vs_trained(diag, grouped_da3, out_dir / "prior_offline_vs_trained.pdf")
    if args.l1_results:
        grouped_l1 = load(args.l1_results)
        report_seed_counts(grouped_l1, L1_REFERENCES + L1_VARIANTS + (L1_COMBINED,), "L1 sweep")
        fig_l1_vs_teacher(grouped_l1, out_dir / "l1_vs_teacher.pdf")
        write_experiment_table(
            grouped_l1, out_dir / "tab_l1.tex", L1_VARIANTS, metrics=L1_METRICS, references=L1_REFERENCES
        )
        # The composition test gets its own table: it answers a different question from the
        # weight sweep, and putting five rows on one slide made neither readable.
        write_experiment_table(
            grouped_l1,
            out_dir / "tab_l1_combined.tex",
            ("pd01da3_trisurfel", "pdl101_trisurfel", L1_COMBINED),
            metrics=L1_METRICS,
            references=L1_REFERENCES,
        )

    if args.fig_root:
        report_rgb_cost(args.fig_root, "emerald-square", "pd01_dn", frame=4)


if __name__ == "__main__":
    main()
