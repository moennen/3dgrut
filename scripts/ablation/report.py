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

"""Summarize an ablation sweep produced by run_ob3d.py.

Quality and cost are shown side by side, because a variant that improves depth while
halving throughput or doubling the primitive count has made a trade rather than a gain.

Two reporting rules are deliberate. Variants are only averaged over scenes they all
completed, since a mean taken over a different scene subset per variant is not a
comparison; and the normal column carries its own control, because an unsupervised
normal buffer loses to a predictor that uses no geometry at all, so the raw angle on
its own invites exactly the wrong conclusion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# (json key, header, format, lower_is_better or None when not a quality metric)
QUALITY_COLUMNS = [
    ("mean_psnr", "psnr", "{:.2f}", False),
    ("mean_ssim", "ssim", "{:.3f}", False),
    ("mean_lpips", "lpips", "{:.3f}", True),
    ("depth_abs_rel", "d_absrel", "{:.4f}", True),
    ("depth_rmse", "d_rmse", "{:.3f}", True),
    ("depth_delta1", "d_delta1", "{:.3f}", False),
    ("depth_bias", "d_bias", "{:+.3f}", None),
    ("depth_covered_frac", "d_cover", "{:.3f}", False),
    ("depth_floater_frac", "d_float", "{:.4f}", True),
]

# Reported for information only; see the module docstring for why it is not ranked on.
NORMAL_COLUMNS = [
    ("normal_mean_deg", "n_mean", "{:.1f}", None),
    ("normal_median_deg", "n_med", "{:.1f}", None),
    ("normal_viewdir_control_deg", "n_control", "{:.1f}", None),
    ("normal_gain_vs_viewdir_deg", "n_gain", "{:+.1f}", None),
]

# The tracer reports 0.0 ms/frame when kernel timings are disabled, which is a sentinel
# for "not measured" rather than an impossibly fast render; show it as absent.
UNMEASURED_WHEN_ZERO = {"mean_inference_time_ms"}

# Metrics in world units. Averaging these across scenes is dominated by whichever scene
# is physically largest, so the scale-free ones are what a ranking should rest on.
SCALE_DEPENDENT = {"depth_rmse", "depth_mae", "depth_bias"}

COST_COLUMNS = [
    ("iteration_speed", "it/s", "{:.1f}", None),
    ("training_time_s", "train_s", "{:.0f}", None),
    ("num_gaussians", "prims", "{:,.0f}", None),
    ("peak_memory_allocated_gb", "mem_gb", "{:.2f}", None),
    ("mean_inference_time_ms", "ms/frame", "{:.2f}", None),
]


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Truncated final line from an interrupted sweep.
    # Later rows win, so a re-run of a failed cell supersedes the earlier failure.
    deduped = {(row.get("variant"), row.get("scene")): row for row in rows}
    return list(deduped.values())


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def render_table(headers: list[str], rows: list[list[str]], highlight: dict[int, int] | None = None) -> str:
    """A markdown table, with optional per-column best-value emphasis."""
    highlight = highlight or {}
    body = [list(row) for row in rows]
    for column, best_row in highlight.items():
        body[best_row][column] = f"**{body[best_row][column]}**"

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i]) for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    lines += ["| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |" for row in body]
    return "\n".join(lines)


def summary_table(rows: list[dict], columns: list[tuple], shared_scenes: set[str]) -> str:
    variants = sorted({row["variant"] for row in rows})
    headers = ["variant", "scenes"] + [header for _, header, _, _ in columns]
    table_rows: list[list[str]] = []
    numeric: dict[str, list[float | None]] = {}

    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant and row["scene"] in shared_scenes]
        cells = [variant, str(len(subset))]
        for key, _, fmt, _ in columns:
            value = mean([row.get(key) for row in subset])
            if key in UNMEASURED_WHEN_ZERO and value == 0.0:
                value = None
            cells.append("-" if value is None else fmt.format(value))
            numeric.setdefault(key, []).append(value)
        table_rows.append(cells)

    # Emphasize the best variant per ranked column so the winner is not left to the eye.
    highlight: dict[int, int] = {}
    for offset, (key, _, _, lower_is_better) in enumerate(columns):
        if lower_is_better is None:
            continue
        values = numeric[key]
        candidates = [(value, index) for index, value in enumerate(values) if value is not None]
        if len(candidates) < 2:
            continue
        best = min(candidates) if lower_is_better else max(candidates)
        highlight[offset + 2] = best[1]
    return render_table(headers, table_rows, highlight)


def per_scene_table(rows: list[dict], key: str, fmt: str) -> str:
    variants = sorted({row["variant"] for row in rows})
    scenes = sorted({row["scene"] for row in rows})
    lookup = {(row["variant"], row["scene"]): row.get(key) for row in rows}

    headers = ["scene"] + variants
    table_rows = []
    for scene in scenes:
        cells = [scene]
        for variant in variants:
            value = lookup.get((variant, scene))
            cells.append("-" if value is None else fmt.format(value))
        table_rows.append(cells)
    return render_table(headers, table_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="results.jsonl written by run_ob3d.py")
    parser.add_argument("--output", type=Path, default=None, help="Write markdown here instead of stdout")
    parser.add_argument("--per-scene", default="depth_abs_rel", help="Metric to break down per scene")
    args = parser.parse_args()

    rows = read_rows(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}")

    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    if not ok:
        raise SystemExit(f"No successful runs in {args.results}; {len(failed)} failed.")

    # Restrict to scenes every variant finished, so the averages compare like with like.
    variants = {row["variant"] for row in ok}
    scenes_by_variant = [{row["scene"] for row in ok if row["variant"] == variant} for variant in variants]
    shared = set.intersection(*scenes_by_variant) if scenes_by_variant else set()
    dropped = sorted({row["scene"] for row in ok} - shared)

    iterations = sorted({row.get("n_iterations") for row in ok})
    parts = [
        "# OB3D ablation",
        "",
        f"{len(ok)} successful run(s) over {len(shared)} shared scene(s), "
        f"{iterations[0] if len(iterations) == 1 else iterations} iterations.",
    ]
    if dropped:
        parts += [
            "",
            f"Excluded from the averages because not every variant completed them: {', '.join(dropped)}.",
        ]

    scale_dependent_headers = [header for key, header, _, _ in QUALITY_COLUMNS if key in SCALE_DEPENDENT]
    parts += [
        "",
        "## Quality (averaged over shared scenes)",
        "",
        summary_table(ok, QUALITY_COLUMNS, shared),
        "",
        f"{', '.join('`' + name + '`' for name in scale_dependent_headers)} are in world units, so an average "
        "across scenes is dominated by the physically largest one; rank on `d_absrel` and `d_delta1`, "
        "which are scale free, and read the world-unit columns per scene.",
    ]
    parts += ["", "## Cost", "", summary_table(ok, COST_COLUMNS, shared)]
    parts += [
        "",
        "## Normals (diagnostic, not ranked)",
        "",
        "`n_control` is the error of pointing every normal back along the view ray, which uses no",
        "geometry at all. `n_gain` is that control minus the rendered error, so a negative value",
        "means the buffer is beaten by the control and its raw angle should not be read as accuracy.",
        "",
        summary_table(ok, NORMAL_COLUMNS, shared),
    ]

    column_format = dict((key, fmt) for key, _, fmt, _ in QUALITY_COLUMNS + NORMAL_COLUMNS + COST_COLUMNS)
    if args.per_scene in column_format:
        parts += [
            "",
            f"## {args.per_scene} per scene",
            "",
            per_scene_table(ok, args.per_scene, column_format[args.per_scene]),
        ]

    if failed:
        parts += ["", "## Failures", ""]
        for row in failed:
            parts.append(f"- `{row.get('variant')}` / `{row.get('scene')}`: {row.get('status')} (`{row.get('log')}`)")

    report = "\n".join(parts) + "\n"
    if args.output:
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
