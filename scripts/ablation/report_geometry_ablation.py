#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create the Markdown tracker and PDF deck for ``run_geometry_ablation.py``."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def records(path: Path) -> list[dict]:
    """Read JSONL safely; later retry rows replace earlier attempts for the same cell."""
    latest = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[(row.get("suite"), row.get("scene"), row.get("variant"))] = row
    return list(latest.values())


def finite_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def metric(row: dict, name: str) -> float | None:
    if name == "psnr":
        return row.get("mean_psnr")
    if name == "absrel":
        return row.get("depth_abs_rel")
    evaluation = row.get("evaluation", {})
    if name == "dtu_recall_5mm":
        recall = evaluation.get("recall", {})
        taus, values = recall.get("taus", []), recall.get("recall", [])
        return values[taus.index(5.0)] if 5.0 in taus else None
    if name == "dtu_chamfer_mm":
        return evaluation.get("surface", {}).get("overall")
    if name == "tnt_recall":
        values = evaluation.get("recall", {}).get("recall", [])
        return values[0] if values else None
    if name == "tnt_f1":
        values = evaluation.get("surface", {}).get("fscore", [])
        return values[0] if values else None
    raise ValueError(name)


METRICS = {
    "ob3d": (("psnr", "PSNR ↑", ".2f"), ("absrel", "AbsRel ↓", ".4f")),
    "dtu": (("dtu_recall_5mm", "visible recall @5 mm ↑", ".3f"), ("dtu_chamfer_mm", "Chamfer (mm) ↓", ".2f")),
    "tnt": (("tnt_recall", "visible recall @official τ ↑", ".3f"), ("tnt_f1", "F1 @official τ ↑", ".3f")),
}


def shared_rows(rows: list[dict], suite: str) -> tuple[list[dict], set[str]]:
    successful = [row for row in rows if row.get("suite") == suite and row.get("status") == "ok"]
    variants = {row["variant"] for row in successful}
    scenes = [{row["scene"] for row in successful if row["variant"] == variant} for variant in variants]
    return successful, set.intersection(*scenes) if scenes else set()


def table(rows: list[dict], suite: str) -> tuple[list[str], list[list[str]], dict[str, list[float | None]]]:
    rows, scenes = shared_rows(rows, suite)
    variants = sorted({row["variant"] for row in rows})
    specs = METRICS[suite]
    numeric, body = {}, []
    for variant in variants:
        cells = [row for row in rows if row["variant"] == variant and row["scene"] in scenes]
        values = [finite_mean(metric(row, key) for row in cells) for key, _, _ in specs]
        numeric[variant] = values
        body.append(
            [variant, str(len(cells))]
            + ["—" if value is None else format(value, fmt) for value, (_, _, fmt) in zip(values, specs)]
        )
    return ["variant", "shared scenes"] + [header for _, header, _ in specs], body, numeric


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def markdown(rows: list[dict]) -> str:
    ok = [row for row in rows if row.get("status") == "ok"]
    parts = [
        "# 30k geometry-improvement ablation",
        "",
        "This report averages only scenes completed by every variant in a suite.  It is therefore safe to compare rows; incomplete scenes are not allowed to make one variant look better by disappearing.",
        "",
        "## Protocol",
        "",
        "- 30,000 iterations per cell (recorded in `results.jsonl`).",
        "- Default preset is the fixed one-third OB3D/DTU/TnT split; `protocol.json` records the exact selection.",
        "- MoGe-3 is the only pseudo-depth backend in this matrix. The feature condition is frozen C-RADIOv4 (`c-radio_v4-h`) PCA reduced to 48 dimensions, decoded by direct NHT.",
        "- DTU/TnT use renderer Euclidean ray depth, shared TSDF fusion, visibility-aware recall, DTU observation masking, and exact chunked surface queries. DTU Chamfer is in mm; TnT thresholds and F1 are in official metres.",
    ]
    for suite, title in (
        ("ob3d", "OB3D: novel-view and rendered-depth diagnostics"),
        ("dtu", "DTU: visible depth recall and mesh"),
        ("tnt", "Tanks and Temples: visible depth recall and mesh"),
    ):
        headers, body, _ = table(rows, suite)
        _, scenes = shared_rows(rows, suite)
        parts += [
            "",
            f"## {title}",
            "",
            f"Shared scenes ({len(scenes)}): {', '.join(sorted(scenes)) or 'none'}.",
            "",
            markdown_table(headers, body),
        ]
    failed = [row for row in rows if row.get("status") != "ok" or row.get("score_status") not in ("ok", "skipped")]
    if failed:
        parts += ["", "## Incomplete cells", ""]
        parts += [
            f"- `{r.get('suite')}/{r.get('scene')}/{r.get('variant')}`: training `{r.get('status')}`, scoring `{r.get('score_status', 'not started')}`; log `{r.get('log')}`."
            for r in failed
        ]
    parts += ["", f"Completed training cells: {len(ok)}/{len(rows)}.", ""]
    return "\n".join(parts)


def add_table_page(pdf: PdfPages, title: str, headers: list[str], body: list[list[str]]) -> None:
    figure, axis = plt.subplots(figsize=(16, max(3, 0.42 * (len(body) + 3))))
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=18, fontweight="bold")
    table_artist = axis.table(
        cellText=body or [["No completed cells"] + [""] * (len(headers) - 1)],
        colLabels=headers,
        loc="center",
        cellLoc="left",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.45)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_bar_page(pdf: PdfPages, rows: list[dict], suite: str) -> None:
    _, _, values = table(rows, suite)
    specs = METRICS[suite]
    names = list(values)
    figure, axes = plt.subplots(1, len(specs), figsize=(15, 5))
    axes = np.atleast_1d(axes)
    for axis, (key, title, _) in zip(axes, specs):
        data = [values[name][[item[0] for item in specs].index(key)] for name in names]
        axis.bar(range(len(names)), [value if value is not None else 0.0 for value in data])
        axis.set_title(title)
        axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"{suite.upper()} 30k geometry ablation", fontsize=16, fontweight="bold")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def pdf(rows: list[dict], output: Path) -> None:
    with PdfPages(output) as deck:
        for suite, title in (("ob3d", "OB3D"), ("dtu", "DTU"), ("tnt", "Tanks and Temples")):
            headers, body, _ = table(rows, suite)
            add_table_page(deck, f"{title}: 30k geometry ablation", headers, body)
            add_bar_page(deck, rows, suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    data = records(args.results)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(data))
    pdf(data, args.pdf)
    print(f"Wrote {args.markdown} and {args.pdf}")


if __name__ == "__main__":
    main()
