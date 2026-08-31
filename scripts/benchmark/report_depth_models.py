# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn ``evaluate_depth_models.py`` records into a Markdown tracker and PDF slide deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def rows_by_suite(records: list[dict], suite: str) -> list[dict]:
    return [record for record in records if record["suite"] == suite]


def pivot(records: list[dict], value) -> tuple[list[str], list[list[str]]]:
    models = sorted({record["model"] for record in records})
    alignments = ["raw", "scale", "affine"]
    table = []
    for model in models:
        cells = []
        for alignment in alignments:
            cell = next(
                (record for record in records if record["model"] == model and record["alignment"] == alignment), None
            )
            cells.append(number(value(cell) if cell else None))
        table.append(cells)
    return models, table


def markdown(records: list[dict], note: str) -> str:
    ob3d, dtu, tnt = (rows_by_suite(records, name) for name in ("ob3d", "dtu", "tnt"))
    text = ["# Depth-model evaluation", "", note, "", "## OB3D depth accuracy", ""]
    models, table = pivot(ob3d, lambda row: row["depth"]["abs_rel"])
    text += ["| model | raw abs-rel | scale abs-rel | affine abs-rel |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    models, table = pivot(dtu, lambda row: row["recall"]["recall"][3])
    text += [
        "",
        "## DTU scan24",
        "",
        "Recall is visibility-corrected recall@5 mm; scale and affine are oracle scan-z-buffer fits.",
        "",
    ]
    text += ["| model | raw recall@5mm | scale recall@5mm | affine recall@5mm |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    models, table = pivot(dtu, lambda row: row["surface"].get("overall"))
    text += ["", "DTU TSDF mesh Chamfer is `(accuracy + completeness) / 2` in millimetres; `—` is an empty mesh.", ""]
    text += ["| model | raw Chamfer | scale Chamfer | affine Chamfer |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    models, table = pivot(tnt, lambda row: row["recall"]["recall"][0])
    text += ["", "## Tanks and Temples Barn", "", "Recall is visibility-corrected at the official 1 cm tolerance.", ""]
    text += ["| model | raw recall@1cm | scale recall@1cm | affine recall@1cm |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    models, table = pivot(tnt, lambda row: row["surface"].get("fscore", [None])[0])
    text += ["", "TnT mesh F1 is at the official 1 cm Barn tolerance; `—` denotes an empty mesh.", ""]
    text += ["| model | raw F1 | scale F1 | affine F1 |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    text += [
        "",
        "## Reading this run",
        "",
        "- Raw DAv2/DA3 are intentionally uncalibrated relative outputs; their absolute scores are not comparable to a metric-depth claim.",
        "- Per-frame scale/affine use a GT scan z-buffer and are explicitly oracle diagnostics, not deployable alignments.",
        "- DTU uses its ground-plane completeness cull and predicted-surface observation mask; TnT uses its official crop plus a GT visibility z-buffer to remove scan self-occlusion.",
        "- TSDF is the shared posed-depth path used by checkpoint extraction and this benchmark. One-view fusion is expected to be incomplete; mesh values here validate the path, not a competitive reconstruction setting.",
    ]
    return "\n".join(text) + "\n"


def table_page(
    pdf: PdfPages, title: str, headers: list[str], labels: list[str], values: list[list[str]], subtitle: str
) -> None:
    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.axis("off")
    axis.set_title(title, fontsize=24, weight="bold", pad=32)
    axis.text(0.5, 0.88, subtitle, transform=axis.transAxes, ha="center", fontsize=12)
    table = axis.table(cellText=values, rowLabels=labels, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 2.3)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def bar_page(pdf: PdfPages, records: list[dict]) -> None:
    """A compact visual comparison; tables retain the exact values beside this graph."""
    ob3d = rows_by_suite(records, "ob3d")
    models = sorted({record["model"] for record in ob3d})
    alignments = ["raw", "scale", "affine"]
    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(models))
    width = 0.24
    for index, alignment in enumerate(alignments):
        values = [
            next(
                record["depth"]["abs_rel"]
                for record in ob3d
                if record["model"] == model and record["alignment"] == alignment
            )
            for model in models
        ]
        axis.bar(x + (index - 1) * width, values, width, label=alignment)
    axis.set_xticks(x, models)
    axis.set_ylabel("absolute relative depth error")
    axis.set_title("OB3D depth alignment ladder", fontsize=24, weight="bold")
    axis.legend(title="alignment")
    axis.grid(axis="y", alpha=0.25)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def qualitative_page(pdf: PdfPages, result_dir: Path, dtu_root: Path, records: list[dict]) -> None:
    dtu = rows_by_suite(records, "dtu")
    models = sorted({record["model"] for record in dtu})
    if not models:
        return
    scene = dtu[0]["scene"]
    image = np.asarray(plt.imread(dtu_root / scene / "images" / "0000.png"))
    figure, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 7))
    if len(models) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    axes[0, 0].set_ylabel("RGB", fontsize=14)
    axes[1, 0].set_ylabel("affine ray depth", fontsize=14)
    for column, model in enumerate(models):
        axes[0, column].imshow(image)
        axes[0, column].set_title(model, fontsize=15)
        depth_path = result_dir / "dtu" / scene / model / "affine" / "depths" / "0000.npy"
        depth = np.load(depth_path)
        axes[1, column].imshow(depth, cmap="turbo", vmin=np.nanpercentile(depth, 2), vmax=np.nanpercentile(depth, 98))
        for axis in axes[:, column]:
            axis.axis("off")
    figure.suptitle("Qualitative sample — DTU scan24, per-frame affine oracle", fontsize=20, weight="bold")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--dtu-root", type=Path, default=Path("/mnt/data/nerf_datasets/dtu_dataset/dtu"))
    parser.add_argument(
        "--note", default="Smoke protocol: one view per scene at a 160 px maximum side; meshes use 1,000 samples."
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(records, args.note))
    with PdfPages(args.pdf) as pdf:
        figure, axis = plt.subplots(figsize=(13.33, 7.5))
        axis.axis("off")
        axis.text(0.5, 0.62, "Depth-model evaluation", ha="center", fontsize=34, weight="bold")
        axis.text(0.5, 0.48, args.note, ha="center", fontsize=17, wrap=True)
        axis.text(0.5, 0.34, "DAv2 · DA3 · MoGe-3\nraw · per-frame scale · per-frame affine", ha="center", fontsize=19)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
        ob3d = rows_by_suite(records, "ob3d")
        models, table = pivot(ob3d, lambda row: row["depth"]["abs_rel"])
        table_page(
            pdf, "OB3D depth", ["raw abs-rel", "scale abs-rel", "affine abs-rel"], models, table, "Lower is better"
        )
        bar_page(pdf, records)
        dtu = rows_by_suite(records, "dtu")
        models, table = pivot(dtu, lambda row: row["recall"]["recall"][3])
        table_page(
            pdf,
            "DTU scan24 recall",
            ["raw @5mm", "scale @5mm", "affine @5mm"],
            models,
            table,
            "Visibility-corrected; higher is better",
        )
        tnt = rows_by_suite(records, "tnt")
        models, table = pivot(tnt, lambda row: row["surface"].get("fscore", [None])[0])
        table_page(
            pdf,
            "Tanks and Temples Barn mesh",
            ["raw F1", "scale F1", "affine F1"],
            models,
            table,
            "Official 1 cm F1; one-view meshes are deliberately incomplete",
        )
        qualitative_page(pdf, args.results.parent, args.dtu_root, records)
    print(f"Wrote {args.markdown} and {args.pdf}")


if __name__ == "__main__":
    main()
