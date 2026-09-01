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


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def pivot(records: list[dict], value) -> tuple[list[str], list[list[str]]]:
    """Average a metric over scenes instead of silently selecting the first matching row."""
    models = sorted({record["model"] for record in records})
    alignments = ["raw", "scale", "affine"]
    table = []
    for model in models:
        cells = []
        for alignment in alignments:
            matches = [record for record in records if record["model"] == model and record["alignment"] == alignment]
            cells.append(number(mean([value(record) for record in matches])))
        table.append(cells)
    return models, table


def scene_view_rows(records: list[dict]) -> list[tuple[str, str, int | None]]:
    """One verified view-count row per benchmark scene."""
    grouped: dict[tuple[str, str], set[int]] = {}
    for record in records:
        count = record.get("views", record.get("frames"))
        if count is not None:
            grouped.setdefault((record["suite"], record["scene"]), set()).add(int(count))
    rows = []
    for (suite, scene), counts in sorted(grouped.items()):
        if len(counts) != 1:
            raise ValueError(f"Inconsistent view counts for {suite}/{scene}: {sorted(counts)}")
        rows.append((suite, scene, counts.pop()))
    return rows


def suite_view_subtitle(records: list[dict], suite: str, fallback: str) -> str:
    rows = [(scene, count) for record_suite, scene, count in scene_view_rows(records) if record_suite == suite]
    return "; ".join(f"{scene}: {count} posed views" for scene, count in rows) if rows else fallback


def recall_curve_data(
    records: list[dict], normalize_taus: bool = False
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    """Return mean per-scene visible-recall curves grouped by model and alignment.

    TnT's official tolerance is scene-specific.  Normalizing its three thresholds by the first
    one makes a cross-scene curve meaningful while preserving the exact raw values in JSONL.
    """
    curves: dict[tuple[str, str], list[np.ndarray]] = {}
    reference_taus: np.ndarray | None = None
    for record in records:
        recall = record["recall"]
        taus = np.asarray(recall["taus"], dtype=np.float64)
        values = np.asarray(recall["recall"], dtype=np.float64)
        if taus.ndim != 1 or values.shape != taus.shape:
            raise ValueError(f"Invalid recall curve in {record['suite']}/{record['scene']}")
        if normalize_taus:
            if taus[0] <= 0:
                raise ValueError(
                    f"Cannot normalize a non-positive recall tolerance in {record['suite']}/{record['scene']}"
                )
            taus = taus / taus[0]
        if reference_taus is None:
            reference_taus = taus
        elif not np.allclose(reference_taus, taus):
            raise ValueError("Cannot combine recall curves with different tolerance grids")
        curves.setdefault((record["model"], record["alignment"]), []).append(values)
    if reference_taus is None:
        return np.empty(0), {}
    return reference_taus, {key: np.mean(values, axis=0) for key, values in curves.items()}


def recall_curve_table(records: list[dict], normalize_taus: bool, unit: str) -> list[str]:
    """Render all curve samples instead of hiding them behind one headline tolerance."""
    taus, curves = recall_curve_data(records, normalize_taus)
    headers = [f"@{tau:g}{unit}" for tau in taus]
    text = [
        "| model | alignment | " + " | ".join(headers) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for (model, alignment), values in sorted(curves.items()):
        text.append(f"| {model} | {alignment} | " + " | ".join(number(value) for value in values) + " |")
    return text


def markdown(records: list[dict], note: str) -> str:
    ob3d, dtu, tnt = (rows_by_suite(records, name) for name in ("ob3d", "dtu", "tnt"))
    text = [
        "# Depth-model evaluation",
        "",
        note,
        "",
        "## Step-by-step workflow",
        "",
        "`evaluate_depth_models.py` is the single runner. It evaluates DAv2 (`dav2`), DA3 (`dav3`) and "
        "MoGe-3 (`moge3`) in `raw`, per-frame `scale`, and per-frame `affine` conditions.",
        "",
        "### 1. Install / verify optional runtimes",
        "",
        "The normal project virtual environment supplies DAv2. DA3 and MoGe-3 are intentionally isolated so their "
        "upstream pins cannot alter the project environment. TSDF fusion and mesh sampling need Open3D.",
        "",
        "```bash",
        "cd /mnt/oss/3dgrut-bernardin",
        ".venv/bin/python -m pip install 'open3d>=0.18'",
        "test -f /mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt",
        "```",
        "",
        "If the MoGe-3 checkpoint is absent, download the official ViT-L checkpoint:",
        "",
        "```bash",
        "mkdir -p /mnt/oss/MoGe/checkpoints/moge-3-vitl",
        ".venv/bin/python -m huggingface_hub.commands.huggingface_cli download Ruicheng/moge-3-vitl model.pt \\",
        "  --local-dir /mnt/oss/MoGe/checkpoints/moge-3-vitl",
        "```",
        "",
        "### 2. Verify dataset mounts",
        "",
        "```bash",
        "test -d /mnt/data/nerf_datasets/ob3d/OB3D_colmap/emerald-square",
        "test -f /mnt/data/nerf_datasets/dtu_dataset/dtu_eval/Points/stl/stl024_total.ply",
        "test -f /mnt/data/nerf_datasets/tnt_dataset/tnt/Barn/Barn.ply",
        "```",
        "",
        "### 3. Export the runtime paths",
        "",
        "```bash",
        "export PYTHONPATH=/mnt/oss/meshdeps:/mnt/oss/MoGe:/mnt/oss/moge3deps:/mnt/oss/Depth-Anything-3/src:/mnt/oss/da3deps",
        "export CUDA_VISIBLE_DEVICES=0",
        "```",
        "",
        "### 4. Run a cheap end-to-end smoke test",
        "",
        "```bash",
        ".venv/bin/python scripts/benchmark/evaluate_depth_models.py \\",
        "  --out-dir /tmp/depth-benchmark-smoke --max-frames 1 --max-image-side 160 \\",
        "  --mesh-samples 1000 --gt-voxel 10 \\",
        "  --ob3d-scenes emerald-square --dtu-scenes scan24 --tnt-scenes Barn",
        "```",
        "",
        "### 5. Run the benchmark",
        "",
        "`--dataset-scale full` is the default and evaluates every supported uploaded sequence. "
        "`--dataset-scale reduced` always selects the same fixed one-third subset: OB3D "
        "`archiviz-flat,classroom,lone-monk,san-miguel`; DTU `scan105,scan114,scan24,scan55,scan69`; "
        "and TnT `Barn,Ignatius`. Omit "
        "`--max-frames` and `--max-image-side` for the full-resolution multi-view run. The default surface sampler "
        "uses two million uniform mesh samples; keep `--gt-voxel` unset for benchmark scoring. The MoGe-3 default "
        "checkpoint is `/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt`.",
        "",
        "```bash",
        ".venv/bin/python scripts/benchmark/evaluate_depth_models.py \\",
        "  --out-dir /mnt/oss/results/depth-benchmark-2026-08-31 \\",
        "  --models dav2,dav3,moge3 \\",
        "  --dataset-scale full",
        "```",
        "",
        "",
        "### 6. Inspect the machine-readable records",
        "",
        "The output directory contains `results.jsonl` (one complete cell per model/suite/alignment), "
        "`protocol.json`, aligned depth maps, visibility z-buffers, and TSDF meshes.",
        "",
        "```bash",
        "wc -l /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl",
        "jq -c '{suite, scene, model, alignment}' /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl",
        "```",
        "",
        "### 7. Generate the Markdown tracker and PDF deck",
        "",
        "```bash",
        ".venv/bin/python scripts/benchmark/report_depth_models.py \\",
        "  /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl \\",
        "  --markdown docs/depth-model-evaluation.md --pdf docs/depth-model-evaluation.pdf",
        "```",
        "",
        "## Protocol",
        "",
        "- Alignment is fit in each model's native quantity: inverse z for DAv2 disparity, z for DA3/MoGe-3.",
        "- Scale and affine benchmark maps are **oracle** per-frame fits to the GT scan z-buffer; they are diagnostics, never deployable results.",
        "- DTU recall uses the official ground-plane GT cull and visibility z-buffer; its mesh score is Chamfer `(accuracy + completeness)/2` in millimetres, with the official observation mask on predictions.",
        "- TnT recall and mesh scoring use the official crop; F1 is reported at each scene's official threshold.",
        "- All meshes are fused through `threedgrut.geometry.tsdf.fuse_depth_frames`, shared with `extract_mesh_tsdf.py`. Source RGB is fused too, so the exported PLY files contain vertex colors; color does not affect the geometry metrics.",
        "- Mesh scoring retains two million samples by default. Exact cKDTree queries are reduced in 100,000-sample batches to bound host RAM; this does not alter the metric.",
        "",
        "## Evaluation coverage",
        "",
        "Each count is the actual number of input camera views used for one condition, recorded by the runner.",
        "",
        "| suite | scene | views |",
        "| --- | --- | ---: |",
    ]
    text += [f"| {suite} | {scene} | {count} |" for suite, scene, count in scene_view_rows(records)]
    text += [
        "",
        "## Evaluate a 3dgrut reconstruction checkpoint",
        "",
        "Use this path for a trained 3dgrut reconstruction rather than a monocular prior. It exports the renderer's "
        "Euclidean ray-distance maps, preserves the camera model actually rendered, and reuses the same TSDF fusion "
        "implementation as the depth-model benchmark.",
        "",
        "### 1. Reconstruct the scene",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config-name apps/colmap_3dgut.yaml \\",
        "  path=/mnt/data/nerf_datasets/dtu_dataset/dtu/scan24 \\",
        "  out_dir=runs experiment_name=scan24_3dgut",
        "# Checkpoint: runs/scan24_3dgut/<timestamp>/ckpt_last.pt",
        "```",
        "",
        "### 2. Export rendered depths and the world transform",
        "",
        "```bash",
        "export CKPT=runs/scan24_3dgut/<timestamp>/ckpt_last.pt",
        "export DEPTH_OUT=/tmp/scan24_3dgut_depth",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/export_depth_maps.py \\",
        '  --checkpoint "$CKPT" --out-dir "$DEPTH_OUT" --report-gt-agreement',
        "```",
        "",
        "The exporter writes `manifest.json` (ray depth plus fitted pinhole cameras), `alignment.npy` (dataset source "
        "world to rendered world), and `export_summary.json`. Do not score these maps as z-depth.",
        "",
        "### 3. Score depth recall — DTU",
        "",
        "The exported depths commonly live in normalized coordinates. Convert the desired millimetre tolerances through "
        "the exported source-to-render scale before passing them to the manifest-driven evaluator.",
        "",
        "```bash",
        "export DTU_EVAL=/mnt/data/nerf_datasets/dtu_dataset/dtu_eval",
        ".venv/bin/python - <<'PY'",
        "import numpy as np",
        "matrix = np.load('/tmp/scan24_3dgut_depth/alignment.npy')",
        "scale = np.linalg.norm(matrix[:3, :3], axis=1).mean()",
        "print('render-space taus for 0.5,1,2,5 mm:', ','.join(f'{v * scale:.9g}' for v in (.5, 1, 2, 5)))",
        "PY",
        "cd tools/depthrecall",
        "PYTHONPATH=$PWD ../../.venv/bin/python -m depthrecall \\",
        '  --manifest "$DEPTH_OUT/manifest.json" --alignment "$DEPTH_OUT/alignment.npy" \\',
        '  --ply "$DTU_EVAL/Points/stl/stl024_total.ply" \\',
        '  --dtu-plane "$DTU_EVAL/ObsMask/Plane24.mat" \\',
        '  --depth-convention ray --taus <render-space-taus> -o "$DEPTH_OUT/dtu_recall.json"',
        "cd ../..",
        "```",
        "",
        "### 4. Score depth recall — Tanks and Temples",
        "",
        "For this section, set `CKPT` and `DEPTH_OUT` to the Barn reconstruction/export rather than the DTU example above.",
        "",
        "First fit the official scan-to-render registration from the exported camera manifest, then rasterize a GT visibility "
        "manifest. The latter removes scan self-occlusion from the recall denominator.",
        "",
        "```bash",
        "export TNT=/mnt/data/nerf_datasets/tnt_dataset/tnt/Barn",
        "PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/align_tnt.py \\",
        '  --manifest "$DEPTH_OUT/manifest.json" --trans "$TNT/Barn_trans.txt" \\',
        '  --log "$TNT/Barn_COLMAP_SfM.log" --out "$DEPTH_OUT/scan_to_render.npy" \\',
        '  --inverse-out "$DEPTH_OUT/render_to_scan.npy" --report "$DEPTH_OUT/tnt_alignment.json"',
        "PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/rasterize_depth.py \\",
        '  --manifest "$DEPTH_OUT/manifest.json" --ply "$TNT/Barn.ply" \\',
        '  --alignment "$DEPTH_OUT/scan_to_render.npy" --crop-json "$TNT/Barn.json" \\',
        '  --out-dir "$DEPTH_OUT/tnt_visibility"',
        "cd tools/depthrecall",
        "PYTHONPATH=$PWD ../../.venv/bin/python -m depthrecall \\",
        '  --manifest "$DEPTH_OUT/manifest.json" --alignment "$DEPTH_OUT/scan_to_render.npy" \\',
        '  --ply "$TNT/Barn.ply" --crop-json "$TNT/Barn.json" \\',
        '  --visibility-manifest "$DEPTH_OUT/tnt_visibility/manifest.json" \\',
        '  --depth-convention ray --taus 0.01,0.02,0.05 -o "$DEPTH_OUT/tnt_recall.json"',
        "cd ../..",
        "```",
        "",
        "### 5. Extract and score the TSDF mesh",
        "",
        "Use a voxel size in the checkpoint's rendered units (for normalized DTU, start with `0.002`). Before standard "
        "DTU/TnT surface scoring, transform the predicted mesh into scan coordinates; the evaluator applies that transform "
        "before the official scan-space masks.",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/extract_mesh_tsdf.py \\",
        '  --checkpoint "$CKPT" --out "$DEPTH_OUT/mesh.ply" --voxel-size 0.002',
        ".venv/bin/python - <<'PY'",
        "import numpy as np",
        "np.save('/tmp/scan24_3dgut_depth/render_to_scan.npy', np.linalg.inv(np.load('/tmp/scan24_3dgut_depth/alignment.npy')))",
        "PY",
        "PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/evaluate_surface.py \\",
        '  --pred "$DEPTH_OUT/mesh.ply" --gt "$DTU_EVAL/Points/stl/stl024_total.ply" \\',
        '  --pred-alignment "$DEPTH_OUT/render_to_scan.npy" \\',
        '  --dtu-obsmask "$DTU_EVAL/ObsMask/ObsMask24_10.mat" \\',
        '  --dtu-plane "$DTU_EVAL/ObsMask/Plane24.mat" --taus 0.2 \\',
        '  --out "$DEPTH_OUT/dtu_surface.json"',
        "# TnT Barn mesh F1: use the render_to_scan.npy written by align_tnt.py",
        "PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/evaluate_surface.py \\",
        '  --pred "$DEPTH_OUT/mesh.ply" --gt "$TNT/Barn.ply" \\',
        '  --pred-alignment "$DEPTH_OUT/render_to_scan.npy" --crop-json "$TNT/Barn.json" \\',
        '  --taus 0.01 --out "$DEPTH_OUT/tnt_surface.json"',
        "```",
        "",
        "## OB3D depth accuracy",
        "",
    ]
    models, table = pivot(ob3d, lambda row: row["depth"]["abs_rel"])
    text += ["| model | raw abs-rel | scale abs-rel | affine abs-rel |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    text += [
        "",
        "## DTU",
        "",
        "Visible-scan recall curves use millimetre tolerances. Values are unweighted means over evaluated scenes; "
        "scale and affine are oracle scan-z-buffer fits.",
        "",
    ]
    text += recall_curve_table(dtu, normalize_taus=False, unit="mm")
    models, table = pivot(dtu, lambda row: row["surface"].get("overall"))
    text += ["", "DTU TSDF mesh Chamfer is `(accuracy + completeness) / 2` in millimetres; `—` is an empty mesh.", ""]
    text += ["| model | raw Chamfer | scale Chamfer | affine Chamfer |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    text += [
        "",
        "## Tanks and Temples",
        "",
        "Visible-scan recall curves are normalized by each scene's official tolerance (1×, 2×, 5×) before "
        "averaging across scenes.",
        "",
    ]
    text += recall_curve_table(tnt, normalize_taus=True, unit="×")
    models, table = pivot(tnt, lambda row: row["surface"].get("fscore", [None])[0])
    text += [
        "",
        "TnT mesh F1 is at each scene's official threshold; `—` denotes an empty mesh.",
        "",
    ]
    text += ["| model | raw F1 | scale F1 | affine F1 |", "| --- | ---: | ---: | ---: |"]
    text += [f"| {model} | {' | '.join(cells)} |" for model, cells in zip(models, table)]
    text += [
        "",
        "## Reading this run",
        "",
        "- Raw DAv2/DA3 are intentionally uncalibrated relative outputs; their absolute scores are not comparable to a metric-depth claim.",
        "- Per-frame scale/affine use a GT scan z-buffer and are explicitly oracle diagnostics, not deployable alignments.",
        "- Affine contains scale and cannot lose on its fitted, unweighted native-space least-squares objective. It can lose on AbsRel, threshold recall, or TSDF mesh metrics, which use different weighting, depth/ray conversion, validity, and nonlinear fusion operations.",
        "- DTU uses its ground-plane completeness cull and predicted-surface observation mask; TnT uses its official crop plus a GT visibility z-buffer to remove scan self-occlusion.",
        "- TSDF is the shared posed-depth path used by checkpoint extraction and this benchmark. Multi-view fusion improves coverage but does not make independently predicted monocular depths mutually consistent; mesh values are a pipeline and prior-quality diagnostic, not a competitive learned reconstruction result.",
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
            mean(
                [
                    record["depth"]["abs_rel"]
                    for record in ob3d
                    if record["model"] == model and record["alignment"] == alignment
                ]
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


def recall_curve_page(pdf: PdfPages, records: list[dict], suite: str) -> None:
    """Plot all tolerances, splitting oracle conditions into readable panels."""
    suite_records = rows_by_suite(records, suite)
    normalize_taus = suite == "tnt"
    taus, curves = recall_curve_data(suite_records, normalize_taus=normalize_taus)
    if not len(taus):
        return
    alignments = ["raw", "scale", "affine"]
    models = sorted({model for model, _ in curves})
    figure, axes = plt.subplots(1, len(alignments), figsize=(13.33, 5.5), sharey=True)
    for axis, alignment in zip(axes, alignments):
        for model in models:
            values = curves.get((model, alignment))
            if values is not None:
                axis.plot(taus, values, marker="o", linewidth=2.5, label=model)
        axis.set_title(alignment)
        axis.set_xlabel("official tolerance multiplier" if normalize_taus else "tolerance (mm)")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1)
        if not normalize_taus and len(taus) > 2 and taus[-1] / taus[0] >= 10:
            axis.set_xscale("log")
    axes[0].set_ylabel("visible-scan recall")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
    title = "DTU visible-depth recall curves" if suite == "dtu" else "Tanks and Temples visible-depth recall curves"
    figure.suptitle(title, fontsize=22, weight="bold", y=0.98)
    subtitle = (
        "GT ground-plane cull + per-view visibility z-buffer; higher is better"
        if suite == "dtu"
        else "Official crop + per-view visibility z-buffer; x-axis is normalized per scene; higher is better"
    )
    figure.text(0.5, 0.90, subtitle, ha="center", fontsize=11)
    figure.tight_layout(rect=(0, 0.10, 1, 0.84))
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def qualitative_page(pdf: PdfPages, result_dir: Path, dtu_root: Path, records: list[dict]) -> None:
    """Add a sample only when the optional local RGB/depth artifacts are present."""
    dtu = rows_by_suite(records, "dtu")
    models = sorted({record["model"] for record in dtu})
    if not models:
        return
    scene = dtu[0]["scene"]
    image_path = dtu_root / scene / "images" / "0000.png"
    depth_paths = [result_dir / "dtu" / scene / model / "affine" / "depths" / "0000.npy" for model in models]
    if not image_path.is_file() or not all(path.is_file() for path in depth_paths):
        return
    image = np.asarray(plt.imread(image_path))
    figure, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 7))
    if len(models) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    axes[0, 0].set_ylabel("RGB", fontsize=14)
    axes[1, 0].set_ylabel("affine ray depth", fontsize=14)
    for column, model in enumerate(models):
        axes[0, column].imshow(image)
        axes[0, column].set_title(model, fontsize=15)
        depth = np.load(depth_paths[column])
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
        "--note",
        default="Report generated from the supplied benchmark records; see the coverage table for actual views.",
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
        recall_curve_page(pdf, records, "dtu")
        tnt = rows_by_suite(records, "tnt")
        recall_curve_page(pdf, records, "tnt")
        models, table = pivot(tnt, lambda row: row["surface"].get("fscore", [None])[0])
        table_page(
            pdf,
            "Tanks and Temples mesh",
            ["raw F1", "scale F1", "affine F1"],
            models,
            table,
            suite_view_subtitle(tnt, "tnt", "Official scene-threshold F1"),
        )
        qualitative_page(pdf, args.results.parent, args.dtu_root, records)
    print(f"Wrote {args.markdown} and {args.pdf}")


if __name__ == "__main__":
    main()
