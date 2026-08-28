#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the geometry-supervision report. Figures are regenerated from the ablation run records
# first, so a plot in the deck cannot drift from the measurements behind it.
#
# Usage:
#   scripts/report/build_report.sh [--render]
#
# `--render` also re-renders the qualitative panels from checkpoints, which needs a GPU and the
# run directories under $ABL_ROOT. Without it, the existing PNGs under $FIG_ROOT are reused.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
ABL_ROOT="${ABL_ROOT:-/tmp/abl_pd}"
# The DA3-vs-DAv2 prior sweep and its offline diagnostic. Separate from ABL_ROOT: that sweep
# re-ran the trisurfel baseline with three seeds, and pooling those into the earlier experiments
# would quietly move numbers the deck has already published.
DA3_ROOT="${DA3_ROOT:-/mnt/oss/da3cmp}"
# Experiment 8's sweep, separate for the same reason: its own three baseline seeds.
L1_ROOT="${L1_ROOT:-/mnt/oss/pdl1cmp}"
PRIOR_DIAG="${PRIOR_DIAG:-/mnt/oss/da3diag/json}"
FIG_ROOT="${FIG_ROOT:-/tmp/report_fig}"
BUILD_DIR="${BUILD_DIR:-/tmp/report_build}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/nerf_datasets/ob3d/OB3D_colmap}"
SCENES=(sponza lone-monk emerald-square)

if [[ "${1:-}" == "--render" ]]; then
  for scene in "${SCENES[@]}"; do
    PATH="$REPO_ROOT/.venv/bin:$PATH" "$PYTHON" "$REPO_ROOT/scripts/report/render_all.py" \
      --scene "$scene" --search-root "$ABL_ROOT" --dataset-root "$DATASET_ROOT" \
      --out-root "$FIG_ROOT" --frames 4
  done
fi

# Every results file, not a hand-picked subset: repeats of a (variant, scene) cell across files
# are the separate seeds, and the error bars come from having all of them.
mapfile -t RESULTS < <(find "$ABL_ROOT" -name results.jsonl | sort)
if [[ ${#RESULTS[@]} -eq 0 ]]; then
  echo "no results.jsonl under $ABL_ROOT" >&2
  exit 1
fi
echo "figures from ${#RESULTS[@]} results file(s)"

# Experiment 7's sources are optional: without them `make_figures.py` skips those figures, and
# the build fails at the \input in the .tex rather than silently emitting a deck with stale
# Experiment 7 numbers.
EXTRA=()
if [[ -d "$DA3_ROOT" ]]; then
  mapfile -t DA3_RESULTS < <(find "$DA3_ROOT" -name results.jsonl | sort)
  if [[ ${#DA3_RESULTS[@]} -gt 0 ]]; then
    EXTRA+=(--da3-results "${DA3_RESULTS[@]}")
    echo "  plus ${#DA3_RESULTS[@]} DA3 results file(s) from $DA3_ROOT"
  fi
fi
if [[ -d "$L1_ROOT" ]]; then
  mapfile -t L1_RESULTS < <(find "$L1_ROOT" -name results.jsonl | sort)
  if [[ ${#L1_RESULTS[@]} -gt 0 ]]; then
    EXTRA+=(--l1-results "${L1_RESULTS[@]}")
    echo "  plus ${#L1_RESULTS[@]} regression-sweep results file(s) from $L1_ROOT"
  fi
fi
if [[ -d "$PRIOR_DIAG" ]]; then
  EXTRA+=(--prior-diagnostic-dir "$PRIOR_DIAG")
  echo "  plus prior diagnostics from $PRIOR_DIAG"
fi

"$PYTHON" "$REPO_ROOT/scripts/report/make_figures.py" "${RESULTS[@]}" \
  --out-dir "$FIG_ROOT/plots" --fig-root "$FIG_ROOT" "${EXTRA[@]+"${EXTRA[@]}"}"

mkdir -p "$BUILD_DIR"
cp "$REPO_ROOT/scripts/report/geometry_report.tex" "$BUILD_DIR/"
# pdflatex resolves \includegraphics relative to the build directory, so the assets are linked
# in rather than referenced by absolute path -- keeps the .tex portable.
for scene in "${SCENES[@]}" plots; do
  rm -rf "$BUILD_DIR/$scene"
  cp -r "$FIG_ROOT/$scene" "$BUILD_DIR/$scene"
done

cd "$BUILD_DIR"
# Twice, for \inserttotalframenumber in the footer.
for _ in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error geometry_report.tex >/dev/null
done
cp "$BUILD_DIR/geometry_report.pdf" "$REPO_ROOT/geometry-supervision-report.pdf"
echo "wrote $REPO_ROOT/geometry-supervision-report.pdf ($(pdfinfo geometry_report.pdf | awk '/^Pages/{print $2}') pages)"
