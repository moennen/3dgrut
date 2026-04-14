#!/bin/bash
# NHT validation suite — runs 3 training experiments + renders and writes report
set -euo pipefail

REPO=/home/nicolasm/dev/3dgrut-laplace-sbx
cd "$REPO"

DATA="${1:-$HOME/dev/data/nerf_synthetic/drums}"
OUTDIR="runs/validate_nht"
REPORT="$HOME/validate_nht_report.md"
PY="micromamba run -n 3dgrut-nht python"
COMMON="path=$DATA out_dir=$OUTDIR strategy.add.max_n_gaussians=150000 strategy.relocate.end_iteration=15000 strategy.add.end_iteration=15000"

mkdir -p "$OUTDIR"

declare -A TRAIN_TIME
declare -A RENDER_TIME

run_experiment() {
    local name=$1
    shift
    local extra_args="$*"

    echo ""
    echo "============================================================"
    echo "TRAINING: $name"
    echo "============================================================"

    local log="$OUTDIR/${name}_train.log"
    local t0 t1

    t0=$(date +%s)
    if ! { $PY train.py --config-name=apps/nerf_synthetic_3dgrt_mcmc_nht \
        $COMMON \
        experiment_name="$name" \
        $extra_args ; } 2>&1 | tee "$log" ; then
        echo "ERROR: training $name failed. Last 60 lines of log:"
        tail -n 60 "$log"
    fi
    t1=$(date +%s)
    TRAIN_TIME[$name]=$(( t1 - t0 ))

    echo "Training $name done in ${TRAIN_TIME[$name]}s"
}

run_render() {
    local name=$1

    echo ""
    echo "============================================================"
    echo "RENDERING: $name"
    echo "============================================================"

    # Find the checkpoint — there may be a timestamped subdir under the experiment dir
    local ckpt
    ckpt=$(find "$OUTDIR/$name" -name "ckpt_last.pt" 2>/dev/null | head -1)
    if [[ -z "$ckpt" ]]; then
        echo "ERROR: no ckpt_last.pt found for $name under $OUTDIR/$name"
        RENDER_TIME[$name]="N/A"
        return
    fi
    echo "Using checkpoint: $ckpt"

    local render_out="$OUTDIR/$name/render"
    local log="$OUTDIR/${name}_render.log"
    local t0 t1

    t0=$(date +%s)
    if ! { $PY render.py --checkpoint "$ckpt" --out-dir "$render_out" ; } 2>&1 | tee "$log" ; then
        echo "ERROR: render $name failed. Last 60 lines of log:"
        tail -n 60 "$log"
    fi
    t1=$(date +%s)
    RENDER_TIME[$name]=$(( t1 - t0 ))

    echo "Render $name done in ${RENDER_TIME[$name]}s"
}

fmt_seconds() {
    local s=$1
    if [[ "$s" == "N/A" ]]; then echo "N/A"; return; fi
    printf "%dm%02ds" $(( s / 60 )) $(( s % 60 ))
}

read_metric() {
    local name=$1 metric=$2
    # Render creates <out-dir>/<experiment_name>/<run_name>/metrics.json
    local mfile
    mfile=$(find "$OUTDIR/$name/render" -name "metrics.json" 2>/dev/null | head -1)
    if [[ -z "$mfile" ]]; then echo "N/A"; return; fi
    python3 -c "import json; d=json.load(open('$mfile')); print(f\"{d['$metric']:.4f}\")" 2>/dev/null || echo "N/A"
}

# ── Experiment 1: SH reference (baseline, no NHT) ──────────────────────────
run_experiment "sh_reference" "model.feature_type=sh"

# ── Experiment 2: NHT reference (default pipeline) ─────────────────────────
run_experiment "nht_reference"

# ── Experiment 3: NHT referenceSlang ───────────────────────────────────────
run_experiment "nht_referenceSlang" \
    "render.pipeline_type=referenceSlang" \
    "render.backward_pipeline_type=referenceSlangBwd"

# ── Renders ─────────────────────────────────────────────────────────────────
run_render "sh_reference"
run_render "nht_reference"
run_render "nht_referenceSlang"

# ── Report ──────────────────────────────────────────────────────────────────
EXPERIMENTS=("sh_reference" "nht_reference" "nht_referenceSlang")

{
    echo "# NHT Validation Report"
    echo ""
    echo "| Experiment | Train time | Render time | PSNR | SSIM | LPIPS |"
    echo "|---|---|---|---|---|---|"
    for name in "${EXPERIMENTS[@]}"; do
        tt=$(fmt_seconds "${TRAIN_TIME[$name]:-N/A}")
        rt=$(fmt_seconds "${RENDER_TIME[$name]:-N/A}")
        psnr=$(read_metric "$name" "mean_psnr")
        ssim=$(read_metric "$name" "mean_ssim")
        lpips=$(read_metric "$name" "mean_lpips")
        echo "| $name | $tt | $rt | $psnr | $ssim | $lpips |"
    done
} > "$REPORT"

echo ""
echo "============================================================"
echo "REPORT"
echo "============================================================"
cat "$REPORT"
