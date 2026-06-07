#!/bin/bash
# Bonsai-only NHT parity runner for 3dgrut vs the reference gsplat NHT codepath.

set -euo pipefail

MODE=${MODE:-all} # all, 3dgrut, gsplat
EXP_ID=${EXP_ID:-e00_reference}
GPU=${GPU:-0}
MAX_STEPS=${MAX_STEPS:-7000}
FEATURE_DIM=${FEATURE_DIM:-48}
CAP_MAX=${CAP_MAX:-1000000}
DATA_ROOT=${DATA_ROOT:-/mnt/gogn/data/nerf_datasets/nerf_360}
GSPLAT_REPO=${GSPLAT_REPO:-/mnt/dev/neural-harmonic-textures}
RESULT_ROOT=${RESULT_ROOT:-results/nht_gsplat_parity}
CONFIG=${CONFIG:-apps/colmap_3dgut_mcmc_nht}
ALLOW_EXISTING=${ALLOW_EXISTING:-0}
PYTHON=${PYTHON:-python}
GSPLAT_TORCH_CUDA_ARCH_LIST=${GSPLAT_TORCH_CUDA_ARCH_LIST:-8.9}
NHT_FEATURES_BWD_LOCAL_GRAD_CUDA=${NHT_FEATURES_BWD_LOCAL_GRAD_CUDA:-1}
THREEDGRUT_EXTRA_ARGS=("$@")

SCENE=bonsai
DATA_DIR="$DATA_ROOT/$SCENE"
FACTOR=2
EXP_DIR="$RESULT_ROOT/$EXP_ID"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Missing bonsai dataset: $DATA_DIR" >&2
    exit 1
fi

if [[ -d "$EXP_DIR" && "$ALLOW_EXISTING" != "1" ]]; then
    echo "Result directory already exists: $EXP_DIR" >&2
    echo "Use ALLOW_EXISTING=1 to append or resume." >&2
    exit 1
fi

mkdir -p "$EXP_DIR"
EXP_DIR_ABS=$(cd "$EXP_DIR" && pwd)
export NHT_FEATURES_BWD_LOCAL_GRAD_CUDA

write_command() {
    local output=$1
    shift
    printf '%q ' "$@" > "$output"
    printf '\n' >> "$output"
}

run_3dgrut() {
    local result_dir="$EXP_DIR_ABS/3dgrut_current"
    local log_dir="$result_dir/logs"
    mkdir -p "$log_dir"
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$RESULT_ROOT/.torch_extensions_3dgrut"}

    local train_cmd=(
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" train.py
        --config-name "$CONFIG"
        "path=$DATA_DIR"
        "out_dir=$result_dir"
        "experiment_name=$SCENE"
        "dataset.downsample_factor=$FACTOR"
        "dataset.load_exif=true"
        "n_iterations=$MAX_STEPS"
        "test_last=false"
        "val_frequency=999999"
        "strategy.add.max_n_gaussians=$CAP_MAX"
        "model.nht_features.dim=$FEATURE_DIM"
        "model.nht_decoder.scheduler.max_steps=$MAX_STEPS"
        "scheduler.positions.max_steps=$MAX_STEPS"
        "scheduler.features.max_steps=$MAX_STEPS"
        "checkpoint.iterations=[$MAX_STEPS]"
        "use_wandb=false"
        "with_gui=false"
        "with_viser_gui=false"
        "${THREEDGRUT_EXTRA_ARGS[@]}"
    )
    write_command "$log_dir/train.command.txt" "${train_cmd[@]}"

    echo ">>> 3dgrut current: $EXP_ID"
    nvidia-smi > "$log_dir/train.log" 2>&1 || true
    "${train_cmd[@]}" >> "$log_dir/train.log" 2>&1

    local ckpt
    ckpt=$(find "$result_dir/$SCENE" -name ckpt_last.pt 2>/dev/null | sort | tail -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "No 3dgrut checkpoint found under $result_dir/$SCENE" >&2
        exit 1
    fi

    local render_cmd=(
        env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" render.py
        --checkpoint "$ckpt"
        --path "$DATA_DIR"
        --out-dir "$result_dir/$SCENE/eval"
    )
    write_command "$log_dir/render.command.txt" "${render_cmd[@]}"
    "${render_cmd[@]}" > "$log_dir/render.log" 2>&1
}

run_gsplat() {
    local result_dir="$EXP_DIR_ABS/gsplat_reference"
    local log_dir="$result_dir/logs"
    local trainer="$GSPLAT_REPO/gsplat/examples/simple_trainer_nht.py"
    mkdir -p "$log_dir"

    if [[ ! -f "$trainer" ]]; then
        echo "Missing gsplat trainer: $trainer" >&2
        exit 1
    fi

    local cuda_home="${CUDA_HOME:-${CONDA_PREFIX:-}}"
    local cuda_include=""
    local cuda_lib=""
    if [[ -n "$cuda_home" ]]; then
        if [[ -f "$cuda_home/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
            cuda_include="$cuda_home/targets/x86_64-linux/include"
            cuda_lib="$cuda_home/targets/x86_64-linux/lib"
        elif [[ -f "$cuda_home/include/cuda_runtime.h" ]]; then
            cuda_include="$cuda_home/include"
            cuda_lib="$cuda_home/lib64"
        fi
    fi
    if [[ -z "$cuda_include" && -f /usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h ]]; then
        cuda_home="/usr/local/cuda"
        cuda_include="/usr/local/cuda/targets/x86_64-linux/include"
        cuda_lib="/usr/local/cuda/targets/x86_64-linux/lib"
    fi
    if [[ -z "$cuda_include" ]]; then
        echo "Could not find cuda_runtime.h. Set CUDA_HOME or activate an environment with CUDA headers." >&2
        exit 1
    fi

    local train_cmd=(
        env
        CUDA_VISIBLE_DEVICES="$GPU"
        "CUDA_HOME=$cuda_home"
        "CPATH=$cuda_include:${CPATH:-}"
        "CPLUS_INCLUDE_PATH=$cuda_include:${CPLUS_INCLUDE_PATH:-}"
        "LIBRARY_PATH=$cuda_lib:${LIBRARY_PATH:-}"
        "LD_LIBRARY_PATH=$cuda_lib:${LD_LIBRARY_PATH:-}"
        "TORCH_CUDA_ARCH_LIST=$GSPLAT_TORCH_CUDA_ARCH_LIST"
        "PYTHONPATH=$GSPLAT_REPO/gsplat:$GSPLAT_REPO:${PYTHONPATH:-}"
        "$PYTHON" "$trainer" default
        --disable_viewer
        --disable_video
        --native_images_factor
        --data_dir "$DATA_DIR/"
        --result_dir "$result_dir/"
        --data_factor "$FACTOR"
        --max_steps "$MAX_STEPS"
        --eval_steps "$MAX_STEPS"
        --save_steps "$MAX_STEPS"
        --ply_steps "$MAX_STEPS"
        --strategy.cap-max "$CAP_MAX"
        --deferred_opt_feature_dim "$FEATURE_DIM"
        --ssim_lambda 0.2
        --opacity_reg 0.02
        --scale_reg 0.005
        --deferred_features_lr 0.015
        --deferred_mlp_lr 0.00068
        --deferred_features_lr_decay_final 0.1
        --deferred_mlp_lr_decay_final 0.1
        --deferred_lr_scheduler cosine
        --deferred_mlp_hidden_dim 128
        --deferred_mlp_num_layers 3
        --deferred_opt_view_encoding_type sh
        --deferred_opt_sh_degree 3
        --deferred_opt_sh_scale 3.0
        --deferred_mlp_ema_decay 0.95
        --deferred_mlp_ema_start_step 0
    )
    write_command "$log_dir/train.command.txt" "${train_cmd[@]}"

    echo ">>> gsplat reference: $EXP_ID"
    (
        cd "$GSPLAT_REPO"
        nvidia-smi > "$log_dir/train.log" 2>&1 || true
        "${train_cmd[@]}" >> "$log_dir/train.log" 2>&1
    )
}

write_summary() {
    EXP_DIR="$EXP_DIR_ABS" EXP_ID="$EXP_ID" "$PYTHON" - <<'PY'
import glob
import json
import os
from pathlib import Path

exp_dir = Path(os.environ["EXP_DIR"])
summary = {"experiment": os.environ["EXP_ID"], "runs": {}}

def latest(paths):
    return max(paths, key=os.path.getmtime) if paths else None

three_metrics = latest(glob.glob(str(exp_dir / "3dgrut_current" / "bonsai" / "eval" / "**" / "metrics.json"), recursive=True))
if three_metrics:
    with open(three_metrics) as f:
        metrics = json.load(f)
    summary["runs"]["3dgrut_current"] = {
        "metrics_path": three_metrics,
        "psnr": metrics.get("mean_psnr"),
        "ssim": metrics.get("mean_ssim"),
        "lpips": metrics.get("mean_lpips"),
        "frame_time_ms": metrics.get("mean_inference_time_ms"),
    }

gsplat_metrics = latest(
    [
        p for p in glob.glob(str(exp_dir / "gsplat_reference" / "stats" / "val_step*.json"))
        if "per_image" not in p
    ]
)
if gsplat_metrics:
    with open(gsplat_metrics) as f:
        metrics = json.load(f)
    summary["runs"]["gsplat_reference"] = {
        "metrics_path": gsplat_metrics,
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"),
        "frame_time_ms": (metrics.get("ellipse_time") or 0) * 1000 if metrics.get("ellipse_time") is not None else None,
        "num_GS": metrics.get("num_GS"),
    }

out = exp_dir / "summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f"Wrote {out}")
print("| Run | PSNR | SSIM | LPIPS | Frame Time ms |")
print("| --- | ---: | ---: | ---: | ---: |")
for name, run in summary["runs"].items():
    def fmt(key, digits):
        value = run.get(key)
        return "TBD" if value is None else f"{value:.{digits}f}"
    print(f"| {name} | {fmt('psnr', 4)} | {fmt('ssim', 4)} | {fmt('lpips', 4)} | {fmt('frame_time_ms', 2)} |")
PY
}

case "$MODE" in
    all)
        run_3dgrut
        run_gsplat
        ;;
    3dgrut)
        run_3dgrut
        ;;
    gsplat)
        run_gsplat
        ;;
    *)
        echo "Unknown MODE=$MODE. Use all, 3dgrut, or gsplat." >&2
        exit 1
        ;;
esac

write_summary
