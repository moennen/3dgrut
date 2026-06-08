#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Full Mip-NeRF 360 benchmark for 3DGUT MCMC SH baseline and 3DGUT MCMC NHT.
#
# Examples:
#   DATA_ROOT=$HOME/data/nerf_datasets/nerf_360 GPU=0 bash benchmark/mipnerf360_3dgut_mcmc_full.sh
#   PYTHON="uv run python" DATA_ROOT=$HOME/data/nerf_datasets/nerf_360 GPU=0 bash benchmark/mipnerf360_3dgut_mcmc_full.sh
#   ALLOW_EXISTING=1 RUN_TRAIN=0 RUN_RENDER=1 DATA_ROOT=$HOME/data/nerf_datasets/nerf_360 bash benchmark/mipnerf360_3dgut_mcmc_full.sh
#
# Extra Hydra overrides may be appended after the script name.

set -euo pipefail

is_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

split_words() {
    local -n out_ref=$1
    local value=$2
    # shellcheck disable=SC2206
    out_ref=($value)
}

PYTHON=${PYTHON:-python}
split_words PYTHON_CMD "$PYTHON"

DATA_ROOT=${DATA_ROOT:-}
if [[ -z "$DATA_ROOT" ]]; then
    if [[ -d "$HOME/data/nerf_datasets/nerf_360" ]]; then
        DATA_ROOT="$HOME/data/nerf_datasets/nerf_360"
    else
        DATA_ROOT="/mnt/gogn/data/nerf_datasets/nerf_360"
    fi
fi

RESULT_ROOT=${RESULT_ROOT:-"results/mipnerf360_3dgut_mcmc_full"}
GPU=${GPU:-0}
BASELINE_GPU=${BASELINE_GPU:-$GPU}
NHT_GPU=${NHT_GPU:-$GPU}
MAX_STEPS=${MAX_STEPS:-30000}
CAP_MAX=${CAP_MAX:-1000000}
FEATURE_DIM=${FEATURE_DIM:-48}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_RENDER=${RUN_RENDER:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
ALLOW_EXISTING=${ALLOW_EXISTING:-0}
REQUIRE_ALL_SCENES=${REQUIRE_ALL_SCENES:-1}
RUN_VARIANTS=${RUN_VARIANTS:-"baseline nht"}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$RESULT_ROOT/.torch_extensions"}
COMMON_EXTRA_ARGS=("$@")

M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("bicycle" "flowers" "garden" "stump" "treehill")
SCENE_LIST=${SCENE_LIST:-"${M360_INDOOR[*]} ${M360_OUTDOOR[*]}"}

VARIANT_KEYS=("baseline" "nht")
VARIANT_NAMES=("3dgut_mcmc" "3dgut_nht_mcmc")
VARIANT_CONFIGS=("apps/colmap_3dgut_mcmc" "apps/colmap_3dgut_mcmc_nht")

if [[ -d "$RESULT_ROOT" ]] && ! is_enabled "$ALLOW_EXISTING" && ! is_enabled "$SKIP_EXISTING"; then
    echo "Result root already exists: $RESULT_ROOT" >&2
    echo "Use ALLOW_EXISTING=1 to append or SKIP_EXISTING=1 to resume partial results." >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
export TORCH_EXTENSIONS_DIR

get_factor() {
    case "$1" in
        bonsai|counter|kitchen|room) echo 2 ;;
        *) echo 4 ;;
    esac
}

get_data_dir() {
    local scene=$1
    for p in \
        "$DATA_ROOT/$scene" \
        "$DATA_ROOT/mipnerf360/$scene" \
        "$DATA_ROOT/360_v2/$scene" \
        "data/mipnerf360/$scene"; do
        if [[ -d "$p" ]]; then
            echo "$p"
            return
        fi
    done
    echo ""
}

latest_ckpt() {
    local scene_dir=$1
    [[ -d "$scene_dir" ]] || return 0
    find "$scene_dir" -name ckpt_last.pt 2>/dev/null | sort | tail -n 1
}

latest_metrics() {
    local eval_dir=$1
    [[ -d "$eval_dir" ]] || return 0
    find "$eval_dir" -name metrics.json 2>/dev/null | sort | tail -n 1
}

variant_enabled() {
    local key=$1
    for enabled in $RUN_VARIANTS; do
        [[ "$enabled" == "$key" ]] && return 0
    done
    return 1
}

variant_gpu() {
    case "$1" in
        baseline) echo "$BASELINE_GPU" ;;
        nht) echo "$NHT_GPU" ;;
        *) echo "$GPU" ;;
    esac
}

write_command() {
    local output=$1
    shift
    printf "%q " "$@" > "$output"
    printf "\n" >> "$output"
}

require_scene_data() {
    local scene=$1
    local data_dir=$2
    local factor=$3

    if [[ -z "$data_dir" ]]; then
        echo "WARNING: data not found for $scene under $DATA_ROOT" >&2
        is_enabled "$REQUIRE_ALL_SCENES" && exit 1
        return 1
    fi

    local image_dir="$data_dir/images_$factor"
    if [[ ! -d "$image_dir" ]]; then
        echo "WARNING: expected image folder not found for $scene: $image_dir" >&2
        is_enabled "$REQUIRE_ALL_SCENES" && exit 1
        return 1
    fi

    if [[ ! -d "$data_dir/sparse/0" ]]; then
        echo "WARNING: expected COLMAP sparse folder not found for $scene: $data_dir/sparse/0" >&2
        is_enabled "$REQUIRE_ALL_SCENES" && exit 1
        return 1
    fi

    return 0
}

write_summary() {
    RESULT_ROOT="$RESULT_ROOT" \
    DATA_ROOT="$DATA_ROOT" \
    SCENE_LIST="$SCENE_LIST" \
    RUN_VARIANTS="$RUN_VARIANTS" \
    VARIANT_KEYS="${VARIANT_KEYS[*]}" \
    VARIANT_NAMES="${VARIANT_NAMES[*]}" \
    VARIANT_CONFIGS="${VARIANT_CONFIGS[*]}" \
    "${PYTHON_CMD[@]}" - <<'PY'
import glob
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
scene_order = os.environ["SCENE_LIST"].split()
enabled_keys = set(os.environ["RUN_VARIANTS"].split())
variant_entries = [
    (key, name, config)
    for key, name, config in zip(
        os.environ["VARIANT_KEYS"].split(),
        os.environ["VARIANT_NAMES"].split(),
        os.environ["VARIANT_CONFIGS"].split(),
    )
    if key in enabled_keys
]
variant_names = [name for _, name, _ in variant_entries]
indoor = {"bonsai", "counter", "kitchen", "room"}
metric_keys = [
    "mean_psnr",
    "mean_ssim",
    "mean_lpips",
    "mean_cc_psnr",
    "mean_cc_ssim",
    "mean_cc_lpips",
    "mean_inference_time_ms",
]

def latest(pattern):
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))

def read_float(path):
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

def fmt(value, digits):
    return "TBD" if value is None else f"{value:.{digits}f}"

combined = {
    "benchmark": "mipnerf360",
    "method": "3dgut_mcmc_full",
    "data_root": os.environ["DATA_ROOT"],
    "scene_order": scene_order,
    "variants": {},
}

for _, variant, config in variant_entries:
    variant_dir = root / variant
    scenes = {}
    for scene in scene_order:
        metrics_path = latest(str(variant_dir / scene / "eval" / "**" / "metrics.json"))
        metrics = {}
        if metrics_path:
            with open(metrics_path) as f:
                metrics = json.load(f)
        train_seconds = read_float(variant_dir / f"train_{scene}_seconds.txt")
        frame_ms = metrics.get("mean_inference_time_ms")
        fps = 1000.0 / frame_ms if frame_ms and frame_ms > 0 else None
        scenes[scene] = {
            "factor": 2 if scene in indoor else 4,
            "metrics_path": metrics_path,
            "train_seconds": train_seconds,
            "fps": fps,
            **metrics,
        }

    averages = {key: mean([scenes[s].get(key) for s in scene_order]) for key in metric_keys}
    averages["train_seconds"] = mean([scenes[s].get("train_seconds") for s in scene_order])
    averages["total_train_seconds"] = sum(
        v for v in (scenes[s].get("train_seconds") for s in scene_order) if v is not None
    )
    averages["fps"] = mean([scenes[s].get("fps") for s in scene_order])

    variant_summary = {
        "name": variant,
        "config": config,
        "result_dir": str(variant_dir),
        "scenes": scenes,
        "averages": averages,
    }
    combined["variants"][variant] = variant_summary
    variant_dir.mkdir(parents=True, exist_ok=True)
    with open(variant_dir / "summary.json", "w") as f:
        json.dump(variant_summary, f, indent=2, sort_keys=True)

with open(root / "summary.json", "w") as f:
    json.dump(combined, f, indent=2, sort_keys=True)

lines = []
lines.append("# Mip-NeRF 360 3DGUT MCMC Full Benchmark")
lines.append("")
lines.append("## Averages")
lines.append("")
lines.append("| Variant | Split | PSNR | SSIM | LPIPS | Time ms/frame | FPS | Train h |")
lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

def split_average(scenes, subset):
    selected = [scenes[s] for s in scene_order if s in subset and scenes[s].get("metrics_path")]
    return {
        key: mean([row.get(key) for row in selected])
        for key in metric_keys
    } | {
        "fps": mean([row.get("fps") for row in selected]),
        "train_seconds": mean([row.get("train_seconds") for row in selected]),
    }

for variant in variant_names:
    entry = combined["variants"][variant]
    splits = [
        ("M360-In", indoor),
        ("M360-Out", set(scene_order) - indoor),
        ("M360", set(scene_order)),
    ]
    for split_name, subset in splits:
        avg = split_average(entry["scenes"], subset)
        train_h = None if avg["train_seconds"] is None else avg["train_seconds"] / 3600.0
        lines.append(
            f"| {variant} | {split_name} | {fmt(avg['mean_psnr'], 3)} | "
            f"{fmt(avg['mean_ssim'], 4)} | {fmt(avg['mean_lpips'], 4)} | "
            f"{fmt(avg['mean_inference_time_ms'], 2)} | {fmt(avg['fps'], 1)} | {fmt(train_h, 2)} |"
        )

lines.append("")
lines.append("## Per Scene")
lines.append("")
lines.append("| Variant | Scene | Factor | PSNR | SSIM | LPIPS | Time ms/frame | FPS | Train h | Metrics |")
lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for variant in variant_names:
    scenes = combined["variants"][variant]["scenes"]
    for scene in scene_order:
        row = scenes[scene]
        train_h = None if row.get("train_seconds") is None else row["train_seconds"] / 3600.0
        metrics_path = row.get("metrics_path")
        metrics_rel = "TBD" if metrics_path is None else str(Path(metrics_path).relative_to(root))
        lines.append(
            f"| {variant} | {scene} | {row['factor']} | {fmt(row.get('mean_psnr'), 3)} | "
            f"{fmt(row.get('mean_ssim'), 4)} | {fmt(row.get('mean_lpips'), 4)} | "
            f"{fmt(row.get('mean_inference_time_ms'), 2)} | {fmt(row.get('fps'), 1)} | "
            f"{fmt(train_h, 2)} | `{metrics_rel}` |"
        )

summary_md = root / "summary.md"
summary_md.write_text("\n".join(lines) + "\n")

print(f"Wrote {root / 'summary.json'}")
print(f"Wrote {summary_md}")
print("| Variant | PSNR | SSIM | LPIPS | Time ms/frame | FPS |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for variant in variant_names:
    avg = combined["variants"][variant]["averages"]
    print(
        f"| {variant} | {fmt(avg.get('mean_psnr'), 3)} | {fmt(avg.get('mean_ssim'), 4)} | "
        f"{fmt(avg.get('mean_lpips'), 4)} | {fmt(avg.get('mean_inference_time_ms'), 2)} | "
        f"{fmt(avg.get('fps'), 1)} |"
    )
PY
}

build_common_overrides() {
    local scene=$1
    local data_dir=$2
    local factor=$3
    COMMON_OVERRIDES=(
        "path=$data_dir"
        "dataset.downsample_factor=$factor"
        "dataset.load_exif=true"
        "dataset.normalize_world_space=true"
        "initialization.use_observation_points=false"
        "n_iterations=$MAX_STEPS"
        "test_last=false"
        "val_frequency=999999"
        "strategy.add.max_n_gaussians=$CAP_MAX"
        "scheduler.positions.max_steps=$MAX_STEPS"
        "checkpoint.iterations=[$MAX_STEPS]"
        "use_wandb=false"
        "with_gui=false"
        "with_viser_gui=false"
    )
}

build_variant_overrides() {
    local key=$1
    VARIANT_OVERRIDES=()
    if [[ "$key" == "nht" ]]; then
        VARIANT_OVERRIDES=(
            "model.nht_features.dim=$FEATURE_DIM"
            "model.nht_decoder.scheduler.max_steps=$MAX_STEPS"
            "scheduler.features.max_steps=$MAX_STEPS"
            "render.particle_feature_half=true"
            "render.feature_output_half=true"
        )
    fi
}

run_variant() {
    local key=$1
    local name=$2
    local config=$3
    local gpu
    gpu=$(variant_gpu "$key")

    local variant_dir="$RESULT_ROOT/$name"
    mkdir -p "$variant_dir"

    echo "============================================================"
    echo "Variant: $name"
    echo "Config:  $config"
    echo "GPU:     $gpu"
    echo "Output:  $variant_dir"
    echo "============================================================"

    for scene in $SCENE_LIST; do
        local data_dir factor
        data_dir=$(get_data_dir "$scene")
        factor=$(get_factor "$scene")

        if ! require_scene_data "$scene" "$data_dir" "$factor"; then
            continue
        fi

        echo
        echo ">>> $name / $scene (factor=$factor) <<<"

        if is_enabled "$RUN_TRAIN"; then
            local ckpt
            ckpt=$(latest_ckpt "$variant_dir/$scene")
            if is_enabled "$SKIP_EXISTING" && [[ -n "$ckpt" ]]; then
                echo "  [skip train] existing checkpoint: $ckpt"
            else
                build_common_overrides "$scene" "$data_dir" "$factor"
                build_variant_overrides "$key"

                local train_log="$variant_dir/train_$scene.log"
                local command_file="$variant_dir/train_$scene.command.txt"
                local start_time
                local train_cmd=(
                    env CUDA_VISIBLE_DEVICES="$gpu"
                    "${PYTHON_CMD[@]}" train.py
                    --config-name "$config"
                    "out_dir=$variant_dir"
                    "experiment_name=$scene"
                    "${COMMON_OVERRIDES[@]}"
                    "${VARIANT_OVERRIDES[@]}"
                    "${COMMON_EXTRA_ARGS[@]}"
                )

                write_command "$command_file" "${train_cmd[@]}"
                start_time=$("${PYTHON_CMD[@]}" -c 'import time; print(time.time())')
                nvidia-smi > "$train_log" 2>&1 || true
                "${train_cmd[@]}" >> "$train_log" 2>&1
                "${PYTHON_CMD[@]}" - "$start_time" "$variant_dir/train_${scene}_seconds.txt" <<'PY'
import sys
import time
from pathlib import Path

Path(sys.argv[2]).write_text(f"{time.time() - float(sys.argv[1]):.6f}\n")
PY
            fi
        fi

        if is_enabled "$RUN_RENDER"; then
            local eval_dir="$variant_dir/$scene/eval"
            local metrics
            metrics=$(latest_metrics "$eval_dir")
            if is_enabled "$SKIP_EXISTING" && [[ -n "$metrics" ]]; then
                echo "  [skip render] existing metrics: $metrics"
            else
                local ckpt
                ckpt=$(latest_ckpt "$variant_dir/$scene")
                if [[ -z "$ckpt" ]]; then
                    echo "WARNING: no ckpt_last.pt found for $name / $scene; skipping render" >&2
                    continue
                fi

                local render_log="$variant_dir/render_$scene.log"
                local command_file="$variant_dir/render_$scene.command.txt"
                local render_cmd=(
                    env CUDA_VISIBLE_DEVICES="$gpu"
                    "${PYTHON_CMD[@]}" render.py
                    --checkpoint "$ckpt"
                    --path "$data_dir"
                    --out-dir "$eval_dir"
                )

                write_command "$command_file" "${render_cmd[@]}"
                "${render_cmd[@]}" > "$render_log" 2>&1
            fi
        fi

        write_summary
    done
}

echo "Mip-NeRF 360 3DGUT MCMC full benchmark"
echo "  Data:          $DATA_ROOT"
echo "  Result:        $RESULT_ROOT"
echo "  Python:        ${PYTHON_CMD[*]}"
echo "  Variants:      $RUN_VARIANTS"
echo "  Baseline GPU:  $BASELINE_GPU"
echo "  NHT GPU:       $NHT_GPU"
echo "  Scenes:        $SCENE_LIST"
echo "  Steps:         $MAX_STEPS"
echo "  Cap:           $CAP_MAX"
echo "  NHT features:  $FEATURE_DIM"
echo "  Cache:         $TORCH_EXTENSIONS_DIR"
echo "  Run train:     $RUN_TRAIN"
echo "  Run render:    $RUN_RENDER"
echo "  Extra args:    ${COMMON_EXTRA_ARGS[*]:-<none>}"

for i in "${!VARIANT_KEYS[@]}"; do
    key=${VARIANT_KEYS[$i]}
    if variant_enabled "$key"; then
        run_variant "$key" "${VARIANT_NAMES[$i]}" "${VARIANT_CONFIGS[$i]}"
    fi
done

write_summary
