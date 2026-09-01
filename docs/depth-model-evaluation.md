# Depth-model evaluation

Smoke protocol: one view per scene at a 160 px maximum side; meshes use 1,000 samples.

## Step-by-step workflow

`evaluate_depth_models.py` is the single runner. It evaluates DAv2 (`dav2`), DA3 (`dav3`) and MoGe-3 (`moge3`) in `raw`, per-frame `scale`, and per-frame `affine` conditions.

### 1. Install / verify optional runtimes

The normal project virtual environment supplies DAv2. DA3 and MoGe-3 are intentionally isolated so their upstream pins cannot alter the project environment. TSDF fusion and mesh sampling need Open3D.

```bash
cd /mnt/oss/3dgrut-bernardin
.venv/bin/python -m pip install 'open3d>=0.18'
test -f /mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt
```

If the MoGe-3 checkpoint is absent, download the official ViT-L checkpoint:

```bash
mkdir -p /mnt/oss/MoGe/checkpoints/moge-3-vitl
.venv/bin/python -m huggingface_hub.commands.huggingface_cli download Ruicheng/moge-3-vitl model.pt \
  --local-dir /mnt/oss/MoGe/checkpoints/moge-3-vitl
```

### 2. Verify dataset mounts

```bash
test -d /mnt/data/nerf_datasets/ob3d/OB3D_colmap/emerald-square
test -f /mnt/data/nerf_datasets/dtu_dataset/dtu_eval/Points/stl/stl024_total.ply
test -f /mnt/data/nerf_datasets/tnt_dataset/tnt/Barn/Barn.ply
```

### 3. Export the runtime paths

```bash
export PYTHONPATH=/mnt/oss/meshdeps:/mnt/oss/MoGe:/mnt/oss/moge3deps:/mnt/oss/Depth-Anything-3/src:/mnt/oss/da3deps
export CUDA_VISIBLE_DEVICES=0
```

### 4. Run a cheap end-to-end smoke test

```bash
.venv/bin/python scripts/benchmark/evaluate_depth_models.py \
  --out-dir /tmp/depth-benchmark-smoke --max-frames 1 --max-image-side 160 \
  --mesh-samples 1000 --gt-voxel 10 \
  --ob3d-scenes emerald-square --dtu-scenes scan24 --tnt-scenes Barn
```

### 5. Run the benchmark

`--dataset-scale full` is the default and evaluates every supported uploaded sequence: 12 OB3D,
15 DTU, and 6 TnT GOF scenes. `--dataset-scale reduced` always selects the same fixed third:
OB3D `archiviz-flat,classroom,lone-monk,san-miguel`; DTU
`scan105,scan114,scan24,scan55,scan69`; and TnT `Barn,Ignatius`. Use the reduced protocol for
development comparisons; use full for reported results. Omit `--max-frames` and
`--max-image-side` for a full-resolution multi-view run. The default surface sampler uses two
million uniform mesh samples; keep `--gt-voxel` unset for benchmark scoring. The MoGe-3 default
checkpoint is `/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt`.

```bash
.venv/bin/python scripts/benchmark/evaluate_depth_models.py \
  --out-dir /mnt/oss/results/depth-benchmark-2026-08-31 \
  --models dav2,dav3,moge3 \
  --dataset-scale full
```

To run the deterministic one-third protocol, change the last option to
`--dataset-scale reduced`. Individual suite lists can still be overridden with
`--ob3d-scenes`, `--dtu-scenes`, or `--tnt-scenes`; this is a custom protocol and should be
recorded as such.

To diagnose a memory limit without altering the protocol, add `--memory-profile`. Each alignment
then writes `memory.jsonl`, including process RSS and cgroup memory before TSDF fusion, after each
integrated frame, after mesh extraction/sampling/release, and after surface scoring. The file is
flushed per sample, so its final line identifies the last completed stage after an OOM kill.


### 6. Inspect the machine-readable records

The output directory contains `results.jsonl` (one complete cell per model/suite/alignment), `protocol.json`, aligned depth maps, visibility z-buffers, and TSDF meshes.

```bash
wc -l /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl
jq -c '{suite, scene, model, alignment}' /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl
```

### 7. Generate the Markdown tracker and PDF deck

```bash
.venv/bin/python scripts/benchmark/report_depth_models.py \
  /mnt/oss/results/depth-benchmark-2026-08-31/results.jsonl \
  --markdown docs/depth-model-evaluation.md --pdf docs/depth-model-evaluation.pdf
```

## Protocol

- Alignment is fit in each model's native quantity: inverse z for DAv2 disparity, z for DA3/MoGe-3.
- Scale and affine benchmark maps are **oracle** per-frame fits to the GT scan z-buffer; they are diagnostics, never deployable results.
- DTU recall uses the official ground-plane GT cull and visibility z-buffer; its mesh score is Chamfer `(accuracy + completeness)/2` in millimetres, with the official observation mask on predictions.
- TnT recall and mesh scoring use the official crop; F1 is reported at each scene's official threshold.
- The generated PDF includes visible-scan recall curves for raw, scale, and affine conditions. DTU uses absolute millimetre tolerances; TnT uses each scene's official tolerance multiplier so scenes can be averaged fairly.
- All meshes are fused through `threedgrut.geometry.tsdf.fuse_depth_frames`, shared with `extract_mesh_tsdf.py`. Source RGB is fused too, so the exported PLY files contain vertex colors; color does not affect the geometry metrics.
- Mesh scoring keeps the two-million-sample default, but executes exact nearest-neighbour queries in 100,000-sample batches to bound host memory. `--surface-query-chunk-size` changes only peak memory, not the metric.
- For DTU/TnT, aligned maps are written one frame at a time and RGB-D frames are lazily loaded into TSDF fusion. The Open3D mesh is released after sampling and before scoring. A 64 GB host should therefore be sufficient for the standard protocol; use `--surface-query-chunk-size 25000` only as an additional exact-query memory guard.
- TSDF depth is capped before Open3D integration. By default the cap is `2 ×` the nearest training camera's distance to the least-squares camera focus, matching AmbiSuR's adaptive extraction rule; it is never inferred from a model-depth maximum. Use `--fusion-max-depth-dtu` or `--fusion-max-depth-tnt` only for an explicit, recorded scene-unit cap. The selected source and cap are saved in each result's `tsdf` record.

## Evaluate a 3dgrut reconstruction checkpoint

Use this path for a trained 3dgrut reconstruction rather than a monocular prior. It exports the renderer's Euclidean ray-distance maps, preserves the camera model actually rendered, and reuses the same TSDF fusion implementation as the depth-model benchmark.

### 1. Reconstruct the scene

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config-name apps/colmap_3dgut.yaml \
  path=/mnt/data/nerf_datasets/dtu_dataset/dtu/scan24 \
  out_dir=runs experiment_name=scan24_3dgut
# Checkpoint: runs/scan24_3dgut/<timestamp>/ckpt_last.pt
```

### 2. Export rendered depths and the world transform

```bash
export CKPT=runs/scan24_3dgut/<timestamp>/ckpt_last.pt
export DEPTH_OUT=/tmp/scan24_3dgut_depth
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/export_depth_maps.py \
  --checkpoint "$CKPT" --out-dir "$DEPTH_OUT" --report-gt-agreement
```

The exporter writes `manifest.json` (ray depth plus fitted pinhole cameras), `alignment.npy` (dataset source world to rendered world), and `export_summary.json`. Do not score these maps as z-depth.

### 3. Score depth recall — DTU

The exported depths commonly live in normalized coordinates. Convert the desired millimetre tolerances through the exported source-to-render scale before passing them to the manifest-driven evaluator.

```bash
export DTU_EVAL=/mnt/data/nerf_datasets/dtu_dataset/dtu_eval
.venv/bin/python - <<'PY'
import numpy as np
matrix = np.load('/tmp/scan24_3dgut_depth/alignment.npy')
scale = np.linalg.norm(matrix[:3, :3], axis=1).mean()
print('render-space taus for 0.5,1,2,5 mm:', ','.join(f'{v * scale:.9g}' for v in (.5, 1, 2, 5)))
PY
cd tools/depthrecall
PYTHONPATH=$PWD ../../.venv/bin/python -m depthrecall \
  --manifest "$DEPTH_OUT/manifest.json" --alignment "$DEPTH_OUT/alignment.npy" \
  --ply "$DTU_EVAL/Points/stl/stl024_total.ply" \
  --dtu-plane "$DTU_EVAL/ObsMask/Plane24.mat" \
  --depth-convention ray --taus <render-space-taus> -o "$DEPTH_OUT/dtu_recall.json"
cd ../..
```

### 4. Score depth recall — Tanks and Temples

For this section, set `CKPT` and `DEPTH_OUT` to the Barn reconstruction/export rather than the DTU example above.

First fit the official scan-to-render registration from the exported camera manifest, then rasterize a GT visibility manifest. The latter removes scan self-occlusion from the recall denominator.

```bash
export TNT=/mnt/data/nerf_datasets/tnt_dataset/tnt/Barn
PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/align_tnt.py \
  --manifest "$DEPTH_OUT/manifest.json" --trans "$TNT/Barn_trans.txt" \
  --log "$TNT/Barn_COLMAP_SfM.log" --out "$DEPTH_OUT/scan_to_render.npy" \
  --inverse-out "$DEPTH_OUT/render_to_scan.npy" --report "$DEPTH_OUT/tnt_alignment.json"
PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/rasterize_depth.py \
  --manifest "$DEPTH_OUT/manifest.json" --ply "$TNT/Barn.ply" \
  --alignment "$DEPTH_OUT/scan_to_render.npy" --crop-json "$TNT/Barn.json" \
  --out-dir "$DEPTH_OUT/tnt_visibility"
cd tools/depthrecall
PYTHONPATH=$PWD ../../.venv/bin/python -m depthrecall \
  --manifest "$DEPTH_OUT/manifest.json" --alignment "$DEPTH_OUT/scan_to_render.npy" \
  --ply "$TNT/Barn.ply" --crop-json "$TNT/Barn.json" \
  --visibility-manifest "$DEPTH_OUT/tnt_visibility/manifest.json" \
  --depth-convention ray --taus 0.01,0.02,0.05 -o "$DEPTH_OUT/tnt_recall.json"
cd ../..
```

### 5. Extract and score the TSDF mesh

Use a voxel size in the checkpoint's rendered units (for normalized DTU, start with `0.002`). Before standard DTU/TnT surface scoring, transform the predicted mesh into scan coordinates; the evaluator applies that transform before the official scan-space masks.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/extract_mesh_tsdf.py \
  --checkpoint "$CKPT" --out "$DEPTH_OUT/mesh.ply" --voxel-size 0.002
.venv/bin/python - <<'PY'
import numpy as np
np.save('/tmp/scan24_3dgut_depth/render_to_scan.npy', np.linalg.inv(np.load('/tmp/scan24_3dgut_depth/alignment.npy')))
PY
PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/evaluate_surface.py \
  --pred "$DEPTH_OUT/mesh.ply" --gt "$DTU_EVAL/Points/stl/stl024_total.ply" \
  --pred-alignment "$DEPTH_OUT/render_to_scan.npy" \
  --dtu-obsmask "$DTU_EVAL/ObsMask/ObsMask24_10.mat" \
  --dtu-plane "$DTU_EVAL/ObsMask/Plane24.mat" --taus 0.2 \
  --out "$DEPTH_OUT/dtu_surface.json"
# TnT Barn mesh F1: use the render_to_scan.npy written by align_tnt.py
PYTHONPATH=tools/depthrecall .venv/bin/python tools/depthrecall/scripts/evaluate_surface.py \
  --pred "$DEPTH_OUT/mesh.ply" --gt "$TNT/Barn.ply" \
  --pred-alignment "$DEPTH_OUT/render_to_scan.npy" --crop-json "$TNT/Barn.json" \
  --taus 0.01 --out "$DEPTH_OUT/tnt_surface.json"
```

## OB3D depth accuracy

| model | raw abs-rel | scale abs-rel | affine abs-rel |
| --- | ---: | ---: | ---: |
| dav2 | 0.987 | 0.164 | 0.067 |
| dav3 | 0.937 | 0.095 | 0.060 |
| moge3 | 0.228 | 0.070 | 0.071 |

## DTU scan24

Recall is visibility-corrected recall@5 mm; scale and affine are oracle scan-z-buffer fits.

| model | raw recall@5mm | scale recall@5mm | affine recall@5mm |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.017 | 0.179 |
| dav3 | 0.000 | 0.067 | 0.267 |
| moge3 | 0.000 | 0.295 | 0.303 |

DTU TSDF mesh Chamfer is `(accuracy + completeness) / 2` in millimetres; `—` is an empty mesh.

| model | raw Chamfer | scale Chamfer | affine Chamfer |
| --- | ---: | ---: | ---: |
| dav2 | — | — | 33.857 |
| dav3 | — | 43.169 | 32.251 |
| moge3 | — | 15.424 | 13.022 |

## Tanks and Temples Barn

Recall is visibility-corrected at the official 1 cm tolerance.

| model | raw recall@1cm | scale recall@1cm | affine recall@1cm |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.000 | 0.010 |
| dav3 | 0.001 | 0.005 | 0.001 |
| moge3 | 0.000 | 0.002 | 0.009 |

TnT mesh F1 is at the official 1 cm Barn tolerance; `—` denotes an empty mesh.

| model | raw F1 | scale F1 | affine F1 |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.000 | 0.000 |
| dav3 | 0.000 | 0.000 | 0.000 |
| moge3 | 0.000 | 0.000 | 0.000 |

## Reading this run

- Raw DAv2/DA3 are intentionally uncalibrated relative outputs; their absolute scores are not comparable to a metric-depth claim.
- Per-frame scale/affine use a GT scan z-buffer and are explicitly oracle diagnostics, not deployable alignments.
- DTU uses its ground-plane completeness cull and predicted-surface observation mask; TnT uses its official crop plus a GT visibility z-buffer to remove scan self-occlusion.
- TSDF is the shared posed-depth path used by checkpoint extraction and this benchmark. One-view fusion is expected to be incomplete; mesh values here validate the path, not a competitive reconstruction setting.
