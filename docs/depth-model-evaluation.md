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
  --mesh-samples 1000 --gt-voxel 10
```

### 5. Run the benchmark

Use one or more comma-separated scenes per suite. Omit `--max-frames` and `--max-image-side` for the full-resolution multi-view run. The default surface sampler uses two million uniform mesh samples; keep `--gt-voxel` unset for benchmark scoring. The MoGe-3 default checkpoint is `/mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt`.

```bash
.venv/bin/python scripts/benchmark/evaluate_depth_models.py \
  --out-dir /mnt/oss/results/depth-benchmark-2026-08-31 \
  --models dav2,dav3,moge3 \
  --ob3d-scenes emerald-square,sponza \
  --dtu-scenes scan24 --tnt-scenes Barn
```


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
- TnT recall and mesh scoring use its official crop; F1 is reported at the scene's official threshold (Barn: 1 cm).
- All meshes are fused through `threedgrut.geometry.tsdf.fuse_depth_frames`, shared with `extract_mesh_tsdf.py`.

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
