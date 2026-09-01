# Frozen image-feature supervision: 7k OB3D smoke

This is a single-seed, 7,000-iteration smoke sweep over `sponza`, `lone-monk`, and
`emerald-square`. It isolates the image target: every condition uses the same direct NHT
representation (32-D per-Gaussian latent, center interpolation, no sinusoidal latent or
directional harmonic expansion), RGB loss, and direct 3-vector RGB view input. The feature head
is direction-free and is trained with a 16-D alpha-aware, low-resolution cosine target at weight
`0.1`.

## Results

| target | PSNR ↑ | depth abs-rel ↓ | depth RMSE ↓ | delta-1 ↑ | coverage ↑ | floater fraction ↓ | normal mean ° ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB only | **31.376** | 0.6041 | 3.4305 | **0.3893** | 0.4503 | 0.0057 | 61.84 |
| DINOv2 + PCA | 30.928 | 0.6035 | 3.4064 | 0.3735 | 0.4501 | **0.0044** | 59.57 |
| DINOv2 + autoencoder | 31.074 | 0.6145 | 3.4256 | 0.3729 | 0.4403 | 0.0047 | **59.25** |
| C-RADIOv4 + PCA | 31.153 | **0.5922** | **3.3925** | 0.3850 | **0.4680** | 0.0048 | 59.90 |

The values are unweighted means over the three scenes. RMSE is in each scene's normalized world
units, so it is reported for context; abs-rel, delta-1, coverage, and floater fraction are the
cross-scene geometry comparisons. Normal error remains diagnostic only: it has no direct normal
supervision in this sweep.

| scene | RGB only: PSNR / abs-rel | DINOv2 PCA: PSNR / abs-rel | DINOv2 AE: PSNR / abs-rel | C-RADIOv4 PCA: PSNR / abs-rel |
|---|---:|---:|---:|---:|
| sponza | **33.341** / 0.2247 | 32.987 / 0.1891 | 33.015 / 0.2055 | 32.983 / **0.1767** |
| lone-monk | **31.901** / 0.8527 | 31.535 / 0.8447 | 30.959 / 0.8579 | 31.428 / **0.8262** |
| emerald-square | 28.885 / **0.7349** | 28.262 / 0.7766 | **29.249** / 0.7800 | 29.050 / 0.7739 |

C-RADIOv4+PCA improves geometry on Sponza and Lone Monk and wins the mean geometry score, at a
0.22 dB mean PSNR trade-off against RGB-only. On Emerald Square, no feature target beat the
RGB-only depth error. DINOv2+PCA outperformed the tested autoencoder compressor on mean
geometry; this is not evidence against nonlinear compression generally, only against this small,
offline 16-D autoencoder and its current settings.

## Reproduction

Install the optional encoder dependencies once. C-RADIOv4 is lazy-loaded, so a DINO-only run does
not import `timm`.

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/pip install -e '.[foundation_features]'
export THREEDGRUT_RADIO_REPO=/mnt/oss/radio-lamarck  # optional local, pinned NVlabs/RADIO checkout
```

Run all four conditions serially, or split their `--variants` lists across GPUs. The harness makes
the chosen virtual environment's `slangc` available to its child training processes.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/ablation/run_ob3d.py \
  --dataset-root /mnt/data/nerf_datasets/ob3d/OB3D_colmap \
  --out-dir runs/feature_smoke_7k \
  --scenes sponza lone-monk emerald-square \
  --variants nht_rgb dinov2_pca dinov2_autoencoder nvradio4_pca \
  --n-iterations 7000 \
  --config-name apps/colmap_3dgut_mcmc_nht.yaml
```

The checked run records are ignored under `runs/`; make the markdown summary after completion with
the generic ablation reporter (it accepts multiple records when the conditions were split across
GPUs):

```bash
.venv/bin/python scripts/ablation/report.py runs/feature_smoke_7k/dinov2/results.jsonl \
  runs/feature_smoke_7k/auto_radio/results.jsonl \
  --output runs/feature_smoke_7k/report.md
```

Focused cache/projection and ablation-harness tests passed for this smoke; the full suite was not
run, consistent with the requested small-change test policy.
