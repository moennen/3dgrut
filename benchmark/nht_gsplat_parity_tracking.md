# NHT gsplat Parity Tracking

This document tracks codepath differences between the reference Neural Harmonic Texture gsplat implementation and the 3dgrut NHT implementation, plus the bonsai A/B experiments used to isolate accuracy gaps.

## Scope

- Branch: `nht-gsplat-parity-ablation`
- Dataset: mip-NeRF 360 `bonsai`
- Images: baked/downscaled `images_2`
- Method: 3DGUT/3DGS MCMC with NHT
- Particle cap: 1M
- Fast A/B budget: 7K iterations unless otherwise noted
- Reference target: gsplat NHT run using the native `images_2` files
- Stop condition: stop the sequence when a 3dgrut 7K A/B result reaches raw PSNR greater than `34.0 dB`, or after all planned experiments are exhausted.
- Interpretation threshold: PSNR deltas below roughly `0.1-0.2 dB` are treated as run-to-run noise, not evidence of a real change.

## Runner

Use the bonsai parity runner for every experiment in this branch:

```bash
micromamba run -n 3dgrut-nht bash benchmark/nht_gsplat_parity_bonsai.sh
```

Common knobs:

```bash
MODE=all|3dgrut|gsplat
EXP_ID=e00_reference
GPU=0
MAX_STEPS=7000
FEATURE_DIM=48
CAP_MAX=1000000
DATA_ROOT=/mnt/gogn/data/nerf_datasets/nerf_360
RESULT_ROOT=results/nht_gsplat_parity
GSPLAT_REPO=/mnt/dev/neural-harmonic-textures
GSPLAT_TORCH_CUDA_ARCH_LIST=8.9
```

For 3dgrut-only A/B rows, pass Hydra overrides after the script name:

```bash
MODE=3dgrut EXP_ID=e01_knn_init_scale \
  micromamba run -n 3dgrut-nht bash benchmark/nht_gsplat_parity_bonsai.sh \
  initialization.use_observation_points=false
```

Each run writes `summary.json`, command files, train logs, render logs, and metrics under `results/nht_gsplat_parity/<EXP_ID>/`.

The gsplat reference runner sets an explicit CUDA build environment for correctness and reproducibility:

- `CUDA_HOME` defaults to the active conda/micromamba prefix when available.
- CUDA headers are added through `CPATH` and `CPLUS_INCLUDE_PATH`.
- CUDA libraries are added through `LIBRARY_PATH` and `LD_LIBRARY_PATH`.
- `TORCH_CUDA_ARCH_LIST` defaults to native Ada `8.9`, override with `GSPLAT_TORCH_CUDA_ARCH_LIST`.
- `TORCH_EXTENSIONS_DIR` defaults to `results/nht_gsplat_parity/.torch_extensions_gsplat` to avoid stale global JIT artifacts.

## Known Reference Points

| Run | Iterations | PSNR | SSIM | LPIPS | Frame Time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gsplat NHT reference | 30K | 34.2183 | 0.9543 | 0.2345 | 9.73 ms | `/mnt/dev/neural-harmonic-textures/results/bonsai_native_images2_1m_48f_ssim02_sm89_cuda_home` |
| 3dgrut NHT current | 30K | 33.4333 | 0.9483 | 0.2518 | 9.52 ms | `results/tmp_mipnerf360_mcmc_nht/3dgut_nht_mcmc/bonsai` |

## Difference Register

| ID | Area | Reference gsplat Codepath | 3dgrut Codepath | Expected Impact | Status |
| --- | --- | --- | --- | --- | --- |
| D01 | Dataset normalization | Normalizes cameras and COLMAP points with focus centering, up alignment, PCA point alignment, and optional flip. | Uses raw COLMAP coordinates. | High: changes position scale, init scale, LR scale, MCMC noise, and regularization. | Open |
| D02 | Scene scale | Effective bonsai scale measured as `1.28054`. | Bonsai scene extent measured as `4.12053`. | High: position LR and MCMC noise are about 3.2x larger in 3dgrut. | Open |
| D03 | Initial Gaussian scale | Uses SFM point KNN distance times `init_scale=0.1`. | Uses point-to-observer distance times `observation_scale_factor=0.01` and `default_scale_factor=0.1`. | High: changes early coverage, opacity dynamics, and densification behavior. | Open |
| D04 | Rasterizer | gsplat 3DGS NHT eval3d/UT rasterizer. | 3DGUT renderer with its own projection, sorting, hit, and backward codepath. | High: may change compositing, ordering, culling, and gradients. | Open |
| D05 | Feature precision | Trainable NHT features and decoder inputs are fp32 Python-side, but the NHT rasterizer casts feature colors to half and launches `at::Half` kernels; loaded feature values are converted back to float for accumulation. Backward accumulates color gradients in fp32 and returns them in the original input dtype. | Current benchmark uses `particle_feature_half=true` and `feature_output_half=true` at the tracer boundary. With `particle_feature_half=true`, 3dgrut casts the feature tensor to half before entering the custom autograd op. | Important in 3dgrut, but not due to the final ray-feature output dtype: E04a showed `feature_output_half=false` alone gives no PSNR recovery, while E04b recovered PSNR with `particle_feature_half=false` even when `feature_output_half=true`. The loss is tied to the particle-feature input boundary. | Measured |
| D06 | Decoder ray direction | Rasterizer appends actual per-pixel ray direction to the NHT output. | Python recomputes world ray directions from cached camera rays and pose. | Medium: should be close for pinhole cameras, but not identical to renderer-generated rays. | Open |
| D07 | Optimizer layout | One Adam optimizer per Gaussian parameter. | One fused Adam with parameter groups, plus separate decoder Adam. | Medium: usually close, but optimizer state mutation during MCMC can differ. | Open |
| D08 | MCMC implementation | gsplat MCMC relocate/add/noise ops. | 3dgrut MCMC strategy and optimizer-state updates. | Medium: can change particle distribution and recovery from dead particles. | Open |
| D09 | Color refinement | Geometry LR is set to zero; strategy still runs with zero LR. | Geometry LR is set to zero and strategy is suspended. | Low to medium: mostly affects the final color-refinement window. | Open |
| D10 | Random streams | Global seeded RNG and gsplat draw order. | Local generator for model init and a different draw order. | Low to medium: affects exact initialization but should not explain systematic gaps alone. | Open |
| D11 | Data loading and split | Native `images_2`, sorted image names, every 8th image for validation. | Same effective split and image source for bonsai. | Low: likely matched. | Matched |
| D12 | Losses | L1 + valid-padding fused SSIM, opacity and scale regularization. | Same loss weights and valid-padding fused SSIM path. | Low: likely matched. | Matched |
| D13 | Decoder architecture | tcnn FullyFusedMLP, ReLU, sigmoid output, SH direction encoding. | Same configured decoder architecture. | Low: likely matched. | Matched |
| D14 | NHT interpolation | Tetrahedral barycentric interpolation with sin/cos harmonic encoding. | Same conceptual tetrahedral NHT interpolation and sin/cos encoding. | Low: likely matched, pending kernel-level parity check. | Mostly matched |

## Planned A/B Experiments

Run cumulative alignment experiments against the same 7K bonsai scene. Each row after E00 keeps earlier accepted alignment knobs unless the row explicitly says it is isolated. Record raw PSNR/SSIM/LPIPS and frame time when available.

| Exp | Difference IDs | Change | Iterations | Expected Signal | Status | Result |
| --- | --- | --- | ---: | --- | --- | --- |
| E00 | Baseline | Current 3dgrut NHT bonsai config and gsplat NHT reference, both at 7K iterations. | 7K | Establish local comparison points. | Done | 3dgrut: 29.3969 / 0.9181 / 0.3121 / 8.77 ms. gsplat: 30.9817 / 0.9322 / 0.2800 / 9.43 ms. |
| E01 | D03 | Set `initialization.use_observation_points=false` to use SFM KNN scale. | 7K | Tests whether initial scale explains early accuracy gap. | Done | 3dgrut: 30.0035 / 0.9183 / 0.3054 / 9.19 ms. Improves E00 3dgrut by +0.6066 dB, still -0.9781 dB vs E00 gsplat. |
| E02 | D02 | Keep E01 KNN init scale and scale configured position LR/final LR by `1.2805438282393957 / 4.120534491539002 = 0.3107712921`. | 7K | Isolates effective scene-scale LR and MCMC-noise mismatch. | Done | 3dgrut: 30.0728 / 0.9167 / 0.3073 / 10.12 ms. Noise-level +0.0693 dB over E01; later MCMC relocation drops to ~6-8%, but initial relocation still ~85%. |
| E03 | D01, D02 | Keep E01 KNN init scale and apply reference world normalization to dataset poses and COLMAP init points; use the normalized scene scale from the dataset. | 7K | Tests full coordinate-system parity. | Done | 3dgrut: 30.5662 / 0.9247 / 0.2930 / 10.36 ms. Clear +0.4934 dB over E02 and +0.5627 dB over E01, still -0.4155 dB vs E00 gsplat. |
| E04 | D05 | Disable 3dgrut NHT half precision: `render.particle_feature_half=false`, `render.feature_output_half=false`. | 7K | Tests 3dgrut tracer boundary/output precision; this does not exactly match gsplat, whose NHT CUDA rasterizer reads features as half internally. | Done | 3dgrut: 30.9332 / 0.9318 / 0.2814 / 10.48 ms. Clear +0.3670 dB over E03 and only -0.0485 dB vs E00 gsplat, which is inside the expected 0.1-0.2 dB run noise. |
| E04a | D05 | Keep gsplat-like half particle-feature reads with `render.particle_feature_half=true`, but disable only the ray-feature output half path with `render.feature_output_half=false`. | 7K | Isolates whether E04 was recovering accuracy by avoiding fp16 integrated ray-feature output quantization. | Done | 3dgrut: 30.5596 / 0.9246 / 0.2930 / 10.69 ms. No recovery versus E03; output quantization is not the measurable 0.3 dB loss. |
| E04b | D05 | Keep fp32 particle-feature input with `render.particle_feature_half=false`, but leave `render.feature_output_half=true`. | 7K | Confirms whether fp32 particle-feature input alone recovers E04. | Done | 3dgrut: 31.0766 / 0.9325 / 0.2800 / 10.58 ms. Recovers and slightly exceeds E04, confirming the E04 gain is from `particle_feature_half=false`, not from fp32 ray-feature output. |
| E05 | D06 | Feed decoder ray directions produced by the renderer path instead of recomputing in Python. | 7K | Tests ray-direction boundary mismatch. | Not run | TBD |
| E06 | D08 | Compare or port gsplat MCMC relocate/add/noise behavior. | 7K | Tests particle-distribution parity after densification. | Not run | TBD |
| E07 | D04 | Renderer parity experiment, if a gsplat rasterizer path can be run with the same 3dgrut model state. | 7K or snapshot | Separates training pipeline issues from rasterizer/kernel issues. | Not run | TBD |

## Result Log Template

Use this table to append completed A/B results.

| Date | Exp | Commit | Command/Config | PSNR | SSIM | LPIPS | Frame Time | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-06-07 | E00 | `14bd2907` | 3dgrut: `MODE=all` initial run, gsplat rerun after env fix: `MODE=gsplat ALLOW_EXISTING=1 MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000` | 29.3969 | 0.9181 | 0.3121 | 8.77 ms | 3dgrut current baseline. Metrics: `results/nht_gsplat_parity/e00_reference/3dgrut_current/bonsai/eval/bonsai/bonsai-0706_104953/metrics.json`. |
| 2026-06-07 | E00 | `14bd2907` | `MODE=gsplat ALLOW_EXISTING=1 MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 GSPLAT_TORCH_CUDA_ARCH_LIST=8.9` | 30.9817 | 0.9322 | 0.2800 | 9.43 ms | gsplat reference at step 6999, 1M Gaussians, color refinement from step 4000. Metrics: `results/nht_gsplat_parity/e00_reference/gsplat_reference/stats/val_step6999.json`. |
| 2026-06-07 | E01 | `93bb6c3e` | `MODE=3dgrut EXP_ID=e01_knn_init_scale MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 initialization.use_observation_points=false` | 30.0035 | 0.9183 | 0.3054 | 9.19 ms | KNN init scale improves PSNR by +0.6066 dB over E00 3dgrut, but first relocation jumps to 86.07%, suggesting a major initialization-scale change. Metrics: `results/nht_gsplat_parity/e01_knn_init_scale/3dgrut_current/bonsai/eval/bonsai/bonsai-0706_112446/metrics.json`. |
| 2026-06-07 | E02 | `5fe802ce` | `MODE=3dgrut EXP_ID=e02_lr_scene_scale MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 initialization.use_observation_points=false optimizer.params.positions.lr=4.972340674225953e-05 scheduler.positions.lr_init=4.972340674225953e-05 scheduler.positions.lr_final=4.972340674225952e-07` | 30.0728 | 0.9167 | 0.3073 | 10.12 ms | Cumulative E01+scene-scale LR/noise alignment. Noise-level +0.0693 dB over E01, still -0.9088 dB vs E00 gsplat. Metrics: `results/nht_gsplat_parity/e02_lr_scene_scale/3dgrut_current/bonsai/eval/bonsai/bonsai-0706_114616/metrics.json`. |
| 2026-06-07 | E03 | `bdbb5d36` | `MODE=3dgrut EXP_ID=e03_gsplat_world_normalization MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 initialization.use_observation_points=false dataset.normalize_world_space=true` | 30.5662 | 0.9247 | 0.2930 | 10.36 ms | Cumulative E01 + gsplat COLMAP world normalization for poses and init points. Pre-run parser check matched gsplat transform within `1.7e-8`, scene extent within `2.1e-7`, and transformed points within `1e-6` fp32. First relocation remained high at 79.80%, but this produced a clear +0.4934 dB over E02. Metrics: `results/nht_gsplat_parity/e03_gsplat_world_normalization/3dgrut_current/bonsai/eval/bonsai/bonsai-0706_121219/metrics.json`. |
| 2026-06-07 | E04 | `64e8cf4c` | `MODE=3dgrut EXP_ID=e04_nht_fp32 MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 TORCH_EXTENSIONS_DIR=results/nht_gsplat_parity/.torch_extensions_3dgrut_e04_fp32 initialization.use_observation_points=false dataset.normalize_world_space=true render.particle_feature_half=false render.feature_output_half=false` | 30.9332 | 0.9318 | 0.2814 | 10.48 ms | Cumulative E03 + disabled 3dgrut NHT half precision at particle-feature input and ray-feature output boundaries. The extension was rebuilt in an isolated cache with `PARTICLE_FEATURE_HALF=0` and `FEATURE_OUTPUT_HALF=0`. This closes the 7K gap to E00 gsplat to -0.0485 dB, within run noise; caveat: gsplat itself casts NHT CUDA rasterizer feature reads to half internally. Training: 554.40 s, 12.63 it/s. Metrics: `results/nht_gsplat_parity/e04_nht_fp32/3dgrut_current/bonsai/eval/bonsai/bonsai-0706_122630/metrics.json`. |
| 2026-06-08 | E04a | `8547eecf` | `MODE=3dgrut EXP_ID=e04a_output_fp32 MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 TORCH_EXTENSIONS_DIR=results/nht_gsplat_parity/.torch_extensions_3dgrut_e04a_output_fp32 initialization.use_observation_points=false dataset.normalize_world_space=true render.particle_feature_half=true render.feature_output_half=false` | 30.5596 | 0.9246 | 0.2930 | 10.69 ms | Cumulative E03 + fp32 ray-feature output only. This is -0.0066 dB versus E03, so output-half quantization is not the E04 gain. The E04 gain is tied to `particle_feature_half=false`. Metrics: `results/nht_gsplat_parity/e04a_output_fp32/3dgrut_current/bonsai/eval/bonsai/bonsai-0806_065917/metrics.json`. |
| 2026-06-08 | E04b | `139cac31` | `MODE=3dgrut EXP_ID=e04b_particle_fp32_output_half MAX_STEPS=7000 FEATURE_DIM=48 CAP_MAX=1000000 TORCH_EXTENSIONS_DIR=results/nht_gsplat_parity/.torch_extensions_3dgrut_e04b_particle_fp32_output_half initialization.use_observation_points=false dataset.normalize_world_space=true render.particle_feature_half=false render.feature_output_half=true` | 31.0766 | 0.9325 | 0.2800 | 10.58 ms | Cumulative E03 + fp32 particle-feature input only, while keeping fp16 ray-feature output. This is +0.5104 dB over E03, +0.1434 dB over E04, and +0.0949 dB over E00 gsplat; those E04/E00 deltas are within the expected 0.1-0.2 dB run noise. Training: 1070.73 s, 6.54 it/s. Metrics: `results/nht_gsplat_parity/e04b_particle_fp32_output_half/3dgrut_current/bonsai/eval/bonsai/bonsai-0806_072252/metrics.json`. |

## Testing Rules

- Change only one variable per experiment unless the row explicitly groups coupled codepaths; cumulative rows must list the inherited knobs in the command/config column.
- Prefer bonsai 7K for quick signal, then rerun promising changes at 30K.
- Treat PSNR deltas below about `0.1-0.2 dB` as run-to-run noise unless repeated runs show the same direction.
- Keep raw and color-corrected metrics separate. The reference target above is raw.
- Record whether the extension was rebuilt when changing compile-time render flags.
- Do not mix remote and local results unless the command, commit, dataset path, and environment are recorded.
- Commit after every completed experiment row, including the logged result and any code/config changes for that row.
- If raw 3dgrut PSNR exceeds `34.0 dB`, stop and do not continue to later A/B rows until the result is reviewed.
