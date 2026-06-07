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
| D05 | Feature precision | NHT features and integrated outputs appear to stay fp32. | Current benchmark uses `particle_feature_half=true` and `feature_output_half=true`. | Medium to high: NHT feature accumulation may lose precision. | Open |
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

Run one change at a time against the same 7K bonsai baseline. Record raw PSNR/SSIM/LPIPS and frame time when available.

| Exp | Difference IDs | Change | Iterations | Expected Signal | Status | Result |
| --- | --- | --- | ---: | --- | --- | --- |
| E00 | Baseline | Current 3dgrut NHT bonsai config and gsplat NHT reference, both at 7K iterations. | 7K | Establish local comparison points. | Done | 3dgrut: 29.3969 / 0.9181 / 0.3121 / 8.77 ms. gsplat: 30.9817 / 0.9322 / 0.2800 / 9.43 ms. |
| E01 | D03 | Set `initialization.use_observation_points=false` to use SFM KNN scale. | 7K | Tests whether initial scale explains early accuracy gap. | Done | 3dgrut: 30.0035 / 0.9183 / 0.3054 / 9.19 ms. Improves E00 3dgrut by +0.6066 dB, still -0.9781 dB vs E00 gsplat. |
| E02 | D02 | Keep raw coordinates, but scale configured position LR and final LR by `1.28054 / 4.12053 ~= 0.311`. | 7K | Isolates effective scene-scale LR and MCMC-noise mismatch. | Not run | TBD |
| E03 | D01, D02 | Apply reference world normalization to dataset poses and COLMAP init points; use reference scene scale. | 7K | Tests full coordinate-system parity. | Not run | TBD |
| E04 | D05 | Disable NHT half precision: `render.particle_feature_half=false`, `render.feature_output_half=false`. | 7K | Tests feature accumulation precision. | Not run | TBD |
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

## Testing Rules

- Change only one variable per experiment unless the row explicitly groups coupled codepaths.
- Prefer bonsai 7K for quick signal, then rerun promising changes at 30K.
- Keep raw and color-corrected metrics separate. The reference target above is raw.
- Record whether the extension was rebuilt when changing compile-time render flags.
- Do not mix remote and local results unless the command, commit, dataset path, and environment are recorded.
- Commit after every completed experiment row, including the logged result and any code/config changes for that row.
- If raw 3dgrut PSNR exceeds `34.0 dB`, stop and do not continue to later A/B rows until the result is reviewed.
