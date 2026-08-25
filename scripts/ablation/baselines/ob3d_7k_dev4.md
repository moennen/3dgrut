# OB3D ablation

8 successful run(s) over 4 shared scene(s), 7000 iterations.

## Quality (averaged over shared scenes)

| variant   | scenes | psnr      | ssim      | lpips     | d_absrel   | d_rmse    | d_delta1  | d_bias | d_cover   |
|-----------|--------|-----------|-----------|-----------|------------|-----------|-----------|--------|-----------|
| gaussian  | 4      | **35.22** | **0.961** | **0.116** | **0.0873** | **4.253** | **0.890** | -1.538 | 0.984     |
| trisurfel | 4      | 33.72     | 0.949     | 0.133     | 0.0998     | 5.642     | 0.855     | -2.579 | **0.991** |

`d_rmse`, `d_bias` are in world units, so an average across scenes is dominated by the physically largest one; rank on `d_absrel` and `d_delta1`, which are scale free, and read the world-unit columns per scene.

## Cost

| variant   | scenes | it/s | train_s | prims   | mem_gb | ms/frame |
|-----------|--------|------|---------|---------|--------|----------|
| gaussian  | 4      | 70.9 | 109     | 513,117 | 0.81   | -        |
| trisurfel | 4      | 85.2 | 83      | 516,408 | 0.81   | -        |

## Normals (diagnostic, not ranked)

`n_control` is the error of pointing every normal back along the view ray, which uses no
geometry at all. `n_gain` is that control minus the rendered error, so a negative value
means the buffer is beaten by the control and its raw angle should not be read as accuracy.

| variant   | scenes | n_mean | n_med | n_control | n_gain |
|-----------|--------|--------|-------|-----------|--------|
| gaussian  | 4      | 56.9   | 55.7  | 40.1      | -16.7  |
| trisurfel | 4      | 39.9   | 38.1  | 40.1      | +0.2   |

## depth_abs_rel per scene

| scene          | gaussian | trisurfel |
|----------------|----------|-----------|
| classroom      | 0.0508   | 0.0528    |
| emerald-square | 0.1389   | 0.1890    |
| lone-monk      | 0.1005   | 0.1003    |
| sponza         | 0.0590   | 0.0569    |
