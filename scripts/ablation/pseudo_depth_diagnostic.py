# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is a monocular pseudo-depth prior worth supervising with, and at what granularity?

Pseudo-depth supervision only helps if the prior is more accurate than the model it is meant
to correct. This script measures that directly, per pixel, against a trained checkpoint.

Two things it is careful about, because both are silent failure modes:

* **Alignment space.** DepthAnything emits *disparity* and is affine-invariant in disparity
  space, so the fit is ``1/z ~ a * disp + b``. Fitting ``z ~ a * disp + b`` instead costs a
  lot of accuracy (measured on sponza: R^2 0.82 vs 0.96) while looking perfectly reasonable.
* **Ray distance vs z.** The tracer's depth is Euclidean distance along the ray; the prior
  predicts z-depth. The two differ by ``||d|| / d_z``, up to 20% at the corners of a 640x360
  sponza frame -- a *radial* error that no global affine can absorb, so it is applied per pixel.

The alignment here fits against ground-truth depth, which makes the reported prior accuracy an
**upper bound** on what COLMAP-sparse-point alignment could achieve. That is deliberate: this
is a go/no-go tool, so the prior is given the benefit of the doubt. Nothing here is training
code, and no ground truth leaks into the supervision path.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from threedgrut.model.model import MixtureOfGaussians  # noqa: E402
from threedgrut.utils.depth_normal_metrics import (  # noqa: E402
    DELTA_THRESHOLDS,
    MIN_ACCUMULATED_OPACITY,
    expected_depth,
    reference_depth_validity,
)

# Patch sizes to fit the affine over. `None` is a single fit per frame -- the "align the whole
# map with the sparse points" formulation. The smaller sizes probe how much of the prior's
# value is local structure that a global fit throws away.
PATCH_SIZES = (None, 64, 32, 16)


def _load_prior_model(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def _predict_disparity(processor, model, rgb: torch.Tensor) -> torch.Tensor:
    """Run the prior on an HWC float image in [0, 1] and return HW disparity at input scale."""
    height, width = rgb.shape[:2]
    inputs = processor(images=(rgb.cpu().numpy() * 255).astype(np.uint8), return_tensors="pt")
    inputs = {k: v.to(rgb.device) for k, v in inputs.items()}
    disparity = model(**inputs).predicted_depth
    return torch.nn.functional.interpolate(
        disparity.unsqueeze(1).float(), size=(height, width), mode="bicubic", align_corners=False
    )[0, 0]


def _fit_affine_inverse(disparity: np.ndarray, inv_z: np.ndarray, trim: float, iters: int) -> np.ndarray:
    """Least squares ``inv_z ~ a * disparity + b``, refit after dropping the worst residuals.

    The trimming matters: COLMAP points and the prior both have outliers, and a plain fit lets a
    few of them drag the whole frame's scale.
    """
    keep = np.ones(len(disparity), dtype=bool)
    coef = np.zeros(2)
    for _ in range(max(iters, 1)):
        if keep.sum() < 2:
            break
        design = np.stack([disparity[keep], np.ones(int(keep.sum()))], axis=1)
        coef, *_ = np.linalg.lstsq(design, inv_z[keep], rcond=None)
        residual = np.abs(np.stack([disparity, np.ones(len(disparity))], axis=1) @ coef - inv_z)
        if trim <= 0:
            break
        keep = residual <= np.quantile(residual, 1.0 - trim)
    return coef


def _align(
    disparity: np.ndarray, inv_z_gt: np.ndarray, valid: np.ndarray, patch: int | None, trim: float, iters: int
) -> np.ndarray:
    """Affine-align disparity to inverse z, globally or per patch. Returns z (NaN where unfit)."""
    height, width = disparity.shape
    inv_z = np.full_like(disparity, np.nan)
    if patch is None:
        blocks = [(slice(0, height), slice(0, width))]
    else:
        blocks = [
            (slice(i, min(i + patch, height)), slice(j, min(j + patch, width)))
            for i in range(0, height, patch)
            for j in range(0, width, patch)
        ]
    for rows, cols in blocks:
        block_valid = valid[rows, cols]
        # A 2-parameter fit needs a handful of points with some depth spread to be meaningful.
        if block_valid.sum() < 10:
            continue
        block_disp = disparity[rows, cols]
        coef = _fit_affine_inverse(block_disp[block_valid], inv_z_gt[rows, cols][block_valid], trim, iters)
        if coef[0] == 0.0:
            continue
        inv_z[rows, cols] = coef[0] * block_disp + coef[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / inv_z


def collect(
    checkpoint_path: str, scene_path: str | None, model_id: str, trim: float, iters: int, max_frames: int
) -> dict:
    from omegaconf import open_dict

    from threedgrut.datasets import make_test

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    if scene_path:
        conf.path = scene_path
    with open_dict(conf):
        conf.dataset.load_depth_gt = True

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    dataset = make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=1, shuffle=False)
    processor, prior = _load_prior_model(model_id, "cuda")

    err_model, err_prior = [], {p: [] for p in PATCH_SIZES}
    frames = 0
    with torch.no_grad():
        for batch in loader:
            if frames >= max_frames:
                break
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            depth_gt = getattr(gpu_batch, "depth_gt", None)
            if depth_gt is None or depth_gt.numel() == 0:
                raise SystemExit(f"{checkpoint_path}: dataset provides no reference depth")

            outputs = model(gpu_batch)
            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], MIN_ACCUMULATED_OPACITY)
            valid = (reference_depth_validity(depth_gt) & confident & (depth > 0)).squeeze(-1).squeeze(0)
            if not bool(valid.any()):
                continue

            # Euclidean ray distance -> z, using the ray directions the tracer actually used, so
            # this holds regardless of whether rays_dir is normalised.
            rays_dir = gpu_batch.rays_dir.squeeze(0)
            to_z = (rays_dir[..., 2] / rays_dir.norm(dim=-1)).abs().clamp_min(1e-8)

            gt_t = depth_gt.squeeze(-1).squeeze(0)
            model_t = depth.squeeze(-1).squeeze(0)
            disparity = _predict_disparity(processor, prior, gpu_batch.rgb_gt.squeeze(0))

            valid_np = valid.cpu().numpy()
            gt_t_np = gt_t.double().cpu().numpy()
            to_z_np = to_z.double().cpu().numpy()
            inv_z_gt = np.where(valid_np & (gt_t_np > 0), 1.0 / np.maximum(gt_t_np * to_z_np, 1e-8), np.nan)
            disparity_np = disparity.double().cpu().numpy()

            rel_model = np.abs(model_t.double().cpu().numpy() - gt_t_np) / np.maximum(gt_t_np, 1e-8)
            err_model.append(rel_model[valid_np])
            for patch in PATCH_SIZES:
                z_prior = _align(disparity_np, inv_z_gt, valid_np & np.isfinite(inv_z_gt), patch, trim, iters)
                prior_t = z_prior / to_z_np  # z -> Euclidean ray distance, per pixel
                rel = np.abs(prior_t - gt_t_np) / np.maximum(gt_t_np, 1e-8)
                err_prior[patch].append(np.where(np.isfinite(rel[valid_np]), rel[valid_np], np.nan))
            frames += 1

    return {
        "model": np.concatenate(err_model),
        "prior": {p: np.concatenate(v) for p, v in err_prior.items()},
        "frames": frames,
    }


def report(stats: dict) -> None:
    rel_model = stats["model"]
    # "Wrong" in the same sense the depth metrics use: outside the delta1 ratio band. These are
    # the pixels supervision is supposed to rescue, so they are where the prior has to win.
    wrong = rel_model >= (DELTA_THRESHOLDS[0] - 1.0)
    print(
        f"\nframes: {stats['frames']}   valid px: {rel_model.size}   "
        f"model wrong (outside delta1): {100 * wrong.mean():.1f}%"
    )
    print(f"\nmodel abs_rel: all {rel_model.mean():.4f}   median {np.median(rel_model):.4f}")
    print("\n                       ---------- abs_rel ----------    P(prior closer than model)")
    print("  alignment            all      median   where-model-wrong    all    where-wrong")
    for patch, rel_prior in stats["prior"].items():
        ok = np.isfinite(rel_prior)
        label = "global (1 fit/frame)" if patch is None else f"{patch}x{patch} patch"
        both = ok & np.isfinite(rel_model)
        closer = rel_prior[both] < rel_model[both]
        w = both & wrong
        closer_wrong = rel_prior[w] < rel_model[w]
        print(
            f"  {label:<20} {np.nanmean(rel_prior):.4f}   {np.nanmedian(rel_prior):.4f}   "
            f"{np.nanmean(rel_prior[w]):.4f}            {closer.mean():.3f}   {closer_wrong.mean():.3f}"
        )
    print("\nA prior only helps where P(prior closer) > 0.5; supervising below that pulls the")
    print("model away from the truth on those pixels.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf")
    parser.add_argument("--trim", type=float, default=0.2, help="fraction of worst residuals dropped before refit")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    report(collect(args.checkpoint, args.scene_path, args.model, args.trim, args.iters, args.max_frames))


if __name__ == "__main__":
    main()
