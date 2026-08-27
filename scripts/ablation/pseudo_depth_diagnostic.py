# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is a monocular pseudo-depth prior worth supervising with, and at what granularity?

Pseudo-depth supervision only helps if the prior is more accurate than the model it is meant
to correct. This script measures that directly, per pixel, against a trained checkpoint.

Two things it is careful about, because both are silent failure modes:

* **Alignment space.** Each prior is affine-invariant in the quantity it emits, and is fitted
  there: ``1/z ~ a * p + b`` for a disparity prior (Depth Anything V2), ``z ~ a * p + b`` for a
  depth prior (Depth Anything 3). Fitting a disparity prior in depth space costs a lot of
  accuracy (measured on sponza: R^2 0.82 vs 0.96) while looking perfectly reasonable, and the
  mistake is symmetric, so the space follows the backend rather than being chosen here.
* **Ray distance vs z.** The tracer's depth is Euclidean distance along the ray; the priors
  predict z-depth. The two differ by ``||d|| / d_z``, up to 20% at the corners of a 640x360
  sponza frame -- a *radial* error that no global affine can absorb, so it is applied per pixel.

Two families of number come out. The ``abs_rel`` table walks the prior up a ladder of alignment
freedom -- no fit at all, one scale per frame, then an affine per frame and per patch -- which
separates what the model predicts from what the fit supplies, and says how much of the prior's
value is local rather than global. The **ordinal
agreement** is the fraction of pixel pairs whose near/far ordering matches ground truth, sampled
exactly as `compute_pseudo_depth_order_loss` samples them, which is literally all the ordinal
loss reads.

It is tempting to conclude that the agreement is therefore the number to judge a prior swap on.
Measured, it is not. Swapping DAv2 for DA3 raised the agreement and improved trained depth on all
three of sponza, lone-monk and emerald-square, so the *sign* agreed -- but it ranked the scenes
backwards: emerald-square gained the least agreement (+0.6pp) and by far the most depth (-11pp).
The change in globally-aligned ``abs_rel``, which the loss never sees, ranked all three
correctly. Both are reported, and neither is a proxy for the trained outcome; the ablation
settles that. Where the agreement *is* decisive is as a veto -- a prior scoring below the trained
model's own agreement, as ``DA3-SMALL`` does on sponza, should not be used at all. See
`docs/normal-supervision.md`.

Predictions come from the same backend classes the training cache uses, so what is measured here
is what training would consume. The alignment fits against ground-truth depth, which makes the
reported prior accuracy an **upper bound** on what COLMAP-sparse-point alignment could achieve.
That is deliberate: this is a go/no-go tool, so the prior is given the benefit of the doubt.
Nothing here is training code, and no ground truth leaks into the supervision path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from threedgrut.datasets.pseudo_depth import BACKENDS  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians  # noqa: E402
from threedgrut.utils.depth_normal_metrics import (  # noqa: E402
    DELTA_THRESHOLDS,
    MIN_ACCUMULATED_OPACITY,
    expected_depth,
    reference_depth_validity,
)
from threedgrut.utils.pseudo_depth_loss import FARTHER_SIGN, sample_pair_offset  # noqa: E402

# How much freedom the prior is given to match ground truth before being scored, in increasing
# order. The progression is the measurement: each row is a claim about the prior that the next
# row stops making.
#
# `raw` fits nothing and reads the prediction as a distance directly, which is only a fair
# question of a *metric* model -- for a scale-free one it quantifies how much of the reported
# accuracy is alignment rather than prediction. `scale` grants the single degree of freedom a
# scale-free prior is entitled to (no offset, so zero still means zero). The affine rows add an
# offset, globally -- the "align the whole map against sparse points" formulation -- and then
# per patch, probing how much of the prior's value is local structure a global fit throws away.
ALIGNMENTS = (
    ("raw (no fit)", None, None),
    ("scale only", "scale", None),
    ("affine, global", "affine", None),
    ("affine, 64x64", "affine", 64),
    ("affine, 32x32", "affine", 32),
    ("affine, 16x16", "affine", 16),
)

# Matches `loss.pseudo_depth_shift_fraction`, so the pairs scored here are drawn from the same
# distribution the trained term draws from. Changing one without the other would make the
# agreement number stop predicting the term's behaviour.
SHIFT_FRACTION = 0.05

# The model each backend is compared at when `--model` is not given. `transformers` matches the
# training default; the DA3 entry is its dedicated monocular model, the like-for-like counterpart.
#
# DA3's any-view checkpoints (`DA3-SMALL` ... `DA3-GIANT`) can also be passed to `--model`. Handed
# a single image they degrade to monocular inference, which is a fair comparison to make and the
# only way to use them through the per-frame cache, but it is not what they are for: their value
# is cross-view consistency, and one view is the case that cannot exhibit it.
DEFAULT_MODELS = {
    "transformers": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_3": "depth-anything/DA3MONO-LARGE",
}


@torch.no_grad()
def _predict(predictor, rgb: torch.Tensor) -> torch.Tensor:
    """Run the prior on an HWC float image in [0, 1] and return its output at input scale.

    Bicubic upsampling to the frame size mirrors `PseudoDepthCache.load`, so the map measured
    here is pixel-for-pixel the one the loss would see.
    """
    height, width = rgb.shape[:2]
    prediction = predictor.predict((rgb.cpu().numpy() * 255).astype(np.uint8))
    tensor = torch.from_numpy(np.asarray(prediction, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False)[0, 0]


def _fit_affine(prior: np.ndarray, target: np.ndarray, trim: float, iters: int) -> np.ndarray:
    """Least squares ``target ~ a * prior + b``, refit after dropping the worst residuals.

    The trimming matters: COLMAP points and the prior both have outliers, and a plain fit lets a
    few of them drag the whole frame's scale.
    """
    keep = np.ones(len(prior), dtype=bool)
    coef = np.zeros(2)
    for _ in range(max(iters, 1)):
        if keep.sum() < 2:
            break
        design = np.stack([prior[keep], np.ones(int(keep.sum()))], axis=1)
        coef, *_ = np.linalg.lstsq(design, target[keep], rcond=None)
        residual = np.abs(np.stack([prior, np.ones(len(prior))], axis=1) @ coef - target)
        if trim <= 0:
            break
        keep = residual <= np.quantile(residual, 1.0 - trim)
    return coef


def _fit_scale(prior: np.ndarray, target: np.ndarray, trim: float, iters: int) -> float:
    """Least squares ``target ~ a * prior``, no intercept, refit after dropping worst residuals.

    Separate from `_fit_affine` rather than a constrained call to it because the question is
    different: an offset lets a prior fix a wrong *near plane*, and withholding it is what makes
    the scale-only row a test of whether the prediction is right up to units.
    """
    keep = np.ones(len(prior), dtype=bool)
    scale = 0.0
    for _ in range(max(iters, 1)):
        if keep.sum() < 1:
            break
        denominator = float(prior[keep] @ prior[keep])
        if denominator <= 0.0:
            return 0.0
        scale = float(prior[keep] @ target[keep]) / denominator
        if trim <= 0:
            break
        residual = np.abs(scale * prior - target)
        keep = residual <= np.quantile(residual, 1.0 - trim)
    return scale


def _align(
    prior: np.ndarray,
    target_gt: np.ndarray,
    valid: np.ndarray,
    mode: str | None,
    patch: int | None,
    trim: float,
    iters: int,
    invert: bool,
) -> np.ndarray:
    """Align the prior to `target_gt`, per `mode`. Returns z (NaN where unfit).

    `mode` is `None` for no fit at all, `"scale"` for one multiplier per frame, or `"affine"` for
    a scale and offset over each `patch`-sized block (`patch=None` being one block per frame).

    `target_gt` is ground truth expressed in the prior's own space -- inverse z for a disparity
    prior, z for a depth prior -- and `invert` says whether the fitted values need inverting to
    become z again. Fitting in the prior's native space is what makes the affine ambiguity the
    *whole* ambiguity.
    """
    height, width = prior.shape
    fitted = np.full_like(prior, np.nan)
    if mode is None:
        # No fit: the prediction *is* the answer, in the prior's own space. For a disparity prior
        # that still means inverting it below, which is a change of variable and not a fit.
        fitted = prior.astype(np.float64, copy=True)
    elif mode == "scale":
        if valid.sum() >= 10:
            scale = _fit_scale(prior[valid], target_gt[valid], trim, iters)
            if scale != 0.0:
                fitted = scale * prior
    elif mode == "affine":
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
            block_prior = prior[rows, cols]
            coef = _fit_affine(block_prior[block_valid], target_gt[rows, cols][block_valid], trim, iters)
            if coef[0] == 0.0:
                continue
            fitted[rows, cols] = coef[0] * block_prior + coef[1]
    else:
        raise ValueError(f"unknown alignment mode {mode!r}")
    if not invert:
        return fitted
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / fitted


def _ordinal_agreement(
    prior: np.ndarray, model_t: np.ndarray, gt_t: np.ndarray, valid: np.ndarray, farther_sign: float, offsets
) -> tuple[int, int, int, int, int]:
    """Pair counts and ordering agreements: the quantity the ordinal loss actually consumes.

    Returns pairs, prior agreements, model agreements, pairs the model orders *wrongly*, and
    prior agreements among those. The last two are the targeted number -- a prior can only move
    training on pairs the model has wrong, so a high overall agreement earned on pairs the model
    already gets right is worth nothing.

    Computed on the raw prior with no alignment at all, since alignment cannot change a sign.
    That is the term's strength, and the reason a more *accurate* prior need not be a
    better-*ordered* one.
    """
    pairs = agree_prior = agree_model = 0
    model_wrong = agree_prior_where_wrong = 0
    for dy, dx in offsets:
        # Crop rather than roll: wrapping would pair opposite edges of the image, inventing
        # orderings neither the prior nor the reference ever claimed.
        top, bottom = max(0, dy), min(prior.shape[0], prior.shape[0] + dy)
        left, right = max(0, dx), min(prior.shape[1], prior.shape[1] + dx)
        base = (slice(top, bottom), slice(left, right))
        shifted = (slice(top - dy, bottom - dy), slice(left - dx, right - dx))

        keep = valid[base] & valid[shifted]
        gt_order = np.sign(gt_t[base] - gt_t[shifted])
        # A tie in the reference carries no ordering to agree with.
        keep &= gt_order != 0
        prior_order = farther_sign * np.sign(prior[base] - prior[shifted])
        model_order = np.sign(model_t[base] - model_t[shifted])
        pairs += int(keep.sum())
        agree_prior += int((prior_order[keep] == gt_order[keep]).sum())
        agree_model += int((model_order[keep] == gt_order[keep]).sum())
        wrong = keep & (model_order != gt_order)
        model_wrong += int(wrong.sum())
        agree_prior_where_wrong += int((prior_order[wrong] == gt_order[wrong]).sum())
    return pairs, agree_prior, agree_model, model_wrong, agree_prior_where_wrong


def collect(
    checkpoint_path: str,
    scene_path: str | None,
    backend: str,
    model_id: str,
    trim: float,
    iters: int,
    max_frames: int,
    pair_samples: int,
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
    predictor = BACKENDS[backend](model_id=model_id, device="cuda")
    quantity = BACKENDS[backend].QUANTITY
    farther_sign = FARTHER_SIGN[quantity]

    err_model, err_prior = [], {label: [] for label, _, _ in ALIGNMENTS}
    ordinal = [0, 0, 0, 0, 0]  # see _ordinal_agreement for the five counts
    # Seeded so that two backends are scored on exactly the same pixel pairs; otherwise a
    # difference of a few tenths of a percent would be sampling noise rather than a result.
    pair_rng = random.Random(0)
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
            prior = _predict(predictor, gpu_batch.rgb_gt.squeeze(0))

            valid_np = valid.cpu().numpy()
            gt_t_np = gt_t.double().cpu().numpy()
            model_t_np = model_t.double().cpu().numpy()
            to_z_np = to_z.double().cpu().numpy()
            z_gt = np.where(valid_np & (gt_t_np > 0), gt_t_np * to_z_np, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_z_gt = 1.0 / np.maximum(z_gt, 1e-8)
            prior_np = prior.double().cpu().numpy()

            # Each prior is fitted in the space it is affine-invariant in; see the module
            # docstring for what fitting in the other one costs.
            target_gt, invert = (inv_z_gt, True) if quantity == "disparity" else (z_gt, False)

            rel_model = np.abs(model_t_np - gt_t_np) / np.maximum(gt_t_np, 1e-8)
            err_model.append(rel_model[valid_np])
            for label, mode, patch in ALIGNMENTS:
                fit_valid = valid_np & np.isfinite(target_gt)
                z_prior = _align(prior_np, target_gt, fit_valid, mode, patch, trim, iters, invert)
                prior_t = z_prior / to_z_np  # z -> Euclidean ray distance, per pixel
                rel = np.abs(prior_t - gt_t_np) / np.maximum(gt_t_np, 1e-8)
                err_prior[label].append(np.where(np.isfinite(rel[valid_np]), rel[valid_np], np.nan))

            # Offsets are redrawn per frame from a seeded generator, matching how the loss draws
            # one per iteration, so the agreement is averaged over the same distribution.
            offsets = [sample_pair_offset(*prior_np.shape, SHIFT_FRACTION, pair_rng) for _ in range(pair_samples)]
            counts = _ordinal_agreement(prior_np, model_t_np, gt_t_np, valid_np, farther_sign, offsets)
            ordinal = [a + b for a, b in zip(ordinal, counts)]
            frames += 1

    return {
        "model": np.concatenate(err_model),
        "prior": {p: np.concatenate(v) for p, v in err_prior.items()},
        "frames": frames,
        "backend": backend,
        "model_id": model_id,
        "quantity": quantity,
        "ordinal": ordinal,
    }


def summarise(stats: dict) -> dict:
    """Reduce the per-pixel arrays to the numbers reported, as plain JSON-serialisable types.

    The deck's figures read this rather than transcribed constants, so a plot cannot drift from
    the run that produced it. `report` prints from the same dict, so the two cannot disagree.
    """
    rel_model = stats["model"]
    # "Wrong" in the same sense the depth metrics use: outside the delta1 ratio band. These are
    # the pixels supervision is supposed to rescue, so they are where the prior has to win.
    wrong = rel_model >= (DELTA_THRESHOLDS[0] - 1.0)
    rows = []
    for label, rel_prior in stats["prior"].items():
        both = np.isfinite(rel_prior) & np.isfinite(rel_model)
        w = both & wrong
        rows.append(
            {
                "alignment": label,
                "abs_rel": float(np.nanmean(rel_prior)),
                "abs_rel_median": float(np.nanmedian(rel_prior)),
                "abs_rel_where_model_wrong": float(np.nanmean(rel_prior[w])),
                "p_closer": float((rel_prior[both] < rel_model[both]).mean()),
                "p_closer_where_wrong": float((rel_prior[w] < rel_model[w]).mean()),
                "fitted_frac": float(both.mean()),
            }
        )

    pairs, agree_prior, agree_model, model_wrong, agree_wrong = stats["ordinal"]
    ordinal = {"pairs": int(pairs)}
    if pairs:
        ordinal.update(
            prior_agreement=agree_prior / pairs,
            model_agreement=agree_model / pairs,
            model_wrong_frac=model_wrong / pairs,
            prior_agreement_where_model_wrong=(agree_wrong / model_wrong) if model_wrong else float("nan"),
        )
    return {
        "backend": stats["backend"],
        "model_id": stats["model_id"],
        "quantity": stats["quantity"],
        "frames": int(stats["frames"]),
        "valid_px": int(rel_model.size),
        "model_abs_rel": float(rel_model.mean()),
        "model_abs_rel_median": float(np.median(rel_model)),
        "model_wrong_frac": float(wrong.mean()),
        "alignments": rows,
        "ordinal": ordinal,
    }


def report(summary: dict) -> None:
    print(f"\nprior: {summary['backend']} / {summary['model_id']}  ({summary['quantity']})")
    print(
        f"frames: {summary['frames']}   valid px: {summary['valid_px']}   "
        f"model wrong (outside delta1): {100 * summary['model_wrong_frac']:.1f}%"
    )
    print(f"\nmodel abs_rel: all {summary['model_abs_rel']:.4f}   median {summary['model_abs_rel_median']:.4f}")
    print("\n                       ---------- abs_rel ----------    P(prior closer than model)")
    print("  alignment            all      median   where-model-wrong    all    where-wrong")
    for row in summary["alignments"]:
        print(
            f"  {row['alignment']:<20} {row['abs_rel']:.4f}   {row['abs_rel_median']:.4f}   "
            f"{row['abs_rel_where_model_wrong']:.4f}            "
            f"{row['p_closer']:.3f}   {row['p_closer_where_wrong']:.3f}"
        )
    print("\nA prior only helps where P(prior closer) > 0.5; supervising below that pulls the")
    print("model away from the truth on those pixels.")
    print("\nThe `raw` row measures the *units*, not the prior: COLMAP's scale is arbitrary, so a")
    print("scale-free prior scores ~1.0 there however good it is. It is here to show how much of")
    print("the rows below it is supplied by the fit. `scale only` is the rung a sparse-point")
    print("alignment runs at, and is where a disparity prior pays for having no shift term.")

    ordinal = summary["ordinal"]
    if ordinal["pairs"]:
        print(f"\nordinal agreement with ground truth over {ordinal['pairs']} pairs (no alignment):")
        print(f"  prior {100 * ordinal['prior_agreement']:.2f}%    " f"model {100 * ordinal['model_agreement']:.2f}%")
        if ordinal["model_wrong_frac"]:
            print(
                f"  on the {100 * ordinal['model_wrong_frac']:.2f}% of pairs the model orders "
                f"wrongly, the prior is right "
                f"{100 * ordinal['prior_agreement_where_model_wrong']:.2f}% of the time"
            )
        print("This is what the ordinal loss consumes. Measured across three OB3D scenes it did")
        print("*not* predict which prior trains better -- see docs/normal-supervision.md -- so")
        print("read it as a description of the term's input, not as a proxy for its outcome.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--backend", default="transformers", choices=sorted(BACKENDS))
    parser.add_argument(
        "--model",
        default=None,
        help="model id for the backend; defaults to the reference model for that backend",
    )
    parser.add_argument("--trim", type=float, default=0.2, help="fraction of worst residuals dropped before refit")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--pair-samples", type=int, default=32, help="pair offsets scored per frame")
    parser.add_argument("--json", default=None, help="also write the summary here, for the report figures")
    parser.add_argument("--scene", default=None, help="scene name recorded in the JSON summary")
    args = parser.parse_args()

    model_id = args.model or DEFAULT_MODELS[args.backend]
    summary = summarise(
        collect(
            args.checkpoint,
            args.scene_path,
            args.backend,
            model_id,
            args.trim,
            args.iters,
            args.max_frames,
            args.pair_samples,
        )
    )
    report(summary)
    if args.json:
        summary["checkpoint"] = args.checkpoint
        summary["scene"] = args.scene
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
