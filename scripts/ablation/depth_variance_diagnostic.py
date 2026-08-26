# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Does the spread of a ray's weight distribution predict where its depth is wrong?

Penalising per-ray spread (item 7) is only worth a backward pass if the spread actually
marks the pixels we want to fix. This measures that on a trained model, without training
anything: `render.enable_depth_variance` is a forward-only flag, so any existing checkpoint
can be re-rendered with it.

Reported per checkpoint, over pixels with reference depth and enough accumulated opacity for
a depth to mean anything:

- `spearman_spread_vs_relerr`: rank correlation between relative spread and relative depth
  error. Rank, not Pearson, because the depth error distribution is heavy-tailed and we care
  about ordering pixels, not about a linear fit.
- `auc_floater`, `auc_delta1_fail`: how well relative spread alone ranks the bad pixels ahead
  of the good ones (0.5 is useless, 1.0 is perfect). This is the number that decides item 7:
  a penalty on spread can only reach the failures that spread can see.

Usage:
    python scripts/ablation/depth_variance_diagnostic.py \
        --checkpoint /tmp/abl_baseline/abl_gaussian_sponza/*/ours_7000/ckpt_7000.pt \
        --label baseline_gaussian_sponza --out-dir /tmp/dv_diag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from threedgrut.datasets.utils import DEFAULT_DEVICE  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians  # noqa: E402
from threedgrut.utils.depth_normal_metrics import (  # noqa: E402
    DELTA_THRESHOLDS,
    FLOATER_RATIO,
    MIN_ACCUMULATED_OPACITY,
    expected_depth,
    reference_depth_validity,
)

# Cap on pixels kept for the rank statistics. Ranking is O(n log n) and the estimates are
# already tight at this size; the cap keeps a many-view scene from exhausting memory.
MAX_PIXELS = 4_000_000


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Average ranks, so ties (many pixels share a spread of exactly 0) are not ordered."""
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
    # Average the ranks within each run of equal values.
    unique, inverse, counts = torch.unique(sorted_values, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    mean_rank = (starts + ends + 1).to(values.dtype) / 2.0
    ranks[order] = mean_rank[inverse]
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if float(denom) > 0 else float("nan")


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    """Mann-Whitney U: the chance a random positive outranks a random negative."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def collect(checkpoint_path: str, scene_path: str | None) -> dict:
    from threedgrut.datasets import make_test

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    if scene_path:
        conf.path = scene_path
    # Forward-only flag, so a model trained without it re-renders unchanged apart from the
    # extra buffer. `open_dict` because a checkpoint saved before the flag existed has a
    # struct config that rejects the new key outright. `enable_normals` is left at whatever
    # the checkpoint trained with: forcing it off changes the compiled variant for no benefit
    # here, and a model trained with `use_depth_normal` rejects the combination outright.
    from omegaconf import open_dict

    with open_dict(conf):
        conf.render.enable_depth_variance = True

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    dataset = make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)

    spreads, rel_errs, floaters, delta1_fails, depths = [], [], [], [], []
    # Controls. Spread is only worth a new accumulator if it beats signals already available
    # from the existing buffers: an incompletely opaque ray, and an occlusion boundary in the
    # rendered depth. Both are plausible confounders -- a high-spread pixel tends to be both.
    transparencies, depth_grads = [], []
    frames = 0
    with torch.no_grad():
        for batch in loader:
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            depth_gt = getattr(gpu_batch, "depth_gt", None)
            if depth_gt is None or depth_gt.numel() == 0:
                raise SystemExit(f"{checkpoint_path}: dataset provides no reference depth")
            outputs = model(gpu_batch)
            if outputs["pred_dist_sq"].numel() == 0:
                raise SystemExit("pred_dist_sq is empty: render.enable_depth_variance did not take effect")

            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], MIN_ACCUMULATED_OPACITY)
            opacity = outputs["pred_opacity"].clamp_min(1e-6)
            # Same normalisation as the first moment, so the two are on the same footing.
            second = outputs["pred_dist_sq"] / opacity
            variance = (second - depth**2).clamp_min(0.0)
            spread = variance.sqrt() / depth.clamp_min(1e-6)

            valid = reference_depth_validity(depth_gt) & confident & (depth > 0)
            mask = valid.squeeze(-1)
            if not bool(mask.any()):
                continue
            gt = depth_gt.squeeze(-1)[mask].double()
            pred = depth.squeeze(-1)[mask].double()
            ratio = torch.maximum(pred / gt, gt / pred)

            # Relative depth gradient: an occlusion boundary, where an expected depth is
            # least likely to land on a surface. Same quantity the metrics use for their
            # high-gradient tail.
            d = depth.squeeze(-1) if depth.dim() == 4 else depth
            dy = torch.zeros_like(d)
            dx = torch.zeros_like(d)
            dy[:, 1:, :] = (d[:, 1:, :] - d[:, :-1, :]).abs()
            dx[:, :, 1:] = (d[:, :, 1:] - d[:, :, :-1]).abs()
            grad = (dx + dy) / d.clamp_min(1e-6)

            spreads.append(spread.squeeze(-1)[mask].double().cpu())
            rel_errs.append(((pred - gt).abs() / gt).cpu())
            floaters.append((pred < FLOATER_RATIO * gt).cpu())
            delta1_fails.append((ratio >= DELTA_THRESHOLDS[0]).cpu())
            depths.append(pred.cpu())
            transparencies.append((1.0 - outputs["pred_opacity"].squeeze(-1)[mask]).double().cpu())
            depth_grads.append(grad[mask].double().cpu())
            frames += 1

    spread = torch.cat(spreads)
    rel_err = torch.cat(rel_errs)
    floater = torch.cat(floaters)
    delta1_fail = torch.cat(delta1_fails)
    transparency = torch.cat(transparencies)
    depth_grad = torch.cat(depth_grads)

    if spread.numel() > MAX_PIXELS:
        keep = torch.randperm(spread.numel(), generator=torch.Generator().manual_seed(0))[:MAX_PIXELS]
        spread, rel_err, floater = spread[keep], rel_err[keep], floater[keep]
        delta1_fail, transparency, depth_grad = delta1_fail[keep], transparency[keep], depth_grad[keep]

    device = DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
    spread_d, rel_err_d = spread.to(device), rel_err.to(device)
    floater_d, delta1_d = floater.to(device), delta1_fail.to(device)
    transparency_d, depth_grad_d = transparency.to(device), depth_grad.to(device)

    quantiles = torch.tensor([0.1, 0.5, 0.9, 0.99], dtype=torch.float64)
    return {
        "checkpoint": checkpoint_path,
        "frames": frames,
        "pixels": int(spread.numel()),
        "spread_p10": float(spread.quantile(quantiles[0])),
        "spread_median": float(spread.quantile(quantiles[1])),
        "spread_p90": float(spread.quantile(quantiles[2])),
        "spread_p99": float(spread.quantile(quantiles[3])),
        "spread_zero_frac": float((spread <= 1e-6).double().mean()),
        "floater_frac": float(floater.double().mean()),
        "delta1_fail_frac": float(delta1_fail.double().mean()),
        "mean_rel_err": float(rel_err.mean()),
        "spearman_spread_vs_relerr": _spearman(spread_d, rel_err_d),
        "auc_floater": _auc(spread_d, floater_d),
        "auc_delta1_fail": _auc(spread_d, delta1_d),
        # Controls: the same AUC from signals that need no new accumulator.
        "auc_floater_transparency": _auc(transparency_d, floater_d),
        "auc_floater_depthgrad": _auc(depth_grad_d, floater_d),
        "auc_delta1_transparency": _auc(transparency_d, delta1_d),
        "auc_delta1_depthgrad": _auc(depth_grad_d, delta1_d),
        "mean_spread_floater": float(spread[floater].mean()) if bool(floater.any()) else float("nan"),
        "mean_spread_ok": float(spread[~floater].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scene-path", default=None, help="override the scene path in the checkpoint config")
    parser.add_argument("--out-dir", default="/tmp/dv_diag")
    args = parser.parse_args()

    result = collect(args.checkpoint, args.scene_path)
    result["label"] = args.label
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
