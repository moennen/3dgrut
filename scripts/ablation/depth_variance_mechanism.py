# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Where does a variance-trained model put the weight it removed from the tail?

The depth-variance term measurably reduces per-ray spread and measurably worsens depth. That
combination means the expected depth moved *away* from the reference while becoming more
confident, which is worth localising rather than narrating: it is equally the signature of a
degenerate optimum and of a sign error somewhere in the accumulation.

Two competing explanations, and the numbers that separate them.

1. The term is correct and the target is degenerate. Zero variance is achievable at *any*
   distance, so the term expresses no preference for the right one. Front-to-back compositing
   makes the nearest hit the cheapest place to collapse onto, because raising one particle's
   opacity drives the transmittance -- and therefore every downstream weight -- to zero at a
   stroke. The prediction is a *confidently wrong* population: rays that are both low-spread
   and outside delta1, concentrated nearer than the reference.
2. Something is wrong in the accumulation. The prediction is inconsistency between the two
   moments, visible as a negative raw variance on more than a float-noise fraction of rays,
   or as a spread that does not track the depth's own units.

Both are checked here, on the same views, for any number of checkpoints.

Usage:
    python scripts/ablation/depth_variance_mechanism.py \
        --checkpoint /tmp/abl/abl_gaussian_sponza/*/ours_7000/ckpt_7000.pt --label baseline \
        --checkpoint /tmp/abl/abl_dv1_gaussian_sponza/*/ours_7000/ckpt_7000.pt --label dv1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from threedgrut.model.model import MixtureOfGaussians  # noqa: E402
from threedgrut.utils.depth_normal_metrics import (  # noqa: E402
    DELTA_THRESHOLDS,
    FLOATER_RATIO,
    MIN_ACCUMULATED_OPACITY,
    expected_depth,
    reference_depth_validity,
)

# A ray whose weight sits within this relative spread of its own expected depth has
# effectively committed to a surface. Applied to every checkpoint alike so the populations
# are comparable; the exact value only sets where "confident" starts.
CONFIDENT_SPREAD = 0.02


def collect(checkpoint_path: str, scene_path: str | None) -> dict:
    from omegaconf import open_dict

    from threedgrut.datasets import make_test

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    if scene_path:
        conf.path = scene_path
    with open_dict(conf):
        conf.render.enable_depth_variance = True

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    dataset = make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)

    spread_all, signed_all, ratio_all, opacity_all, negative_all, absvar_all = [], [], [], [], [], []
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
            opacity = outputs["pred_opacity"]
            safe = opacity.clamp_min(1e-6)
            # The *unclamped* variance, so an inconsistency between the two accumulators shows
            # up as a negative value instead of being silently floored at zero.
            raw_variance = outputs["pred_dist_sq"] / safe - depth**2
            variance = raw_variance.clamp_min(0.0)

            valid = reference_depth_validity(depth_gt) & confident & (depth > 0)
            mask = valid.squeeze(-1)
            if not bool(mask.any()):
                continue

            gt = depth_gt.squeeze(-1)[mask].double()
            pred = depth.squeeze(-1)[mask].double()
            std = variance.sqrt().squeeze(-1)[mask].double()

            spread_all.append((std / pred.clamp_min(1e-6)).cpu())
            absvar_all.append(std.cpu())
            signed_all.append(((pred - gt) / gt).cpu())
            ratio_all.append(torch.maximum(pred / gt, gt / pred).cpu())
            opacity_all.append(opacity.squeeze(-1)[mask].double().cpu())
            # Relative to the scale of the two terms being differenced, so the test is about
            # cancellation rather than about world units.
            scale = (outputs["pred_dist_sq"] / safe).squeeze(-1)[mask].double().abs().clamp_min(1e-12)
            negative_all.append(((raw_variance.squeeze(-1)[mask].double() / scale) < -1e-3).cpu())

    spread = torch.cat(spread_all)
    signed = torch.cat(signed_all)
    ratio = torch.cat(ratio_all)
    opacity = torch.cat(opacity_all)
    negative = torch.cat(negative_all)
    std = torch.cat(absvar_all)

    wrong = ratio >= DELTA_THRESHOLDS[0]
    tight = spread < CONFIDENT_SPREAD
    floater = signed < (FLOATER_RATIO - 1.0)  # pred < 0.5 * gt

    return {
        "checkpoint": checkpoint_path,
        "pixels": int(spread.numel()),
        # Did the term do its job?
        "spread_median": float(spread.median()),
        "std_median_world": float(std.median()),
        # Did the mean move, and which way?
        "signed_rel_err_mean": float(signed.mean()),
        "signed_rel_err_median": float(signed.median()),
        "abs_rel_err_mean": float(signed.abs().mean()),
        "delta1_fail_frac": float(wrong.double().mean()),
        "floater_frac": float(floater.double().mean()),
        # Explanation 1: a confidently wrong population, and where it sits.
        "tight_frac": float(tight.double().mean()),
        "tight_and_wrong_frac": float((tight & wrong).double().mean()),
        "wrong_given_tight": float(wrong[tight].double().mean()) if bool(tight.any()) else float("nan"),
        "signed_err_given_tight": float(signed[tight].mean()) if bool(tight.any()) else float("nan"),
        "signed_err_given_tight_and_wrong": (
            float(signed[tight & wrong].mean()) if bool((tight & wrong).any()) else float("nan")
        ),
        "opacity_mean": float(opacity.mean()),
        # Explanation 2: the accumulators disagreeing.
        "negative_variance_frac": float(negative.double().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--scene-path", default=None, help="Override the scene path in the checkpoint config")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if len(args.checkpoint) != len(args.label):
        raise SystemExit("--checkpoint and --label must be given the same number of times")

    rows = []
    for path, label in zip(args.checkpoint, args.label):
        row = collect(path, args.scene_path)
        row["label"] = label
        rows.append(row)
        print(f"{label}: {json.dumps({k: v for k, v in row.items() if k != 'checkpoint'}, indent=2)}")

    columns = [
        ("spread_median", "spread"),
        ("negative_variance_frac", "neg_var"),
        ("signed_rel_err_mean", "signed_err"),
        ("abs_rel_err_mean", "abs_rel"),
        ("delta1_fail_frac", "d1_fail"),
        ("floater_frac", "float"),
        ("tight_frac", "tight"),
        ("wrong_given_tight", "wrong|tight"),
        ("signed_err_given_tight", "sgn|tight"),
        ("opacity_mean", "opacity"),
    ]
    print("\n| label | " + " | ".join(header for _, header in columns) + " |")
    print("|" + "---|" * (len(columns) + 1))
    for row in rows:
        print(f"| {row['label']} | " + " | ".join(f"{row[key]:.4f}" for key, _ in columns) + " |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
