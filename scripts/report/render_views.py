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

"""Render one checkpoint's RGB / depth / normal buffers and their error maps, as PNGs.

Deliberately one checkpoint per process. `render.enable_depth_variance`, `enable_normals` and
`primitive_type` are compiled into `lib3dgut_cc`, and a process holds exactly one of those
binaries, so rendering a depth-variance variant and a plain one in the same process would either
raise or -- before that guard existed -- silently render the second against the first's binary.
The driver loops over variants with subprocesses for that reason; see `make_report.py`.

Colour scales are passed in rather than fitted per image, because a per-image scale makes two
variants look different when only their range changed. `--depth-range` and the fixed error caps
below are what make these panels comparable across variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Error-map caps, shared by every panel so the colours mean the same thing in each. Depth is
# capped at the delta1 boundary, so anything saturated is a pixel the metrics count as wrong.
DEPTH_ERR_CAP = 0.25
NORMAL_ERR_CAP = 90.0

# Percentiles of valid reference depth used to derive a scene's depth colour range, when one is
# not supplied. Clipping the tails keeps the sky sentinel and a few near-camera pixels from
# flattening everything else.
DEPTH_RANGE_PERCENTILES = (2.0, 98.0)


def _turbo(values: np.ndarray) -> np.ndarray:
    """Turbo colormap on [0, 1], via matplotlib. Returns float RGB in [0, 1]."""
    import matplotlib

    return matplotlib.colormaps["turbo"](np.clip(values, 0.0, 1.0))[..., :3]


def _save(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def _encode_normals(normals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map unit normals to RGB by the usual n*0.5+0.5, with invalid pixels black.

    Applied identically to rendered and reference normals: both are world-space in this
    codebase, so the same encoding makes them directly comparable by eye.
    """
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit = np.divide(normals, np.maximum(norm, 1e-8))
    return np.where(valid[..., None], unit * 0.5 + 0.5, 0.0)


def render(args) -> dict:
    from omegaconf import open_dict

    from threedgrut.datasets import make_test
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.depth_normal_metrics import (
        MIN_ACCUMULATED_OPACITY,
        expected_depth,
        normal_metrics,
        reference_depth_validity,
        world_view_dirs,
    )
    from threedgrut.utils.render import apply_background

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    if args.scene_path:
        conf.path = args.scene_path
    # The reference maps are needed for the error panels; the training split of these runs did
    # not read them, and enabling them here is evaluation-only.
    with open_dict(conf):
        conf.dataset.load_depth_gt = True
        conf.dataset.load_normal_gt = True

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    dataset = make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=1, shuffle=False)

    wanted = set(args.frames)
    summary: dict[str, dict] = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index not in wanted:
                continue
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = model(gpu_batch)
            # Composite the background exactly as validation does, so the RGB panel is the
            # image the reported PSNR was computed on rather than the raw accumulation.
            outputs = apply_background(model.background, outputs, gpu_batch, training=False)

            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], MIN_ACCUMULATED_OPACITY)
            depth_gt = gpu_batch.depth_gt
            valid_t = reference_depth_validity(depth_gt) & confident & (depth > 0)

            depth_np = depth.squeeze(-1).squeeze(0).float().cpu().numpy()
            gt_np = depth_gt.squeeze(-1).squeeze(0).float().cpu().numpy()
            valid = valid_t.squeeze(-1).squeeze(0).cpu().numpy()

            lo, hi = args.depth_range if args.depth_range else np.percentile(gt_np[valid], DEPTH_RANGE_PERCENTILES)
            span = max(float(hi) - float(lo), 1e-8)

            prefix = f"{args.tag}_f{index}"
            # `pred_features` is the radiance channel; it is named for the general case where a
            # decoder produces more than three channels, and is RGB once composited.
            rgb = outputs["pred_features"].squeeze(0).clamp(0, 1).float().cpu().numpy()
            _save(out_dir / f"{prefix}_rgb.png", rgb)
            _save(out_dir / f"{prefix}_depth.png", _turbo((depth_np - float(lo)) / span))

            # Relative error, the same quantity abs_rel averages, so a saturated pixel here is
            # one the reported metric is penalising.
            rel = np.where(valid, np.abs(depth_np - gt_np) / np.maximum(gt_np, 1e-8), 0.0)
            _save(out_dir / f"{prefix}_depth_err.png", np.where(valid[..., None], _turbo(rel / DEPTH_ERR_CAP), 0.0))

            frame: dict = {
                "depth_range": [float(lo), float(hi)],
                "abs_rel": float(rel[valid].mean()),
                "valid_px": int(valid.sum()),
            }

            normal_gt = getattr(gpu_batch, "normal_gt", None)
            if "pred_normals" in outputs and normal_gt is not None and normal_gt.numel():
                pred_n = outputs["pred_normals"]
                normals_np = pred_n.squeeze(0).float().cpu().numpy()
                gt_n_np = normal_gt.squeeze(0).float().cpu().numpy()
                # Reference normals are only meaningful where the reference has a surface;
                # `normal_metrics` uses the same condition, so the panels and the numbers agree.
                normal_valid = (np.linalg.norm(gt_n_np, axis=-1) > 0.5) & valid
                _save(out_dir / f"{prefix}_normal.png", _encode_normals(normals_np, normal_valid))
                _save(out_dir / f"{prefix}_normal_gt.png", _encode_normals(gt_n_np, normal_valid))

                cos = (normals_np * gt_n_np).sum(-1) / np.maximum(
                    np.linalg.norm(normals_np, axis=-1) * np.linalg.norm(gt_n_np, axis=-1), 1e-8
                )
                ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                _save(
                    out_dir / f"{prefix}_normal_err.png",
                    np.where(normal_valid[..., None], _turbo(ang / NORMAL_ERR_CAP), 0.0),
                )
                view = world_view_dirs(gpu_batch.rays_dir, gpu_batch.T_to_world)
                # `normal_metrics` takes the mask without the trailing channel, matching the
                # `[B, H, W]` shape of the reference normal's own length.
                frame.update(normal_metrics(pred_n, normal_gt, valid_t.squeeze(-1), view))

            # Reference panels, written once per frame: they do not depend on the variant, but
            # they must use this frame's colour range to sit next to the rendered ones.
            _save(out_dir / f"gt_f{index}_depth.png", np.where(valid[..., None], _turbo((gt_np - lo) / span), 0.0))
            if gpu_batch.rgb_gt is not None:
                _save(out_dir / f"gt_f{index}_rgb.png", gpu_batch.rgb_gt.squeeze(0).clamp(0, 1).float().cpu().numpy())

            summary[str(index)] = frame

    (out_dir / f"{args.tag}_frames.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True, help="prefix for this variant's PNGs")
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--scene-path", default=None, help="override the checkpoint's dataset path")
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="shared depth colour range; without it, percentiles of this frame's reference depth",
    )
    args = parser.parse_args()
    summary = render(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
