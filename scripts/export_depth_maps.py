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

"""Render a checkpoint's depth maps into a folder that `tools/depthrecall` can score.

This is the only piece of the depth-recall evaluation that knows about 3dgrut; the metric
itself is a standalone package with no dependency on this repo, so it can score any method
that produces depth maps.

Three things have to cross the boundary intact, and each has a silent failure mode:

- **The depth convention.** The tracers emit Euclidean ray distance, not z-depth, so the
  manifest records `"ray"` and `depthrecall` is told the same. A plane is not constant depth
  under this convention, so scoring ray distance as z-depth produces an error that grows
  towards the image corners: a plausible curve rather than a crash.
- **The camera model.** `depthrecall` projects points with a pinhole `K`. Rather than
  rebuilding `K` from the dataset's intrinsics -- a chain of downscale factors and distortion
  parameters, any one of which could be misread -- this script *fits* a pinhole model to the
  ray directions the renderer actually used, and refuses to export a view whose fit residual
  exceeds `--max-pinhole-residual` pixels. A distorted or fisheye camera therefore fails
  loudly instead of being scored against the wrong projection.
- **The world space.** `normalize_world_space` rescales and recentres the poses, so rendered
  depth is in normalized units while a ground-truth scan is not. The transform is exported to
  `alignment.npy` for `depthrecall --alignment`; without it the scan sits at the wrong scale
  and recall is near zero for reasons unrelated to reconstruction quality.

Usage:
    python scripts/export_depth_maps.py --checkpoint runs/x/ckpt_last.pt --out-dir /tmp/depths
    python -m depthrecall --manifest /tmp/depths/manifest.json --ply scan.ply \\
        --depth-convention ray --alignment /tmp/depths/alignment.npy --taus 1,2,5 -o m.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# A projected pixel that is off by this much would move the sampled depth by roughly one
# pixel of depth gradient, which is the resolution the metric itself has.
DEFAULT_MAX_PINHOLE_RESIDUAL = 0.1


def fit_pinhole(rays_dir: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a pinhole K to camera-space ray directions laid out as (H, W, 3).

    Returns (K, max_residual_pixels). The residual is what makes this trustworthy: a camera
    the pinhole model cannot represent shows up as a large number rather than as a subtly
    wrong K.
    """
    H, W = rays_dir.shape[:2]
    d = rays_dir.astype(np.float64)
    z = d[..., 2]
    if np.any(z <= 0):
        raise ValueError("Ray directions must have positive z in camera space to fit a pinhole model")
    x_over_z = d[..., 0] / z
    y_over_z = d[..., 1] / z

    # Pixel centres, in COLMAP's corner-based convention: pixel (0, 0) is centred at (0.5, 0.5).
    u = np.broadcast_to(np.arange(W, dtype=np.float64) + 0.5, (H, W))
    v = np.broadcast_to((np.arange(H, dtype=np.float64) + 0.5)[:, None], (H, W))

    # u = fx * (x/z) + cx, solved by least squares over every pixel independently in each axis.
    def solve(ratio: np.ndarray, target: np.ndarray) -> tuple[float, float, np.ndarray]:
        A = np.stack([ratio.ravel(), np.ones(ratio.size)], axis=1)
        (f, c), *_ = np.linalg.lstsq(A, target.ravel(), rcond=None)
        return float(f), float(c), A @ np.array([f, c]) - target.ravel()

    fx, cx, res_u = solve(x_over_z, u)
    fy, cy, res_v = solve(y_over_z, v)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, float(max(np.abs(res_u).max(), np.abs(res_v).max()))


def world_to_camera(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 camera-to-world pose into COLMAP's (R, t): X_cam = R @ X_world + t."""
    R_c2w = pose[:3, :3].astype(np.float64)
    center = pose[:3, 3].astype(np.float64)
    R = R_c2w.T
    return R, -R @ center


def export(args) -> dict:
    from omegaconf import open_dict

    from threedgrut.datasets import make_test
    from threedgrut.datasets.protocols import get_dataset_world_transform
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.depth_normal_metrics import MIN_ACCUMULATED_OPACITY, expected_depth

    out_dir = Path(args.out_dir)
    depth_dir = out_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    if args.scene_path:
        conf.path = args.scene_path
    with open_dict(conf):
        # Evaluation-only: the reference depth is not needed for the export itself, but loading
        # it lets --report-gt-agreement cross-check the export against the dataset's own GT.
        conf.dataset.load_depth_gt = bool(args.report_gt_agreement)

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    dataset = make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=1, shuffle=False)

    views: list[dict] = []
    agreement: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = model(gpu_batch)

            # `expected_depth` returns ray distance along the *normalized* ray direction, with
            # `confident` false where too little opacity accumulated to place a surface. Those
            # pixels are written as 0, which depthrecall reads as "no surface" -- and counts as
            # a failure rather than excluding the point, so a method that renders nothing
            # cannot score well.
            depth, confident = expected_depth(outputs["pred_dist"], outputs["pred_opacity"], MIN_ACCUMULATED_OPACITY)
            valid = confident & (depth > 0)
            depth_np = np.where(
                valid.squeeze(-1).squeeze(0).cpu().numpy(),
                depth.squeeze(-1).squeeze(0).float().cpu().numpy(),
                0.0,
            ).astype(np.float32)

            rays_dir = gpu_batch.rays_dir[0].float().cpu().numpy()
            rays_ori = gpu_batch.rays_ori[0].float().cpu().numpy()
            if np.abs(rays_ori).max() > 1e-6:
                raise ValueError(
                    f"View {index} has non-zero ray origins in camera space (max "
                    f"{np.abs(rays_ori).max():.3g}); depthrecall assumes a single centre of "
                    "projection per view."
                )
            K, residual = fit_pinhole(rays_dir)
            if residual > args.max_pinhole_residual:
                raise ValueError(
                    f"View {index} is not a pinhole camera to within {args.max_pinhole_residual} px "
                    f"(residual {residual:.4f} px). depthrecall projects points with a pinhole K, so "
                    "scoring this camera would compare the wrong pixels. Undistort the dataset first."
                )

            pose = gpu_batch.T_to_world[0].float().cpu().numpy().astype(np.float64)
            R, t = world_to_camera(pose)

            name = f"{index:05d}"
            np.save(depth_dir / f"{name}.npy", depth_np)
            views.append(
                {
                    "name": name,
                    "width": int(depth_np.shape[1]),
                    "height": int(depth_np.shape[0]),
                    "K": K.tolist(),
                    "R": R.tolist(),
                    "t": t.tolist(),
                    "depth_path": f"depths/{name}.npy",
                    "depth_convention": "ray",
                    "pinhole_residual_px": residual,
                }
            )

            if args.report_gt_agreement and getattr(gpu_batch, "depth_gt", None) is not None:
                gt = gpu_batch.depth_gt.squeeze(-1).squeeze(0).float().cpu().numpy()
                both = (gt > 0) & (depth_np > 0)
                if both.any():
                    agreement.append(float(np.abs(depth_np[both] - gt[both]).mean()))

    (out_dir / "manifest.json").write_text(json.dumps({"views": views}, indent=2))

    # The transform from the source (COLMAP) world into the world the depth maps live in. It is
    # exactly what a ground-truth scan in source coordinates has to be multiplied by, so it is
    # written whether or not it is the identity -- an absent file would be indistinguishable
    # from a forgotten one.
    transform = get_dataset_world_transform(dataset)
    if transform is None:
        # None is returned both for "the transform is the identity" and for "this dataset does
        # not report one", which are not the same claim. Treating the second as identity would
        # export depth in a rescaled frame while asserting the scan needs no transform, and the
        # only symptom is a recall near zero. Only the first is safe to assume.
        from threedgrut.datasets.protocols import WorldTransformProvider

        if conf.dataset.get("normalize_world_space") and not isinstance(dataset, WorldTransformProvider):
            raise ValueError(
                f"{type(dataset).__name__} normalizes world space but does not implement "
                "get_world_normalization_transform, so the depth maps are in a frame this script "
                "cannot describe. Export with dataset.normalize_world_space=false, or implement "
                "the protocol."
            )
        transform = np.eye(4)
    np.save(out_dir / "alignment.npy", np.asarray(transform, dtype=np.float64))

    summary = {
        "views": len(views),
        "depth_convention": "ray",
        "max_pinhole_residual_px": max((v["pinhole_residual_px"] for v in views), default=0.0),
        "world_transform_is_identity": bool(np.allclose(transform, np.eye(4))),
        "out_dir": str(out_dir),
    }
    if agreement:
        summary["mean_abs_depth_vs_dataset_gt"] = float(np.mean(agreement))
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-path", default=None, help="override the checkpoint's dataset path")
    parser.add_argument(
        "--max-pinhole-residual",
        type=float,
        default=DEFAULT_MAX_PINHOLE_RESIDUAL,
        help="reject a view whose rays deviate from a pinhole model by more than this (pixels)",
    )
    parser.add_argument(
        "--report-gt-agreement",
        action="store_true",
        help="also load the dataset's reference depth and report mean absolute agreement",
    )
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2))


if __name__ == "__main__":
    main()
