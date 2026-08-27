# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a *regression* loss on a pseudo-depth prior could actually be aligned with.

`pseudo_depth_diagnostic.py` answers "which prior", and it fits every alignment against
**ground-truth depth**, which makes its numbers an upper bound rather than a plan. This script
answers the two questions that stand between that upper bound and a working L1 term:

1. **Where do the alignment parameters come from?** A trained loss cannot see ground truth. The
   honest source is the COLMAP sparse points already used to initialise the scene, so every
   alignment here is fitted twice -- once on ground truth, once on the sparse points visible in
   that frame -- and the gap between them is the cost of the alignment being *estimable*.

2. **How much freedom does the alignment need?** A *per-frame* fit is the obvious choice and the
   wrong default for a metric prior: it hands every frame its own scale and offset, which is
   exactly the per-frame freedom that makes a monocular prior mutually inconsistent across views.
   A metric prior should need one alignment for the entire scene -- COLMAP units to metres -- and
   if it does, the depth it supervises with is automatically multi-view consistent. That is a
   claim about the checkpoint, not the loss, so it is measured: `global` fits one alignment across
   all frames, `per-frame` fits one each, and the gap says whether "metric" survives contact with
   these scenes.

Everything is scored the way the renderer sees it. `depth_gt` and the tracer's depth are
**Euclidean ray distance**, while DA3 emits **z**, so alignment is fitted in z (the prior's own
space, where its ambiguity is affine) and the result is converted per pixel before scoring. The
sparse points are converted the same way, and the chain is validated rather than assumed: their
own Euclidean depth is compared against `depth_gt` at the same pixel, and a large disagreement
there means the transform is wrong and no other number in the output should be believed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo_depth_diagnostic import _fit_affine, _fit_scale, _predict  # noqa: E402

from threedgrut.datasets.pseudo_depth import BACKENDS  # noqa: E402
from threedgrut.datasets.utils import qvec_to_so3  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians  # noqa: E402
from threedgrut.utils.depth_normal_metrics import (  # noqa: E402
    MIN_ACCUMULATED_OPACITY,
    expected_depth,
    reference_depth_validity,
)

# (label, mode, scope). `scope` is what the fit is shared over: one alignment per frame, or one
# for the whole scene. The per-frame rows are the upper bound a regression loss could reach; the
# global rows are the ones that leave the prior multi-view consistent.
VARIANTS = (
    ("scale, per-frame", "scale", "frame"),
    ("scale, global", "scale", "global"),
    ("affine, per-frame", "affine", "frame"),
    ("affine, global", "affine", "global"),
)

# Fitting `target ~ a * prior (+ b)` needs enough points that the fit is not interpolating noise.
# COLMAP frames vary by an order of magnitude in how many points they see, so a frame below this
# is reported as unfittable rather than silently given a garbage alignment.
MIN_SPARSE_POINTS = {"scale": 8, "affine": 16}


def read_points3d_with_ids(sparse_dir: Path) -> dict[int, np.ndarray]:
    """``{point_id: xyz}`` from COLMAP's points3D file.

    The repo's own readers in `threedgrut/datasets/utils.py` drop the ids -- they only ever need
    the cloud for initialisation -- and the id is precisely what links a point to the keypoint
    that observed it, so this reads the file again rather than extending them.
    """
    binary, text = sparse_dir / "points3D.bin", sparse_dir / "points3D.txt"
    points: dict[int, np.ndarray] = {}
    if binary.exists():
        with open(binary, "rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                point_id, x, y, z = struct.unpack("<QdddBBBd", handle.read(43))[:4]
                (track_length,) = struct.unpack("<Q", handle.read(8))
                handle.read(8 * track_length)
                points[int(point_id)] = np.array([x, y, z], dtype=np.float64)
        return points
    if not text.exists():
        raise SystemExit(f"no points3D.bin or points3D.txt under {sparse_dir}")
    for line in text.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        points[int(fields[0])] = np.array([float(v) for v in fields[1:4]], dtype=np.float64)
    return points


def sparse_observations(image, points3d, scaling_factor: float, world_scale: float, shape) -> np.ndarray:
    """The frame's sparse points as rows of ``(col, row, z_cam, dist_cam)``.

    `image.xys` are pixels at COLMAP's *full* resolution while everything else here is at the
    loaded resolution, so they are divided by the dataset's downscale factor. `z_cam` is what the
    prior is fitted against (the prior predicts z); `dist_cam` is Euclidean and is what `depth_gt`
    can be checked against.
    """
    height, width = shape
    rotation, translation = qvec_to_so3(image.qvec), np.asarray(image.tvec, dtype=np.float64)
    rows = []
    for (x_full, y_full), point_id in zip(image.xys, image.point3D_ids):
        xyz = points3d.get(int(point_id))
        if point_id == -1 or xyz is None:
            continue
        camera = (rotation @ xyz + translation) * world_scale
        if camera[2] <= 0.0:  # behind the camera; COLMAP tracks can include these
            continue
        col, row = x_full / scaling_factor, y_full / scaling_factor
        if not (0 <= col < width and 0 <= row < height):
            continue
        rows.append((col, row, camera[2], float(np.linalg.norm(camera))))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def _apply(prior_z: np.ndarray, mode: str, coef) -> np.ndarray:
    return coef[0] * prior_z + coef[1] if mode == "affine" else coef * prior_z


def _fit(prior_values: np.ndarray, target_values: np.ndarray, mode: str, trim: float, iters: int):
    if mode == "affine":
        return _fit_affine(prior_values, target_values, trim, iters)
    return _fit_scale(prior_values, target_values, trim, iters)


def _abs_rel(predicted_dist: np.ndarray, gt_dist: np.ndarray) -> np.ndarray:
    return np.abs(predicted_dist - gt_dist) / np.maximum(gt_dist, 1e-8)


def collect(
    checkpoint_path: str,
    scene_path: str | None,
    backend: str,
    model_id: str,
    trim: float,
    iters: int,
    max_frames: int,
    gt_samples: int,
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
    if quantity != "depth":
        raise SystemExit(
            f"{model_id} emits {quantity}; this script fits in z and is only meaningful for a "
            "depth prior. Use pseudo_depth_diagnostic.py for a disparity prior."
        )

    # The sparse points live in raw COLMAP world units. `depth_gt` is scaled by the world
    # normalization's similarity scale when that is enabled, so the points must be too.
    world_scale = 1.0
    if getattr(dataset, "normalize_world_space", False):
        from threedgrut.datasets.gt_geometry import similarity_scale

        world_scale = float(similarity_scale(dataset.world_normalization_transform))
    sparse_dir = Path(conf.path) / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = Path(conf.path) / "colmap"
    points3d = read_points3d_with_ids(sparse_dir)

    frames: list[dict] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if len(frames) >= max_frames:
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

            rays_dir = gpu_batch.rays_dir.squeeze(0)
            to_z = (rays_dir[..., 2] / rays_dir.norm(dim=-1)).abs().clamp_min(1e-8)
            prior = _predict(predictor, gpu_batch.rgb_gt.squeeze(0))

            valid_np = valid.cpu().numpy()
            gt_dist = depth_gt.squeeze(-1).squeeze(0).double().cpu().numpy()
            model_dist = depth.squeeze(-1).squeeze(0).double().cpu().numpy()
            to_z_np = to_z.double().cpu().numpy()
            prior_np = prior.double().cpu().numpy()

            # `xys` are at COLMAP's full resolution; the loaded frame may be downsampled. The
            # dataset computes this factor as a local (dataset_colmap.py:539) and does not keep
            # it, so it is recovered the same way, per frame, from the frame's own intrinsic.
            image = dataset.cam_extrinsics[index]
            scaling_factor = dataset.cam_intrinsics[image.camera_id].height / prior_np.shape[0]
            observations = sparse_observations(image, points3d, scaling_factor, world_scale, prior_np.shape)
            frames.append(
                {
                    "valid": valid_np,
                    "gt_dist": gt_dist,
                    "gt_z": gt_dist * to_z_np,
                    "model_dist": model_dist,
                    "to_z": to_z_np,
                    "prior": prior_np,
                    "sparse": observations,
                }
            )

    if not frames:
        raise SystemExit("no frames with usable reference depth")
    return summarise(frames, trim, iters, gt_samples, model_id, backend, conf)


def _sparse_fit_pairs(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(prior_z, target_z, gt_dist_at_point)`` at the frame's sparse observations."""
    observations = frame["sparse"]
    if not len(observations):
        return (np.empty(0),) * 3
    cols = np.clip(np.round(observations[:, 0]).astype(int), 0, frame["prior"].shape[1] - 1)
    rows = np.clip(np.round(observations[:, 1]).astype(int), 0, frame["prior"].shape[0] - 1)
    keep = frame["valid"][rows, cols]
    return (
        frame["prior"][rows[keep], cols[keep]],
        observations[keep, 2],
        frame["gt_dist"][rows[keep], cols[keep]],
    )


def summarise(frames, trim: float, iters: int, gt_samples: int, model_id: str, backend: str, conf) -> dict:
    rng = np.random.default_rng(0)
    model_error = np.concatenate([_abs_rel(f["model_dist"], f["gt_dist"])[f["valid"]] for f in frames])

    # Validate the transform chain before anything is built on it: the sparse points carry their
    # own Euclidean depth, and it must agree with `depth_gt` sampled at the same pixel.
    check, counts = [], []
    for frame in frames:
        observations = frame["sparse"]
        counts.append(int(len(observations)))
        if not len(observations):
            continue
        cols = np.clip(np.round(observations[:, 0]).astype(int), 0, frame["prior"].shape[1] - 1)
        rows = np.clip(np.round(observations[:, 1]).astype(int), 0, frame["prior"].shape[0] - 1)
        keep = frame["valid"][rows, cols]
        if keep.any():
            check.append(_abs_rel(observations[keep, 3], frame["gt_dist"][rows[keep], cols[keep]]))
    check_all = np.concatenate(check) if check else np.zeros(1)

    # Pixels the global fits are estimated from, subsampled per frame so one large frame cannot
    # dominate the single alignment shared by all of them.
    gt_pool_prior, gt_pool_target, sparse_pool_prior, sparse_pool_target = [], [], [], []
    for frame in frames:
        fit_valid = frame["valid"] & np.isfinite(frame["gt_z"])
        indices = np.flatnonzero(fit_valid.ravel())
        if len(indices) > gt_samples:
            indices = rng.choice(indices, gt_samples, replace=False)
        gt_pool_prior.append(frame["prior"].ravel()[indices])
        gt_pool_target.append(frame["gt_z"].ravel()[indices])
        prior_values, target_values, _ = _sparse_fit_pairs(frame)
        sparse_pool_prior.append(prior_values)
        sparse_pool_target.append(target_values)
    pools = {
        "ground truth": (np.concatenate(gt_pool_prior), np.concatenate(gt_pool_target)),
        "sparse points": (np.concatenate(sparse_pool_prior), np.concatenate(sparse_pool_target)),
    }

    rows = []
    for label, mode, scope in VARIANTS:
        for source in ("ground truth", "sparse points"):
            global_coef = _fit(*pools[source], mode, trim, iters) if scope == "global" else None
            errors, model_errors, unfittable = [], [], 0
            for frame in frames:
                if scope == "global":
                    coef = global_coef
                elif source == "ground truth":
                    fit_valid = frame["valid"] & np.isfinite(frame["gt_z"])
                    coef = _fit(frame["prior"][fit_valid], frame["gt_z"][fit_valid], mode, trim, iters)
                else:
                    prior_values, target_values, _ = _sparse_fit_pairs(frame)
                    if len(prior_values) < MIN_SPARSE_POINTS[mode]:
                        unfittable += 1
                        continue
                    coef = _fit(prior_values, target_values, mode, trim, iters)
                predicted = _apply(frame["prior"], mode, coef) / frame["to_z"]
                errors.append(_abs_rel(predicted, frame["gt_dist"])[frame["valid"]])
                model_errors.append(_abs_rel(frame["model_dist"], frame["gt_dist"])[frame["valid"]])
            joined = np.concatenate(errors) if errors else np.full(1, np.nan)
            # Paired against the model on exactly the pixels that contributed, so a variant that
            # dropped frames as unfittable is not compared against the model's error elsewhere.
            paired_model = np.concatenate(model_errors) if model_errors else np.full(1, np.nan)
            rows.append(
                {
                    "alignment": label,
                    "mode": mode,
                    "scope": scope,
                    "fitted_on": source,
                    "abs_rel": float(np.nanmean(joined)),
                    "abs_rel_median": float(np.nanmedian(joined)),
                    "p_closer": float(np.nanmean(joined < paired_model)),
                    "frames_scored": len(errors),
                    "frames_unfittable": unfittable,
                }
            )

    return {
        "backend": backend,
        "model_id": model_id,
        "scene": os.path.basename(str(conf.path).rstrip("/")),
        "frames": len(frames),
        "model_abs_rel": float(np.mean(model_error)),
        "model_abs_rel_median": float(np.median(model_error)),
        "sparse_points_per_frame": {
            "min": int(np.min(counts)),
            "median": float(np.median(counts)),
            "max": int(np.max(counts)),
        },
        "sparse_vs_gt_abs_rel": {
            "mean": float(np.mean(check_all)),
            "median": float(np.median(check_all)),
        },
        "alignments": rows,
    }


def report(summary: dict) -> None:
    print(f"\n{summary['model_id']}  on  {summary['scene']}   ({summary['frames']} frames)")
    counts = summary["sparse_points_per_frame"]
    print(f"sparse points per frame: min {counts['min']}, median {counts['median']:.0f}, max {counts['max']}")

    check = summary["sparse_vs_gt_abs_rel"]
    print(
        f"\nTRANSFORM CHECK -- sparse-point depth vs depth_gt at the same pixel: "
        f"mean {check['mean']:.4f}, median {check['median']:.4f}"
    )
    if check["median"] > 0.05:
        print("  WARNING: the sparse points disagree with reference depth. Either the transform")
        print("  chain is wrong or this reconstruction is poor; the rows below inherit it.")
    else:
        print("  Consistent, so the projection, units and downscale factor are right.")

    print(f"\ntrained model abs_rel {summary['model_abs_rel']:.4f} (median {summary['model_abs_rel_median']:.4f})")
    print("A regression loss can only teach where the aligned prior beats this.\n")
    print(f"{'alignment':<20}{'fitted on':<16}{'abs_rel':>9}{'median':>9}{'vs model':>13}{'p_closer':>10}  unfit")
    for row in summary["alignments"]:
        verdict = "better" if row["abs_rel"] < summary["model_abs_rel"] else "WORSE"
        ratio = summary["model_abs_rel"] / row["abs_rel"] if row["abs_rel"] else float("nan")
        print(
            f"{row['alignment']:<20}{row['fitted_on']:<16}{row['abs_rel']:>9.4f}"
            f"{row['abs_rel_median']:>9.4f}{f'{verdict} {ratio:.2f}x':>13}"
            f"{row['p_closer']:>10.3f}  {row['frames_unfittable']}"
        )
    print("\n`global` shares one alignment across every frame, which is the only variant that")
    print("leaves the prior multi-view consistent. `sparse points` is the only variant a trained")
    print("loss could actually estimate; `ground truth` is its upper bound.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--backend", default="depth_anything_3", choices=sorted(BACKENDS))
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--trim", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--gt-samples", type=int, default=50000, help="pixels per frame pooled for a global fit")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    summary = collect(
        args.checkpoint,
        args.scene_path,
        args.backend,
        args.model,
        args.trim,
        args.iters,
        args.max_frames,
        args.gt_samples,
    )
    report(summary)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
