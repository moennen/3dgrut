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

"""Putting a monocular depth prior into the scene's units, using the COLMAP points already there.

A monocular prior predicts z up to an unknown affine transform, which is why the shipped
supervision term reads only its *ordering*. A regression term needs the transform resolved, and
the only reference available at training time -- ground truth being reserved for evaluation -- is
the sparse point cloud COLMAP triangulated to initialise the scene. Each frame records which
keypoints it observed and which 3D point each one matched, so every frame comes with a few
thousand (pixel, depth) samples for free.

Measured on three OB3D scenes, that is enough. A per-frame affine fitted to these points instead
of to ground truth costs +11% to +14% `abs_rel` on sponza and lone-monk, and is 4-7% *better* on
emerald-square, because a least-squares fit against every ground-truth pixel optimises something
other than `abs_rel` while the sparse points sit on the near textured geometry that metric
weights. So the alignment is not the weak link it was assumed to be. See
`docs/normal-supervision.md` and `scripts/ablation/sparse_align_diagnostic.py`.

Two properties of this fit are deliberate and worth not undoing:

* **It is fitted per frame, not once per scene.** One global alignment -- the form a genuinely
  metric prior would need, and the form that would make the supervision multi-view consistent for
  free -- is 1.2x to 3.5x worse on every scene tested, including for `DA3METRIC-LARGE`, whose
  metric output is the entire reason to expect otherwise. The per-frame fit is still anchored to a
  globally consistent point cloud, so the frames do not drift apart; the extra freedom absorbs the
  prior's own per-frame error rather than introducing any.
* **It is fitted in z and converted to ray distance afterwards.** The prior's ambiguity is affine
  in z, and ray distance is z divided by a per-pixel cosine, so an affine in one is not an affine
  in the other. Fitting in distance space would be solving the wrong problem, quietly.

Three conversions have to be right and none of them announce themselves when wrong -- each yields
a plausible-looking depth map. `xys` are pixels at COLMAP's *full* resolution while the training
frames may be downsampled; the comparison target is Euclidean ray distance, not z; and
`normalize_world_space` rescales poses and depth, so the points must be rescaled to match.
`observation_consistency` exists to catch all three against a reference depth map, and is what
the diagnostic and the tests check.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from threedgrut.datasets.utils import qvec_to_so3

# Below this many usable observations a frame's affine is fitted to noise, so the frame is
# reported unfittable and callers drop it rather than supervise against a fabricated alignment.
# Two would suffice to solve the system; the margin is because COLMAP tracks contain outliers and
# the trimming needs something left to trim.
MIN_OBSERVATIONS = 16

# Fraction of the worst residuals dropped before the fit is repeated, and how many times. Matches
# `scripts/ablation/pseudo_depth_diagnostic.py`, so the alignment used in training is the one that
# was measured offline; changing one without the other decouples them.
DEFAULT_TRIM = 0.2
DEFAULT_ITERS = 3


def read_points3d_with_ids(sparse_dir: str | Path) -> dict[int, np.ndarray]:
    """``{point_id: xyz}`` from COLMAP's points3D file, binary or text.

    `read_colmap_points3D_binary`/`_text` in `threedgrut/datasets/utils.py` drop the ids, because
    initialising a point cloud never needs them. The id is exactly what links a 3D point to the
    keypoint that observed it, so this reads the file again rather than changing their contract.
    """
    sparse_dir = Path(sparse_dir)
    binary, text = sparse_dir / "points3D.bin", sparse_dir / "points3D.txt"
    points: dict[int, np.ndarray] = {}
    if binary.exists():
        with open(binary, "rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                point_id, x, y, z = struct.unpack("<QdddBBBd", handle.read(43))[:4]
                (track_length,) = struct.unpack("<Q", handle.read(8))
                handle.read(8 * track_length)  # the track itself is not needed here
                points[int(point_id)] = np.array([x, y, z], dtype=np.float64)
        return points
    if not text.exists():
        raise FileNotFoundError(f"no points3D.bin or points3D.txt under {sparse_dir}")
    for line in text.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        points[int(fields[0])] = np.array([float(v) for v in fields[1:4]], dtype=np.float64)
    return points


def sparse_observations(
    image,
    points3d: dict[int, np.ndarray],
    scaling_factor: float,
    world_scale: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """The frame's sparse points as rows of ``(col, row, z_camera, distance_camera)``.

    `image` is a COLMAP `Image` record (`qvec`, `tvec`, `xys`, `point3D_ids`). `scaling_factor` is
    how much the training frames are downsampled relative to COLMAP's resolution, and
    `world_scale` is the uniform scale `normalize_world_space` applied to poses and depth, or 1.

    Pixels come from the recorded keypoint rather than from re-projecting the point, which keeps
    this independent of the camera's distortion model. Both depths are returned because they are
    for different jobs: `z_camera` is what the prior is fitted against, `distance_camera` is what
    a rendered depth map can be compared with.
    """
    height, width = shape
    rotation = qvec_to_so3(image.qvec)
    translation = np.asarray(image.tvec, dtype=np.float64)
    rows = []
    for (x_full, y_full), point_id in zip(image.xys, image.point3D_ids):
        xyz = points3d.get(int(point_id))
        if point_id == -1 or xyz is None:
            continue
        camera = (rotation @ xyz + translation) * world_scale
        if camera[2] <= 0.0:  # behind the camera; COLMAP tracks can contain these
            continue
        col, row = x_full / scaling_factor, y_full / scaling_factor
        if not (0 <= col < width and 0 <= row < height):
            continue
        rows.append((col, row, camera[2], float(np.linalg.norm(camera))))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def fit_trimmed_affine(
    prior: np.ndarray,
    target: np.ndarray,
    trim: float = DEFAULT_TRIM,
    iters: int = DEFAULT_ITERS,
) -> tuple[float, float]:
    """Least squares ``target ~ a * prior + b``, refitted after dropping the worst residuals.

    The trimming is not cosmetic: COLMAP points include mistriangulations, and a plain fit lets a
    handful of them set the scale for the whole frame.
    """
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.ones(len(prior), dtype=bool)
    coefficients = np.zeros(2)
    for _ in range(max(iters, 1)):
        if keep.sum() < 2:
            break
        design = np.stack([prior[keep], np.ones(int(keep.sum()))], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, target[keep], rcond=None)
        if trim <= 0:
            break
        residual = np.abs(prior * coefficients[0] + coefficients[1] - target)
        keep = residual <= np.quantile(residual, 1.0 - trim)
    return float(coefficients[0]), float(coefficients[1])


def fit_frame_alignment(
    prior: np.ndarray,
    observations: np.ndarray,
    trim: float = DEFAULT_TRIM,
    iters: int = DEFAULT_ITERS,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[tuple[float, float]]:
    """The affine taking this frame's prior into scene z, or `None` if it cannot be fitted.

    `prior` is the prior map at the training resolution and `observations` comes from
    `sparse_observations` at that same resolution. Returns `None` when the frame has too few
    usable points, or when the fitted scale is non-positive -- a negative scale would mirror the
    prior's ordering, which is a sign the prior and the points are not describing the same thing,
    and supervising against it would actively teach the scene inside out.
    """
    if len(observations) < min_observations:
        return None
    columns = np.clip(np.round(observations[:, 0]).astype(int), 0, prior.shape[1] - 1)
    rows = np.clip(np.round(observations[:, 1]).astype(int), 0, prior.shape[0] - 1)
    sampled = prior[rows, columns]
    finite = np.isfinite(sampled) & np.isfinite(observations[:, 2])
    if int(finite.sum()) < min_observations:
        return None
    scale, offset = fit_trimmed_affine(sampled[finite], observations[finite, 2], trim, iters)
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0.0:
        return None
    return scale, offset


def observation_consistency(observations: np.ndarray, reference_distance: np.ndarray) -> Optional[float]:
    """Median relative disagreement between the points' own depth and a reference depth map.

    This validates the projection rather than the prior: it uses only `distance_camera` from the
    observations, so a wrong downscale factor, a missed world scale, or z used where ray distance
    was wanted all show up here as a large number. On the three OB3D scenes measured it is
    0.004-0.007 when correct. `None` when nothing overlaps.
    """
    if not len(observations):
        return None
    columns = np.clip(np.round(observations[:, 0]).astype(int), 0, reference_distance.shape[1] - 1)
    rows = np.clip(np.round(observations[:, 1]).astype(int), 0, reference_distance.shape[0] - 1)
    reference = reference_distance[rows, columns]
    usable = np.isfinite(reference) & (reference > 0)
    if not usable.any():
        return None
    relative = np.abs(observations[usable, 3] - reference[usable]) / reference[usable]
    return float(np.median(relative))
