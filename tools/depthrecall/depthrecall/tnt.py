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

"""Tanks and Temples support: the ground-truth alignment and the evaluation crop volume.

TnT differs from DTU in the one way that matters here: **the alignment is not given, it has to
be estimated, and its error is comparable to the evaluation threshold.**

Each training-set scene ships

- ``<scene>.ply``: the laser scan, in metres, `double` precision, 12.7M points for Barn.
- ``<scene>_trans.txt``: a 4x4 similarity mapping the *official* COLMAP SfM frame to the scan
  frame. For Barn its scale is 4.1385, so the official SfM is not metric.
- ``<scene>_COLMAP_SfM.log``: the official SfM camera-to-world poses, one per image.
- ``<scene>.json``: an Open3D ``SelectionPolygonVolume`` delimiting the region the official
  benchmark scores.

The trap is that a model is trained on *a* COLMAP reconstruction of the same images, and that
reconstruction is not in the same frame as the official log even when it uses the same images:
for Barn the two differ by a similarity of scale 1.0036, and applying ``inv(trans)`` alone
leaves the scan misaligned by a median 0.19 SfM units, which is 0.78 m of scan -- 78x Barn's
official 0.01 m threshold. Recall then reads ~0 for reasons that have nothing to do with the
reconstruction.

So the alignment is composed in two steps, GT -> official SfM -> the target reconstruction,
where the second step is fitted from the camera centres. Those correspondences are known
exactly (images match by name), which makes this better posed than the ICP the official toolbox
runs on the clouds. It is still not exact: see ``GtAlignment.residual_*``, which are reported in
scan units so they can be compared against tau, because **a recall at a tau near the residual
is measuring the registration, not the reconstruction.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The official per-scene thresholds from the Tanks and Temples evaluation toolbox, in metres.
# Reproduced here so a caller can pick the same tau the benchmark reports F-scores at -- but see
# GtAlignment: on a re-run reconstruction the registration residual can exceed these.
OFFICIAL_TAU_METRES = {
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
}


def read_sfm_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``.log`` trajectory: returns (indices, camera-to-world 4x4 matrices).

    The format is a 3-integer header line per entry followed by the 4x4 matrix, and the
    matrices are camera-to-world, so the translation column is the camera centre.
    """
    tokens = Path(path).read_text().split("\n")
    indices: list[int] = []
    poses: list[np.ndarray] = []
    i = 0
    while i < len(tokens):
        header = tokens[i].split()
        if len(header) != 3:
            i += 1
            continue
        if i + 4 >= len(tokens):
            raise ValueError(f"Truncated .log entry at line {i} of {path}")
        rows = [[float(x) for x in tokens[i + 1 + k].split()] for k in range(4)]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 pose at line {i + 1} of {path}, got {matrix.shape}")
        indices.append(int(header[0]))
        poses.append(matrix)
        i += 5
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return np.asarray(indices), np.stack(poses)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama) mapping ``source`` onto ``target``, as a 4x4.

    A similarity, not a rigid transform: two COLMAP runs of the same images agree only up to
    scale, and forcing scale 1 would leave a residual proportional to the scene size.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) arrays, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError(f"A similarity needs at least 3 correspondences, got {len(source)}")

    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src_c, dst_c = source - src_mean, target - dst_mean
    covariance = src_c.T @ dst_c / len(source)
    U, singular, Vt = np.linalg.svd(covariance)
    # Reflections are not similarities of a right-handed frame; flip the smallest axis instead.
    sign = np.sign(np.linalg.det(U @ Vt))
    correction = np.diag([1.0, 1.0, sign])
    rotation = (U @ correction @ Vt).T
    variance = (src_c**2).sum() / len(source)
    if variance <= 0:
        raise ValueError("Degenerate correspondences: all source points coincide")
    scale = float((singular * np.array([1.0, 1.0, sign])).sum() / variance)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = dst_mean - scale * rotation @ src_mean
    return matrix


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (N, 3) points."""
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


@dataclass
class GtAlignment:
    """The scan-to-render-world transform, with the evidence for how good it is.

    ``residual_median``/``residual_max`` are the camera-centre disagreement *in scan units*
    (metres for TnT) after fitting. Compare them against tau before reading a recall: they are
    a lower bound on the geometric error this measurement can resolve.
    """

    matrix: np.ndarray
    scale: float
    residual_median: float
    residual_max: float
    n_correspondences: int

    def warn_if_coarse(self, taus: np.ndarray) -> str | None:
        """Return a warning when the smallest tau is not clear of the registration residual."""
        smallest = float(np.min(taus))
        if smallest >= 3.0 * self.residual_median:
            return None
        return (
            f"Registration residual (median {self.residual_median:.4g}, max {self.residual_max:.4g} "
            f"scan units over {self.n_correspondences} cameras) is not small against the smallest "
            f"tau ({smallest:.4g}): recall there reflects alignment error as much as geometry. "
            f"Prefer taus above {3.0 * self.residual_median:.4g}."
        )


def gt_to_render_alignment(
    trans_path: str | Path,
    sfm_log_path: str | Path,
    render_centres: np.ndarray,
    *,
    log_indices: np.ndarray | None = None,
) -> GtAlignment:
    """Compose the scan -> render-world transform for a TnT scene.

    ``render_centres`` are the camera centres of the reconstruction the depth maps were
    rendered in, ordered to correspond to the ``.log`` entries (image-name order for a COLMAP
    model built from ``000001.jpg`` upwards). ``trans_path`` maps official-SfM -> scan, so its
    inverse is the first leg; the second is fitted from the centres.
    """
    trans = np.loadtxt(trans_path, dtype=np.float64)
    if trans.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 in {trans_path}, got {trans.shape}")

    indices, poses = read_sfm_log(sfm_log_path)
    log_centres = poses[:, :3, 3]
    if log_indices is not None:
        order = {int(v): i for i, v in enumerate(indices)}
        missing = [int(v) for v in log_indices if int(v) not in order]
        if missing:
            raise ValueError(f"{sfm_log_path} has no entry for log indices {missing[:5]}")
        log_centres = log_centres[[order[int(v)] for v in log_indices]]

    render_centres = np.asarray(render_centres, dtype=np.float64)
    if len(render_centres) != len(log_centres):
        raise ValueError(
            f"{len(render_centres)} rendered cameras against {len(log_centres)} .log poses. "
            "Pass log_indices to select, or check that the reconstruction covers the same images: "
            "a similarity fitted to mismatched correspondences is silently wrong."
        )

    log_to_render = fit_similarity(log_centres, render_centres)
    residuals = np.linalg.norm(apply_transform(log_to_render, log_centres) - render_centres, axis=1)

    scan_to_log = np.linalg.inv(trans)
    scan_to_render = log_to_render @ scan_to_log
    render_scale = float(np.linalg.norm(log_to_render[:3, :3], axis=1).mean())
    # Report the residual in scan units so it is comparable with a tau expressed in metres.
    scan_per_render = float(np.linalg.norm(trans[:3, :3], axis=1).mean()) / max(render_scale, 1e-30)
    return GtAlignment(
        matrix=scan_to_render,
        scale=float(np.linalg.norm(scan_to_render[:3, :3], axis=1).mean()),
        residual_median=float(np.median(residuals)) * scan_per_render,
        residual_max=float(residuals.max()) * scan_per_render,
        n_correspondences=len(log_centres),
    )


def read_crop_volume(path: str | Path) -> tuple[int, float, float, np.ndarray]:
    """Read an Open3D ``SelectionPolygonVolume``: (axis, axis_min, axis_max, polygon)."""
    data = json.loads(Path(path).read_text())
    if data.get("class_name") != "SelectionPolygonVolume":
        raise ValueError(f"{path} is not a SelectionPolygonVolume (got {data.get('class_name')!r})")
    axis = {"X": 0, "Y": 1, "Z": 2}[data["orthogonal_axis"].upper()]
    polygon = np.asarray(data["bounding_polygon"], dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) bounding polygon in {path}, got {polygon.shape}")
    return axis, float(data["axis_min"]), float(data["axis_max"]), polygon


def crop_volume_mask(points: np.ndarray, path: str | Path) -> np.ndarray:
    """Mask of the points inside a scene's evaluation crop volume, in *scan* coordinates.

    The official benchmark scores only this region, so the scan must be cropped for a recall to
    be comparable with a published F-score -- an uncropped scan puts surroundings the method was
    never asked to reconstruct into the denominator. For Barn the crop keeps 92%.
    """
    axis, axis_min, axis_max, polygon = read_crop_volume(path)
    points = np.asarray(points, dtype=np.float64)
    inside = (points[:, axis] >= axis_min) & (points[:, axis] <= axis_max)

    others = [i for i in range(3) if i != axis]
    poly = polygon[:, others]
    query = points[:, others]
    # Even-odd ray crossing test, vectorised over points.
    crossings = np.zeros(len(query), dtype=bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if y1 == y2:
            continue
        straddles = (y1 > query[:, 1]) != (y2 > query[:, 1])
        x_at_y = (x2 - x1) * (query[:, 1] - y1) / (y2 - y1) + x1
        crossings ^= straddles & (query[:, 0] < x_at_y)
    return inside & crossings
