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

"""Core recall metric over projected GT points."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_cameras import View, viewset_hash
from .io_depth import load_depth, set_invalid_value
from .io_points import Normalizer


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for a single evaluation run."""

    taus: np.ndarray  # (T,) float64; monotonically increasing from 0
    depth_convention: str  # "ray" or "z"
    max_valid_depth: float | None = None
    invalid_value: float | None = None
    depth_scale: float = 1.0
    # COLMAP's principal point is corner-based: the centre of pixel (0, 0) sits at (0.5, 0.5).
    # Array index i therefore corresponds to continuous coordinate i + 0.5, so a projected
    # coordinate must have this subtracted before it indexes the depth array. Dropping it
    # biases every residual by half a pixel of depth gradient, which is invisible on flat
    # surfaces and largest exactly where the geometry is interesting.
    pixel_center_offset: float = 0.5
    # Nearest by default: bilinear interpolation blends neighbouring pixels, and at a hole
    # boundary or a silhouette one of those neighbours is invalid or belongs to a different
    # surface. That manufactures residuals precisely where reconstructions differ from each
    # other, so the smoother option is the less faithful one. Bilinear is offered for smooth
    # surfaces and requires all four neighbours to be valid.
    sampling: str = "nearest"
    # Per-view rasterized GT depth. When supplied, only GT points that agree with this z-buffer
    # are scored. This removes self-occluded scan points, essential for meaningful TnT recall.
    visibility_depths: dict[str, Path] | None = None
    visibility_tolerance: float = 1e-3

    def __post_init__(self):
        if self.depth_convention not in ("ray", "z"):
            raise ValueError(f"depth_convention must be 'ray' or 'z', got {self.depth_convention}")
        if self.sampling not in ("nearest", "bilinear"):
            raise ValueError(f"sampling must be 'nearest' or 'bilinear', got {self.sampling}")
        if self.taus.ndim != 1 or not np.all(np.diff(self.taus) >= 0):
            raise ValueError("taus must be a 1-D non-decreasing array")
        if self.taus.size == 0:
            raise ValueError("taus must not be empty")
        if np.any(self.taus < 0):
            raise ValueError("taus must be non-negative")
        if self.visibility_tolerance < 0:
            raise ValueError("visibility_tolerance must be non-negative")

    @property
    def tau_list(self) -> list[float]:
        return self.taus.tolist()


@dataclass
class ViewResult:
    """Per-view diagnostics."""

    name: str
    in_frustum: int
    no_surface: int
    recall: np.ndarray  # (T,)
    too_near: np.ndarray  # fraction of finite pairs with pred < gt - tau
    too_far: np.ndarray  # fraction of finite pairs with pred > gt + tau
    median_signed_delta: float | None
    coverage: float  # in_frustum / total points


@dataclass
class EvaluationResult:
    """Pooled result across all views and points."""

    config: dict[str, Any]
    gt_points: int
    gt_voxel_size: float | None
    views_count: int
    taus: list[float]
    in_frustum_pairs: int
    no_surface_pairs: int
    recall: np.ndarray  # (T,)
    too_near: np.ndarray
    too_far: np.ndarray
    median_signed_delta: float | None
    per_view: list[ViewResult]
    viewset_hash: str
    gt_hash: str

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "config": self.config,
            "gt_points": self.gt_points,
            "gt_voxel_size": self.gt_voxel_size,
            "views_count": self.views_count,
            "taus": self.taus,
            "in_frustum_pairs": self.in_frustum_pairs,
            "no_surface_pairs": self.no_surface_pairs,
            "recall": self.recall.tolist(),
            "too_near": self.too_near.tolist(),
            "too_far": self.too_far.tolist(),
            # Stated explicitly because the two are different: a no-surface pair counts as a
            # recall failure but is excluded from the signed split, so the three series do not
            # sum to one and cannot be compared without knowing this.
            "denominators": {
                "recall": "in_frustum_pairs",
                "too_near_too_far": "in_frustum_pairs - no_surface_pairs",
                "pairs_with_surface": self.in_frustum_pairs - self.no_surface_pairs,
            },
            "median_signed_delta": self.median_signed_delta,
            "per_view": [
                {
                    "name": v.name,
                    "in_frustum": v.in_frustum,
                    "no_surface": v.no_surface,
                    "recall": v.recall.tolist(),
                    "too_near": v.too_near.tolist(),
                    "too_far": v.too_far.tolist(),
                    "median_signed_delta": v.median_signed_delta,
                    "coverage": v.coverage,
                }
                for v in self.per_view
            ],
            "viewset_hash": self.viewset_hash,
            "gt_hash": self.gt_hash,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.asdict(), indent=2))


def _sample_nearest(
    depth: np.ndarray, valid: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the pixel containing the projected point. Returns (depth, valid)."""
    H, W = depth.shape
    # floor(x + 0.5) rather than np.round, whose half-to-even rule would send u = i + 0.5
    # to i or i + 1 depending on the parity of i.
    ui = np.clip(np.floor(u + 0.5).astype(np.int64), 0, W - 1)
    vi = np.clip(np.floor(v + 0.5).astype(np.int64), 0, H - 1)
    return depth[vi, ui], valid[vi, ui]


def _sample_bilinear(
    depth: np.ndarray, valid: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample depth, requiring all four neighbours to be valid."""
    H, W = depth.shape
    u0 = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)

    du = np.clip(u - u0.astype(np.float64), 0.0, 1.0)
    dv = np.clip(v - v0.astype(np.float64), 0.0, 1.0)

    d00, d10, d01, d11 = depth[v0, u0], depth[v0, u1], depth[v1, u0], depth[v1, u1]
    all_valid = valid[v0, u0] & valid[v0, u1] & valid[v1, u0] & valid[v1, u1]

    sampled = (1 - du) * (1 - dv) * d00 + du * (1 - dv) * d10 + (1 - du) * dv * d01 + du * dv * d11
    return sampled, all_valid


def _approx_median(values: np.ndarray, n_bins: int = 20001) -> float:
    """Approximate median via a symmetric histogram; exact enough for diagnostics."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    vmax = np.max(np.abs(finite))
    if vmax == 0:
        return 0.0
    edges = np.linspace(-vmax, vmax, n_bins)
    counts, _ = np.histogram(finite, bins=edges)
    cumsum = np.cumsum(counts)
    half = finite.size / 2.0
    idx = int(np.searchsorted(cumsum, half, side="right"))
    idx = min(idx, len(edges) - 2)
    return float((edges[idx] + edges[idx + 1]) / 2)


def _evaluate_view(
    view: View,
    points_world: np.ndarray,
    config: MetricConfig,
) -> ViewResult:
    """Evaluate one view and return per-view curves."""
    depth, valid = load_depth(view.depth_path, depth_scale=config.depth_scale)
    if config.invalid_value is not None:
        depth, valid = set_invalid_value(depth, config.invalid_value)
    if config.max_valid_depth is not None:
        valid = valid & (depth <= config.max_valid_depth)

    H, W = depth.shape
    n_total = points_world.shape[0]

    # Camera-space coordinates.
    X_cam = (view.R @ points_world.T).T + view.t
    z = X_cam[:, 2]
    positive_z = z > 0
    x_proj = X_cam[:, 0] / np.where(positive_z, z, 1.0)
    y_proj = X_cam[:, 1] / np.where(positive_z, z, 1.0)
    # Corner-based projected coordinates, then shifted into array-index space.
    u = view.K[0, 0] * x_proj + view.K[0, 2] - config.pixel_center_offset
    v = view.K[1, 1] * y_proj + view.K[1, 2] - config.pixel_center_offset
    # Array coordinate i covers the pixel extent [i - 0.5, i + 0.5], so the image spans
    # [-0.5, W - 0.5]. Using [0, W - 1] instead would discard the outer half-pixel ring, and
    # since a projected boundary coordinate lands there only up to floating-point noise, a
    # principal point recovered as 31.999999999999975 silently dropped an entire column.
    in_image = (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
    in_frustum = positive_z & in_image
    if config.visibility_depths is not None:
        visibility_path = config.visibility_depths.get(view.name)
        if visibility_path is None:
            raise ValueError(f"No GT visibility depth for view {view.name}")
        visibility, visibility_valid = load_depth(visibility_path)
        if visibility.shape != (H, W):
            raise ValueError(f"Visibility depth for {view.name} has shape {visibility.shape}, expected {(H, W)}")
        # A rasterized GT map is the visibility oracle: a point survives only if it is the
        # nearest scan surface at its pixel. The relative tolerance absorbs point-rasterization
        # discretization without admitting a genuinely occluded back surface.
        vis_sample, vis_valid = _sample_nearest(visibility, visibility_valid, u, v)
        gt_all = np.linalg.norm(X_cam, axis=1) if config.depth_convention == "ray" else z
        in_frustum &= vis_valid & (np.abs(vis_sample - gt_all) <= config.visibility_tolerance * gt_all)
    n_in = int(in_frustum.sum())

    taus = config.taus
    if n_in == 0:
        empty = np.zeros(len(taus), dtype=np.float64)
        return ViewResult(
            name=view.name,
            in_frustum=0,
            no_surface=0,
            recall=empty.copy(),
            too_near=empty.copy(),
            too_far=empty.copy(),
            median_signed_delta=None,
            coverage=0.0,
        )

    u_in = u[in_frustum]
    v_in = v[in_frustum]
    z_in = z[in_frustum]
    gt_range = np.linalg.norm(X_cam[in_frustum], axis=1) if config.depth_convention == "ray" else z_in

    sampler = _sample_nearest if config.sampling == "nearest" else _sample_bilinear
    sampled, sampled_valid = sampler(depth, valid, u_in, v_in)
    sampled_valid = sampled_valid & np.isfinite(sampled)
    n_no_surface = int((~sampled_valid).sum())

    delta = sampled - gt_range
    delta[~sampled_valid] = np.nan
    finite_mask = np.isfinite(delta)
    n_finite = int(finite_mask.sum())

    # Absolute recall: denominator is all in-frustum pairs (including no-surface).
    abs_delta = np.abs(delta[finite_mask])
    abs_sorted = np.sort(abs_delta)
    recall_counts = np.searchsorted(abs_sorted, taus, side="right")
    recall = recall_counts.astype(np.float64) / max(n_in, 1)

    # Signed diagnostics: denominator is finite (valid-surface) pairs.
    signed = delta[finite_mask]
    signed_sorted = np.sort(signed)
    too_near = np.searchsorted(signed_sorted, -taus, side="left").astype(np.float64) / max(n_finite, 1)
    lte_tau = np.searchsorted(signed_sorted, taus, side="right")
    too_far = (signed_sorted.size - lte_tau).astype(np.float64) / max(n_finite, 1)

    median_signed_delta = _approx_median(signed) if n_finite > 0 else None

    return ViewResult(
        name=view.name,
        in_frustum=n_in,
        no_surface=n_no_surface,
        recall=recall,
        too_near=too_near,
        too_far=too_far,
        median_signed_delta=median_signed_delta,
        coverage=n_in / max(n_total, 1),
    )


def _pooled_result(
    per_view: list[ViewResult],
    taus: np.ndarray,
    config_dict: dict[str, Any],
) -> EvaluationResult:
    """Pool per-view results by summing counts."""
    in_frustum_total = sum(v.in_frustum for v in per_view)
    no_surface_total = sum(v.no_surface for v in per_view)
    finite_total = in_frustum_total - no_surface_total

    passes = np.zeros(len(taus), dtype=np.float64)
    near_sum = np.zeros(len(taus), dtype=np.float64)
    far_sum = np.zeros(len(taus), dtype=np.float64)
    for v in per_view:
        if v.in_frustum == 0:
            continue
        passes += v.recall * v.in_frustum
        if v.in_frustum > v.no_surface:
            w = v.in_frustum - v.no_surface
            near_sum += v.too_near * w
            far_sum += v.too_far * w

    recall = passes / max(in_frustum_total, 1)
    too_near = near_sum / max(finite_total, 1)
    too_far = far_sum / max(finite_total, 1)

    weights = np.array([v.in_frustum for v in per_view if v.median_signed_delta is not None], dtype=np.float64)
    medians = [v.median_signed_delta for v in per_view if v.median_signed_delta is not None]
    pooled_median = None
    if len(medians) > 0 and weights.sum() > 0:
        medians = np.asarray(medians)
        order = np.argsort(medians)
        cumw = np.cumsum(weights[order])
        half = cumw[-1] / 2.0
        idx = int(np.searchsorted(cumw, half, side="right"))
        pooled_median = float(medians[order][min(idx, len(order) - 1)])

    return EvaluationResult(
        config=config_dict,
        gt_points=0,
        gt_voxel_size=None,
        views_count=len(per_view),
        taus=taus.tolist(),
        in_frustum_pairs=in_frustum_total,
        no_surface_pairs=no_surface_total,
        recall=recall,
        too_near=too_near,
        too_far=too_far,
        median_signed_delta=pooled_median,
        per_view=per_view,
        viewset_hash="",
        gt_hash="",
    )


def evaluate(
    views: list[View],
    gt_points: np.ndarray,
    config: MetricConfig,
    alignment: np.ndarray | None = None,
    downsample_voxel: float | None = None,
    gt_masks: list[str] | None = None,
) -> EvaluationResult:
    """Run the recall evaluation over all views.

    ``gt_masks`` names the ground-truth culls the caller already applied, purely so the JSON
    records them: whether a benchmark's GT-side cull was applied changes the denominator by
    tens of percent, and two runs would otherwise be indistinguishable in the output.
    """
    normalizer = Normalizer(gt_points, alignment=alignment, downsample_voxel=downsample_voxel)
    points_world = normalizer.points

    per_view = [_evaluate_view(view, points_world, config) for view in views]

    config_dict = {
        "taus": config.tau_list,
        "depth_convention": config.depth_convention,
        "max_valid_depth": config.max_valid_depth,
        "invalid_value": config.invalid_value,
        "depth_scale": config.depth_scale,
        "sampling": config.sampling,
        "pixel_center_offset": config.pixel_center_offset,
        "downsample_voxel": downsample_voxel,
        "gt_masks": list(gt_masks) if gt_masks else [],
        "visibility": {
            "enabled": config.visibility_depths is not None,
            "tolerance_relative": config.visibility_tolerance if config.visibility_depths is not None else None,
        },
    }
    result = _pooled_result(per_view, config.taus, config_dict)
    result.gt_points = normalizer.count
    result.gt_voxel_size = downsample_voxel
    result.viewset_hash = viewset_hash(views)
    result.gt_hash = normalizer.hash()
    return result
