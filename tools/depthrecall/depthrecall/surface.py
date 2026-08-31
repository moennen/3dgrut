# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional point-to-surface metrics used by DTU and Tanks and Temples."""

from dataclasses import dataclass

import numpy as np

DEFAULT_QUERY_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class SurfaceMetrics:
    """Distances and thresholded precision/recall, all in the input coordinates' units."""

    accuracy: float
    completeness: float
    overall: float
    precision: np.ndarray
    recall: np.ndarray
    fscore: np.ndarray

    def asdict(self, taus: np.ndarray) -> dict:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "overall": self.overall,
            "taus": taus.tolist(),
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "fscore": self.fscore.tolist(),
        }


def _nearest_summary(
    source: np.ndarray,
    target: np.ndarray,
    taus: np.ndarray,
    *,
    query_chunk_size: int,
    workers: int,
) -> tuple[float, np.ndarray]:
    """Return exact nearest-distance mean and threshold fractions without retaining all distances.

    Large benchmark meshes contain millions of samples. Querying them all at once with every CPU
    worker can create enough temporary allocations to OOM a 64 GB host. The KD-tree remains exact;
    only its queries are processed in bounded batches and immediately reduced to the statistics
    required by DTU/TnT metrics.
    """
    if len(source) == 0 or len(target) == 0:
        raise ValueError("Both predicted and reference surface samples must be non-empty")
    if query_chunk_size < 1:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Surface metrics need scipy; install depthrecall[surface]") from exc
    source = np.asarray(source)
    tree = cKDTree(np.asarray(target, dtype=np.float64))
    distance_sum = 0.0
    threshold_counts = np.zeros(len(taus), dtype=np.int64)
    for start in range(0, len(source), query_chunk_size):
        distances = tree.query(np.asarray(source[start : start + query_chunk_size], dtype=np.float64), workers=workers)[
            0
        ]
        distance_sum += float(distances.sum())
        threshold_counts += np.count_nonzero(distances[:, None] <= taus[None, :], axis=0)
    return distance_sum / len(source), threshold_counts / len(source)


def evaluate_surface(
    predicted: np.ndarray,
    reference: np.ndarray,
    taus: np.ndarray,
    *,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    workers: int = 1,
) -> SurfaceMetrics:
    """Compute DTU/TnT surface metrics with bounded-memory exact nearest-neighbour queries.

    ``query_chunk_size`` changes only peak memory and execution scheduling; it does not subsample
    either surface or approximate nearest-neighbour distances. ``workers=1`` is the safe default
    for multi-million-sample benchmark scoring; callers can raise it when host memory permits.
    """
    taus = np.asarray(taus, dtype=np.float64)
    if taus.ndim != 1 or len(taus) == 0 or np.any(taus <= 0):
        raise ValueError("taus must be a non-empty 1-D array of positive thresholds")
    accuracy, precision = _nearest_summary(
        predicted, reference, taus, query_chunk_size=query_chunk_size, workers=workers
    )
    completeness, recall = _nearest_summary(
        reference, predicted, taus, query_chunk_size=query_chunk_size, workers=workers
    )
    fscore = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return SurfaceMetrics(accuracy, completeness, (accuracy + completeness) / 2, precision, recall, fscore)
