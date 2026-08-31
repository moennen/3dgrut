# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional point-to-surface metrics used by DTU and Tanks and Temples."""

from dataclasses import dataclass

import numpy as np


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


def _nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        raise ValueError("Both predicted and reference surface samples must be non-empty")
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Surface metrics need scipy; install depthrecall[surface]") from exc
    return cKDTree(np.asarray(target, dtype=np.float64)).query(np.asarray(source, dtype=np.float64), workers=-1)[0]


def evaluate_surface(predicted: np.ndarray, reference: np.ndarray, taus: np.ndarray) -> SurfaceMetrics:
    """Compute DTU accuracy/completeness and TnT precision/recall/F-score from surface samples."""
    taus = np.asarray(taus, dtype=np.float64)
    if taus.ndim != 1 or len(taus) == 0 or np.any(taus <= 0):
        raise ValueError("taus must be a non-empty 1-D array of positive thresholds")
    predicted_to_reference = _nearest(predicted, reference)
    reference_to_predicted = _nearest(reference, predicted)
    accuracy = float(predicted_to_reference.mean())
    completeness = float(reference_to_predicted.mean())
    precision = np.array([(predicted_to_reference <= tau).mean() for tau in taus])
    recall = np.array([(reference_to_predicted <= tau).mean() for tau in taus])
    fscore = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return SurfaceMetrics(accuracy, completeness, (accuracy + completeness) / 2, precision, recall, fscore)
