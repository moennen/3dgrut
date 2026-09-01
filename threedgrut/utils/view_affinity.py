"""Persistent sparse camera-pair affinity graphs for multi-view supervision."""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from threedgrut.datasets.protocols import Batch, BoundedMultiViewDataset


@dataclass(frozen=True)
class ViewAffinityGraph:
    """CSR rows plus Vose alias tables, all indexed by training-frame index."""

    offsets: np.ndarray
    indices: np.ndarray
    probabilities: np.ndarray
    alias_probabilities: np.ndarray
    aliases: np.ndarray

    @property
    def num_views(self) -> int:
        return int(self.offsets.size - 1)

    def neighbours(self, source: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.offsets[source]), int(self.offsets[source + 1])
        return self.indices[start:end], self.probabilities[start:end]

    def sample(self, source: int, rng: random.Random) -> int | None:
        """Draw one target in O(1), returning None for a source without valid neighbours."""
        start, end = int(self.offsets[source]), int(self.offsets[source + 1])
        if start == end:
            return None
        column = start + rng.randrange(end - start)
        if rng.random() >= float(self.alias_probabilities[column]):
            column = int(self.aliases[column])
        return int(self.indices[column])

    def save(self, path: str | Path, fingerprint: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            offsets=self.offsets,
            indices=self.indices,
            probabilities=self.probabilities,
            alias_probabilities=self.alias_probabilities,
            aliases=self.aliases,
            fingerprint=np.asarray(fingerprint),
        )

    @classmethod
    def load(cls, path: str | Path, fingerprint: str) -> "ViewAffinityGraph | None":
        path = Path(path)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) != fingerprint:
                return None
            return cls(
                offsets=archive["offsets"],
                indices=archive["indices"],
                probabilities=archive["probabilities"],
                alias_probabilities=archive["alias_probabilities"],
                aliases=archive["aliases"],
            )


class PairedViewSampler:
    """Deterministically sample graph neighbours and keep a small GPU batch LRU cache."""

    def __init__(self, dataset: BoundedMultiViewDataset, graph: ViewAffinityGraph, *, seed: int, cache_size: int):
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.dataset = dataset
        self.graph = graph
        self.rng = random.Random(seed)
        self.cache_size = cache_size
        self._cache: OrderedDict[int, Batch] = OrderedDict()

    def sample(self, source_frame_idx: int) -> tuple[int, Batch] | None:
        target_frame_idx = self.graph.sample(source_frame_idx, self.rng)
        if target_frame_idx is None:
            return None
        if target_frame_idx in self._cache:
            batch = self._cache.pop(target_frame_idx)
            self._cache[target_frame_idx] = batch
            return target_frame_idx, batch
        # Dataset implementations expect the normal DataLoader collation shape even though a
        # paired target is fetched synchronously in the trainer.
        get_paired_item = getattr(self.dataset, "get_paired_item", None)
        item = get_paired_item(target_frame_idx) if get_paired_item is not None else self.dataset[target_frame_idx]
        batch = self.dataset.get_gpu_batch_with_intrinsics(default_collate([item]))
        if self.cache_size:
            self._cache[target_frame_idx] = batch
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return target_frame_idx, batch


def affinity_fingerprint(poses: np.ndarray, scene_bbox: tuple[np.ndarray, np.ndarray], config: object) -> str:
    """Stable cache key for pose, scene-bound, and affinity-configuration changes."""
    digest = hashlib.sha256()
    for value in (*np.asarray(poses, dtype=np.float32).shape, *np.asarray(scene_bbox, dtype=np.float32).shape):
        digest.update(str(value).encode())
    digest.update(np.asarray(poses, dtype=np.float32).tobytes())
    digest.update(np.asarray(scene_bbox, dtype=np.float32).tobytes())
    digest.update(repr(config).encode())
    return digest.hexdigest()


def _scene_samples(scene_bbox: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    minimum, maximum = (np.asarray(value, dtype=np.float32) for value in scene_bbox)
    corners = np.stack(np.meshgrid(*zip(minimum, maximum), indexing="ij"), axis=-1).reshape(-1, 3)
    return np.concatenate(((minimum + maximum)[None] * 0.5, corners), axis=0)


def _alias_table(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vose alias table; aliases are local row offsets until CSR assembly remaps them."""
    count = len(probabilities)
    scaled = probabilities.astype(np.float64) * count
    threshold = np.zeros(count, dtype=np.float32)
    aliases = np.arange(count, dtype=np.int64)
    small = [index for index, probability in enumerate(scaled) if probability < 1.0]
    large = [index for index, probability in enumerate(scaled) if probability >= 1.0]
    while small and large:
        low, high = small.pop(), large.pop()
        threshold[low] = scaled[low]
        aliases[low] = high
        scaled[high] -= 1.0 - scaled[low]
        (small if scaled[high] < 1.0 else large).append(high)
    for index in (*small, *large):
        threshold[index] = 1.0
    return threshold, aliases


def build_view_affinity_graph(
    poses: np.ndarray,
    scene_bbox: tuple[np.ndarray, np.ndarray],
    *,
    top_k: int,
    max_view_angle_deg: float,
    min_baseline_ratio: float,
    overlap_weight: float,
    baseline_weight: float,
    angle_weight: float,
) -> ViewAffinityGraph:
    """Build a sparse camera graph from frustum-like AABB intersection and view geometry.

    Each camera is approximated by a cone facing its C2W +Z direction.  The cone's intersection
    score is the fraction of scene AABB centre/corners visible to both cameras; this works for
    all datasets before any rendered depth exists.  The remaining factors prefer meaningful
    parallax and compatible views of the same scene points.  Runtime depth visibility still
    decides whether an individual reprojected pixel is supervised.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 < max_view_angle_deg < 180.0:
        raise ValueError("max_view_angle_deg must be in (0, 180)")
    if min_baseline_ratio < 0.0:
        raise ValueError("min_baseline_ratio must be non-negative")

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be [N, 4, 4], got {poses.shape}")
    centres = poses[:, :3, 3]
    forwards = poses[:, :3, 2]
    forwards /= np.linalg.norm(forwards, axis=-1, keepdims=True).clip(min=1e-8)
    points = _scene_samples(scene_bbox)
    camera_to_points = points[None] - centres[:, None]
    distances = np.linalg.norm(camera_to_points, axis=-1).clip(min=1e-8)
    directions = camera_to_points / distances[..., None]
    cos_limit = np.cos(np.deg2rad(max_view_angle_deg))
    visible = np.einsum("nc,npc->np", forwards, directions) >= cos_limit

    offsets, indices, probabilities, alias_probabilities, aliases = [0], [], [], [], []
    for source in range(len(poses)):
        candidates: list[tuple[int, float]] = []
        for target in range(len(poses)):
            if source == target:
                continue
            common = visible[source] & visible[target]
            overlap = float(common.mean())
            if overlap == 0.0:
                continue
            baseline = float(np.linalg.norm(centres[source] - centres[target]))
            reference_distance = float(np.median((distances[source, common] + distances[target, common]) * 0.5))
            baseline_ratio = baseline / max(reference_distance, 1e-8)
            if baseline_ratio < min_baseline_ratio:
                continue
            # Peaks at baseline_ratio=1 and decays smoothly for very wide pairs.
            baseline_score = min(baseline_ratio, 1.0 / max(baseline_ratio, 1e-8))
            view_cos = np.clip((directions[source, common] * directions[target, common]).sum(axis=-1), -1.0, 1.0)
            view_angle = float(np.arccos(view_cos).mean())
            angle_score = max(0.0, 1.0 - view_angle / np.deg2rad(max_view_angle_deg))
            score = overlap_weight * overlap * (baseline_weight * baseline_score + angle_weight * angle_score)
            if score > 0.0:
                candidates.append((target, score))
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
        candidates = candidates[:top_k]
        row_indices = np.asarray([candidate[0] for candidate in candidates], dtype=np.int64)
        row_scores = np.asarray([candidate[1] for candidate in candidates], dtype=np.float32)
        row_probabilities = row_scores / row_scores.sum() if len(row_scores) else row_scores
        row_alias_probabilities, row_aliases = (
            _alias_table(row_probabilities) if len(row_probabilities) else (row_probabilities, row_indices)
        )
        row_start = len(indices)
        indices.extend(row_indices.tolist())
        probabilities.extend(row_probabilities.tolist())
        alias_probabilities.extend(row_alias_probabilities.tolist())
        aliases.extend((row_start + row_aliases).tolist())
        offsets.append(len(indices))
    return ViewAffinityGraph(
        offsets=np.asarray(offsets, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float32),
        alias_probabilities=np.asarray(alias_probabilities, dtype=np.float32),
        aliases=np.asarray(aliases, dtype=np.int64),
    )
