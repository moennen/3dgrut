import random

import numpy as np
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.view_affinity import (
    PairedViewSampler,
    ViewAffinityGraph,
    affinity_fingerprint,
    build_view_affinity_graph,
)


def _pose(center, forward):
    forward = np.asarray(forward, dtype=np.float32)
    forward /= np.linalg.norm(forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 2] = forward
    pose[:3, 3] = center
    return pose


def _graph(poses):
    return build_view_affinity_graph(
        np.asarray(poses),
        (np.array([-1.0, -1.0, 2.0]), np.array([1.0, 1.0, 4.0])),
        top_k=2,
        max_view_angle_deg=80.0,
        min_baseline_ratio=0.01,
        overlap_weight=1.0,
        baseline_weight=1.0,
        angle_weight=1.0,
    )


def test_affinity_keeps_overlapping_compatible_views_and_rejects_back_facing_view():
    graph = _graph(
        [
            _pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            _pose([0.3, 0.0, 0.0], [-0.1, 0.0, 1.0]),
            _pose([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ]
    )
    neighbours, probabilities = graph.neighbours(0)
    assert neighbours.tolist() == [1]
    np.testing.assert_allclose(probabilities, [1.0])


def test_alias_sampling_is_deterministic_and_never_leaves_the_source_row():
    graph = _graph(
        [
            _pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            _pose([0.3, 0.0, 0.0], [-0.1, 0.0, 1.0]),
            _pose([-0.25, 0.0, 0.0], [0.1, 0.0, 1.0]),
        ]
    )
    first_rng, second_rng = random.Random(7), random.Random(7)
    first = [graph.sample(0, first_rng) for _ in range(20)]
    second = [graph.sample(0, second_rng) for _ in range(20)]
    neighbours, _ = graph.neighbours(0)
    assert first == second
    assert set(first).issubset(set(neighbours.tolist()))


def test_graph_round_trip_respects_fingerprint(tmp_path):
    poses = np.asarray([_pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), _pose([0.3, 0.0, 0.0], [0.0, 0.0, 1.0])])
    bbox = (np.array([-1.0, -1.0, 2.0]), np.array([1.0, 1.0, 4.0]))
    fingerprint = affinity_fingerprint(poses, bbox, {"top_k": 2})
    graph = _graph(poses)
    path = tmp_path / "affinity.npz"
    graph.save(path, fingerprint)
    loaded = ViewAffinityGraph.load(path, fingerprint)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.offsets, graph.offsets)
    np.testing.assert_array_equal(loaded.indices, graph.indices)
    assert ViewAffinityGraph.load(path, "different") is None


class _ExactPairDataset:
    def __init__(self):
        self.requested: int | None = None

    def __getitem__(self, index):
        raise AssertionError(f"random dataset lookup {index} must not be used for an exact paired frame")

    def get_paired_item(self, index):
        self.requested = index
        return {"frame_idx": index}

    def get_gpu_batch_with_intrinsics(self, batch):
        assert batch["frame_idx"].item() == self.requested
        return Batch(
            rays_ori=torch.zeros(1, 1, 1, 3),
            rays_dir=torch.tensor([[[[0.0, 0.0, 1.0]]]]),
            T_to_world=torch.eye(4).unsqueeze(0),
            frame_idx=int(self.requested),
            intrinsics=[1.0, 1.0, 0.5, 0.5],
        )


def test_paired_sampler_uses_exact_frame_hook_when_available():
    graph = ViewAffinityGraph(
        offsets=np.array([0, 1, 1]),
        indices=np.array([1]),
        probabilities=np.array([1.0], dtype=np.float32),
        alias_probabilities=np.array([1.0], dtype=np.float32),
        aliases=np.array([0]),
    )
    dataset = _ExactPairDataset()
    sampled = PairedViewSampler(dataset, graph, seed=0, cache_size=0).sample(0)
    assert sampled is not None
    target, batch = sampled
    assert target == 1
    assert dataset.requested == 1
    assert batch.frame_idx == 1
