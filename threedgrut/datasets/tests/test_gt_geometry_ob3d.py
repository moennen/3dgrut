# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-validation of the ground-truth readers against real OB3D scenes.

The synthetic tests pin down the reader in isolation; these check the conventions it
assumes actually hold in the dataset. Both work through the camera geometry, so a wrong
channel mapping, a plane-distance depth or a camera-space normal all show up here.

Skipped when the dataset is not mounted.
"""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.datasets.gt_geometry import depth_validity

OB3D_ROOT = "/mnt/data/nerf_datasets/ob3d/OB3D_colmap"

pytestmark = pytest.mark.skipif(not os.path.isdir(OB3D_ROOT), reason=f"OB3D not available at {OB3D_ROOT}")

# `barbershop` names its normal channels B,G,R; `sponza` uses X,Y,Z and stores depth as a
# lone V channel. Together they cover every naming OB3D uses.
SCENES = ["barbershop", "sponza"]


def load_frame(scene: str, normalize_world_space: bool = False):
    dataset = ColmapDataset(
        os.path.join(OB3D_ROOT, scene),
        split="val",
        load_depth_gt=True,
        load_normal_gt=True,
        normalize_world_space=normalize_world_space,
    )
    batch = dataset.get_gpu_batch_with_intrinsics(default_collate([dataset[0]]))

    depth = batch.depth_gt[0, ..., 0].cpu().numpy().astype(np.float64)
    normal = batch.normal_gt[0].cpu().numpy().astype(np.float64)
    pose = batch.T_to_world[0].cpu().numpy().astype(np.float64)
    rotation, translation = pose[:3, :3], pose[:3, 3]

    origins = batch.rays_ori[0].cpu().numpy().astype(np.float64) @ rotation.T + translation
    directions = batch.rays_dir[0].cpu().numpy().astype(np.float64) @ rotation.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    valid = depth_validity(depth) & (np.linalg.norm(normal, axis=-1) > 0.5)
    return depth, normal, origins, directions, valid


def surface_agreement(depth, normal, origins, directions, valid):
    """Cosine between stored normals and normals differentiated from the depth point cloud."""
    points = origins + directions * depth[..., None]
    across = points[1:-1, 2:] - points[1:-1, :-2]
    down = points[2:, 1:-1] - points[:-2, 1:-1]
    estimated = np.cross(across, down)
    estimated /= np.maximum(np.linalg.norm(estimated, axis=-1, keepdims=True), 1e-12)

    core = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    cosine = np.abs(np.sum(estimated * normal[1:-1, 1:-1], axis=-1))
    # Finite differences are meaningless across a depth edge, so keep locally flat pixels.
    span = np.linalg.norm(across, axis=-1) + np.linalg.norm(down, axis=-1)
    flat = core & (span < np.percentile(span[core], 40))
    return cosine[flat]


@pytest.mark.parametrize("scene", SCENES)
def test_normals_are_unit_length_and_world_space(scene) -> None:
    _, normal, _, _, valid = load_frame(scene)

    np.testing.assert_allclose(np.linalg.norm(normal[valid], axis=-1), 1.0, atol=1e-3)


@pytest.mark.parametrize("scene", SCENES)
def test_visible_surfaces_face_the_camera(scene) -> None:
    """A transposed channel mapping or a camera-space convention breaks this immediately."""
    _, normal, _, directions, valid = load_frame(scene)

    facing = np.sum(normal[valid] * directions[valid], axis=-1)
    assert np.mean(facing < 0.0) > 0.99


@pytest.mark.parametrize("scene", SCENES)
def test_depth_and_normals_describe_the_same_surface(scene) -> None:
    agreement = surface_agreement(*load_frame(scene))

    assert np.median(agreement) > 0.95


@pytest.mark.parametrize("scene", SCENES)
def test_ray_distance_beats_the_plane_distance_reading(scene) -> None:
    """Confirms the documented Euclidean convention, and that this check can tell them apart."""
    depth, normal, origins, directions, valid = load_frame(scene)

    euclidean = np.median(surface_agreement(depth, normal, origins, directions, valid))
    # Reading the same values as a distance to the image plane stretches every ray by
    # 1/cos(angle to the optical axis).
    forward = np.mean(directions[valid], axis=0)
    forward /= np.linalg.norm(forward)
    cosine = np.clip(np.abs(np.einsum("ijk,k->ij", directions, forward)), 1e-9, None)
    planar = np.median(surface_agreement(depth / cosine, normal, origins, directions, valid))

    assert euclidean > planar


def test_world_normalization_preserves_the_depth_normal_relationship() -> None:
    """Scaling depth and rotating normals must keep them describing one surface."""
    scene = SCENES[0]
    raw = np.median(surface_agreement(*load_frame(scene, normalize_world_space=False)))
    normalized = np.median(surface_agreement(*load_frame(scene, normalize_world_space=True)))

    assert normalized == pytest.approx(raw, abs=1e-3)


def test_world_normalization_actually_rescales_depth() -> None:
    """Guards against the transform being silently ignored, which the check above would pass."""
    scene = SCENES[0]
    raw_depth, _, _, _, raw_valid = load_frame(scene, normalize_world_space=False)
    scaled_depth, _, _, _, scaled_valid = load_frame(scene, normalize_world_space=True)

    assert np.median(scaled_depth[scaled_valid]) < 0.5 * np.median(raw_depth[raw_valid])
