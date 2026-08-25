# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth geometry loading in `ColmapDataset`, and the opt-in guarantee."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.datasets.tests.test_gt_geometry import similarity, write_exr

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs")


def write_images(tmp_path, frames: int, height=2, width=3) -> None:
    """RGB frames, which `__getitem__` reads to discover the resolution."""
    os.makedirs(tmp_path / "images", exist_ok=True)
    for i in range(frames):
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / "images" / f"{i:05d}_rgb.png")


def build_dataset(tmp_path, monkeypatch, *, frames=4, split="val", **kwargs) -> ColmapDataset:
    """A `ColmapDataset` whose camera loading is stubbed, so only GT wiring is exercised."""
    write_images(tmp_path, frames)
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.path = str(tmp_path)
    dataset.device = "cpu"
    dataset.split = split
    dataset.normalize_world_space = False
    dataset.world_normalization_transform = np.eye(4, dtype=np.float32)
    dataset._worker_gpu_cache = {}
    dataset._all_exif_exposures = None
    dataset.test_split_interval = 0
    dataset.load_depth_gt = False
    dataset.load_normal_gt = False
    for key, value in kwargs.items():
        setattr(dataset, key, value)

    image_paths = np.array([os.path.join(str(tmp_path), "images", f"{i:05d}_rgb.png") for i in range(frames)])

    def load_intrinsics_and_extrinsics() -> None:
        dataset.cam_intrinsics = {1: object()}
        dataset.cam_extrinsics = [SimpleNamespace(camera_id=1) for _ in range(frames)]

    def load_camera_data() -> None:
        dataset.poses = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        dataset.image_paths = image_paths
        dataset.mask_paths = np.array(["missing.png"] * frames)
        dataset.camera_centers = np.zeros((frames, 3), dtype=np.float32)

    monkeypatch.setattr(dataset, "load_intrinsics_and_extrinsics", load_intrinsics_and_extrinsics)
    monkeypatch.setattr(dataset, "_filter_cameras", lambda: list(range(frames)))
    monkeypatch.setattr(dataset, "load_camera_data", load_camera_data)
    monkeypatch.setattr(
        dataset,
        "compute_spatial_extents",
        lambda: (torch.zeros(3), torch.tensor(1.0), (torch.zeros(3), torch.zeros(3))),
    )
    dataset.reload()
    return dataset


def write_gt(tmp_path, frames: int, *, depth=True, normal=True, height=2, width=3) -> None:
    if depth:
        os.makedirs(tmp_path / "depths", exist_ok=True)
    if normal:
        os.makedirs(tmp_path / "normals", exist_ok=True)
    for i in range(frames):
        if depth:
            values = np.full((height, width), float(i + 1), dtype=np.float32)
            write_exr(tmp_path / "depths" / f"{i:05d}_depth.exr", {"V": values})
        if normal:
            zeros = np.zeros((height, width), dtype=np.float32)
            ones = np.ones((height, width), dtype=np.float32)
            write_exr(tmp_path / "normals" / f"{i:05d}_normal.exr", {"X": zeros, "Y": zeros, "Z": ones})


# --- the opt-in guarantee ------------------------------------------------------------


def test_ground_truth_is_not_touched_when_disabled(tmp_path, monkeypatch) -> None:
    """With the flags off, a scene whose ground truth is unreadable must still load."""
    os.makedirs(tmp_path / "depths")
    (tmp_path / "depths" / "00000_depth.exr").write_text("not an exr")

    dataset = build_dataset(tmp_path, monkeypatch)

    assert dataset.depth_gt_available is False
    assert dataset.normal_gt_available is False
    assert "depth_gt" not in dataset[0] and "normal_gt" not in dataset[0]


def test_colmap_defaults_keep_ground_truth_off() -> None:
    """The shipped configuration must not change behaviour for existing users."""
    config = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", "colmap.yaml"))

    assert config.load_depth_gt is False
    assert config.load_normal_gt is False


# --- loading -------------------------------------------------------------------------


def test_depth_and_normals_reach_the_sample(tmp_path, monkeypatch) -> None:
    write_gt(tmp_path, 4)

    dataset = build_dataset(tmp_path, monkeypatch, load_depth_gt=True, load_normal_gt=True)
    sample = dataset[2]

    assert dataset.depth_gt_available and dataset.normal_gt_available
    assert sample["depth_gt"].shape == (1, 2, 3, 1)
    assert sample["normal_gt"].shape == (1, 2, 3, 3)
    # Frame 2 was written with the constant value 3.0.
    assert torch.allclose(sample["depth_gt"], torch.full((1, 2, 3, 1), 3.0))
    assert torch.allclose(sample["normal_gt"][0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))


def test_each_flag_works_on_its_own(tmp_path, monkeypatch) -> None:
    write_gt(tmp_path, 2, normal=False)

    dataset = build_dataset(tmp_path, monkeypatch, frames=2, load_depth_gt=True)

    assert dataset.depth_gt_available and not dataset.normal_gt_available
    assert "depth_gt" in dataset[0] and "normal_gt" not in dataset[0]


def test_ground_truth_is_index_aligned_with_the_split(tmp_path, monkeypatch) -> None:
    """The val split keeps every 8th frame, so GT must be re-indexed alongside the images."""
    write_gt(tmp_path, 16)

    dataset = build_dataset(tmp_path, monkeypatch, frames=16, split="val", test_split_interval=8, load_depth_gt=True)

    assert len(dataset.depth_paths) == dataset.n_frames == 2
    for idx in range(dataset.n_frames):
        stem = os.path.basename(str(dataset.image_paths[idx])).removesuffix("_rgb.png")
        assert os.path.basename(dataset.depth_paths[idx]) == f"{stem}_depth.exr"
        # Frame i was written with depth i+1, which pins the alignment numerically.
        expected = float(int(stem) + 1)
        assert dataset[idx]["depth_gt"].flatten()[0].item() == pytest.approx(expected)


def test_missing_ground_truth_fails_loudly(tmp_path, monkeypatch) -> None:
    write_gt(tmp_path, 4)
    os.remove(tmp_path / "depths" / "00002_depth.exr")

    with pytest.raises(FileNotFoundError, match="1 of 4 frames"):
        build_dataset(tmp_path, monkeypatch, load_depth_gt=True)


def test_absent_folder_names_the_flag_to_disable(tmp_path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError, match="dataset.load_depth_gt"):
        build_dataset(tmp_path, monkeypatch, load_depth_gt=True)


def test_ground_truth_is_resampled_to_the_image_size(tmp_path, monkeypatch) -> None:
    """Downsampled RGB must not silently mismatch full-resolution ground truth."""
    write_gt(tmp_path, 1, height=4, width=6)

    dataset = build_dataset(tmp_path, monkeypatch, frames=1, load_depth_gt=True, load_normal_gt=True)
    depth = dataset._load_depth_gt(0, 2, 3)
    normal = dataset._load_normal_gt(0, 2, 3)

    assert depth.shape == (2, 3, 1)
    assert normal.shape == (2, 3, 3)


# --- world normalization -------------------------------------------------------------


def test_ground_truth_follows_the_normalized_world(tmp_path, monkeypatch) -> None:
    """Depth must be rescaled and normals rotated, or metrics compare different spaces."""
    write_gt(tmp_path, 1)
    transform, rotation = similarity(0.25)

    dataset = build_dataset(tmp_path, monkeypatch, frames=1, load_depth_gt=True, load_normal_gt=True)
    dataset.normalize_world_space = True
    dataset.world_normalization_transform = transform.astype(np.float32)

    # Frame 0 holds depth 1.0 and the +Z normal.
    np.testing.assert_allclose(dataset._load_depth_gt(0, 2, 3)[0, 0, 0], 0.25, atol=1e-6)
    np.testing.assert_allclose(dataset._load_normal_gt(0, 2, 3)[0, 0], rotation @ np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_ground_truth_is_untouched_without_normalization(tmp_path, monkeypatch) -> None:
    write_gt(tmp_path, 1)

    dataset = build_dataset(tmp_path, monkeypatch, frames=1, load_depth_gt=True, load_normal_gt=True)

    np.testing.assert_allclose(dataset._load_depth_gt(0, 2, 3)[0, 0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(dataset._load_normal_gt(0, 2, 3)[0, 0], [0.0, 0.0, 1.0], atol=1e-6)
