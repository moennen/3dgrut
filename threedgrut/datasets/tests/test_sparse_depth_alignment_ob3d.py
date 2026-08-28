# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-validation of the sparse-point alignment against real OB3D scenes.

The synthetic tests pin the arithmetic; this checks the number that decided the loss was worth
building. `scripts/ablation/sparse_align_diagnostic.py` measured a per-frame affine fitted to
COLMAP points at `abs_rel` 0.042 on lone-monk, 0.057 on sponza and 0.079 on emerald-square, and
the whole case for a regression term rests on those. The dataset fits the affine and the loss
converts it to ray distance, neither of which is the code the diagnostic ran, so this walks the
*training* path and asserts it lands in the same place.

That makes it the regression guard for the three silent failures the arithmetic tests can only
simulate: a downscale factor recovered wrongly from real intrinsics, a missed
`normalize_world_space` scale, and z compared against a renderer that emits ray distance. Each
leaves a plausible depth map and would move these numbers by tens of percent.

Skipped when the dataset or the prior cache is not available. Building the cache needs the
`depth_anything_3` package, so this does not build one.
"""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.utils.depth_normal_metrics import reference_depth_validity
from threedgrut.utils.pseudo_depth_loss import aligned_prior_distance

OB3D_ROOT = "/mnt/data/nerf_datasets/ob3d/OB3D_colmap"
MODEL_ID = "depth-anything/DA3MONO-LARGE"

pytestmark = pytest.mark.skipif(not os.path.isdir(OB3D_ROOT), reason=f"OB3D not available at {OB3D_ROOT}")

# Measured on the val split with `scripts/ablation/sparse_align_diagnostic.py`. The tolerance is
# wide enough to absorb the pixel population differing from the diagnostic's -- which additionally
# masks on the trained model's own opacity -- and far tighter than any of the failures above.
EXPECTED_ABS_REL = {"lone-monk": 0.042, "sponza": 0.058, "emerald-square": 0.072}
TOLERANCE = 0.010


def _dataset(scene: str) -> ColmapDataset:
    return ColmapDataset(
        os.path.join(OB3D_ROOT, scene),
        split="val",
        load_depth_gt=True,
        pseudo_depth={
            "enabled": True,
            "backend": "depth_anything_3",
            "model": MODEL_ID,
            "align_to_sparse_points": True,
        },
    )


@pytest.mark.parametrize("scene", sorted(EXPECTED_ABS_REL))
def test_training_path_alignment_matches_the_measured_quality(scene):
    try:
        dataset = _dataset(scene)
    except ImportError as exc:  # the cache is incomplete and the predictor cannot be loaded
        pytest.skip(f"pseudo-depth cache for {scene} is not populated: {exc}")

    assert dataset.pseudo_depth_alignment is not None
    fitted = [c for c in dataset.pseudo_depth_alignment if c is not None]
    assert len(fitted) == len(dataset), "every OB3D frame has enough COLMAP points to align"
    # A per-frame fit that collapsed to one shared value would mean the observations are not
    # actually varying per frame -- i.e. the wrong frame's points are being read.
    assert len(set(round(scale, 4) for scale, _ in fitted)) > 1

    errors = []
    for index in range(len(dataset)):
        batch = dataset.get_gpu_batch_with_intrinsics(default_collate([dataset[index]]))
        target = aligned_prior_distance(batch.pseudo_depth_prior, batch.pseudo_depth_affine, batch.rays_dir)
        valid = reference_depth_validity(batch.depth_gt) & torch.isfinite(target)
        if not bool(valid.any()):
            continue
        gt = batch.depth_gt[valid]
        errors.append(((target[valid] - gt).abs() / gt).double().cpu().numpy())

    abs_rel = float(np.concatenate(errors).mean())
    assert abs_rel == pytest.approx(EXPECTED_ABS_REL[scene], abs=TOLERANCE), (
        f"{scene}: aligned prior abs_rel {abs_rel:.4f} against the measured "
        f"{EXPECTED_ABS_REL[scene]:.3f}. The alignment or the z-to-distance conversion has moved."
    )
