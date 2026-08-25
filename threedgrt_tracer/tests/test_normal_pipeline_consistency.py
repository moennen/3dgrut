# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every 3DGRT pipeline must report the same rendered normal.

`reference` (hand-written CUDA) and `referenceSlang` (Slang) accumulate the normal buffer in
separate implementations, and they silently drifted apart: the CUDA one computed a ray-ellipsoid
surface normal instead of the flat-disk normal, dropped the particle rotation entirely in its
surfel branch (emitting a world-z-aligned vector), left that branch unnormalized, and used the
opposite orientation convention. Rendered normals still looked plausible, so only a
cross-pipeline comparison catches it -- the agreement below was originally ~80 degrees for
`instances` and ~147 degrees for `trisurfel`.

The shared definition is the flat-disk normal: the particle's third canonical axis carried into
world space, oriented to face the ray. See `canonicalRayNormal()` in
include/3dgrt/kernels/slang/models/gaussianParticles.slang and `processHit()` in
include/3dgrt/kernels/cuda/gaussianParticles.cuh.

Process isolation: the compiled extension is cached in a module global, so one interpreter can
only hold one build variant. Each pipeline therefore renders in its own subprocess (this file
doubles as its own worker).
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

PIPELINES = ("reference", "referenceSlang")
PRIMITIVE_TYPES = ("instances", "trisurfel")

RESOLUTION = 32
# Only compare where both pipelines actually accumulated something.
MIN_OPACITY = 0.05
# The two pipelines share the normal definition but not the arithmetic, so expect fp noise only.
MAX_ANGLE_DEGREES = 1.0


# ---------------------------------------------------------------------------
# Worker: everything below runs inside the per-pipeline subprocess.
# ---------------------------------------------------------------------------


def _render_normals(pipeline_type: str, primitive_type: str):
    import torch

    from threedgrut.datasets.protocols import Batch
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.build_variants import compose_config

    conf = compose_config(
        "apps/colmap_3dgrt.yaml",
        [
            "render.enable_normals=true",
            f"render.pipeline_type={pipeline_type}",
            f"render.primitive_type={primitive_type}",
        ],
        str(CONFIG_DIR),
    )

    device = "cuda"
    parameter = torch.nn.Parameter

    # Oriented so that no particle's third axis lines up with a world axis: a normal that drops
    # the rotation would otherwise still look correct.
    positions = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.3, 0.1, 2.4],
            [-0.25, 0.2, 2.2],
            [0.1, -0.3, 2.6],
            [-0.15, -0.1, 2.9],
            [0.2, 0.25, 3.1],
            [-0.05, 0.05, 2.05],
            [0.12, -0.08, 2.75],
        ],
        device=device,
    )
    rotation = torch.tensor(
        [
            [0.92, 0.13, -0.25, 0.28],
            [0.80, -0.30, 0.40, 0.32],
            [0.71, 0.50, 0.10, -0.48],
            [0.86, 0.20, 0.35, 0.31],
            [0.65, -0.45, 0.30, 0.53],
            [0.77, 0.34, -0.40, 0.37],
            [0.90, -0.10, 0.20, -0.35],
            [0.68, 0.42, -0.30, 0.51],
        ],
        device=device,
    )
    rotation = rotation / rotation.norm(dim=1, keepdim=True)
    # Flattened: the third axis is the meaningful normal direction.
    scale = torch.log(
        torch.tensor(
            [
                [0.30, 0.26, 0.05],
                [0.24, 0.30, 0.04],
                [0.28, 0.22, 0.06],
                [0.26, 0.28, 0.05],
                [0.22, 0.24, 0.04],
                [0.30, 0.20, 0.06],
                [0.27, 0.25, 0.05],
                [0.23, 0.29, 0.04],
            ],
            device=device,
        )
    )
    count = positions.shape[0]
    density = torch.logit(torch.full((count, 1), 0.6, device=device))
    generator = torch.Generator(device=device).manual_seed(0)

    model = MixtureOfGaussians(conf)
    checkpoint = {
        "positions": parameter(positions),
        "rotation": parameter(rotation),
        "scale": parameter(scale),
        "density": parameter(density),
        "feature_type": model.feature_type.name.lower(),
        "particle_feature_dim": model.particle_feature_dim,
        "ray_feature_dim": model.ray_feature_dim,
        "n_active_features": model.n_active_features,
        "max_n_features": model.max_n_features,
        "scene_extent": torch.tensor(1.0),
        "background": {},
        "progressive_training": False,
        "feature_dim_increase_interval": 1000,
        "feature_dim_increase_step": 1,
        "config": conf,
    }
    for field in model.feature_fields():
        reference = getattr(model, field)
        checkpoint[field] = parameter(
            torch.rand((count, reference.shape[1]), device=device, generator=generator) * 0.5 + 0.25
        )
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    half = RESOLUTION / 2
    focal = RESOLUTION / (2 * math.tan(0.45))
    ys, xs = torch.meshgrid(
        torch.arange(RESOLUTION, device=device, dtype=torch.float32),
        torch.arange(RESOLUTION, device=device, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack([(xs + 0.5 - half) / focal, (ys + 0.5 - half) / focal, torch.ones_like(xs)], -1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    batch = Batch(
        rays_ori=torch.zeros(1, RESOLUTION, RESOLUTION, 3, device=device),
        rays_dir=directions.unsqueeze(0).contiguous(),
        T_to_world=torch.eye(4, device=device).unsqueeze(0),
        intrinsics=[focal, focal, half, half],
    )

    with torch.no_grad():
        out = model(batch)
    return out["pred_normals"][0], out["pred_opacity"][0, ..., 0]


def _run_worker(pipeline_type: str, primitive_type: str, destination: str) -> None:
    import torch

    normals, opacity = _render_normals(pipeline_type, primitive_type)
    torch.save({"normals": normals.cpu(), "opacity": opacity.cpu()}, destination)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _render_in_subprocess(pipeline_type: str, primitive_type: str, destination: pathlib.Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            pipeline_type,
            primitive_type,
            str(destination),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"rendering {pipeline_type}/{primitive_type} failed\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )


@pytest.mark.parametrize("primitive_type", PRIMITIVE_TYPES)
def test_pipelines_agree_on_rendered_normals(primitive_type: str, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    rendered = {}
    for pipeline in PIPELINES:
        destination = tmp_path / f"{pipeline}_{primitive_type}.pt"
        _render_in_subprocess(pipeline, primitive_type, destination)
        rendered[pipeline] = torch.load(destination)

    first, second = (rendered[p] for p in PIPELINES)
    covered = (first["opacity"] > MIN_OPACITY) & (second["opacity"] > MIN_OPACITY)
    assert covered.sum() > 100, "scene barely covers the image; the comparison would be vacuous"

    lhs, rhs = first["normals"][covered], second["normals"][covered]

    # Both pipelines must emit unit normals; the old surfel branch was unnormalized.
    torch.testing.assert_close(lhs.norm(dim=-1), torch.ones(lhs.shape[0]), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rhs.norm(dim=-1), torch.ones(rhs.shape[0]), rtol=1e-4, atol=1e-4)

    angles = torch.rad2deg(torch.acos((lhs * rhs).sum(-1).clamp(-1.0, 1.0)))
    worst = angles.max().item()
    assert worst < MAX_ANGLE_DEGREES, (
        f"{PIPELINES[0]} and {PIPELINES[1]} disagree on {primitive_type} normals: "
        f"max={worst:.2f}deg mean={angles.mean().item():.2f}deg "
        f"({(angles > MAX_ANGLE_DEGREES).float().mean():.1%} of pixels beyond tolerance)"
    )


@pytest.mark.parametrize("pipeline_type", PIPELINES)
def test_normals_depend_on_particle_rotation(pipeline_type: str, tmp_path) -> None:
    """A normal that drops the rotation still looks smooth, so check it is not axis-aligned.

    The old CUDA surfel branch emitted `(0, 0, +-scaleRotated.z)`, giving normals along world z
    for every particle no matter how it was oriented; the x and y channels were exactly zero.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    destination = tmp_path / f"{pipeline_type}_trisurfel.pt"
    _render_in_subprocess(pipeline_type, "trisurfel", destination)
    rendered = torch.load(destination)

    normals = rendered["normals"][rendered["opacity"] > MIN_OPACITY]
    assert normals.shape[0] > 100
    for channel, name in enumerate("xy"):
        spread = normals[:, channel].abs().mean().item()
        assert spread > 0.05, (
            f"{pipeline_type} normals are world-z aligned: mean |{name}| = {spread:.4f}. "
            "The particle rotation is being dropped."
        )


if __name__ == "__main__":
    _run_worker(sys.argv[1], sys.argv[2], sys.argv[3])
