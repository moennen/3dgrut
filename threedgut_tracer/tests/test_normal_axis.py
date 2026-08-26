# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins which particle axis the rendered normal follows.

`canonicalRayNormal` returns the local z axis rotated into world space, for ellipsoids as
well as surfels -- the `Surfel` template parameter is accepted and never branched on, and the
`scale` argument is unused. So the normal is a fixed body axis, unrelated to the ellipsoid's
shape.

`loss.use_scale_flatten` depends on exactly this: it penalises `scale.z` because z is the
normal direction. Penalising `min(scale)` instead flattens along an axis the normal does not
track, producing a disk whose reported normal lies in its own plane, which measured ~30
degrees worse than not regularising at all. If someone redefines the normal as the shortest
axis (the CUDA path in `gaussianParticles.cuh` computes a true ray-ellipsoid normal, and is
dead on this path), this test fails and the loss must be updated to match.
"""

import math

import pytest

RESOLUTION = 32


def _render_single_particle(short_axis: int, primitive_type: str):
    import torch

    from threedgrut.datasets.protocols import Batch
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.build_variants import compose_config

    conf = compose_config(
        "apps/colmap_3dgut.yaml",
        ["render.enable_normals=true", f"render.primitive_type={primitive_type}"],
        "configs",
    )
    device = "cuda"
    parameter = torch.nn.Parameter

    scale = [0.30, 0.26, 0.22]
    scale[short_axis] = 0.02
    # Off-axis, so a normal that dropped the rotation would not coincidentally match.
    quaternion = torch.tensor([[0.92, 0.13, -0.25, 0.28]], device=device)
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)

    model = MixtureOfGaussians(conf)
    checkpoint = {
        "positions": parameter(torch.tensor([[0.0, 0.0, 2.0]], device=device)),
        "rotation": parameter(quaternion),
        "scale": parameter(torch.log(torch.tensor([scale], device=device))),
        "density": parameter(torch.logit(torch.full((1, 1), 0.99, device=device))),
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
    generator = torch.Generator(device=device).manual_seed(0)
    for field in model.feature_fields():
        reference = getattr(model, field)
        checkpoint[field] = parameter(
            torch.rand((1, reference.shape[1]), device=device, generator=generator) * 0.5 + 0.25
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

    covered = out["pred_opacity"][0, ..., 0] > 0.5
    assert covered.any(), "particle not visible; the probe would prove nothing"
    normal = torch.nn.functional.normalize(out["pred_normals"][0][covered].mean(0), dim=0)

    w, x, y, z = model.get_rotation()[0].tolist()
    axes = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        device=device,
    )
    return [abs(float(torch.dot(normal, axes[:, axis]))) for axis in range(3)]


@pytest.mark.parametrize("short_axis", [0, 1, 2])
def test_the_rendered_normal_is_the_z_axis_whichever_axis_is_shortest(short_axis: int) -> None:
    """The normal must not follow the shortest axis, or `use_scale_flatten` is wrong."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    alignment = _render_single_particle(short_axis, "instances")
    assert alignment[2] > 0.99, f"normal is not the z axis: alignments {alignment}"
