"""Tracer contract for the alpha-composited appearance second moment.

The rendered vector is RGB in SH mode and the pre-decoder latent in NHT mode.  This test
pins the renderer-side identity that is independent of either representation: a ray with
one contributor has ``sum(w*f²) == sum(w*f)² / sum(w)``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.build_variants import compose_config


def _check_single_hit_feature_second_moment(config_name: str, overrides: list[str] | None = None):
    conf = compose_config(
        config_name,
        ["render.enable_appearance_variance=true", "render.particle_kernel_min_response=0.0001", *(overrides or [])],
        "configs",
    )
    model = MixtureOfGaussians(conf)
    device = "cuda"
    parameter = torch.nn.Parameter
    checkpoint = {
        "positions": parameter(torch.tensor([[0.0, 0.0, 2.0]], device=device)),
        "rotation": parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)),
        "scale": parameter(torch.log(torch.full((1, 3), 0.3, device=device))),
        "density": parameter(torch.logit(torch.full((1, 1), 0.7, device=device))),
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
        checkpoint[field] = parameter(torch.full((1, getattr(model, field).shape[1]), 0.4, device=device))
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    resolution = 16
    focal = resolution / (2 * math.tan(0.4))
    ys, xs = torch.meshgrid(
        torch.arange(resolution, device=device), torch.arange(resolution, device=device), indexing="ij"
    )
    directions = torch.stack(
        [(xs + 0.5 - resolution / 2) / focal, (ys + 0.5 - resolution / 2) / focal, torch.ones_like(xs)], dim=-1
    ).float()
    directions = directions / directions.norm(dim=-1, keepdim=True)
    batch = Batch(
        rays_ori=torch.zeros(1, resolution, resolution, 3, device=device),
        rays_dir=directions.unsqueeze(0),
        T_to_world=torch.eye(4, device=device).unsqueeze(0),
        intrinsics=[focal, focal, resolution / 2, resolution / 2],
    )
    outputs = model(batch)
    valid = outputs["pred_opacity"] > 0.1
    expected = outputs["pred_features"].square() / outputs["pred_opacity"].clamp_min(1e-8)
    assert valid.any()
    # NHT's primary feature buffer is intentionally fp16 in this app, whereas the moment is
    # fp32.  The identity is therefore checked to the quantization envelope of the primary.
    torch.testing.assert_close(
        outputs["pred_feature_sq"][valid.expand_as(expected)],
        expected[valid.expand_as(expected)],
        rtol=2e-3,
        atol=4e-4,
    )
    # Exercise the renderer's reverse compositing too. In NHT this reaches the interpolated
    # latent feature parameters (rather than pretending the decoded RGB has a usable moment).
    outputs["pred_feature_sq"].sum().backward()
    assert sum(getattr(model, field).grad.abs().sum() for field in model.feature_fields()) > 0


def test_single_hit_feature_second_moment_matches_primary_feature():
    # 3DGUT extensions are process-global and the two feature modes compile distinct binaries.
    # Isolate each mode so this test validates the same one-variant-per-process rule as training.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    configurations = (
        ("apps/colmap_3dgut.yaml", []),
        ("apps/colmap_3dgut.yaml", ["render.primitive_type=trisurfel"]),
        ("apps/colmap_3dgut_mcmc_nht.yaml", []),
        ("apps/colmap_3dgut_mcmc_nht.yaml", ["render.primitive_type=trisurfel", "render.splat.k_buffer_size=4"]),
    )
    for config_name, overrides in configurations:
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), json.dumps([config_name, overrides])],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]


if __name__ == "__main__":
    _config_name, _overrides = json.loads(sys.argv[1])
    _check_single_hit_feature_second_moment(_config_name, _overrides)
