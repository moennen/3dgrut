# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The hit-distance second moment agrees with the compositing that produced the depth.

`sum(w * t^2)` is accumulated in the renderer next to the depth's `sum(w * t)`, so the two
can disagree in ways nothing else would notice: a wrong weight, a stale `t`, or an
accumulator that is simply never written. Both numeric checks here are closed-form, so they
pin the buffer to a value rather than to whatever it happened to produce first.

The two-particle case predicts the moments from two *single*-particle renders and the
front-to-back transmittance rule, rather than recomputing the Gaussian response -- a
reimplementation of the kernel's own math would drift with it and could agree with a bug.

`render.enable_depth_variance` selects a distinct compiled variant, and a process holds only
one `lib3dgut_cc` (see `load_3dgut_plugin`), so every render here runs in a child process.
Sharing the interpreter with the rest of the suite would mean rendering against whichever
binary was loaded first, which is exactly the silent failure this file is meant to catch.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

RES = 16


def _render(positions, scale: float = 0.3, opacity: float = 0.6):
    """Render in a child process; returns (depth, second moment, opacity, requires_grad)."""
    import numpy as np

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), json.dumps(positions), str(scale), str(opacity)],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=1800,
    )
    if out.returncode != 0:
        pytest.fail(f"child render failed:\n{out.stdout[-4000:]}\n{out.stderr[-4000:]}")
    payload = np.load(os.path.join(root, ".pytest_cache", "depth_variance_render.npz"))
    return payload["m1"], payload["m2"], payload["opacity"], bool(payload["requires_grad"])


def _child_render(positions, scale, opacity):
    import numpy as np
    import torch

    from threedgrut.datasets.protocols import Batch
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.build_variants import compose_config

    conf = compose_config("apps/colmap_3dgut.yaml", ["render.enable_depth_variance=true"], "configs")
    device = "cuda"
    n = len(positions)
    model = MixtureOfGaussians(conf)
    parameter = torch.nn.Parameter
    checkpoint = {
        "positions": parameter(torch.tensor(positions, device=device, dtype=torch.float32)),
        "rotation": parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, device=device)),
        "scale": parameter(torch.log(torch.full((n, 3), scale, device=device))),
        "density": parameter(torch.logit(torch.full((n, 1), opacity, device=device))),
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
        checkpoint[field] = parameter(torch.full((n, getattr(model, field).shape[1]), 0.4, device=device))
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    half = RES / 2
    focal = RES / (2 * math.tan(0.4))
    ys, xs = torch.meshgrid(
        torch.arange(RES, device=device, dtype=torch.float32),
        torch.arange(RES, device=device, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack([(xs + 0.5 - half) / focal, (ys + 0.5 - half) / focal, torch.ones_like(xs)], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    batch = Batch(
        rays_ori=torch.zeros(1, RES, RES, 3, device=device),
        rays_dir=directions.unsqueeze(0).contiguous(),
        T_to_world=torch.eye(4, device=device).unsqueeze(0),
        intrinsics=[focal, focal, half, half],
    )
    out = model(batch)
    target = os.path.join(os.getcwd(), ".pytest_cache", "depth_variance_render.npz")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    np.savez(
        target,
        m1=out["pred_dist"].detach().float().cpu().numpy(),
        m2=out["pred_dist_sq"].detach().float().cpu().numpy(),
        opacity=out["pred_opacity"].detach().float().cpu().numpy(),
        requires_grad=out["pred_dist_sq"].requires_grad,
    )


def test_second_moment_buffer_is_written():
    """An unwritten accumulator is the failure this whole term would silently survive."""
    _, m2, opacity, _ = _render([[0.0, 0.0, 2.0]])
    assert m2.shape == opacity.shape
    assert (m2[opacity > 0.1] > 0).all()


def test_single_particle_has_zero_variance():
    """All of a ray's weight sits at one distance, so the spread is exactly zero.

    This holds whatever the weight is, which is what makes it sharp: it fails if the
    accumulator uses a different weight from the depth's, or squares the wrong quantity.
    """
    m1, m2, opacity, _ = _render([[0.0, 0.0, 2.0]])
    valid = opacity > 0.05
    mean = m1[valid] / opacity[valid]
    variance = m2[valid] / opacity[valid] - mean**2
    assert valid.sum() > 0
    assert abs(variance).max() < 2e-4, f"single particle should have no spread, got {abs(variance).max()}"


def test_two_particles_match_front_to_back_prediction():
    """Predict both moments from single-particle renders and the transmittance rule.

    Rendering each particle alone gives its alpha (the accumulated opacity) and its distance
    (depth / alpha). Front-to-back compositing then fixes the two weights exactly:
    `w_near = a_near`, `w_far = a_far * (1 - a_near)`.
    """
    import numpy as np

    near, far = [0.0, 0.0, 2.0], [0.0, 0.0, 3.2]
    m1_n, _, a_n, _ = _render([near])
    m1_f, _, a_f, _ = _render([far])
    m1_both, m2_both, a_both, _ = _render([near, far])

    valid = (a_n > 0.05) & (a_f > 0.05)
    assert valid.sum() > 0
    a_near, a_far = a_n[valid], a_f[valid]
    t_near, t_far = m1_n[valid] / a_near, m1_f[valid] / a_far

    w_near = a_near
    w_far = a_far * (1 - a_near)
    np.testing.assert_allclose(a_both[valid], w_near + w_far, atol=2e-3)
    np.testing.assert_allclose(m1_both[valid], w_near * t_near + w_far * t_far, atol=5e-3)
    np.testing.assert_allclose(m2_both[valid], w_near * t_near**2 + w_far * t_far**2, atol=2e-2)


def test_second_moment_is_not_differentiable():
    """Stage 0 renders the moment but does not back-propagate it, so a loss must fail loudly.

    The forward marks the output non-differentiable, which is what keeps a later loss from
    quietly training against a zero gradient.
    """
    _, _, _, requires_grad = _render([[0.0, 0.0, 2.0]])
    assert not requires_grad


if __name__ == "__main__":
    _child_render(json.loads(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
