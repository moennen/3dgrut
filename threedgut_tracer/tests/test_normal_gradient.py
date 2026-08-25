# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference validation of the rendered-normal backward pass.

The normal buffer is accumulated in CUDA and differentiated by Slang autodiff, so nothing in
the Python layer can tell us whether the gradient it returns is actually the derivative of the
forward it ships with. These checks compare the analytic gradient against central finite
differences of the forward render, which is the only thing that catches a *wrong* Jacobian
rather than merely a non-zero one.

Two distinct code paths produce the normal gradient and both are covered:

* `k_buffer_size=0` -- the warp-reduced path. It cannot use the fused `processHitBwd` (that
  one carries no normal), so it takes a split features/density backward that accumulates into
  thread-local raw-parameter gradients.
* `k_buffer_size>0` -- the k-buffer path, which back-props through the global gradient buffer.

Process isolation: `load_3dgut_plugin` caches the compiled extension in a module global, so a
single interpreter can only ever hold one build variant. Each variant therefore runs in its own
subprocess (this file doubles as its own worker); running them in-process would silently test
the first-built variant four times over.

On discontinuities: the renderer rejects hits below a hard response/alpha threshold, so the
render is piecewise smooth in the particle parameters rather than smooth. A perturbation that
steps across such a threshold changes the loss by a finite amount, making the central
difference blow up like 1/eps. `_finite_difference` detects this by evaluating at eps and eps/2
and discarding the entry when the two disagree: a step inside the bracket doubles the quotient,
whereas a genuinely smooth entry gives the same answer twice.
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# Parameter groups the normal buffer must be differentiable with respect to.
PARAM_GROUPS = ("positions", "rotation", "scale", "density")

RESOLUTION = 16
# Small enough to stay on the smooth side of the hit threshold for these particles;
# _finite_difference also probes eps/2 and rejects the entry if the two disagree.
FD_EPS = 2.5e-4
# Relative tolerance between analytic and finite difference. Loose because the render is fp32
# built with `-use_fast_math`; observed agreement on smooth entries is 1e-3 or better.
FD_RTOL = 0.05
# Two FD estimates at eps and eps/2 must agree this closely for the entry to count as smooth.
FD_STABILITY_RTOL = 0.02

VARIANTS = [
    (primitive, k, feature_type)
    for feature_type in ("sh", "nht")
    for primitive in ("gaussian", "trisurfel")
    for k in (0, 4)
]


# ---------------------------------------------------------------------------
# Worker: everything below runs inside the per-variant subprocess.
# ---------------------------------------------------------------------------


def _make_conf(primitive_type: str, k_buffer_size: int, feature_type: str, enable_normals: bool = True):
    from threedgrut.utils.build_variants import compose_config

    app = "apps/colmap_3dgut.yaml" if feature_type == "sh" else "apps/colmap_3dgut_mcmc_nht.yaml"
    overrides = [
        f"render.primitive_type={primitive_type}",
        f"render.enable_normals={str(enable_normals).lower()}",
        f"render.splat.k_buffer_size={k_buffer_size}",
    ]
    if feature_type == "nht":
        # The shipped NHT config stores features in fp16; central differences cannot resolve
        # anything through that. The normal backward branches on feature type, not on feature
        # precision, so fp32 exercises the same code path with usable numerics.
        overrides += ["render.particle_feature_half=false", "render.feature_output_half=false"]
    return compose_config(app, overrides, str(CONFIG_DIR))


def _make_model(conf):
    """A handful of particles, placed and oriented so no parameter group is degenerate."""
    import torch

    from threedgrut.model.model import MixtureOfGaussians

    device = "cuda"
    parameter = torch.nn.Parameter

    positions = torch.tensor(
        [
            [0.00, 0.00, 2.00],
            [0.30, 0.10, 2.40],
            [-0.25, 0.20, 2.20],
            [0.10, -0.30, 2.60],
            [-0.15, -0.10, 2.90],
            [0.20, 0.25, 3.10],
            [-0.05, 0.05, 2.05],
            [0.12, -0.08, 2.75],
        ],
        device=device,
    )
    # Off-axis rotations, so the normal depends on all four quaternion components.
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
    # Anisotropic, and flat enough that the surfel plane normal is well conditioned.
    # Raw scale is log-space and raw density is logit-space (see get_scale / get_density).
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
    # Feature parameters are named differently per feature type (see model.feature_fields).
    for field in model.feature_fields():
        reference = getattr(model, field)
        values = torch.rand((count, reference.shape[1]), device=device, generator=generator) * 0.5 + 0.25
        checkpoint[field] = parameter(values)

    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()
    return model


def _make_batch():
    """A single pinhole view looking down +z, framing all the particles."""
    import torch

    from threedgrut.datasets.protocols import Batch

    device = "cuda"
    center = RESOLUTION / 2
    focal = RESOLUTION / (2 * math.tan(0.45))
    ys, xs = torch.meshgrid(
        torch.arange(RESOLUTION, device=device, dtype=torch.float32),
        torch.arange(RESOLUTION, device=device, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack([(xs + 0.5 - center) / focal, (ys + 0.5 - center) / focal, torch.ones_like(xs)], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return Batch(
        rays_ori=torch.zeros(1, RESOLUTION, RESOLUTION, 3, device=device),
        rays_dir=directions.unsqueeze(0).contiguous(),
        T_to_world=torch.eye(4, device=device).unsqueeze(0),
        intrinsics=[focal, focal, center, center],
    )


def _make_loss(model, batch):
    """A fixed random projection of the *normals only*.

    Any gradient reaching the particle parameters must therefore have travelled through the
    normal buffer, which is what makes this a test of the normal backward specifically.
    """
    import torch

    weights = torch.randn(
        1, RESOLUTION, RESOLUTION, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )

    def loss_fn():
        return (model(batch)["pred_normals"] * weights).sum()

    return loss_fn


def _analytic_gradients(model, loss_fn):
    params = {name: getattr(model, name) for name in PARAM_GROUPS}
    for tensor in params.values():
        tensor.grad = None
    loss_fn().backward()
    return params, {name: tensor.grad.clone() for name, tensor in params.items()}


def _finite_difference(tensor, index, loss_fn):
    """Central difference at FD_EPS, or None if the bracket straddles a discontinuity.

    Returns the estimate at the smaller step, which is the more accurate of the two when the
    function is smooth there.
    """
    import torch

    estimates = []
    for eps in (FD_EPS, FD_EPS / 2):
        with torch.no_grad():
            original = tensor[index].item()
            tensor[index] = original + eps
            plus = float(loss_fn())
            tensor[index] = original - eps
            minus = float(loss_fn())
            tensor[index] = original
        estimates.append((plus - minus) / (2 * eps))

    coarse, fine = estimates
    scale = max(abs(coarse), abs(fine), 1e-6)
    if abs(coarse - fine) / scale > FD_STABILITY_RTOL:
        return None  # a hit threshold flips inside the bracket: not differentiable here
    return fine


def _check_gradient_reaches_every_group(analytic):
    """A loss on the normals alone must move positions, rotation, scale and density.

    Density matters as much as the geometric groups: it is what lets a normal loss make a
    badly-oriented particle transparent instead of only rotating it.
    """
    import torch

    for name, gradient in analytic.items():
        assert torch.isfinite(gradient).all(), f"{name} gradient has non-finite entries"
        assert gradient.abs().sum() > 0, f"{name} received no gradient from the normal buffer"


def _check_matches_finite_differences(params, analytic, loss_fn):
    """The analytic normal gradient must be the derivative of the normal forward."""
    import torch

    checked = 0
    for name, gradient in analytic.items():
        tensor = params[name]
        # Probe the entries carrying the most signal: finite differences are meaningless
        # where the true gradient sits at the level of fp32 render noise.
        magnitudes = gradient.abs().flatten()
        for flat_index in torch.topk(magnitudes, min(4, magnitudes.numel())).indices.tolist():
            index = (flat_index // gradient.shape[1], flat_index % gradient.shape[1])
            expected = _finite_difference(tensor, index, loss_fn)
            if expected is None:
                continue
            actual = float(gradient[index])
            scale = max(abs(actual), abs(expected), 1e-6)
            assert (
                abs(actual - expected) / scale < FD_RTOL
            ), f"{name}{list(index)}: analytic {actual:.6e} but finite difference {expected:.6e}"
            checked += 1

    # Guard against passing vacuously because every probe was discarded as discontinuous.
    assert checked >= 2 * len(PARAM_GROUPS), f"only {checked} entries were differentiable enough to check"
    return checked


def _check_backward_leaves_forward_unchanged(model, batch, loss_fn):
    """The backward replays the compositing accumulators in place, so it must not corrupt them."""
    import torch

    before = model(batch)["pred_normals"].detach().clone()
    loss_fn().backward()
    after = model(batch)["pred_normals"].detach().clone()

    torch.testing.assert_close(before, after, rtol=0, atol=0)

    magnitudes = before.norm(dim=3)
    hit = magnitudes > 0
    assert hit.any(), "no ray hit a particle; the scene or camera is misconfigured"
    torch.testing.assert_close(magnitudes[hit], torch.ones_like(magnitudes[hit]), rtol=1e-5, atol=1e-5)


def _run_variant(primitive_type: str, k_buffer_size: int, feature_type: str) -> None:
    conf = _make_conf(primitive_type, k_buffer_size, feature_type)
    model = _make_model(conf)
    batch = _make_batch()
    loss_fn = _make_loss(model, batch)

    params, analytic = _analytic_gradients(model, loss_fn)
    _check_gradient_reaches_every_group(analytic)
    checked = _check_matches_finite_differences(params, analytic, loss_fn)
    _check_backward_leaves_forward_unchanged(model, batch, loss_fn)

    print(
        f"{primitive_type} {feature_type} k_buffer_size={k_buffer_size}: "
        f"{checked} gradient entries matched finite differences"
    )


# ---------------------------------------------------------------------------
# Test driver: one subprocess per build variant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primitive_type", "k_buffer_size", "feature_type"),
    VARIANTS,
    ids=[f"{f}-{p}-kbuffer{k}" for p, k, f in VARIANTS],
)
def test_normal_gradient_matches_finite_differences(primitive_type: str, k_buffer_size: int, feature_type: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), primitive_type, str(k_buffer_size), feature_type],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{primitive_type} {feature_type} k_buffer_size={k_buffer_size} failed\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("skipped: requires CUDA")
        sys.exit(0)
    _run_variant(sys.argv[1], int(sys.argv[2]), sys.argv[3])
