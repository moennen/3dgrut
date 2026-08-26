# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference validation of the depth second-moment backward pass.

`sum(w * t^2)` is composited alongside the depth's `sum(w * t)`, and on the hand-written CUDA
path its backward is folded into the *same* two intermediate gradients the depth uses
(`galphaRayHitGrd` and `grdsRayHitGrd` in `gaussianParticles.cuh`). That sharing is what makes
the term cheap, and also what makes it dangerous: a sign error or a missing factor would leak
into the depth gradient rather than merely producing a wrong moment gradient. Central
differences of the forward render are the only check that distinguishes the correct Jacobian
from a plausible one.

Three losses are differentiated, which is the point of the file:

* the moment alone -- isolates the new backward, since no gradient can arrive by another route;
* the depth alone -- pins that adding the moment did not disturb the depth gradient, which
  shares the accumulators;
* the two summed -- pins that the shared folding adds rather than overwrites. A backward that
  handled each loss correctly in isolation but assigned instead of accumulating into the
  shared gradient would pass the first two checks and fail this one.

The renderer has *three* backward compositing paths and the moment had to be taught to all
three, so each is covered (`BACKWARD_PATHS`). They are selected by configuration rather than
named in the API, which is exactly why a test that exercised only the default would have left
two thirds of the feature unverified:

* `cuda` -- `k_buffer_size=0`, normals off: the fused hand-written `processHitBwd`.
* `slang_raw` -- `k_buffer_size=0`, normals on: Slang `...BwdToRawParameters`, warp-reduced.
* `slang_buffer` -- `k_buffer_size>0`: Slang `...BwdToBuffer`, via the global gradient buffer.
  Reached with normals either way, so both are checked for the moment-only loss.

Process isolation and the discontinuity handling follow `test_normal_gradient.py`: one
subprocess per build variant because `load_3dgut_plugin` caches a single compiled extension,
and finite differences evaluated at both eps and eps/2 so an entry whose bracket straddles a
hit-acceptance threshold is discarded rather than reported as a mismatch.
"""

from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

PARAM_GROUPS = ("positions", "rotation", "scale", "density")

RESOLUTION = 16
FD_EPS = 2.5e-4
# Loose because the render is fp32 with `-use_fast_math`; smooth entries agree far better.
FD_RTOL = 0.05
FD_STABILITY_RTOL = 0.02

LOSS_KINDS = ("moment", "depth", "both")
PRIMITIVES = ("gaussian", "trisurfel")

# name -> (enable_normals, k_buffer_size); see the module docstring.
BACKWARD_PATHS = {
    "cuda": (False, 0),
    "slang_raw": (True, 0),
    "slang_buffer": (False, 4),
    "slang_buffer_normals": (True, 4),
}


# ---------------------------------------------------------------------------
# Worker: everything below runs inside the per-variant subprocess.
# ---------------------------------------------------------------------------


def _make_conf(primitive_type: str, path: str = "cuda", enable_depth_variance: bool = True):
    from threedgrut.utils.build_variants import compose_config

    enable_normals, k_buffer_size = BACKWARD_PATHS[path]
    return compose_config(
        "apps/colmap_3dgut.yaml",
        [
            f"render.primitive_type={primitive_type}",
            f"render.enable_depth_variance={str(enable_depth_variance).lower()}",
            f"render.enable_normals={str(enable_normals).lower()}",
            f"render.splat.k_buffer_size={k_buffer_size}",
        ],
        str(CONFIG_DIR),
    )


def _make_model(conf):
    """Particles spread along the ray, so a ray accumulates weight at several distances.

    A single particle per ray would give every ray zero variance and a moment gradient that
    is degenerate with the depth's, which would hide exactly the errors this file looks for.
    """
    import torch

    from threedgrut.model.model import MixtureOfGaussians

    device = "cuda"
    parameter = torch.nn.Parameter

    positions = torch.tensor(
        [
            [0.00, 0.00, 2.00],
            [0.03, 0.02, 2.45],
            [-0.02, 0.04, 2.85],
            [0.30, 0.10, 2.40],
            [-0.25, 0.20, 2.20],
            [0.10, -0.30, 2.60],
            [-0.15, -0.10, 2.90],
            [0.20, 0.25, 3.10],
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
    # Anisotropic so the rotation genuinely affects the hit distance.
    scale = torch.log(
        torch.tensor(
            [
                [0.30, 0.20, 0.12],
                [0.24, 0.30, 0.10],
                [0.28, 0.18, 0.14],
                [0.26, 0.28, 0.11],
                [0.22, 0.24, 0.13],
                [0.30, 0.20, 0.10],
                [0.27, 0.25, 0.12],
                [0.23, 0.29, 0.14],
            ],
            device=device,
        )
    )
    count = positions.shape[0]
    # Partly transparent, so several particles contribute to each ray's moment.
    density = torch.logit(torch.full((count, 1), 0.45, device=device))
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
        values = torch.rand((count, reference.shape[1]), device=device, generator=generator) * 0.5 + 0.25
        checkpoint[field] = parameter(values)

    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()
    return model


def _make_batch():
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


def _make_loss(model, batch, kind: str):
    """A fixed random projection of the moment, the depth, or their sum.

    Random weights rather than a plain sum: a uniform weighting would let a per-pixel sign
    error cancel in the total and go unnoticed.
    """
    import torch

    generator = torch.Generator(device="cuda").manual_seed(1)
    weights_moment = torch.randn(1, RESOLUTION, RESOLUTION, 1, device="cuda", generator=generator)
    weights_depth = torch.randn(1, RESOLUTION, RESOLUTION, 1, device="cuda", generator=generator)

    def loss_fn():
        out = model(batch)
        total = 0.0
        if kind in ("moment", "both"):
            total = total + (out["pred_dist_sq"] * weights_moment).sum()
        if kind in ("depth", "both"):
            total = total + (out["pred_dist"] * weights_depth).sum()
        return total

    return loss_fn


def _analytic_gradients(model, loss_fn):
    params = {name: getattr(model, name) for name in PARAM_GROUPS}
    for tensor in params.values():
        tensor.grad = None
    loss_fn().backward()
    return params, {name: tensor.grad.clone() for name, tensor in params.items()}


def _finite_difference(tensor, index, loss_fn):
    """Central difference at FD_EPS, or None if the bracket straddles a discontinuity."""
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
        return None
    return fine


def _check_matches_finite_differences(params, analytic, loss_fn, probes: int = 4):
    import torch

    checked = 0
    mismatches = []
    for name, gradient in analytic.items():
        tensor = params[name]
        assert torch.isfinite(gradient).all(), f"{name} gradient has non-finite entries"
        magnitudes = gradient.abs().flatten()
        for flat_index in torch.topk(magnitudes, min(probes, magnitudes.numel())).indices.tolist():
            index = (flat_index // gradient.shape[1], flat_index % gradient.shape[1])
            expected = _finite_difference(tensor, index, loss_fn)
            if expected is None:
                continue
            actual = float(gradient[index])
            scale = max(abs(actual), abs(expected), 1e-6)
            if abs(actual - expected) / scale >= FD_RTOL:
                mismatches.append(f"{name}{list(index)}: analytic {actual:.6e} vs finite diff {expected:.6e}")
            checked += 1

    assert not mismatches, "analytic gradient disagrees with finite differences:\n" + "\n".join(mismatches)
    assert checked >= 4, f"only {checked} entries were differentiable enough to check"
    return checked


def _check_moment_gradient_reaches_geometry(analytic):
    """A loss on the moment alone must move position, scale and density.

    Those three are what a variance penalty has to act on to concentrate a ray: move the
    particle along the ray, shrink it, or fade it out.
    """
    import torch

    for name in ("positions", "scale", "density"):
        gradient = analytic[name]
        assert torch.isfinite(gradient).all(), f"{name} gradient has non-finite entries"
        assert gradient.abs().sum() > 0, f"{name} received no gradient from the second moment"


def _check_backward_leaves_forward_unchanged(model, batch, loss_fn):
    """The backward unwinds the moment accumulator in place, so it must not corrupt it."""
    import torch

    before = model(batch)["pred_dist_sq"].detach().clone()
    loss_fn().backward()
    after = model(batch)["pred_dist_sq"].detach().clone()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert (before > 0).any(), "no ray accumulated a second moment; the scene is misconfigured"


def _check_depth_gradient_unaffected_by_the_moment(primitive_type: str, path: str):
    """Compiling the moment in must not change the depth gradient it shares accumulators with.

    Rendered with the moment enabled and disabled, a depth-only loss must produce the same
    parameter gradient. This is the regression the shared `galphaRayHitGrd` folding risks, and
    it cannot be checked inside one process because the two are different compiled variants --
    so the disabled reference is produced by a nested subprocess.
    """
    import json
    import subprocess as sp

    import torch

    conf = _make_conf(primitive_type, path, enable_depth_variance=True)
    model = _make_model(conf)
    batch = _make_batch()
    _, analytic = _analytic_gradients(model, _make_loss(model, batch, "depth"))

    completed = sp.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "depth-reference", primitive_type, path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(f"reference render failed\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
    reference = json.loads(completed.stdout.strip().splitlines()[-1])

    for name, gradient in analytic.items():
        expected = torch.tensor(reference[name], device=gradient.device).reshape(gradient.shape)
        torch.testing.assert_close(gradient, expected, rtol=2e-3, atol=1e-6)


def _run_variant(primitive_type: str, kind: str, path: str) -> None:
    conf = _make_conf(primitive_type, path)
    model = _make_model(conf)
    batch = _make_batch()
    loss_fn = _make_loss(model, batch, kind)

    params, analytic = _analytic_gradients(model, loss_fn)
    if kind == "moment":
        _check_moment_gradient_reaches_geometry(analytic)
    checked = _check_matches_finite_differences(params, analytic, loss_fn)
    if kind == "moment":
        _check_backward_leaves_forward_unchanged(model, batch, loss_fn)

    print(f"{primitive_type} {path} loss={kind}: {checked} gradient entries matched finite differences")


def _run_depth_reference(primitive_type: str, path: str) -> None:
    """Print the depth-only gradient from a build with the moment compiled *out*."""
    import json

    conf = _make_conf(primitive_type, path, enable_depth_variance=False)
    model = _make_model(conf)
    batch = _make_batch()
    _, analytic = _analytic_gradients(model, _make_loss(model, batch, "depth"))
    print(json.dumps({name: gradient.flatten().tolist() for name, gradient in analytic.items()}))


def _run_disabled_refusal() -> None:
    """With the feature off the buffer is empty, so a loss on it must raise, not read zero.

    This is the failure mode a config typo produces: `enable_depth_variance` left unset while a
    variance loss is switched on. An empty tensor sums to 0.0 and would train as a no-op, so the
    forward marks it non-differentiable and autograd refuses instead.
    """
    conf = _make_conf("gaussian", "cuda", enable_depth_variance=False)
    model = _make_model(conf)
    batch = _make_batch()
    out = model(batch)
    assert out["pred_dist_sq"].numel() == 0, "the moment should not be rendered when disabled"
    assert not out["pred_dist_sq"].requires_grad, "a disabled moment must not claim to be differentiable"
    try:
        out["pred_dist_sq"].sum().backward()
    except RuntimeError:
        print("disabled moment correctly refused a gradient")
        return
    raise AssertionError("a loss on the disabled moment silently succeeded")


# ---------------------------------------------------------------------------
# Test driver: one subprocess per build variant.
# ---------------------------------------------------------------------------


def _run_worker(*args: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(f"{' '.join(args)} failed\n--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}")


# The full loss matrix on one primitive and every backward path; the second primitive is
# covered for the moment-only loss below. Each distinct (primitive, path) pair is a separate
# compiled variant, so widening this multiplies build time rather than test time.
@pytest.mark.parametrize("path", ["cuda", "slang_raw", "slang_buffer"])
@pytest.mark.parametrize("kind", LOSS_KINDS)
def test_second_moment_gradient_matches_finite_differences(kind: str, path: str) -> None:
    _run_worker("gaussian", kind, path)


@pytest.mark.parametrize("path", sorted(BACKWARD_PATHS))
@pytest.mark.parametrize("primitive_type", PRIMITIVES)
def test_second_moment_gradient_on_every_backward_path(primitive_type: str, path: str) -> None:
    """Each compositing path must differentiate the moment, for both primitives.

    `slang_buffer_normals` is here rather than in the matrix above because it is the
    configuration a depth-variance loss has to share with the depth-normal loss of item 5,
    which needs normals compiled in.
    """
    _run_worker(primitive_type, "moment", path)


@pytest.mark.parametrize("primitive_type", PRIMITIVES)
def test_depth_gradient_unchanged_by_compiling_the_moment_in(primitive_type: str) -> None:
    _run_worker("depth-unaffected", primitive_type, "cuda")


def test_second_moment_gradient_is_refused_when_disabled() -> None:
    _run_worker("disabled")


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("skipped: requires CUDA")
        sys.exit(0)
    if sys.argv[1] == "depth-reference":
        _run_depth_reference(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "depth-unaffected":
        _check_depth_gradient_unaffected_by_the_moment(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "disabled":
        _run_disabled_refusal()
    else:
        _run_variant(sys.argv[1], sys.argv[2], sys.argv[3])
