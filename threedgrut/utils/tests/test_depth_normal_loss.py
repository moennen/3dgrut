# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from threedgrut.utils.depth_geometry import normals_from_points, unproject_to_world, world_ray_dirs
from threedgrut.utils.depth_normal_loss import depth_normal_consistency_loss

# The term runs every training iteration on device, so it is exercised on the GPU whenever
# one is present rather than only on the CPU fallback.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def scene(device: str, height: int = 8, width: int = 8, plane_z: float = 2.0, opacity: float = 1.0):
    """A fronto-parallel plane rendered opaquely, with its exact depth-implied normals.

    Returns the tracer-shaped buffers plus the normal the loss should be driving towards,
    so a test can construct both the agreeing and the disagreeing case from one fixture.
    """
    rays_dir = torch.zeros(1, height, width, 3, device=device)
    rays_dir[..., 0] = torch.linspace(-0.2, 0.2, width, device=device)[None, None, :]
    rays_dir[..., 1] = torch.linspace(-0.2, 0.2, height, device=device)[None, :, None]
    rays_dir[..., 2] = 1.0
    rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
    rays_ori = torch.zeros_like(rays_dir)
    T_to_world = torch.eye(4, device=device).unsqueeze(0)

    depth = (plane_z / rays_dir[..., 2]).unsqueeze(-1)
    pred_opacity = torch.full((1, height, width, 1), opacity, device=device)
    # The tracer accumulates alpha-premultiplied distance, so pred_dist is depth * opacity.
    pred_dist = depth * pred_opacity

    points = unproject_to_world(depth, rays_ori, rays_dir, T_to_world)
    derived, _ = normals_from_points(
        points, world_ray_dirs(rays_dir, T_to_world), torch.ones(1, height, width, dtype=torch.bool, device=device)
    )
    return dict(
        pred_dist=pred_dist,
        pred_opacity=pred_opacity,
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=T_to_world,
        derived=derived,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_agreeing_normal_costs_nothing(device: str) -> None:
    """A normal already matching the depth-implied one must give exactly zero loss."""
    data = scene(device)
    loss, count = depth_normal_consistency_loss(
        data["pred_dist"],
        data["pred_opacity"],
        data["derived"].clone(),
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
    )

    assert int(count) > 0
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_inverted_normal_costs_the_maximum(device: str) -> None:
    """`1 - cos` is 2 for a normal pointing exactly the wrong way, which bounds the term."""
    data = scene(device)
    loss, _ = depth_normal_consistency_loss(
        data["pred_dist"],
        data["pred_opacity"],
        -data["derived"],
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
    )
    assert float(loss) == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_tilted_normal_costs_one_minus_cosine(device: str) -> None:
    """The value is the actual angle, not merely something monotone in it."""
    data = scene(device)
    angle = torch.tensor(0.3, device=device)
    # Rotate the target normal about the x axis by a known angle.
    tilted = data["derived"].clone()
    tilted[..., 1] = torch.sin(angle) * data["derived"][..., 2].abs()
    tilted[..., 2] = torch.cos(angle) * data["derived"][..., 2]
    tilted = tilted / tilted.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    loss, _ = depth_normal_consistency_loss(
        data["pred_dist"],
        data["pred_opacity"],
        tilted,
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
    )
    # The plane's implied normal is close to -z, so the tilt is very nearly `angle`.
    assert float(loss) == pytest.approx(float(1.0 - torch.cos(angle)), abs=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_transparent_rays_are_excluded_not_averaged_in(device: str) -> None:
    """A ray too transparent to carry a surface must not dilute the mean.

    This is the normalisation choice the reference implementation makes differently: it
    fills invalid pixels with zero and takes a full mean, so the effective weight drifts
    with the valid fraction. Here half the image going transparent must leave the value of
    the remaining half untouched.
    """
    opaque = scene(device)
    dense, _ = depth_normal_consistency_loss(
        opaque["pred_dist"],
        opaque["pred_opacity"],
        -opaque["derived"],
        opaque["rays_ori"],
        opaque["rays_dir"],
        opaque["T_to_world"],
    )

    holed = scene(device)
    opacity = holed["pred_opacity"].clone()
    opacity[:, :, :4] = 0.0  # left half sees nothing at all
    sparse, count = depth_normal_consistency_loss(
        holed["pred_dist"],
        opacity,
        -holed["derived"],
        holed["rays_ori"],
        holed["rays_dir"],
        holed["T_to_world"],
    )

    assert int(count) > 0
    assert float(sparse) == pytest.approx(float(dense), abs=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_a_fully_transparent_frame_is_zero_and_still_differentiable(device: str) -> None:
    """No valid pixel must yield zero without dividing by zero or detaching the graph."""
    data = scene(device, opacity=0.0)
    normals = data["derived"].clone().requires_grad_(True)
    loss, count = depth_normal_consistency_loss(
        data["pred_dist"],
        data["pred_opacity"],
        normals,
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
    )

    assert int(count) == 0
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert normals.grad is not None
    assert torch.isfinite(normals.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_reaches_both_the_depth_and_the_normal(device: str) -> None:
    """The term is a mutual constraint, so neither side may be detached.

    Measurement found the depth-implied normal to be a worse target than trisurfel's own
    rendered normal, so the term is only justified if it also acts on the depth. A
    regression that detached the depth would leave this test as the only witness.
    """
    data = scene(device)
    pred_dist = data["pred_dist"].clone().requires_grad_(True)
    # A generic tilt, not the inverted normal: `1 - cos` is stationary at exact
    # antiparallelism, so a test posed there would report a zero gradient that says
    # nothing about whether the term is connected.
    tilted = data["derived"].clone()
    tilted[..., 0] += 0.35
    normals = (tilted / tilted.norm(dim=-1, keepdim=True)).clone().requires_grad_(True)

    loss, _ = depth_normal_consistency_loss(
        pred_dist,
        data["pred_opacity"],
        normals,
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
    )
    loss.backward()

    assert pred_dist.grad is not None and torch.isfinite(pred_dist.grad).all()
    assert normals.grad is not None and torch.isfinite(normals.grad).all()
    assert float(pred_dist.grad.abs().sum()) > 0.0
    assert float(normals.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_percentile_drops_the_steep_tail(device: str) -> None:
    """The occlusion-boundary trim must exclude a depth step, not merely reweight it."""
    data = scene(device)
    pred_dist = data["pred_dist"].clone()
    # A step down the middle: a boundary whose implied normal describes neither surface.
    pred_dist[:, :, 4:] *= 2.0

    unmasked, unmasked_count = depth_normal_consistency_loss(
        pred_dist,
        data["pred_opacity"],
        data["derived"],
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
        grad_percentile=None,
    )
    masked, masked_count = depth_normal_consistency_loss(
        pred_dist,
        data["pred_opacity"],
        data["derived"],
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
        grad_percentile=0.5,
    )

    assert int(masked_count) < int(unmasked_count)
    assert float(masked) < float(unmasked)


@pytest.mark.parametrize("device", DEVICES)
def test_loss_and_count_stay_on_the_input_device(device: str) -> None:
    """Returning device tensors is what keeps the per-iteration call free of a host sync."""
    data = scene(device)
    loss, count = depth_normal_consistency_loss(
        data["pred_dist"],
        data["pred_opacity"],
        data["derived"],
        data["rays_ori"],
        data["rays_dir"],
        data["T_to_world"],
        grad_percentile=0.9,
    )
    assert loss.device.type == device
    assert count.device.type == device
