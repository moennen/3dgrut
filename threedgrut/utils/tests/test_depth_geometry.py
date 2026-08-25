# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from threedgrut.utils.depth_geometry import depth_gradient_magnitude, normals_from_points, unproject_to_world


def test_unproject_plane_gives_world_points() -> None:
    """A fronto-parallel plane unprojects to itself up to camera translation."""
    h, w = 4, 6
    depth = torch.full((1, h, w, 1), 2.0)
    # Rays in camera space pointing along +z, origin at zero.
    rays_dir = torch.zeros(1, h, w, 3)
    rays_dir[..., 2] = 1.0
    rays_ori = torch.zeros_like(rays_dir)
    identity = torch.eye(4).unsqueeze(0)

    points = unproject_to_world(depth, rays_ori, rays_dir, identity)
    assert points.shape == (1, h, w, 3)
    assert torch.allclose(points[..., 2], torch.full_like(points[..., 2], 2.0), atol=1e-6)


def test_unproject_with_pose_applies_transform() -> None:
    """A pure +1 world y translation moves the fronto-parallel plane up."""
    depth = torch.full((1, 2, 2, 1), 1.0)
    rays_dir = torch.zeros(1, 2, 2, 3)
    rays_dir[..., 2] = 1.0
    rays_ori = torch.zeros_like(rays_dir)
    pose = torch.eye(4).unsqueeze(0).clone()
    pose[0, :3, 3] = torch.tensor([0.0, 1.0, 0.0])

    points = unproject_to_world(depth, rays_ori, rays_dir, pose)
    assert points[0, 0, 0, 1] == pytest.approx(1.0)
    assert points[0, 0, 0, 2] == pytest.approx(1.0)


def _pinhole_rays(height: int = 6, width: int = 6):
    """Unit ray directions of a simple pinhole camera looking along +z."""
    rays_dir = torch.zeros(1, height, width, 3)
    rays_dir[..., 0] = torch.linspace(-0.2, 0.2, width)[None, None, :]
    rays_dir[..., 1] = torch.linspace(-0.2, 0.2, height)[None, :, None]
    rays_dir[..., 2] = 1.0
    return rays_dir / rays_dir.norm(dim=-1, keepdim=True)


def _fronto_parallel_plane(height: int = 6, width: int = 6, plane_z: float = 1.5):
    """Points on the plane z = `plane_z`, plus the matching unit view directions.

    Depth here is the Euclidean ray distance, as everywhere else in this codebase, so a
    plane is *not* a constant depth: the distance grows with the angle off the axis as
    `plane_z / dir_z`. Getting this wrong is what makes a depth-derived normal subtly
    curved, so the fixture derives it rather than assuming a constant.
    """
    rays_dir = _pinhole_rays(height, width)
    rays_ori = torch.zeros_like(rays_dir)
    depth = (plane_z / rays_dir[..., 2]).unsqueeze(-1)
    points = unproject_to_world(depth, rays_ori, rays_dir, torch.eye(4).unsqueeze(0))
    return points, rays_dir


def test_plane_normal_faces_the_camera_not_away() -> None:
    """A camera-facing normal points against the view ray, matching the renderer.

    The sign of a cross product of central differences depends on the pixel ordering and
    the camera handedness, so the only assertion worth making is the one the metrics rely
    on: the normal is in the opposite hemisphere to the ray that saw it.
    """
    points, rays_dir = _fronto_parallel_plane()
    normals, valid = normals_from_points(points, rays_dir, torch.ones(1, 6, 6, dtype=torch.bool))

    interior = valid[0, 1:-1, 1:-1]
    assert interior.all()
    facing = (normals * rays_dir).sum(-1)[0, 1:-1, 1:-1]
    assert bool((facing < 0.0).all())
    assert torch.allclose(normals.norm(dim=-1)[0, 1:-1, 1:-1], torch.ones(4, 4), atol=1e-5)


def test_plane_normal_is_constant_over_the_plane() -> None:
    """Every pixel of one plane must imply the same normal, up to numerical noise.

    This is the test that catches confusing ray distance with z-depth: treating the
    distance as planar bends the plane into a bowl, and the implied normals then fan out
    by tens of degrees across the image instead of agreeing.
    """
    points, rays_dir = _fronto_parallel_plane()
    normals, _ = normals_from_points(points, rays_dir, torch.ones(1, 6, 6, dtype=torch.bool))

    interior = normals[0, 1:-1, 1:-1].reshape(-1, 3)
    spread = torch.rad2deg(torch.acos((interior @ interior[0]).clamp(-1.0, 1.0)))
    assert float(spread.max()) < 1e-3
    assert torch.allclose(interior[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)


def test_constant_ray_distance_is_a_sphere_not_a_plane() -> None:
    """Constant depth means a sphere centred on the camera, whose normal is -dir exactly.

    Included because it is the one case with a closed-form answer that differs from the
    plane, so it pins the ray-distance convention from the other side.
    """

    def worst_angle(resolution: int) -> float:
        rays_dir = _pinhole_rays(resolution, resolution)
        rays_ori = torch.zeros_like(rays_dir)
        depth = torch.full((1, resolution, resolution, 1), 3.0)
        points = unproject_to_world(depth, rays_ori, rays_dir, torch.eye(4).unsqueeze(0))
        normals, _ = normals_from_points(points, rays_dir, torch.ones(1, resolution, resolution, dtype=torch.bool))
        interior = slice(1, -1)
        cosine = (normals[0, interior, interior] * -rays_dir[0, interior, interior]).sum(-1)
        return float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).max())

    # A central difference measures a secant plane rather than the tangent plane, so a
    # small residual is expected and is discretization rather than a wrong formula. It
    # does not fall off quadratically here because a pinhole grid is uniform in the image
    # plane and therefore not uniform in angle, which leaves a first-order asymmetry;
    # both resolutions are bounded instead of a convergence rate being asserted.
    assert worst_angle(8) < 0.1
    assert worst_angle(32) < 0.1


def test_orientation_is_independent_of_pixel_ordering() -> None:
    """Mirroring the point grid must not flip the normal.

    A cross product changes sign under a mirror, so a normal that relied on it would
    invert here. The view-direction orientation is what makes the result well defined.
    """
    points, rays_dir = _fronto_parallel_plane()
    normals, _ = normals_from_points(points, rays_dir, torch.ones(1, 6, 6, dtype=torch.bool))
    mirrored, _ = normals_from_points(points.flip(2), rays_dir.flip(2), torch.ones(1, 6, 6, dtype=torch.bool))

    assert torch.allclose(normals[0, 2, 2], mirrored.flip(2)[0, 2, 2], atol=1e-5)


def test_border_and_invalid_neighbours_are_excluded() -> None:
    """A normal differenced across a hole is the normal of the hole, so it is dropped."""
    points, rays_dir = _fronto_parallel_plane(height=7, width=7)
    valid = torch.ones(1, 7, 7, dtype=torch.bool)
    valid[0, 3, 3] = False

    _, usable = normals_from_points(points, rays_dir, valid)
    assert not bool(usable[0, 0, :].any())  # border has no central difference
    assert not bool(usable[0, 3, 3])  # the hole itself
    assert not bool(usable[0, 3, 2])  # and every pixel that would difference across it
    assert not bool(usable[0, 2, 3])
    assert bool(usable[0, 1, 1])


def test_gradient_flows_back_to_depth() -> None:
    """The loss differentiates the normal with respect to depth, so this must have grad."""
    rays_dir = torch.zeros(1, 5, 5, 3)
    rays_dir[..., 2] = 1.0
    rays_ori = torch.zeros_like(rays_dir)
    depth = torch.full((1, 5, 5, 1), 2.0, requires_grad=True)

    points = unproject_to_world(depth, rays_ori, rays_dir, torch.eye(4).unsqueeze(0))
    normals, _ = normals_from_points(points, rays_dir, torch.ones(1, 5, 5, dtype=torch.bool))
    normals.sum().backward()

    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


def test_depth_gradient_zero_on_flat_plane() -> None:
    depth = torch.full((1, 8, 8, 1), 2.0)
    gradient = depth_gradient_magnitude(depth)
    assert gradient.shape == (1, 8, 8)
    # The interior of a flat plane has zero relative gradient.
    assert torch.allclose(gradient[0, 1:-1, 1:-1], torch.zeros(6, 6), atol=1e-6)


def test_depth_gradient_detects_step() -> None:
    depth = torch.zeros(1, 8, 8, 1)
    depth[..., :4, :] = 1.0
    depth[..., 4:, :] = 2.0
    gradient = depth_gradient_magnitude(depth)
    assert gradient[0, 3:5, 3:5].max() > 0.5
