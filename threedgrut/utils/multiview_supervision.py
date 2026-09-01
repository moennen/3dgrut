"""Primitive-agnostic reprojection and configurable multi-view losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.depth_geometry import unproject_to_world


@dataclass(frozen=True)
class MultiViewLossWeights:
    point: float = 0.0
    normal: float = 0.0
    raw_feature_l2: float = 0.0
    zncc: float = 0.0


def _intrinsics(
    batch: Batch, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pinhole intrinsics, including COLMAP OpenCV-pinhole camera batches.

    The renderer supports additional lens models, but a first reprojector must not silently
    treat a fisheye ray as pinhole. Those configurations fail clearly until their camera-model
    inverse projection is added here.
    """
    if batch.intrinsics is not None:
        fx, fy, cx, cy = batch.intrinsics
        values = (fx, fy, cx, cy)
    elif batch.intrinsics_OpenCVPinholeCameraModelParameters is not None:
        params = batch.intrinsics_OpenCVPinholeCameraModelParameters
        focal = params["focal_length"]
        principal = params["principal_point"]
        values = (focal[0], focal[1], principal[0], principal[1])
    else:
        raise ValueError(
            "multi-view reprojection currently requires pinhole or OpenCV-pinhole intrinsics; "
            "fisheye/F-theta support needs the corresponding inverse camera model."
        )
    return tuple(torch.as_tensor(value, device=device, dtype=dtype) for value in values)  # type: ignore[return-value]


def _sample(map_: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    return F.grid_sample(map_.permute(0, 3, 1, 2), grid, mode=mode, padding_mode="zeros", align_corners=True).permute(
        0, 2, 3, 1
    )


def reproject_to_target(
    source_batch: Batch, source_depth: torch.Tensor, target_batch: Batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map source pixels to target sampling coordinates using Euclidean ray distance.

    Returns normalized ``grid_sample`` coordinates, the source world point and an in-bounds,
    front-facing validity mask.  It is intentionally independent of the rendered primitive.
    """
    if source_depth.shape[0] != target_batch.T_to_world.shape[0]:
        raise ValueError("source and target batches must have the same batch size")
    if target_batch.rays_in_world_space:
        raise ValueError(
            "multi-view reprojection does not yet support target rays with per-pixel world poses; "
            "use global-shutter camera-space rays."
        )
    if target_batch.T_to_world_end is not None and not torch.equal(
        target_batch.T_to_world, target_batch.T_to_world_end
    ):
        raise ValueError(
            "multi-view reprojection does not yet support rolling-shutter target poses; use global-shutter captures."
        )
    source_pose = None if source_batch.rays_in_world_space else source_batch.T_to_world
    points_world = unproject_to_world(source_depth, source_batch.rays_ori, source_batch.rays_dir, source_pose)
    target_pose = target_batch.T_to_world
    camera_points = torch.einsum(
        "bji,bhwj->bhwi", target_pose[:, :3, :3], points_world - target_pose[:, None, None, :3, 3]
    )
    fx, fy, cx, cy = _intrinsics(target_batch, source_depth.dtype, source_depth.device)
    z = camera_points[..., 2:3]
    pixel_center_x = fx * camera_points[..., 0:1] / z.clamp_min(1e-8) + cx
    pixel_center_y = fy * camera_points[..., 1:2] / z.clamp_min(1e-8) + cy
    target_h, target_w = target_batch.rays_dir.shape[1:3]
    # Camera parameters and our ray generator use +0.5 pixel centres, while grid_sample's
    # align_corners convention addresses centre index 0 at -1.
    pixel_x = pixel_center_x - 0.5
    pixel_y = pixel_center_y - 0.5
    grid = torch.cat(
        (
            2.0 * pixel_x / max(target_w - 1, 1) - 1.0,
            2.0 * pixel_y / max(target_h - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    boundary_epsilon = 1e-6
    in_bounds = (
        (z > 1e-8)
        & (grid[..., 0:1] >= -1.0 - boundary_epsilon)
        & (grid[..., 0:1] <= 1.0 + boundary_epsilon)
        & (grid[..., 1:2] >= -1.0 - boundary_epsilon)
        & (grid[..., 1:2] <= 1.0 + boundary_epsilon)
    )
    return grid.clamp(-1.0, 1.0), points_world, in_bounds


def _masked_mean(value: torch.Tensor, valid: torch.Tensor, confidence: torch.Tensor | None = None) -> torch.Tensor:
    weight = valid.to(value.dtype) if confidence is None else valid.to(value.dtype) * confidence
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _charbonnier(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon) - epsilon


def _feature_map(outputs: dict[str, torch.Tensor], source: str) -> torch.Tensor:
    key = {"rgb": "pred_features", "latent": "pred_latent", "decoded_image_feature": "pred_image_features"}.get(source)
    if key is None:
        raise ValueError(f"unknown multi-view feature source {source!r}")
    if key not in outputs:
        raise ValueError(f"multi-view feature source {source!r} requires {key}, which this render did not produce")
    return outputs[key]


def multiview_supervision_loss(
    source_batch: Batch,
    source_outputs: dict[str, torch.Tensor],
    target_batch: Batch,
    target_outputs: dict[str, torch.Tensor],
    *,
    scene_extent: float,
    min_opacity: float,
    visibility_relative_tolerance: float,
    visibility_absolute_tolerance_scene: float,
    weights: MultiViewLossWeights,
    feature_source: str = "rgb",
    zncc_patch_size: int = 5,
    zncc_min_patch_valid_fraction: float = 0.7,
    signed_normals: bool = False,
    source_confidence: torch.Tensor | None = None,
    target_confidence: torch.Tensor | None = None,
    agreement_weight: float = 0.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Compute selectable losses on visible reprojections, optionally reliability-weighted.

    Structural source/target confidence maps are expected to be detached. ``agreement_weight``
    further downweights pixels whose rendered target distance only barely passes the visibility
    tolerance; it is detached too, preventing a model from reducing its own loss by making
    disagreement look uncertain.
    """
    grid, source_points, projected_valid = reproject_to_target(source_batch, source_outputs["pred_dist"], target_batch)
    target_depth = _sample(target_outputs["pred_dist"], grid)
    target_opacity = _sample(target_outputs["pred_opacity"], grid)
    target_rays_ori = _sample(target_batch.rays_ori, grid)
    target_rays_dir = _sample(target_batch.rays_dir, grid)
    target_pose = None if target_batch.rays_in_world_space else target_batch.T_to_world
    target_points = unproject_to_world(target_depth, target_rays_ori, target_rays_dir, target_pose)
    # A rolling-shutter/world-space ray may not start at the pose translation.  Transform the
    # sampled ray origin itself so the depth-visibility test remains a true Euclidean distance.
    target_origins_world = unproject_to_world(
        torch.zeros_like(target_depth), target_rays_ori, target_rays_dir, target_pose
    )
    expected_target_distance = (source_points - target_origins_world).norm(dim=-1, keepdim=True)
    depth_error = (target_depth - expected_target_distance).abs()
    depth_tolerance = (
        visibility_absolute_tolerance_scene * scene_extent + visibility_relative_tolerance * expected_target_distance
    )
    valid = (
        projected_valid
        & (source_outputs["pred_opacity"] >= min_opacity)
        & (target_opacity >= min_opacity)
        & (depth_error <= depth_tolerance)
    )

    if agreement_weight < 0.0:
        raise ValueError("agreement_weight must be non-negative")
    confidence = torch.ones_like(source_outputs["pred_dist"])
    if source_confidence is not None:
        if source_confidence.shape != confidence.shape:
            raise ValueError("source_confidence must match pred_dist [B, H, W, 1]")
        confidence = confidence * source_confidence
    if target_confidence is not None:
        if target_confidence.shape != target_outputs["pred_dist"].shape:
            raise ValueError("target_confidence must match target pred_dist [B, H, W, 1]")
        confidence = confidence * _sample(target_confidence, grid)
    if agreement_weight:
        agreement = torch.exp(-agreement_weight * (depth_error / depth_tolerance.clamp_min(1e-8)).square())
        confidence = confidence * agreement.detach()
    confidence = torch.nan_to_num(confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    zero = source_outputs["pred_dist"].new_zeros(())
    losses = {"point": zero, "normal": zero, "raw_feature_l2": zero, "zncc": zero}
    if weights.point:
        point_error = (source_points - target_points).norm(dim=-1, keepdim=True) / max(scene_extent, 1e-8)
        losses["point"] = _masked_mean(_charbonnier(point_error), valid, confidence)
    if weights.normal:
        target_normals = F.normalize(_sample(target_outputs["pred_normals"], grid), dim=-1)
        source_normals = F.normalize(source_outputs["pred_normals"], dim=-1)
        dot = (source_normals * target_normals).sum(dim=-1, keepdim=True)
        losses["normal"] = _masked_mean(1.0 - (dot if signed_normals else dot.abs()), valid, confidence)

    source_features = target_features = None
    if weights.raw_feature_l2 or weights.zncc:
        source_features = _feature_map(source_outputs, feature_source)
        target_features = _sample(_feature_map(target_outputs, feature_source), grid)
    if weights.raw_feature_l2:
        losses["raw_feature_l2"] = _masked_mean(
            (source_features - target_features).square().mean(dim=-1, keepdim=True), valid, confidence
        )
    if weights.zncc:
        if zncc_patch_size < 1 or zncc_patch_size % 2 == 0:
            raise ValueError("zncc_patch_size must be a positive odd integer")
        source_nchw, target_nchw = source_features.permute(0, 3, 1, 2), target_features.permute(0, 3, 1, 2)
        unfold = lambda tensor: F.unfold(tensor, zncc_patch_size, padding=zncc_patch_size // 2)
        source_patch, target_patch = unfold(source_nchw), unfold(target_nchw)
        source_patch = source_patch - source_patch.mean(dim=1, keepdim=True)
        target_patch = target_patch - target_patch.mean(dim=1, keepdim=True)
        zncc = (source_patch * target_patch).sum(dim=1) / (
            source_patch.square().sum(dim=1).sqrt() * target_patch.square().sum(dim=1).sqrt()
        ).clamp_min(1e-8)
        patch_valid = F.avg_pool2d(
            valid.to(source_nchw.dtype).permute(0, 3, 1, 2), zncc_patch_size, stride=1, padding=zncc_patch_size // 2
        )
        patch_valid = patch_valid.permute(0, 2, 3, 1) >= zncc_min_patch_valid_fraction
        losses["zncc"] = _masked_mean((1.0 - zncc).view_as(valid), valid & patch_valid, confidence)
    return losses, valid
