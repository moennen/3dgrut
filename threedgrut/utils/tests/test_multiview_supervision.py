import pytest
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.multiview_supervision import MultiViewLossWeights, multiview_supervision_loss, reproject_to_target


def _batch(height: int = 7, width: int = 9) -> Batch:
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    rays = torch.stack(
        ((xs + 0.5 - width / 2) / width, (ys + 0.5 - height / 2) / width, torch.ones_like(xs)), dim=-1
    ).float()
    return Batch(
        rays_ori=torch.zeros(1, height, width, 3),
        rays_dir=rays.unsqueeze(0),
        T_to_world=torch.eye(4).unsqueeze(0),
        intrinsics=[float(width), float(width), width / 2, height / 2],
    )


def _outputs(batch: Batch, feature_dim: int = 3) -> dict[str, torch.Tensor]:
    height, width = batch.rays_dir.shape[1:3]
    features = torch.linspace(0.1, 0.9, height * width * feature_dim).reshape(1, height, width, feature_dim)
    return {
        "pred_dist": torch.full((1, height, width, 1), 3.0),
        "pred_opacity": torch.ones((1, height, width, 1)),
        "pred_normals": torch.tensor([0.0, 0.0, -1.0]).expand(1, height, width, 3).clone(),
        "pred_features": features,
    }


def test_identity_reprojection_and_all_matching_terms_are_zero():
    source, target = _batch(), _batch()
    source_outputs, target_outputs = _outputs(source), _outputs(target)
    grid, _, valid = reproject_to_target(source, source_outputs["pred_dist"], target)
    assert valid.all()
    torch.testing.assert_close(
        grid[..., 0], torch.linspace(-1.0, 1.0, grid.shape[2]).view(1, 1, -1).expand_as(grid[..., 0])
    )
    losses, visibility = multiview_supervision_loss(
        source,
        source_outputs,
        target,
        target_outputs,
        scene_extent=10.0,
        min_opacity=0.5,
        visibility_relative_tolerance=0.01,
        visibility_absolute_tolerance_scene=0.0,
        weights=MultiViewLossWeights(point=1.0, normal=1.0, raw_feature_l2=1.0, zncc=1.0),
    )
    assert visibility.all()
    assert losses["point"].item() == 0.0
    assert losses["normal"].item() == 0.0
    assert losses["raw_feature_l2"].item() < 1e-12
    assert losses["zncc"].item() < 1e-5


def test_visibility_rejects_target_depth_disagreement():
    source, target = _batch(), _batch()
    source_outputs, target_outputs = _outputs(source), _outputs(target)
    target_outputs["pred_dist"] = torch.full_like(target_outputs["pred_dist"], 4.0)
    losses, visibility = multiview_supervision_loss(
        source,
        source_outputs,
        target,
        target_outputs,
        scene_extent=10.0,
        min_opacity=0.5,
        visibility_relative_tolerance=0.01,
        visibility_absolute_tolerance_scene=0.0,
        weights=MultiViewLossWeights(point=1.0),
    )
    assert not visibility.any()
    assert losses["point"].item() == 0.0


def test_latent_feature_source_is_selectable():
    source, target = _batch(), _batch()
    source_outputs, target_outputs = _outputs(source), _outputs(target)
    source_outputs["pred_latent"] = torch.rand(1, 7, 9, 5)
    target_outputs["pred_latent"] = source_outputs["pred_latent"].clone()
    losses, _ = multiview_supervision_loss(
        source,
        source_outputs,
        target,
        target_outputs,
        scene_extent=10.0,
        min_opacity=0.5,
        visibility_relative_tolerance=0.01,
        visibility_absolute_tolerance_scene=0.0,
        weights=MultiViewLossWeights(raw_feature_l2=1.0),
        feature_source="latent",
    )
    assert losses["raw_feature_l2"].item() < 1e-12


def test_raw_feature_l2_is_channelwise_mean_squared_error():
    source, target = _batch(), _batch()
    source_outputs, target_outputs = _outputs(source), _outputs(target)
    target_outputs["pred_features"] = source_outputs["pred_features"] + 0.25
    losses, _ = multiview_supervision_loss(
        source,
        source_outputs,
        target,
        target_outputs,
        scene_extent=10.0,
        min_opacity=0.5,
        visibility_relative_tolerance=0.01,
        visibility_absolute_tolerance_scene=0.0,
        weights=MultiViewLossWeights(raw_feature_l2=1.0),
    )
    torch.testing.assert_close(losses["raw_feature_l2"], torch.tensor(0.25**2))


def test_detached_confidence_excludes_an_ambiguous_multiview_residual():
    source, target = _batch(), _batch()
    source_outputs, target_outputs = _outputs(source), _outputs(target)
    target_outputs["pred_features"] = source_outputs["pred_features"] + 0.25
    confidence = torch.zeros_like(source_outputs["pred_dist"])
    losses, _ = multiview_supervision_loss(
        source,
        source_outputs,
        target,
        target_outputs,
        scene_extent=10.0,
        min_opacity=0.5,
        visibility_relative_tolerance=0.01,
        visibility_absolute_tolerance_scene=0.0,
        weights=MultiViewLossWeights(raw_feature_l2=1.0),
        source_confidence=confidence,
    )
    assert losses["raw_feature_l2"].item() == 0.0


def test_rolling_shutter_target_is_rejected_instead_of_projected_with_a_wrong_pose():
    source, target = _batch(), _batch()
    target.T_to_world_end = target.T_to_world.clone()
    target.T_to_world_end[:, 0, 3] = 0.1
    with pytest.raises(ValueError, match="rolling-shutter"):
        reproject_to_target(source, _outputs(source)["pred_dist"], target)
