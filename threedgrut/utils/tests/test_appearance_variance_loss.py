import pytest
import torch

from threedgrut.utils.appearance_variance_loss import appearance_feature_variance_loss


def _moments(features: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        (weights[:, None] * features).sum(dim=0).reshape(1, 1, 1, -1),
        (weights[:, None] * features.square()).sum(dim=0).reshape(1, 1, 1, -1),
        weights.sum().reshape(1, 1, 1, 1),
    )


def test_coherent_appearance_has_zero_loss_for_rgb_and_nht_dimensions():
    for dim in (3, 16):
        feature = torch.linspace(0.1, 1.0, dim)
        first, second, opacity = _moments(feature.repeat(2, 1), torch.tensor([0.4, 0.5]))
        loss, valid = appearance_feature_variance_loss(first, second, opacity)
        assert valid.item()
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_mixed_features_are_penalised():
    first, second, opacity = _moments(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0.5, 0.5]))
    loss, valid = appearance_feature_variance_loss(first, second, opacity)
    assert valid.item()
    assert loss.item() == pytest.approx(0.5)


def test_loss_is_invariant_to_uniform_opacity_fade_and_feature_scale():
    features = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 2.0, 0.0]])
    first, second, opacity = _moments(features, torch.tensor([0.4, 0.4]))
    faded_first, faded_second, faded_opacity = _moments(features * 7.0, torch.tensor([0.08, 0.08]))
    loss, _ = appearance_feature_variance_loss(first, second, opacity)
    faded_loss, _ = appearance_feature_variance_loss(faded_first, faded_second, faded_opacity, min_opacity=0.1)
    assert faded_loss.item() == pytest.approx(loss.item())


def test_transparent_and_empty_moments_are_excluded():
    first = torch.ones(1, 1, 2, 3)
    second = torch.ones_like(first)
    opacity = torch.tensor([[[[0.49], [0.0]]]])
    loss, valid = appearance_feature_variance_loss(first, second, opacity)
    assert not valid.any()
    assert loss.item() == 0.0


def test_gradients_reach_both_feature_moments_and_opacity():
    first = torch.tensor([[[[0.3, 0.1]]]], requires_grad=True)
    second = torch.tensor([[[[0.4, 0.2]]]], requires_grad=True)
    opacity = torch.tensor([[[[0.7]]]], requires_grad=True)
    loss, _ = appearance_feature_variance_loss(first, second, opacity)
    loss.backward()
    assert first.grad.abs().sum() > 0
    assert second.grad.abs().sum() > 0
    assert opacity.grad.abs().sum() > 0
