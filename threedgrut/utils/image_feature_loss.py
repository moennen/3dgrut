"""Alpha-aware low-resolution supervision from frozen foundation-model image features."""

import torch
import torch.nn.functional as F


def image_feature_loss(predicted: torch.Tensor, opacity: torch.Tensor, target: torch.Tensor, alpha_min: float = 0.05):
    """Area-pool decoded features with opacity before cosine comparison to ``target``."""
    if opacity.ndim == 3:
        opacity = opacity.unsqueeze(-1)
    size = target.shape[1:3]
    numerator = F.adaptive_avg_pool2d((predicted * opacity).permute(0, 3, 1, 2), size).permute(0, 2, 3, 1)
    pooled_alpha = F.adaptive_avg_pool2d(opacity.permute(0, 3, 1, 2), size).permute(0, 2, 3, 1)
    predicted = F.normalize(numerator / pooled_alpha.clamp_min(1e-8), dim=-1)
    target = F.normalize(target, dim=-1)
    valid = pooled_alpha[..., 0] >= alpha_min
    if not valid.any():
        return predicted.sum() * 0
    return (1 - (predicted * target).sum(dim=-1))[valid].mean()
