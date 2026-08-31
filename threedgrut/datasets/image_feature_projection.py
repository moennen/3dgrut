# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen, backend-independent compression for dense foundation-model image features.

Projectors are fitted offline on training images and then frozen.  They deliberately do not
participate in reconstruction optimisation: otherwise the target representation could move to
match an incorrect render rather than remain a robust external data term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def l2_normalize(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass(frozen=True)
class PCAProjector:
    """A frozen PCA feature projector with optional whitening and unit-length outputs."""

    mean: torch.Tensor  # [C]
    components: torch.Tensor  # [C, D]
    scales: torch.Tensor  # [D], square roots of eigenvalues
    whiten: bool = False
    normalize_output: bool = True

    @classmethod
    def fit(
        cls, samples: torch.Tensor, output_dim: int, *, whiten: bool = False, normalize_output: bool = True
    ) -> "PCAProjector":
        samples = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if samples.ndim != 2 or samples.shape[0] < 2:
            raise ValueError(f"PCA samples must be [N, C] with N >= 2, got {tuple(samples.shape)}")
        if output_dim < 1 or output_dim > min(samples.shape):
            raise ValueError(f"output_dim must be in [1, {min(samples.shape)}], got {output_dim}")
        mean = samples.mean(dim=0)
        _, singular, right = torch.pca_lowrank(samples - mean, q=output_dim, center=False)
        scales = singular[:output_dim] / (samples.shape[0] - 1) ** 0.5
        return cls(mean, right[:, :output_dim], scales, whiten=whiten, normalize_output=normalize_output)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32)
        projected = (features - self.mean.to(features)) @ self.components.to(features)
        if self.whiten:
            projected = projected / self.scales.to(features).clamp_min(1e-8)
        return l2_normalize(projected) if self.normalize_output else projected

    def state_dict(self) -> dict:
        return {
            "type": "pca",
            "mean": self.mean,
            "components": self.components,
            "scales": self.scales,
            "whiten": self.whiten,
            "normalize_output": self.normalize_output,
        }


class FeatureAutoEncoder(nn.Module):
    """Small offline nonlinear compressor. It is frozen before reconstruction training."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.decoder = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.encoder(features))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(features)
        return latent, self.decoder(latent)


def fit_autoencoder(
    samples: torch.Tensor,
    output_dim: int,
    *,
    hidden_dim: int = 128,
    steps: int = 1_000,
    batch_size: int = 4_096,
    learning_rate: float = 1e-3,
    neighborhood_weight: float = 0.1,
    seed: int = 0,
) -> FeatureAutoEncoder:
    """Fit a frozen autoencoder with reconstruction and pairwise-cosine preservation losses."""
    samples = l2_normalize(torch.as_tensor(samples, dtype=torch.float32, device="cpu"))
    if samples.ndim != 2 or output_dim < 1 or output_dim > samples.shape[1]:
        raise ValueError(f"Expected samples [N, C] and output_dim in [1, C], got {tuple(samples.shape)}, {output_dim}")
    generator = torch.Generator().manual_seed(seed)
    model = FeatureAutoEncoder(samples.shape[1], output_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        index = torch.randint(len(samples), (min(batch_size, len(samples)),), generator=generator)
        batch = samples[index]
        latent, reconstruction = model(batch)
        reconstruction_loss = 1 - (l2_normalize(reconstruction) * batch).sum(dim=-1).mean()
        pairs = torch.randperm(len(batch), generator=generator)
        latent_similarity = (latent * latent[pairs]).sum(dim=-1)
        source_similarity = (batch * batch[pairs]).sum(dim=-1)
        neighbor_loss = (latent_similarity - source_similarity).abs().mean()
        loss = reconstruction_loss + neighborhood_weight * neighbor_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
