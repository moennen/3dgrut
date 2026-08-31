# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from threedgrut.datasets.image_feature_projection import PCAProjector, fit_autoencoder, l2_normalize


def test_pca_projector_returns_unit_compact_features():
    torch.manual_seed(3)
    samples = torch.randn(64, 12)
    projector = PCAProjector.fit(samples, 4)
    output = projector.transform(samples[:5])
    assert output.shape == (5, 4)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(5))
    assert projector.state_dict()["type"] == "pca"


def test_autoencoder_is_frozen_and_preserves_normalized_shape():
    torch.manual_seed(4)
    samples = l2_normalize(torch.randn(32, 8))
    model = fit_autoencoder(samples, 3, hidden_dim=12, steps=4, batch_size=16)
    latent = model.encode(samples[:3])
    assert latent.shape == (3, 3)
    torch.testing.assert_close(latent.norm(dim=-1), torch.ones(3))
    assert not any(parameter.requires_grad for parameter in model.parameters())
