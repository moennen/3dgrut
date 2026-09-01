# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
from PIL import Image

from threedgrut.datasets import image_features


def test_feature_cache_builds_compact_pca_maps_and_reuses_them(tmp_path, monkeypatch):
    class Backend:
        calls = 0

        def __init__(self, model, device="cuda"):
            self.model = model

        def identity(self):
            return {"backend": "fake", "model": self.model}

        def encode(self, image):
            Backend.calls += 1
            grid = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
            return image_features.torch.from_numpy(grid + image.mean())

    monkeypatch.setitem(image_features.BACKENDS, "fake", Backend)
    image = tmp_path / "images" / "frame.png"
    image.parent.mkdir()
    Image.fromarray(np.full((28, 28, 3), 7, dtype=np.uint8)).save(image)
    cache = image_features.ImageFeatureCache(
        str(tmp_path), backend="fake", model="fake/model", output_dim=2, feature_stride=14, device="cpu"
    )
    cache.ensure([str(image)], fit_samples=16)
    target = cache.load(image)
    assert target.shape == (2, 2, 2)
    assert Backend.calls == 2  # fit pass + compressed-cache pass
    cache.ensure([str(image)], fit_samples=16)
    assert Backend.calls == 2


def test_feature_cache_can_use_a_frozen_autoencoder(tmp_path, monkeypatch):
    class Backend:
        def __init__(self, model, device="cuda"):
            pass

        def identity(self):
            return {"backend": "fake-auto", "model": "fake"}

        def encode(self, image):
            return image_features.torch.arange(64, dtype=image_features.torch.float32).reshape(4, 4, 4)

    monkeypatch.setitem(image_features.BACKENDS, "fake-auto", Backend)
    image = tmp_path / "images" / "frame.png"
    image.parent.mkdir()
    Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8)).save(image)
    cache = image_features.ImageFeatureCache(
        str(tmp_path),
        backend="fake-auto",
        model="fake",
        output_dim=2,
        feature_stride=14,
        device="cpu",
        projector="autoencoder",
        autoencoder_steps=2,
    )
    cache.ensure([str(image)], fit_samples=16)
    assert cache.load(image).shape == (2, 2, 2)


def test_nvradio4_backend_uses_spatial_backbone_features(monkeypatch):
    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def get_nearest_supported_resolution(self, height, width):
            return height, width

        def __call__(self, pixels, feature_fmt):
            assert feature_fmt == "NCHW"
            return None, image_features.torch.ones((1, 5, 2, 3), device=pixels.device)

    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return Model()

    monkeypatch.setattr(image_features.torch.hub, "load", load)
    backend = image_features.NVRadio4Backend("c-radio_v4-h", device="cpu")
    encoded = backend.encode(np.zeros((8, 12, 3), dtype=np.uint8))
    assert encoded.shape == (2, 3, 5)
    assert calls[0][0][:2] == ("NVlabs/RADIO", "radio_model")
    assert calls[0][1]["version"] == "c-radio_v4-h"
