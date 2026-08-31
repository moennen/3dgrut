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
