# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cached frozen dense image features for robust reconstruction supervision."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from threedgrut.datasets.image_feature_projection import PCAProjector, fit_autoencoder

CACHE_FORMAT_VERSION = 1


class DenseFeatureBackend(Protocol):
    """Frozen image encoder; implementations return an ``[H, W, C]`` CPU tensor."""

    def encode(self, image: np.ndarray) -> torch.Tensor: ...

    def identity(self) -> dict: ...


class DINOv2Backend:
    """Dense DINOv2 patch tokens through Transformers, imported only on cache misses."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id, self.device = model_id, device
        self._processor = self._model = None

    def identity(self) -> dict:
        return {"backend": "dinov2", "model": self.model_id}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> torch.Tensor:
        self._ensure_loaded()
        inputs = self._processor(images=image, return_tensors="pt", do_center_crop=False)
        pixels = inputs["pixel_values"].to(self.device)
        hidden = self._model(pixel_values=pixels).last_hidden_state[0]
        patch = int(getattr(self._model.config, "patch_size", 14))
        height, width = pixels.shape[-2] // patch, pixels.shape[-1] // patch
        tokens = hidden[-height * width :]
        return tokens.reshape(height, width, -1).float().cpu()


BACKENDS = {"dinov2": DINOv2Backend}


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


class ImageFeatureCache:
    """Build and load compact frozen image feature maps.

    PCA requires two encoder passes: a reservoir sample fits the projector, then every map is
    projected and written.  This avoids retaining raw high-dimensional maps on disk or in RAM.
    """

    def __init__(
        self,
        scene_path: str,
        *,
        backend: str,
        model: str,
        output_dim: int = 16,
        feature_stride: int = 14,
        cache_dir: str | None = None,
        device: str = "cuda",
        projector: str = "pca",
        autoencoder_hidden_dim: int = 128,
        autoencoder_steps: int = 1_000,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unknown image-feature backend {backend!r}; expected {sorted(BACKENDS)}")
        if output_dim < 1 or feature_stride < 1 or projector not in {"pca", "autoencoder"}:
            raise ValueError("output_dim and feature_stride must be positive")
        self.backend = BACKENDS[backend](model, device=device)
        self.output_dim, self.feature_stride = output_dim, feature_stride
        self.identity = {
            "format_version": CACHE_FORMAT_VERSION,
            **self.backend.identity(),
            "output_dim": output_dim,
            "feature_stride": feature_stride,
            "projector": projector,
            "autoencoder_hidden_dim": autoencoder_hidden_dim if projector == "autoencoder" else None,
            "autoencoder_steps": autoencoder_steps if projector == "autoencoder" else None,
        }
        digest = hashlib.sha1(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:8]
        root = Path(cache_dir) if cache_dir else Path(scene_path) / "image_feature_cache"
        self.directory = root / f"{_slug(model)}__{digest}"
        self.projector_type = projector
        self.autoencoder_hidden_dim, self.autoencoder_steps = autoencoder_hidden_dim, autoencoder_steps
        self.projector = None

    def entry_path(self, image_path: str | Path) -> Path:
        image_path = Path(image_path)
        return self.directory / f"{image_path.parent.name}__{image_path.stem}.npy"

    def _meta_path(self) -> Path:
        return self.directory / "meta.json"

    def _write_or_verify_meta(self) -> None:
        path = self._meta_path()
        if path.exists():
            if json.loads(path.read_text()) != self.identity:
                raise RuntimeError(f"Image feature cache identity mismatch at {self.directory}")
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.identity, sort_keys=True, indent=2))

    def _canonicalize(self, raw: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
        height, width = image_shape
        out_shape = (
            (height + self.feature_stride - 1) // self.feature_stride,
            (width + self.feature_stride - 1) // self.feature_stride,
        )
        return F.interpolate(raw.permute(2, 0, 1)[None], size=out_shape, mode="bilinear", align_corners=False)[
            0
        ].permute(1, 2, 0)

    def _features(self, image_path: str | Path) -> torch.Tensor:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        return self._canonicalize(self.backend.encode(image), image.shape[:2])

    def ensure(self, image_paths: Sequence[str], *, fit_samples: int = 250_000, seed: int = 0) -> None:
        self._write_or_verify_meta()
        if all(self.entry_path(path).exists() for path in image_paths):
            return
        generator = torch.Generator().manual_seed(seed)
        reservoir = None
        for path in image_paths:
            features = self._features(path)
            features = features.reshape(-1, features.shape[-1])
            # Keep a bounded uniform reservoir without storing complete encoder outputs.
            if reservoir is None:
                reservoir = features[:fit_samples].clone()
                continue
            combined = torch.cat([reservoir, features], dim=0)
            if len(combined) > fit_samples:
                index = torch.randperm(len(combined), generator=generator)[:fit_samples]
                combined = combined[index]
            reservoir = combined
        if reservoir is None:
            raise ValueError("Cannot build image features for an empty image list")
        if self.projector_type == "pca":
            self.projector = PCAProjector.fit(reservoir, self.output_dim)
            torch.save(self.projector.state_dict(), self.directory / "projector.pt")
        else:
            self.projector = fit_autoencoder(
                reservoir,
                self.output_dim,
                hidden_dim=self.autoencoder_hidden_dim,
                steps=self.autoencoder_steps,
                seed=seed,
            )
            torch.save(
                {"type": "autoencoder", "state_dict": self.projector.state_dict()}, self.directory / "projector.pt"
            )
        for path in image_paths:
            target = self.entry_path(path)
            if target.exists():
                continue
            features = self._features(path)
            if self.projector_type == "pca":
                compressed = self.projector.transform(features)
            else:
                compressed = self.projector.encode(features)
            compressed = compressed.numpy().astype(np.float16)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.save(handle, compressed)
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, image_path: str | Path) -> np.ndarray:
        return np.asarray(np.load(self.entry_path(image_path)), dtype=np.float32)
