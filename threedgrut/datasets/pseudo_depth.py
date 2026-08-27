# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monocular pseudo-depth priors from a foundation model, cached per scene.

This is *pseudo* depth: a prediction from an image-only model, used as a training signal. It is
deliberately independent of the ground-truth depth that ships with OB3D, which stays reserved
for evaluation -- see `dataset.load_depth_gt`, which the training split does not enable.

Three properties of these priors shape the design:

* **The output is disparity, not depth.** Depth Anything predicts a quantity that is affine to
  *inverse* depth, so it is monotonically *decreasing* in depth. Consumers must not read it as
  a distance. Measured on sponza, an affine fit to `1/z` reaches R^2 0.96 where a fit to `z`
  reaches only 0.82, so treating it as depth loses real accuracy while looking plausible.
* **Only the ordering is trustworthy at scene scale.** The prior's local structure is excellent
  and its global structure drifts: aligning one affine per frame gives `abs_rel` 0.068 against
  ground truth, worse than the model being trained (0.058), while a per-16x16-patch affine
  reaches 0.011. Hence the cache stores the raw disparity and leaves it to the loss to be
  invariant to the unknown transform, rather than baking in a scale that is not reliable.
* **Inference is far too slow to repeat.** The prediction depends only on the image and the
  model, so it is computed once and cached on disk.

The cache directory embeds the model identity, so switching `dataset.pseudo_depth.model` cannot
silently reuse another model's predictions -- the failure this would otherwise cause (training
against the wrong prior) is invisible in the loss curve.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from threedgrut.utils.logger import logger

# Bumped when the stored representation changes in a way that makes existing files wrong.
# It is part of the cache identity, so a bump invalidates old caches instead of misreading them.
CACHE_FORMAT_VERSION = 1


@dataclass
class PseudoDepthPredictor:
    """Lazily-loaded monocular depth model. Returns **disparity** at the model's native size.

    The model is only instantiated on the first cache miss, so a run whose cache is already
    complete never pays for loading it (or for having `transformers` importable at all).
    """

    model_id: str
    device: str = "cuda"
    _processor: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Pseudo-depth supervision needs `transformers` to build its cache. Install it, "
                "or point dataset.pseudo_depth.cache_dir at an already-populated cache."
            ) from exc
        logger.info(f"Loading pseudo-depth model {self.model_id}")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def predict_disparity(self, image: np.ndarray) -> np.ndarray:
        """Disparity for an HWC uint8 RGB image, at the model's native output resolution.

        Kept at native resolution because that is exactly what the model produced; resampling to
        the training resolution is the consumer's business and costs nothing to redo.
        """
        self._ensure_loaded()
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        disparity = self._model(**inputs).predicted_depth
        return disparity.squeeze(0).float().cpu().numpy()


def _cache_identity(model_id: str) -> dict:
    return {"model_id": model_id, "format_version": CACHE_FORMAT_VERSION, "quantity": "disparity"}


def _slugify(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)


class PseudoDepthCache:
    """On-disk store of per-frame pseudo-depth, keyed by the model that produced it.

    The directory name carries the model slug *and* a hash of the full identity, and a
    `meta.json` inside records that identity. Two different models therefore cannot share a
    directory, and a stale directory whose identity no longer matches is rejected loudly rather
    than read as if it were current.
    """

    def __init__(self, scene_path: str, model_id: str, cache_dir: Optional[str] = None, device: str = "cuda"):
        self.model_id = model_id
        identity = _cache_identity(model_id)
        digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
        root = Path(cache_dir) if cache_dir else Path(scene_path) / "pseudo_depth_cache"
        self.directory = root / f"{_slugify(model_id)}__{digest}"
        self._identity = identity
        self._predictor = PseudoDepthPredictor(model_id=model_id, device=device)

    # -- identity ---------------------------------------------------------------------------

    def _meta_path(self) -> Path:
        return self.directory / "meta.json"

    def _verify_or_write_meta(self) -> None:
        path = self._meta_path()
        if path.exists():
            stored = json.loads(path.read_text())
            if stored != self._identity:
                raise RuntimeError(
                    f"Pseudo-depth cache at {self.directory} was written by a different "
                    f"configuration ({stored}) than the one requested ({self._identity}). "
                    "Delete it or choose another dataset.pseudo_depth.cache_dir."
                )
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._identity, sort_keys=True, indent=2))

    # -- population -------------------------------------------------------------------------

    def entry_path(self, image_path: str) -> Path:
        # Frames are addressed by image stem; a scene with two images of the same stem in
        # different folders would collide, so the parent folder name is included.
        image_path = Path(image_path)
        return self.directory / f"{image_path.parent.name}__{image_path.stem}.npy"

    def ensure(self, image_paths: Sequence[str]) -> None:
        """Predict and store anything missing. A complete cache makes this nearly free."""
        self._verify_or_write_meta()
        missing = [p for p in image_paths if not self.entry_path(p).exists()]
        if not missing:
            return
        logger.info(f"Building pseudo-depth cache for {len(missing)}/{len(image_paths)} frames in {self.directory}")
        for image_path in logger.track(missing, description="Pseudo-depth"):
            image = np.asarray(Image.open(image_path).convert("RGB"))
            disparity = self._predictor.predict_disparity(image)
            # Write via a temporary file so an interrupted run cannot leave a truncated entry
            # that later looks like a valid cache hit.
            target = self.entry_path(image_path)
            temporary = target.with_suffix(".npy.tmp")
            # Written through an open handle rather than by path: `np.save` silently appends
            # `.npy` to a name that lacks it, which would leave the rename with nothing to move.
            with open(temporary, "wb") as handle:
                np.save(handle, disparity.astype(np.float32))
            os.replace(temporary, target)

    # -- reading ----------------------------------------------------------------------------

    def load(self, image_path: str, height: int, width: int) -> np.ndarray:
        """Disparity for one frame, resampled to `height` x `width`.

        Bicubic, unlike ground-truth maps which must use nearest neighbour: this is a smooth
        prediction being brought back to full resolution, which is what the upstream model's own
        inference code does, and the alternative would quantise the ordering the loss reads.
        """
        disparity = np.load(self.entry_path(image_path))
        tensor = torch.from_numpy(disparity)[None, None].float()
        resized = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False)
        return np.ascontiguousarray(resized[0, 0].numpy())
