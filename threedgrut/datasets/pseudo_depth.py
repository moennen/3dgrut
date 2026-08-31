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

* **Whether the output is disparity or depth depends on the model.** Depth Anything V2 predicts
  a quantity affine to *inverse* depth, monotonically *decreasing* in distance; Depth Anything
  3's monocular model predicts depth directly, increasing in distance. Measured on sponza frame
  0, DAv2 correlates +0.977 with `1/z` and -0.907 with `z`, while DA3 correlates +0.986 with `z`.
  Neither may be read as the other, so each backend declares its quantity, the cache records it,
  and the ordinal loss requires it to be passed explicitly.
* **Only the ordering is trustworthy at scene scale.** The prior's local structure is excellent
  and its global structure drifts: aligning one affine per frame gives `abs_rel` 0.068 against
  ground truth, worse than the model being trained (0.058), while a per-16x16-patch affine
  reaches 0.011. Hence the cache stores the raw prediction and leaves it to the loss to be
  invariant to the unknown transform, rather than baking in a scale that is not reliable.
* **Inference is far too slow to repeat.** The prediction depends only on the image and the
  model, so it is computed once and cached on disk.

The cache directory embeds the full identity -- backend, model id, quantity and format version --
so switching `dataset.pseudo_depth.model` or its backend cannot silently reuse another model's
predictions. The failure this would otherwise cause (training against the wrong prior, or against
the right one with the sign inverted) is invisible in the loss curve.

Two backends are available. `transformers` covers any Depth Anything V2 checkpoint through
`AutoModelForDepthEstimation`. `depth_anything_3` runs Depth Anything 3, which needs the upstream
package importable and so is imported lazily, only on a cache miss.

`transformers` remains the default for availability rather than quality: it installs with the
venv, where DA3 needs a source checkout on `PYTHONPATH`. Licence is not the reason -- DA3's
*monocular* weights are Apache-2.0, like DAv2's. Only DA3's any-view checkpoints (`DA3-LARGE`
and larger) are CC BY-NC 4.0, which would matter if the pose-conditioned path is taken up.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
class TransformersPredictor:
    """Depth Anything V2 (or any `AutoModelForDepthEstimation`) via `transformers`.

    Emits **disparity**: larger is nearer.

    The model is only instantiated on the first cache miss, so a run whose cache is already
    complete never pays for loading it (or for having `transformers` importable at all).
    """

    QUANTITY = "disparity"

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
    def predict(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        disparity = self._model(**inputs).predicted_depth
        return disparity.squeeze(0).float().cpu().numpy()


@dataclass
class DepthAnything3Predictor:
    """Depth Anything 3 through its own `depth_anything_3` package.

    Emits **depth**: larger is farther. DA3 deliberately dropped the disparity parameterisation
    its predecessors used, so this is the opposite convention to `TransformersPredictor` and the
    two are not interchangeable.

    `DA3MONO-LARGE` is the largest published *monocular* DA3 checkpoint -- there is no mono
    GIANT -- and is Apache-2.0. The larger any-view models are CC BY-NC 4.0, so a switch to one
    of those would be a licence change as well as a model change.
    """

    QUANTITY = "depth"

    model_id: str
    device: str = "cuda"
    # Longest side the upstream preprocessing resizes to before enforcing a multiple of its patch
    # size. `None` keeps DA3's own default, which is what a plain user of the model would get;
    # raising it trades inference time for a prediction closer to the source resolution.
    process_res: Optional[int] = None
    _model: object = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The depth_anything_3 backend needs the upstream `depth_anything_3` package on "
                "PYTHONPATH (https://github.com/ByteDance-Seed/Depth-Anything-3). Install it, or "
                "point dataset.pseudo_depth.cache_dir at an already-populated cache."
            ) from exc
        logger.info(f"Loading pseudo-depth model {self.model_id}")
        self._model = DepthAnything3.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        # A single-image call is monocular inference: DA3 can consume several views at once, but
        # doing so here would make a frame's prior depend on which frames it was batched with,
        # and the cache is addressed per frame.
        options = {} if self.process_res is None else {"process_res": self.process_res}
        prediction = self._model.inference([image], **options)
        return np.asarray(prediction.depth[0], dtype=np.float32)


@dataclass
class MoGe3Predictor:
    """MoGe-3 metric monocular depth from its upstream local-checkpoint API.

    MoGe-3 publishes metric depth, normals and point maps. Training currently consumes its depth
    only; it is still cached per image and sparse-aligned like DA3 so a dataset's scale convention
    never becomes an untested assumption. MoGe-3 checkpoints are local paths in the upstream API.
    """

    QUANTITY = "depth"

    model_id: str
    device: str = "cuda"
    _model: object = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from moge.model.v3 import MoGeModel
        except ImportError as exc:  # pragma: no cover - external optional package
            raise ImportError(
                "The moge3 backend needs the upstream MoGe package (`pip install git+https://github.com/microsoft/MoGe.git`)."
            ) from exc
        logger.info(f"Loading MoGe-3 checkpoint {self.model_id}")
        self._model = MoGeModel.from_pretrained(self.model_id).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, dtype=torch.float32)
        output = self._model.infer((tensor / 255.0).permute(2, 0, 1))
        return output["depth"].detach().float().cpu().numpy()


# Each backend fixes the quantity it emits, so a config cannot pair a model with the wrong one.
BACKENDS = {
    "transformers": TransformersPredictor,
    "depth_anything_3": DepthAnything3Predictor,
    "moge3": MoGe3Predictor,
}


def _cache_identity(backend: str, model_id: str) -> dict:
    return {
        "backend": backend,
        "model_id": model_id,
        "format_version": CACHE_FORMAT_VERSION,
        "quantity": BACKENDS[backend].QUANTITY,
    }


def _slugify(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)


class PseudoDepthCache:
    """On-disk store of per-frame pseudo-depth, keyed by the model that produced it.

    The directory name carries the model slug *and* a hash of the full identity, and a
    `meta.json` inside records that identity. Two different models therefore cannot share a
    directory, and a stale directory whose identity no longer matches is rejected loudly rather
    than read as if it were current.
    """

    def __init__(
        self,
        scene_path: str,
        model_id: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        backend: str = "transformers",
        **predictor_options,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unknown pseudo-depth backend {backend!r}; expected one of {sorted(BACKENDS)}")
        self.model_id = model_id
        self.backend = backend
        identity = _cache_identity(backend, model_id)
        digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
        root = Path(cache_dir) if cache_dir else Path(scene_path) / "pseudo_depth_cache"
        self.directory = root / f"{_slugify(model_id)}__{digest}"
        self._identity = identity
        self._predictor = BACKENDS[backend](model_id=model_id, device=device, **predictor_options)

    @property
    def quantity(self) -> str:
        """`"disparity"` or `"depth"`, as the stored values are to be read.

        Consumers must pass this to the loss rather than assume it; the two supported backends
        disagree, and the disagreement is a sign error that no loss curve reveals.
        """
        return self._identity["quantity"]

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
            prediction = self._predictor.predict(image)
            # Write via a temporary file so an interrupted run cannot leave a truncated entry
            # that later looks like a valid cache hit.
            target = self.entry_path(image_path)
            # More than one training process can prebuild a shared cache. A fixed temporary
            # name lets one process rename the other process's prediction (or make its rename
            # fail); a unique file keeps the final replace atomic for each writer.
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                # Written through an open handle rather than by path: `np.save` silently
                # appends `.npy` to a name that lacks it, which would leave the rename with
                # nothing to move.
                np.save(handle, prediction.astype(np.float32))
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    # -- reading ----------------------------------------------------------------------------

    def load(self, image_path: str, height: int, width: int) -> np.ndarray:
        """The stored prior for one frame, in `self.quantity`, resampled to `height` x `width`.

        Bicubic, unlike ground-truth maps which must use nearest neighbour: this is a smooth
        prediction being brought back to full resolution, which is what the upstream model's own
        inference code does, and the alternative would quantise the ordering the loss reads.
        """
        prediction = np.load(self.entry_path(image_path))
        tensor = torch.from_numpy(prediction)[None, None].float()
        resized = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bicubic", align_corners=False)
        return np.ascontiguousarray(resized[0, 0].numpy())
