# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pseudo-depth cache.

The cache's job is to be indistinguishable from rerunning the model, and to refuse to be
anything else. Training against another model's predictions would not show up in a loss curve,
so the identity checks below matter more than the arithmetic.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from threedgrut.datasets.pseudo_depth import BACKENDS, CACHE_FORMAT_VERSION, PseudoDepthCache
from threedgrut.utils.pseudo_depth_loss import FARTHER_SIGN

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


class _StubPredictor:
    """Stands in for the foundation model; counts calls so cache hits are observable."""

    def __init__(self, height: int = 6, width: int = 8):
        self.calls: list[str] = []
        self._height, self._width = height, width

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.calls.append(str(image.shape))
        ramp = np.linspace(0.0, 1.0, self._width, dtype=np.float32)
        return np.tile(ramp, (self._height, 1))


def _scene(tmp_path, count: int = 3):
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = images / f"{index:05d}_rgb.png"
        Image.fromarray(np.full((12, 16, 3), index * 10, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def _cache(tmp_path, model_id: str = "vendor/model-Base-hf", **kwargs) -> PseudoDepthCache:
    cache = PseudoDepthCache(str(tmp_path), model_id, device="cpu", **kwargs)
    cache._predictor = _StubPredictor()
    return cache


def test_cache_directory_identifies_the_model(tmp_path):
    first = _cache(tmp_path, "vendor/model-Base-hf")
    second = _cache(tmp_path, "vendor/model-Large-hf")
    assert first.directory != second.directory
    assert "model-Base-hf" in first.directory.name
    # A slug alone could collide after sanitisation, so a hash of the full identity is appended.
    assert first.directory.name.split("__")[-1] != second.directory.name.split("__")[-1]


def test_model_ids_differing_only_in_punctuation_do_not_share_a_directory(tmp_path):
    """Sanitising `/` and `-` to `_` makes distinct ids collide unless the hash disambiguates."""
    first = _cache(tmp_path, "vendor/model-base")
    second = _cache(tmp_path, "vendor-model_base")
    assert first.directory != second.directory


def test_ensure_populates_then_hits_the_cache(tmp_path):
    paths = _scene(tmp_path)
    cache = _cache(tmp_path)
    cache.ensure(paths)
    assert len(cache._predictor.calls) == len(paths)
    assert all(cache.entry_path(p).exists() for p in paths)

    # A second cache over the same directory must not invoke the model at all.
    reopened = _cache(tmp_path)
    reopened.ensure(paths)
    assert reopened._predictor.calls == []


def test_ensure_only_predicts_the_missing_frames(tmp_path):
    paths = _scene(tmp_path, count=4)
    cache = _cache(tmp_path)
    cache.ensure(paths[:2])
    resumed = _cache(tmp_path)
    resumed.ensure(paths)
    assert len(resumed._predictor.calls) == 2


def test_meta_records_the_identity(tmp_path):
    cache = _cache(tmp_path)
    cache.ensure(_scene(tmp_path, count=1))
    meta = json.loads((cache.directory / "meta.json").read_text())
    assert meta == {
        "backend": "transformers",
        "model_id": "vendor/model-Base-hf",
        "format_version": CACHE_FORMAT_VERSION,
        "quantity": "disparity",
    }


def test_mismatched_meta_is_rejected_rather_than_reused(tmp_path):
    """The directory hash makes this unreachable by config alone; a hand-edited cache must fail."""
    cache = _cache(tmp_path)
    cache.ensure(_scene(tmp_path, count=1))
    (cache.directory / "meta.json").write_text(json.dumps({"model_id": "someone/else", "format_version": 1}))
    with pytest.raises(RuntimeError, match="written by a different configuration"):
        _cache(tmp_path).ensure(_scene(tmp_path, count=1))


def test_load_resamples_to_the_requested_resolution(tmp_path):
    paths = _scene(tmp_path, count=1)
    cache = _cache(tmp_path)
    cache.ensure(paths)
    # Stored at the stub's native 6x8; requested at the training resolution.
    loaded = cache.load(paths[0], 12, 16)
    assert loaded.shape == (12, 16)
    assert loaded.dtype == np.float32
    assert loaded.flags["C_CONTIGUOUS"]


def test_load_preserves_the_ordering_the_loss_reads(tmp_path):
    """Resampling may over/undershoot, but the monotone ramp must stay monotone."""
    paths = _scene(tmp_path, count=1)
    cache = _cache(tmp_path)
    cache.ensure(paths)
    loaded = cache.load(paths[0], 12, 16)
    assert np.all(np.diff(loaded, axis=1) > 0)


def test_no_temporary_files_are_left_behind(tmp_path):
    """Entries are written via a temporary path so an interrupted run cannot look like a hit."""
    paths = _scene(tmp_path)
    cache = _cache(tmp_path)
    cache.ensure(paths)
    assert list(cache.directory.glob("*.tmp")) == []


def test_entries_from_different_folders_do_not_collide(tmp_path):
    """Two frames sharing a stem in different folders must map to different entries."""
    cache = _cache(tmp_path)
    left = cache.entry_path(str(tmp_path / "left" / "00000_rgb.png"))
    right = cache.entry_path(str(tmp_path / "right" / "00000_rgb.png"))
    assert left != right


def test_explicit_cache_dir_is_honoured(tmp_path):
    """A read-only scene directory has to be servable from elsewhere."""
    elsewhere = tmp_path / "somewhere_else"
    cache = _cache(tmp_path, cache_dir=str(elsewhere))
    cache.ensure(_scene(tmp_path, count=1))
    assert elsewhere in cache.directory.parents


class TestBackends:
    """Each backend owns the convention it emits, so a config cannot pair the two wrongly."""

    def test_the_two_backends_disagree_about_the_quantity(self):
        """This is the point of the abstraction: DAv2 emits disparity, DA3 emits depth."""
        assert BACKENDS["transformers"].QUANTITY == "disparity"
        assert BACKENDS["depth_anything_3"].QUANTITY == "depth"

    def test_every_quantity_is_one_the_loss_understands(self):
        """A backend naming a quantity the loss has no sign for would fail only at train time."""
        assert {b.QUANTITY for b in BACKENDS.values()} <= set(FARTHER_SIGN)

    def test_cache_reports_the_backends_quantity(self, tmp_path):
        assert _cache(tmp_path).quantity == "disparity"
        assert _cache(tmp_path, backend="depth_anything_3").quantity == "depth"

    def test_switching_backend_cannot_reuse_the_other_cache(self, tmp_path):
        """Same model id, different backend: the predictions are not interchangeable."""
        transformers = _cache(tmp_path, backend="transformers")
        da3 = _cache(tmp_path, backend="depth_anything_3")
        assert transformers.directory != da3.directory

    def test_unknown_backend_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown pseudo-depth backend"):
            PseudoDepthCache(str(tmp_path), "vendor/model", device="cpu", backend="depth_anything_2")

    def test_default_backend_is_the_permissively_licensed_one(self):
        """DA3's weights are CC BY-NC 4.0, so it must never become the default by accident."""
        from omegaconf import OmegaConf

        config = OmegaConf.load(_REPO_ROOT / "configs" / "dataset" / "colmap.yaml")
        assert config.pseudo_depth.backend == "transformers"


class TestTrainingSplitOptIn:
    """The dataset-side flag is derived from the loss, so the two cannot disagree.

    Both directions of disagreement are silent: a loss with no prior would read zero forever,
    and a prior with no loss would pay for a cache nothing consumes.
    """

    @staticmethod
    def _config(use_loss: bool, enabled: bool = False, model: str = "vendor/model"):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "dataset": {"pseudo_depth": {"enabled": enabled, "model": model, "cache_dir": None}},
                "loss": {"use_pseudo_depth_order": use_loss},
            }
        )

    def test_loss_switches_the_prior_on(self):
        from threedgrut.datasets import _pseudo_depth_config

        assert _pseudo_depth_config(self._config(use_loss=True))["enabled"] is True

    def test_prior_stays_off_when_no_loss_wants_it(self):
        from threedgrut.datasets import _pseudo_depth_config

        assert _pseudo_depth_config(self._config(use_loss=False))["enabled"] is False

    def test_prior_can_still_be_forced_on_for_inspection(self):
        from threedgrut.datasets import _pseudo_depth_config

        assert _pseudo_depth_config(self._config(use_loss=False, enabled=True))["enabled"] is True

    def test_model_choice_is_carried_through(self):
        from threedgrut.datasets import _pseudo_depth_config

        resolved = _pseudo_depth_config(self._config(use_loss=True, model="vendor/other"))
        assert resolved["model"] == "vendor/other"

    def test_backend_choice_is_carried_through(self):
        from threedgrut.datasets import _pseudo_depth_config

        config = self._config(use_loss=True)
        config.dataset.pseudo_depth.backend = "depth_anything_3"
        assert _pseudo_depth_config(config)["backend"] == "depth_anything_3"

    def test_backend_defaults_when_a_scene_config_predates_the_option(self):
        from threedgrut.datasets import _pseudo_depth_config

        assert _pseudo_depth_config(self._config(use_loss=True))["backend"] == "transformers"

    def test_dataset_without_a_pseudo_depth_section_is_tolerated(self):
        """Only the colmap dataset defines it; the others must not break."""
        from omegaconf import OmegaConf

        from threedgrut.datasets import _pseudo_depth_config

        config = OmegaConf.create({"dataset": {}, "loss": {"use_pseudo_depth_order": True}})
        assert _pseudo_depth_config(config) == {}
