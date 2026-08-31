# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_depth_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_depth_models", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_depth_alignment_respects_disparity_parameterization():
    disparity = np.array([[0.5, 0.25], [0.125, 0.0625]], dtype=np.float32)
    truth = 2.0 / disparity
    aligned = benchmark.align_prediction(disparity, truth, "disparity", "scale")
    np.testing.assert_allclose(aligned, truth)


def test_depth_alignment_recovers_depth_scale_and_offset():
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    truth = 3.0 * prediction + 2.0
    aligned = benchmark.align_prediction(prediction, truth, "depth", "affine")
    np.testing.assert_allclose(aligned, truth)


def test_empty_mesh_is_a_reported_zero_fscore_not_a_missing_cell():
    metrics = benchmark._empty_surface_metrics(np.array([0.001, 0.01]))
    assert metrics["empty_prediction"]
    assert metrics["fscore"] == [0.0, 0.0]
    assert metrics["overall"] is None


def test_predict_frames_keeps_one_resized_map_per_frame(tmp_path):
    class Predictor:
        def __init__(self):
            self.calls = 0

        def predict(self, image):
            self.calls += 1
            return np.full(image.shape[:2], self.calls, dtype=np.float32)

    image = tmp_path / "image.png"
    benchmark.Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image)
    view = benchmark.View("view", 3, 2, np.eye(3), np.eye(3), np.zeros(3), tmp_path / "unused.npy")
    frame = benchmark.Frame("view", image, view)
    predictor = Predictor()

    predictions = benchmark.predict_frames(predictor, [frame])
    assert predictor.calls == 1
    np.testing.assert_array_equal(predictions[0], np.ones((2, 3), dtype=np.float32))


def test_prepare_aligned_depths_streams_one_frame_at_a_time(tmp_path):
    view = benchmark.View("view", 2, 2, np.eye(3), np.eye(3), np.zeros(3), tmp_path / "unused.npy")
    frame = benchmark.Frame("view", tmp_path / "unused.png", view)
    visibility_path = tmp_path / "visibility.npy"
    prediction = np.full((2, 2), 2.0, dtype=np.float32)
    np.save(visibility_path, np.full((2, 2), 4.0, dtype=np.float32))

    views, max_depth = benchmark.prepare_aligned_depths(
        [frame], [prediction], {"view": visibility_path}, "depth", "scale", tmp_path / "aligned"
    )

    expected_z = benchmark.align_prediction(
        prediction, benchmark._z_from_ray(np.load(visibility_path), view.K), "depth", "scale"
    )
    assert max_depth == np.max(expected_z)
    np.testing.assert_allclose(np.load(views[0].depth_path), benchmark.z_depth_to_ray_distance(expected_z, view.K))
