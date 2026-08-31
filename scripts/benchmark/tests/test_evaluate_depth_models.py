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
