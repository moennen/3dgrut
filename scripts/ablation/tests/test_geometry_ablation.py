# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cheap contract tests for the 30k geometry-ablation driver and its report."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ABLATON = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ABLATON / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("run_geometry_ablation")
report = load("report_geometry_ablation")
scorer = load("score_geometry_checkpoint")


def test_matrix_is_one_factor_plus_a_compatible_full_stack():
    variants = {variant.name: variant for variant in runner.VARIANTS}
    assert "full_geometry" in variants
    assert "radioc4_pca48" in variants
    feature = variants["radioc4_pca48"].overrides
    assert "dataset.image_features.backend=nvradio4" in feature
    assert "dataset.image_features.model=c-radio_v4-h" in feature
    assert "dataset.image_features.projector=pca" in feature
    assert "dataset.image_features.output_dim=48" in feature
    assert "model.nht_decoder.image_feature_dim=48" in feature
    assert "dataset.pseudo_depth.backend=moge3" in variants["full_geometry"].overrides
    assert len(variants) == 13


def test_default_protocol_remains_30k_and_reduced():
    parser_source = (ABLATON / "run_geometry_ablation.py").read_text()
    assert 'default="reduced"' in parser_source
    assert "default=30000" in parser_source


def test_reference_image_maps_are_requested_only_for_ob3d(tmp_path):
    args = Namespace(
        ob3d_root=tmp_path / "ob3d",
        dtu_root=tmp_path / "dtu",
        tnt_reconstruction_root=tmp_path / "tnt",
        out_dir=tmp_path / "out",
        n_iterations=30000,
        moge3_model="moge.pt",
        config_name="apps/colmap_3dgut.yaml",
        override=[],
    )
    variant = next(item for item in runner.VARIANTS if item.name == "trisurfel")
    ob3d, _ = runner.train_command(args, "ob3d", "scene", variant)
    dtu, _ = runner.train_command(args, "dtu", "scan24", variant)
    assert "dataset.load_depth_gt=true" in ob3d
    assert "dataset.load_normal_gt=true" in ob3d
    assert "dataset.load_depth_gt=true" not in dtu


def test_surface_and_recall_report_official_source_units_after_normalization():
    surface = scorer.surface_in_source_units(
        {"accuracy": 0.6, "completeness": 0.9, "overall": 0.75, "taus": [0.06], "fscore": [0.4]},
        scorer.np.asarray([0.2]),
        3.0,
    )
    assert surface["taus"] == [0.2]
    assert surface["overall"] == pytest.approx(0.25)
    recall = scorer.recall_in_source_units(
        {"taus": [1.5], "median_signed_delta": 0.3, "per_view": [{"median_signed_delta": 0.6}]},
        scorer.np.asarray([0.5]),
        3.0,
    )
    assert recall["taus"] == [0.5]
    assert recall["median_signed_delta"] == pytest.approx(0.1)
    assert recall["per_view"][0]["median_signed_delta"] == pytest.approx(0.2)


def test_report_keeps_only_shared_scenes_and_reads_benchmark_fields(tmp_path):
    rows = [
        {
            "suite": "dtu",
            "scene": "scan24",
            "variant": "a",
            "status": "ok",
            "score_status": "ok",
            "evaluation": {"recall": {"taus": [0.5, 5.0], "recall": [0.1, 0.4]}, "surface": {"overall": 2.0}},
        },
        {
            "suite": "dtu",
            "scene": "scan24",
            "variant": "b",
            "status": "ok",
            "score_status": "ok",
            "evaluation": {"recall": {"taus": [0.5, 5.0], "recall": [0.2, 0.5]}, "surface": {"overall": 1.0}},
        },
        {
            "suite": "dtu",
            "scene": "scan55",
            "variant": "a",
            "status": "ok",
            "score_status": "ok",
            "evaluation": {"recall": {"taus": [0.5, 5.0], "recall": [0.1, 0.1]}, "surface": {"overall": 9.0}},
        },
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = report.records(path)
    headers, body, _ = report.table(data, "dtu")
    assert headers[-2:] == ["visible recall @5 mm ↑", "Chamfer (mm) ↓"]
    assert body == [["a", "1", "0.400", "2.00"], ["b", "1", "0.500", "1.00"]]
