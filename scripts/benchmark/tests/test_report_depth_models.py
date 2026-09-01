# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for the depth-model benchmark report."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_depth_models import markdown, pivot, qualitative_page, suite_view_subtitle


def _record(suite: str, scene: str, model: str, alignment: str, value: float, views: int) -> dict:
    record = {"suite": suite, "scene": scene, "model": model, "alignment": alignment, "views": views}
    if suite == "ob3d":
        record["frames"] = views
        record["depth"] = {"abs_rel": value}
    else:
        record["recall"] = {"recall": [0.0, 0.0, 0.0, value], "taus": [0.0, 0.0, 0.0, 5.0]}
        record["surface"] = {"overall": value, "fscore": [value]}
    return record


def test_pivot_averages_all_requested_scenes():
    records = [
        _record("ob3d", "a", "model", "raw", 0.1, 10),
        _record("ob3d", "b", "model", "raw", 0.3, 10),
    ]
    assert pivot(records, lambda row: row["depth"]["abs_rel"]) == (["model"], [["0.200", "—", "—"]])


def test_report_uses_actual_multiview_count_and_never_claims_one_view():
    records = [
        _record("ob3d", "emerald-square", "model", "raw", 0.1, 100),
        _record("dtu", "scan24", "model", "raw", 0.1, 49),
        _record("tnt", "Barn", "model", "raw", 0.1, 410),
    ]
    report = markdown(records, "test")
    assert "| tnt | Barn | 410 |" in report
    assert "One-view fusion" not in report
    assert suite_view_subtitle(records, "tnt", "fallback") == "Barn: 410 posed views"


def test_qualitative_page_is_optional_when_only_records_are_fetched(tmp_path):
    records = [_record("dtu", "scan24", "model", "raw", 0.1, 49)]
    with PdfPages(tmp_path / "report.pdf") as pdf:
        qualitative_page(pdf, tmp_path / "results", tmp_path / "dtu", records)
