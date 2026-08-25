# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sweep bookkeeping, which is where a silent error would be costly.

A wrong number in a report is worse than a crash: it gets believed. These cover the
rules that keep a partially completed or partially failed sweep from being read as a
clean comparison.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import UNMEASURED_WHEN_ZERO, mean, read_rows, summary_table  # noqa: E402
from run_ob3d import BASELINE_VARIANTS, Cell, Variant, completed_keys, scene_dirs, tail_of  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def ok_row(variant: str, scene: str, **extra) -> dict:
    return {"variant": variant, "scene": scene, "status": "ok", "n_iterations": 100, **extra}


def test_resume_skips_only_successful_cells(tmp_path: Path) -> None:
    """A failed cell must be retried, or a transient OOM silently drops from the grid."""
    path = write_jsonl(
        tmp_path / "r.jsonl",
        [
            ok_row("a", "s1"),
            {"variant": "a", "scene": "s2", "status": "failed"},
            {"variant": "b", "scene": "s1", "status": "timeout"},
        ],
    )
    assert completed_keys(path) == {("a", "s1")}


def test_resume_survives_a_truncated_final_line(tmp_path: Path) -> None:
    """A sweep killed mid-write must still be resumable rather than crashing the next run."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(ok_row("a", "s1")) + "\n" + '{"variant": "a", "scene')
    assert completed_keys(path) == {("a", "s1")}


def test_resume_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert completed_keys(tmp_path / "absent.jsonl") == set()


def test_rerun_of_a_cell_supersedes_the_earlier_attempt(tmp_path: Path) -> None:
    """Re-running a failure must replace it, not leave both rows in the report."""
    path = write_jsonl(
        tmp_path / "r.jsonl",
        [{"variant": "a", "scene": "s1", "status": "failed"}, ok_row("a", "s1", mean_psnr=30.0)],
    )
    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"


def test_averages_use_only_scenes_every_variant_completed(tmp_path: Path) -> None:
    """Averaging over a different scene subset per variant is not a comparison.

    Here variant `b` failed the hard scene. If its mean were taken over only the easy
    scene it would look better than `a` purely by having skipped the hard one.
    """
    rows = [
        ok_row("a", "easy", depth_abs_rel=0.10),
        ok_row("a", "hard", depth_abs_rel=0.50),
        ok_row("b", "easy", depth_abs_rel=0.20),
    ]
    shared = {"easy"}
    table = summary_table(rows, [("depth_abs_rel", "d_absrel", "{:.4f}", True)], shared)

    assert "0.1000" in table  # variant a on the shared scene only
    assert "0.3000" not in table  # not the mean of 0.10 and 0.50
    assert "0.2000" in table


def test_best_value_is_highlighted_in_the_right_direction() -> None:
    rows = [ok_row("a", "s", mean_psnr=30.0, mean_lpips=0.30), ok_row("b", "s", mean_psnr=25.0, mean_lpips=0.10)]
    columns = [("mean_psnr", "psnr", "{:.2f}", False), ("mean_lpips", "lpips", "{:.3f}", True)]
    table = summary_table(rows, columns, {"s"})

    assert "**30.00**" in table  # higher psnr wins
    assert "**0.100**" in table  # lower lpips wins
    assert "**25.00**" not in table


def test_unranked_columns_are_never_highlighted() -> None:
    """Depth bias has no better direction; marking one as best would assert a claim."""
    rows = [ok_row("a", "s", depth_bias=-0.5), ok_row("b", "s", depth_bias=0.1)]
    table = summary_table(rows, [("depth_bias", "d_bias", "{:+.3f}", None)], {"s"})
    assert "**" not in table


def test_missing_metric_renders_as_absent_not_zero() -> None:
    """A missing metric shown as 0.0000 would read as a perfect score."""
    table = summary_table(
        rows=[ok_row("a", "s")], columns=[("depth_abs_rel", "d", "{:.4f}", True)], shared_scenes={"s"}
    )
    assert "0.0000" not in table
    assert "-" in table


def test_unmeasured_inference_time_is_shown_as_absent() -> None:
    """The tracer reports 0.0 ms/frame when timings are off; that is not a fast render."""
    assert "mean_inference_time_ms" in UNMEASURED_WHEN_ZERO
    rows = [ok_row("a", "s", mean_inference_time_ms=0.0)]
    table = summary_table(rows, [("mean_inference_time_ms", "ms", "{:.2f}", None)], {"s"})
    assert "0.00" not in table


def test_mean_ignores_missing_and_non_finite_values() -> None:
    assert mean([1.0, None, 3.0]) == pytest.approx(2.0)
    assert mean([float("nan"), 2.0]) == pytest.approx(2.0)
    assert mean([]) is None
    assert mean([None]) is None


def test_cell_command_includes_geometry_and_variant_overrides(tmp_path: Path) -> None:
    """Depth metrics need the reference maps, so a cell without them measures nothing."""
    cell = Cell(
        variant=Variant("trisurfel", ("render.primitive_type=trisurfel",)),
        scene=tmp_path / "sponza",
        n_iterations=7000,
        out_dir=tmp_path / "out",
        config_name="apps/colmap_3dgut.yaml",
    )
    command = cell.command()

    assert "dataset.load_depth_gt=true" in command
    assert "dataset.load_normal_gt=true" in command
    assert "render.enable_normals=true" in command
    assert "render.primitive_type=trisurfel" in command
    assert "n_iterations=7000" in command
    assert cell.experiment == "abl_trisurfel_sponza"


def test_cell_experiment_names_are_unique_per_variant_and_scene(tmp_path: Path) -> None:
    """Colliding names would make two cells overwrite each other's run directory."""
    names = {
        Cell(variant, tmp_path / scene, 100, tmp_path, "c.yaml").experiment
        for variant in BASELINE_VARIANTS
        for scene in ("a", "b")
    }
    assert len(names) == 2 * len(BASELINE_VARIANTS)


def test_unknown_scene_is_rejected_up_front(tmp_path: Path) -> None:
    """A typo must fail before the sweep starts, not appear as a gap hours later."""
    (tmp_path / "real").mkdir()
    with pytest.raises(SystemExit, match="No such scene directory"):
        scene_dirs(tmp_path, ["real", "typo"])


def test_scene_discovery_requires_a_colmap_reconstruction(tmp_path: Path) -> None:
    (tmp_path / "good" / "sparse").mkdir(parents=True)
    (tmp_path / "not_a_scene").mkdir()
    assert [path.name for path in scene_dirs(tmp_path, None)] == ["good"]


def test_tail_of_a_missing_log_is_empty(tmp_path: Path) -> None:
    assert tail_of(tmp_path / "nope.log") == ""


def test_tail_of_returns_the_end_of_the_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("\n".join(str(index) for index in range(100)))
    assert tail_of(log, lines=3) == "97\n98\n99"
