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

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pseudo_depth_diagnostic import ALIGNMENTS, _align  # noqa: E402
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


class TestPriorAlignment:
    """The alignment ladder in `pseudo_depth_diagnostic`.

    Each rung grants the prior one more degree of freedom, and the whole point of reporting
    them side by side is that the gaps between them are attributable. A bug that quietly let
    an offset into the scale-only row, or a fit into the raw row, would not crash -- it would
    just make a scale-free prior look metric.
    """

    @staticmethod
    def _grid():
        rng = np.random.default_rng(0)
        z = 2.0 + rng.random((48, 48)) * 8.0
        return z, np.ones_like(z, dtype=bool)

    @pytest.mark.parametrize("mode,patch", [(None, None), ("scale", None), ("affine", None), ("affine", 16)])
    def test_an_exact_prior_survives_every_rung(self, mode, patch) -> None:
        z, valid = self._grid()
        out = _align(z.copy(), z, valid, mode, patch, trim=0.2, iters=3, invert=False)
        assert np.nanmax(np.abs(out - z)) < 1e-9

    def test_raw_does_not_fit_anything(self) -> None:
        """Otherwise the row that exists to expose a missing scale would hide it."""
        z, valid = self._grid()
        out = _align(3.0 * z, z, valid, None, None, trim=0.2, iters=3, invert=False)
        assert np.nanmean(out / z) == pytest.approx(3.0)

    def test_scale_only_recovers_a_pure_scale_but_not_an_offset(self) -> None:
        z, valid = self._grid()
        scaled = _align(3.0 * z, z, valid, "scale", None, trim=0.2, iters=3, invert=False)
        assert np.nanmax(np.abs(scaled - z)) < 1e-9
        # An offset is exactly what this rung withholds, so it must *not* be absorbed.
        offset = _align(2.0 * z + 5.0, z, valid, "scale", None, trim=0.2, iters=3, invert=False)
        assert np.nanmax(np.abs(offset - z)) > 0.1
        affine = _align(2.0 * z + 5.0, z, valid, "affine", None, trim=0.2, iters=3, invert=False)
        assert np.nanmax(np.abs(affine - z)) < 1e-9

    def test_inverting_a_disparity_prior_is_a_change_of_variable_not_a_fit(self) -> None:
        """The raw row for a disparity prior reads 1/p as a distance, with nothing fitted."""
        z, valid = self._grid()
        out = _align(1.0 / z, 1.0 / z, valid, None, None, trim=0.2, iters=3, invert=True)
        assert np.nanmax(np.abs(out - z)) < 1e-9

    def test_unknown_mode_is_rejected(self) -> None:
        z, valid = self._grid()
        with pytest.raises(ValueError, match="unknown alignment mode"):
            _align(z, z, valid, "quadratic", None, trim=0.2, iters=3, invert=False)

    def test_the_ladder_is_ordered_from_least_to_most_freedom(self) -> None:
        """`report` prints these in order, so a reader compares adjacent rungs."""
        labels = [label for label, _, _ in ALIGNMENTS]
        assert labels[:3] == ["raw (no fit)", "scale only", "affine, global"]
        patches = [patch for _, mode, patch in ALIGNMENTS if mode == "affine" and patch]
        assert patches == sorted(patches, reverse=True)


class TestSparseObservations:
    """COLMAP sparse points projected into a frame, in `sparse_align_diagnostic`.

    This is the part of the regression-prior measurement with no natural sanity check inside the
    numbers it produces: a wrong downscale factor, a transposed pixel or z-instead-of-distance
    all yield plausible-looking `abs_rel`. The script validates itself at runtime by comparing
    the points against reference depth, and these pin the arithmetic that check relies on.
    """

    @staticmethod
    def _identity_image(xys, ids):
        from threedgrut.datasets.utils import Image

        return Image(
            id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),  # identity rotation
            tvec=np.zeros(3),
            camera_id=1,
            name="frame.png",
            xys=np.asarray(xys, dtype=np.float64),
            point3D_ids=np.asarray(ids),
        )

    def test_a_point_on_the_axis_gives_its_own_depth(self) -> None:
        from sparse_align_diagnostic import sparse_observations

        image = self._identity_image([[8.0, 6.0]], [7])
        rows = sparse_observations(image, {7: np.array([0.0, 0.0, 5.0])}, 1.0, 1.0, (12, 16))
        assert rows.shape == (1, 4)
        col, row, z, dist = rows[0]
        assert (col, row) == (8.0, 6.0)
        # On the optical axis z and Euclidean distance coincide; off it they must not.
        assert z == pytest.approx(5.0) and dist == pytest.approx(5.0)

    def test_distance_exceeds_z_off_axis(self) -> None:
        """`depth_gt` is ray distance, so returning z would understate depth everywhere but the centre."""
        from sparse_align_diagnostic import sparse_observations

        image = self._identity_image([[0.0, 0.0]], [1])
        rows = sparse_observations(image, {1: np.array([3.0, 4.0, 12.0])}, 1.0, 1.0, (12, 16))
        assert rows[0, 2] == pytest.approx(12.0)
        assert rows[0, 3] == pytest.approx(13.0)  # sqrt(9 + 16 + 144)

    def test_the_downscale_factor_divides_the_pixel_but_not_the_depth(self) -> None:
        from sparse_align_diagnostic import sparse_observations

        image = self._identity_image([[8.0, 6.0]], [1])
        rows = sparse_observations(image, {1: np.array([0.0, 0.0, 5.0])}, 2.0, 1.0, (12, 16))
        assert (rows[0, 0], rows[0, 1]) == (4.0, 3.0)
        assert rows[0, 3] == pytest.approx(5.0)

    def test_world_scale_multiplies_the_depth_but_not_the_pixel(self) -> None:
        """`normalize_world_space` rescales poses and `depth_gt`; the points must follow."""
        from sparse_align_diagnostic import sparse_observations

        image = self._identity_image([[8.0, 6.0]], [1])
        rows = sparse_observations(image, {1: np.array([0.0, 0.0, 5.0])}, 1.0, 0.5, (12, 16))
        assert (rows[0, 0], rows[0, 1]) == (8.0, 6.0)
        assert rows[0, 3] == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "xy,xyz,ids,why",
        [
            ([[8.0, 6.0]], {1: np.array([0.0, 0.0, -5.0])}, [1], "behind the camera"),
            ([[8.0, 6.0]], {}, [1], "point id absent from points3D"),
            ([[8.0, 6.0]], {1: np.array([0.0, 0.0, 5.0])}, [-1], "keypoint never triangulated"),
            ([[99.0, 6.0]], {1: np.array([0.0, 0.0, 5.0])}, [1], "outside the frame"),
        ],
    )
    def test_unusable_observations_are_dropped(self, xy, xyz, ids, why) -> None:
        from sparse_align_diagnostic import sparse_observations

        rows = sparse_observations(self._identity_image(xy, ids), xyz, 1.0, 1.0, (12, 16))
        assert rows.shape == (0, 4), why

    def test_points3d_text_reader_keeps_the_ids(self, tmp_path: Path) -> None:
        """The repo's own readers drop them, which is why this one exists."""
        from sparse_align_diagnostic import read_points3d_with_ids

        (tmp_path / "points3D.txt").write_text(
            "# comment\n" "5 1.0 2.0 3.0 255 0 0 0.5 1 0 2 1\n" "9 -1.0 0.0 4.0 0 255 0 0.25 1 3\n"
        )
        points = read_points3d_with_ids(tmp_path)
        assert sorted(points) == [5, 9]
        assert points[5] == pytest.approx(np.array([1.0, 2.0, 3.0]))
        assert points[9] == pytest.approx(np.array([-1.0, 0.0, 4.0]))
