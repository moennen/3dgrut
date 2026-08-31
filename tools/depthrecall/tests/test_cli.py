# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI tests over a synthetic COLMAP scene."""

import json
from pathlib import Path

import numpy as np
import pytest

from depthrecall.cli import main

FX = FY = 100.0
CX = CY = 50.0
SIZE = 100
PLANE_Z = 3.0


def build_scene(root: Path, n_views: int = 4, depth_offset: float = 0.0) -> tuple[Path, Path, Path]:
    """Write a COLMAP sparse/0, a depth folder and a GT ply for a plane at z = PLANE_Z."""
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    depths = root / "depths"
    depths.mkdir()

    i = np.arange(SIZE) + 0.5
    u, v = np.meshgrid(i, i)
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    ray_len = np.linalg.norm(rays, axis=-1)

    camera_lines = ["# Camera list", f"1 PINHOLE {SIZE} {SIZE} {FX} {FY} {CX} {CY}"]
    image_lines = ["# Image list"]
    all_points = []
    for k in range(n_views):
        # Translate the camera along x so views differ but the plane stays in frame.
        shift = 0.05 * k
        name = f"{k:04d}.png"
        image_lines.append(f"{k + 1} 1 0 0 0 {shift} 0 0 1 {name}")
        image_lines.append("")
        np.save(depths / f"{k:04d}.npy", ray_len * PLANE_Z + depth_offset)
        # Points on the pixel-centre rays of this view, expressed in world coordinates.
        pts_cam = (rays * PLANE_Z)[::10, ::10].reshape(-1, 3)
        all_points.append(pts_cam - np.array([shift, 0.0, 0.0]))

    (sparse / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    ply = root / "gt.ply"
    with open(ply, "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
            ).encode()
        )
        f.write(points.astype("<f4").tobytes())

    return sparse, depths, ply


def build_manifest(root: Path, depths: Path, n_views: int = 4) -> Path:
    """A manifest over the same cameras as `build_scene`, naming each view's depth file."""
    views = []
    for k in range(n_views):
        views.append(
            {
                "name": f"{k:04d}.png",
                "width": SIZE,
                "height": SIZE,
                "K": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
                "R": np.eye(3).tolist(),
                "t": [0.05 * k, 0.0, 0.0],
                "depth_path": str(depths / f"{k:04d}.npy"),
                "depth_convention": "ray",
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"views": views}))
    return manifest


def test_cli_scores_a_manifest_without_a_depths_directory(tmp_path: Path, capsys) -> None:
    """A manifest names its own depth files, so --depths must not be required alongside it."""
    _sparse, depths, ply = build_scene(tmp_path)
    manifest = build_manifest(tmp_path, depths)
    out = tmp_path / "metrics.json"
    main(
        [
            "--manifest",
            str(manifest),
            "--ply",
            str(ply),
            "--taus",
            "0.01",
            "--depth-convention",
            "ray",
            "-o",
            str(out),
        ]
    )
    assert json.loads(out.read_text())["recall"][0] == pytest.approx(1.0)


def test_cli_rejects_depths_alongside_a_manifest(tmp_path: Path) -> None:
    """--depths is ignored on the manifest path, so accepting it would score the wrong folder.

    The failure this guards against is silent: the run succeeds and reports the manifest's
    depths under the name of whatever directory was passed.
    """
    _sparse, depths, ply = build_scene(tmp_path)
    manifest = build_manifest(tmp_path, depths)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="would be ignored"):
        main(
            [
                "--manifest",
                str(manifest),
                "--depths",
                str(elsewhere),
                "--ply",
                str(ply),
                "--taus",
                "0.01",
                "--depth-convention",
                "ray",
                "-o",
                str(tmp_path / "m.json"),
            ]
        )


def test_cli_still_requires_depths_without_a_manifest(tmp_path: Path) -> None:
    sparse, _depths, ply = build_scene(tmp_path)
    with pytest.raises(ValueError, match="--depths is required"):
        main(
            [
                "--colmap",
                str(sparse),
                "--ply",
                str(ply),
                "--taus",
                "0.01",
                "--depth-convention",
                "ray",
                "-o",
                str(tmp_path / "m.json"),
            ]
        )


def test_cli_writes_metrics_for_an_exact_reconstruction(tmp_path: Path, capsys) -> None:
    sparse, depths, ply = build_scene(tmp_path)
    out = tmp_path / "metrics.json"
    main(
        [
            "--colmap",
            str(sparse),
            "--depths",
            str(depths),
            "--ply",
            str(ply),
            "--taus",
            "1e-6,1e-3,0.01",
            "--depth-convention",
            "ray",
            "-o",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["views_count"] == 4
    assert data["no_surface_pairs"] == 0
    # Each view reproduces its *own* points exactly, so at a tau far below one pixel of depth
    # gradient at least 1/n_views of the pairs pass (more, where another view's points happen
    # to quantise onto the same pixel), and the rest come in once tau exceeds that
    # quantisation.
    assert 0.25 <= data["recall"][0] < 1.0
    assert data["recall"][-1] > 0.99
    assert data["config"]["depth_convention"] == "ray"
    assert data["config"]["sampling"] == "nearest"
    assert len(data["per_view"]) == 4
    assert data["viewset_hash"] and data["gt_hash"]

    captured = capsys.readouterr().out
    # Each threshold is reported, and the two denominators are named: recall counts
    # no-surface pairs as failures while the signed split excludes them, so a reader who
    # assumes one denominator misreads a sparse result badly.
    assert "1e-06:" in captured
    assert "recall (of in-frustum pairs)" in captured
    assert "too_near, too_far (of pairs with a surface)" in captured
    assert data["denominators"]["recall"] == "in_frustum_pairs"


def test_cli_view_stride_selects_a_subset(tmp_path: Path) -> None:
    sparse, depths, ply = build_scene(tmp_path, n_views=4)
    out = tmp_path / "m.json"
    main(
        [
            "--colmap",
            str(sparse),
            "--depths",
            str(depths),
            "--ply",
            str(ply),
            "--taus",
            "0.01",
            "--depth-convention",
            "ray",
            "--view-stride",
            "2",
            "-o",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["views_count"] == 2
    assert [v["name"] for v in data["per_view"]] == ["0000.png", "0002.png"]


def test_cli_records_a_degraded_reconstruction_as_lower_recall(tmp_path: Path) -> None:
    """A depth map biased behind the scan must read as reduced recall and positive sign."""
    good_root = tmp_path / "good"
    good_root.mkdir()
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    sparse_g, depths_g, ply_g = build_scene(good_root)
    sparse_b, depths_b, ply_b = build_scene(bad_root, depth_offset=0.02)

    results = {}
    for tag, (sparse, depths, ply) in {
        "good": (sparse_g, depths_g, ply_g),
        "bad": (sparse_b, depths_b, ply_b),
    }.items():
        out = tmp_path / f"{tag}.json"
        main(
            [
                "--colmap",
                str(sparse),
                "--depths",
                str(depths),
                "--ply",
                str(ply),
                "--taus",
                "0.005",
                "--depth-convention",
                "ray",
                "-o",
                str(out),
            ]
        )
        results[tag] = json.loads(out.read_text())

    assert results["good"]["recall"][0] > results["bad"]["recall"][0]
    assert results["bad"]["too_far"][0] > 0.9
    assert results["bad"]["median_signed_delta"] > 0
    # Same cameras and same scan in both runs, so a comparison between them is legitimate.
    assert results["good"]["viewset_hash"] == results["bad"]["viewset_hash"]
    assert results["good"]["gt_hash"] == results["bad"]["gt_hash"]


def test_cli_rejects_unsorted_taus(tmp_path: Path) -> None:
    sparse, depths, ply = build_scene(tmp_path)
    with pytest.raises(ValueError):
        main(
            [
                "--colmap",
                str(sparse),
                "--depths",
                str(depths),
                "--ply",
                str(ply),
                "--taus",
                "1.0,0.5",
                "--depth-convention",
                "ray",
                "-o",
                str(tmp_path / "m.json"),
            ]
        )


def write_dtu_cameras(path: Path, n_views: int, scale: float) -> None:
    """Write a minimal DTU cameras.npz: world_mat = K [R|t] in scan units, plus scale_mat."""
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    entries = {}
    scale_mat = np.eye(4)
    scale_mat[:3, :3] *= scale
    for k in range(n_views):
        Rt = np.eye(4)
        Rt[0, 3] = 0.05 * k * scale  # the same lateral shift as build_scene, in scan units
        world_mat = np.eye(4)
        world_mat[:3, :4] = K @ Rt[:3, :4]
        entries[f"world_mat_{k}"] = world_mat
        entries[f"scale_mat_{k}"] = scale_mat
    np.savez(path, **entries)


def build_dtu_scene(root: Path, n_views: int = 4, scale: float = 10.0) -> tuple[Path, Path, Path]:
    """A DTU-shaped scene: depths in the normalized frame, GT in scan units."""
    _, depths, _ = build_scene(root, n_views=n_views)
    cameras = root / "cameras.npz"
    write_dtu_cameras(cameras, n_views, scale)

    # GT in scan units: the same plane, scaled up, plus points well below a ground plane.
    i = np.arange(0, SIZE, 10) + 0.5
    u, v = np.meshgrid(i, i)
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1).reshape(-1, 3)
    object_points = rays * PLANE_Z * scale
    # The "below the ground" half must stay *inside* the frustum, or it never reaches the
    # denominator and culling it cannot change recall -- which is exactly what a first version
    # of this test did, leaving both sides at 1.0 and asserting nothing.
    below = rays * 1.5 * PLANE_Z * scale
    points = np.concatenate([object_points, below], axis=0)
    ply = root / "gt_scan.npy"
    np.save(ply, points)
    return cameras, depths, ply


def test_cli_dtu_requires_an_explicit_decision_about_the_ground_plane(tmp_path: Path) -> None:
    """DTU's official completeness culls the GT below the ground plane.

    Silently skipping it inflates recall by several points, so the DTU path refuses to run
    until the caller either supplies the plane or says it does not want the cull.
    """
    cameras, depths, ply = build_dtu_scene(tmp_path)
    args = [
        "--dtu-cameras",
        str(cameras),
        "--dtu-space",
        "normalized",
        "--depths",
        str(depths),
        "--depth-ext",
        ".npy",
        "--ply",
        str(ply),
        "--taus",
        "0.01",
        "--depth-convention",
        "ray",
        "-o",
        str(tmp_path / "m.json"),
    ]
    with pytest.raises(ValueError, match="ground plane"):
        main(args)

    # Opting out explicitly is allowed, and is recorded as such.
    main(args + ["--no-gt-mask"])
    assert json.loads((tmp_path / "m.json").read_text())["config"]["gt_masks"] == []


def test_cli_dtu_plane_cull_changes_the_denominator_and_is_recorded(tmp_path: Path, capsys) -> None:
    cameras, depths, ply = build_dtu_scene(tmp_path)
    # Keep z < 37.5 (the object at z = 30), culling the copy at z = 45: half the cloud.
    plane_path = tmp_path / "plane.txt"
    np.savetxt(plane_path, np.array([0.0, 0.0, -1.0, 37.5]))

    def run(extra: list[str], out: str) -> dict:
        main(
            [
                "--dtu-cameras",
                str(cameras),
                "--dtu-space",
                "normalized",
                "--depths",
                str(depths),
                "--depth-ext",
                ".npy",
                "--ply",
                str(ply),
                "--taus",
                "0.01",
                "--depth-convention",
                "ray",
                "-o",
                str(tmp_path / out),
            ]
            + extra
        )
        return json.loads((tmp_path / out).read_text())

    culled = run(["--dtu-plane", str(plane_path)], "culled.json")
    full = run(["--no-gt-mask"], "full.json")

    assert culled["gt_points"] * 2 == full["gt_points"]
    assert culled["config"]["gt_masks"] == ["ground plane (0.5000 kept)"]
    assert "ground plane keeps" in capsys.readouterr().out
    # The culled half never projects onto the plane's depth, so removing it must raise recall.
    assert culled["recall"][0] > full["recall"][0]


def test_cli_rejects_no_gt_mask_together_with_a_mask(tmp_path: Path) -> None:
    cameras, depths, ply = build_dtu_scene(tmp_path)
    plane_path = tmp_path / "plane.txt"
    np.savetxt(plane_path, np.array([0.0, 0.0, -1.0, 37.5]))
    with pytest.raises(ValueError, match="conflicts"):
        main(
            [
                "--dtu-cameras",
                str(cameras),
                "--depths",
                str(depths),
                "--depth-ext",
                ".npy",
                "--ply",
                str(ply),
                "--taus",
                "0.01",
                "--depth-convention",
                "ray",
                "-o",
                str(tmp_path / "m.json"),
                "--dtu-plane",
                str(plane_path),
                "--no-gt-mask",
            ]
        )
