# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `scripts/align_affine.py`, the per-frame affine alignment of a monocular prior."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_cli import CX, CY, FX, FY, SIZE

from depthrecall.cli import main
from depthrecall.metric import MetricConfig

_spec = importlib.util.spec_from_file_location(
    "align_affine", Path(__file__).resolve().parents[1] / "scripts" / "align_affine.py"
)
align_affine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(align_affine)

K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
Z0 = 3.0
TILT = 0.3
STRIDE = 10


def build_tilted_scene(root: Path, n_views: int = 2) -> tuple[Path, Path, Path]:
    """A plane tilted about the y axis, its ray-distance depth maps, a manifest and a GT ply.

    The tilt is what makes the scene usable: a fronto-parallel plane has *constant* z, so a least
    squares affine onto it is degenerate -- scale goes to zero and the shift returns the constant,
    which absorbs any error a test tries to inject and reports a perfect fit.
    """
    depths = root / "depths"
    depths.mkdir()
    j, i = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5)
    x = (j - CX) / FX
    y = (i - CY) / FY
    ray_length = np.sqrt(1.0 + x**2 + y**2)
    z = Z0 * (1.0 + TILT * x)

    views, all_points = [], []
    for k in range(n_views):
        shift = 0.05 * k
        name = f"{k:04d}.png"
        np.save(depths / f"{k:04d}.npy", (z * ray_length).astype(np.float32))
        points = np.stack([x * z, y * z, z], axis=-1)[::STRIDE, ::STRIDE].reshape(-1, 3)
        all_points.append(points - np.array([shift, 0.0, 0.0]))
        views.append(
            {
                "name": name,
                "width": SIZE,
                "height": SIZE,
                "K": K.tolist(),
                "R": np.eye(3).tolist(),
                "t": [shift, 0.0, 0.0],
                "depth_path": str(depths / f"{k:04d}.npy"),
                "depth_convention": "ray",
            }
        )

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"views": views}))

    pts = np.concatenate(all_points, axis=0).astype(np.float32)
    ply = root / "gt.ply"
    with open(ply, "wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
            ).encode()
        )
        f.write(pts.astype("<f4").tobytes())
    return manifest, depths, ply


def test_ray_length_grid_uses_the_metrics_pixel_centre_convention() -> None:
    """Array index i is the pixel at continuous coordinate i + 0.5, as `MetricConfig` documents.

    Evaluating on the integer grid instead offsets the z-to-ray factor by half a pixel, which is
    invisible on a flat surface and largest where the geometry is interesting.
    """
    offset = MetricConfig(taus=np.array([1.0]), depth_convention="ray").pixel_center_offset
    grid = align_affine.ray_length_grid(K, SIZE, SIZE)
    for i, j in [(0, 0), (SIZE - 1, SIZE - 1), (SIZE // 2, SIZE // 3)]:
        x = (j + offset - CX) / FX
        y = (i + offset - CY) / FY
        assert grid[i, j] == pytest.approx(np.sqrt(1.0 + x**2 + y**2))
    # The principal point sits between pixels for an even size, so no pixel is exactly 1.0.
    assert grid.min() > 1.0


def test_emitting_ray_undoes_the_z_conversion(tmp_path: Path) -> None:
    """Fitting a reference's own z-depth and emitting ray distance must return the reference.

    This is the end-to-end check on the two conversions: a prior is fitted in z, because that is
    what it predicts, but scored in ray distance, because that is what the reference and the
    renderer hold. If either conversion is wrong the round trip does not close, and the error is
    a smooth field over the image that no eyeball on a depth map would catch.
    """
    manifest, depths, _ply = build_tilted_scene(tmp_path)
    views = json.loads(manifest.read_text())["views"]

    as_z = tmp_path / "pred_z"
    as_z.mkdir()
    ray_length = align_affine.ray_length_grid(K, SIZE, SIZE)
    for view in views:
        reference = np.load(view["depth_path"]).astype(np.float64)
        np.save(as_z / f"{view['name']}.npy", (reference / ray_length).astype(np.float32))

    out = tmp_path / "aligned"
    # fmt: off
    align_affine.main([
        "--pred-dir", str(as_z), "--reference-dir", str(depths),
        "--manifest", str(manifest), "--out-dir", str(out),
        "--quantity", "z", "--emit", "ray",
    ])
    # fmt: on
    for view in views:
        recovered = np.load(out / f"{Path(view['name']).stem}.npy")
        np.testing.assert_allclose(recovered, np.load(view["depth_path"]), rtol=1e-5)


def test_emit_choice_changes_the_score_it_is_read_at(tmp_path: Path) -> None:
    """A tau in z is a looser tolerance in distance, so the two emissions are not interchangeable.

    Scoring a z prediction against a ray-distance reference silently applies a threshold scaled by
    the per-pixel ray length -- 6% at DTU's image corner, 39% at Barn's -- which flatters the
    prediction by an amount that grows with field of view.
    """
    manifest, depths, ply = build_tilted_scene(tmp_path)
    ray_length = align_affine.ray_length_grid(K, SIZE, SIZE)
    tau = 0.03

    # The error has to be one the affine cannot absorb, or the fit removes it and both
    # conventions score a perfect prediction. Random signs of a fixed magnitude survive it,
    # leaving |error| = delta in z and delta * ray_length in distance -- and since ray_length > 1
    # at every pixel, the ray-scored recall can only be the lower of the two. The fit does not
    # come back as exactly identity: noise in the predictor attenuates the least squares scale to
    # var(z) / (var(z) + delta**2), so the comparison is between the two conventions rather than
    # against a known-perfect prediction.
    delta = 0.028
    assert delta < tau < delta * ray_length.max()
    rng = np.random.default_rng(0)
    pred = tmp_path / "pred"
    pred.mkdir()
    for view in json.loads(manifest.read_text())["views"]:
        reference = np.load(view["depth_path"]).astype(np.float64)
        perturbed = reference / ray_length + delta * rng.choice([-1.0, 1.0], size=reference.shape)
        np.save(pred / f"{view['name']}.npy", perturbed.astype(np.float32))

    recalls = {}
    for emit in ("z", "ray"):
        out = tmp_path / f"aligned_{emit}"
        # fmt: off
        align_affine.main([
            "--pred-dir", str(pred), "--reference-dir", str(depths),
            "--manifest", str(manifest), "--out-dir", str(out),
            "--quantity", "z", "--emit", emit,
            "--manifest-out", str(tmp_path / f"m_{emit}.json"),
        ])
        # fmt: on
        assert json.loads((tmp_path / f"m_{emit}.json").read_text())["views"][0]["depth_convention"] == emit
        metrics = tmp_path / f"metrics_{emit}.json"
        # fmt: off
        main([
            "--manifest", str(tmp_path / f"m_{emit}.json"), "--ply", str(ply),
            "--taus", str(tau), "--depth-convention", emit, "-o", str(metrics),
        ])
        # fmt: on
        recalls[emit] = json.loads(metrics.read_text())["recall"][0]

    # Both strictly inside (0, 1), or the tau is in a saturated regime and would not register a
    # difference between the conventions at all.
    assert 0.0 < recalls["ray"] < recalls["z"] < 1.0
    assert (
        recalls["z"] - recalls["ray"] > 0.05
    ), f"the same prediction and tau should score higher in z than in distance, got {recalls}"
