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

"""Fit a per-frame affine from predicted depth to a reference depth, and write the result.

A relative monocular prior is defined only up to a per-frame scale and shift, so it cannot be
scored directly -- unaligned it reads as ~0 recall regardless of how good its geometry is. This
puts it in the reference's units so the recall measures shape rather than units.

The reference is a depth folder in the render frame, normally produced by `rasterize_depth.py`:

- the ground-truth scan, giving an *oracle* fit that uses information a method would not have;
- the sparse SfM points, giving what could actually be estimated without ground truth.

Report both. The gap between them is the cost of not knowing the alignment, and the oracle is
not even guaranteed to be an upper bound: it is least squares over all pixels, while a recall
threshold weights near geometry differently.

Fitting and scoring want *different* quantities, so they are separate options:

- ``--quantity`` is what the prediction holds and what the affine is fitted in. DA3 emits z, and
  z and ray distance differ by a per-pixel factor, so a scale and shift in one is not a scale and
  shift in the other; fitting in the wrong one degrades the fit instead of failing.
- ``--emit`` is what gets written, and defaults to ``ray``. Recall at a fixed tau is **not the
  same metric** in the two conventions: under ``--depth-convention z`` the tool compares the
  point's camera-frame z, so a tau in z is a tolerance of ``tau * ||[x', y', 1]||`` in true
  distance -- up to 6% looser at DTU's image corner and 39% at Barn's. Scoring a prediction in z
  against a reference measured in ray therefore compares two different thresholds, and flatters
  the prediction by a factor that grows with field of view.

The output folder is a drop-in replacement for the reference folder: files are named exactly as
the reference names them, so whatever already reads the reference reads this. That matters
because the two conventions in play differ -- ``read_colmap_views`` keys depth files by image
*stem* (``000001.npy``) while a manifest written by ``rasterize_depth.py`` keys them by view
*name* (``000001.jpg.npy``) -- and a prediction folder produced elsewhere may use either.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def ray_length_grid(K: np.ndarray, height: int, width: int, pixel_center_offset: float = 0.5) -> np.ndarray:
    """Per-pixel ``||[x', y', 1]||``, the factor between z-depth and ray distance.

    Array index ``i`` is the pixel whose continuous coordinate is ``i + pixel_center_offset``,
    matching `MetricConfig.pixel_center_offset` and COLMAP's corner-based principal point.
    Evaluating on the integer grid instead offsets the factor by half a pixel of depth gradient,
    which is invisible on a flat surface and largest exactly where the geometry is interesting.
    """
    u = np.arange(width) + pixel_center_offset
    v = np.arange(height) + pixel_center_offset
    x = (u[None, :] - K[0, 2]) / K[0, 0]
    y = (v[:, None] - K[1, 2]) / K[1, 1]
    return np.sqrt(1.0 + x**2 + y**2)


def resolve_depth(directory: Path, name: str, extension: str = ".npy") -> Path:
    """Find a view's depth file, accepting either the view name or the image stem as the key."""
    candidates = [directory / f"{name}{extension}", directory / f"{Path(name).stem}{extension}"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No depth for view {name!r}: tried {', '.join(str(c) for c in candidates)}")


def fit_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least squares ``a * prediction + b ~= target``."""
    A = np.stack([prediction, np.ones_like(prediction)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, target, rcond=None)
    return float(a), float(b)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Raw predictions, one .npy per view")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference depths (ray distance)")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest carrying K per view")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--quantity",
        choices=["z", "ray"],
        default="z",
        help=(
            "What the prediction holds, and so what the affine is fitted in. z-depth and ray "
            "distance differ by a per-pixel factor, not an affine, so fitting in the wrong one "
            "degrades the fit silently instead of failing"
        ),
    )
    parser.add_argument(
        "--emit",
        choices=["z", "ray"],
        default="ray",
        help=(
            "Quantity to write. Defaults to ray distance, the convention a rasterized reference "
            "and a renderer both use: a tau in z is a looser tolerance in true distance (39%% at "
            "Barn's corner), so scoring a z prediction against a ray reference flatters it"
        ),
    )
    parser.add_argument("--report", type=Path, default=None, help="Write per-frame fit statistics as JSON")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write a copy of the manifest pointing at the aligned depths, for the manifest-driven path",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    views = manifest["views"]

    stats = []
    pixels = 0
    for view in views:
        name = view["name"]
        prediction = np.load(resolve_depth(args.pred_dir, name)).astype(np.float64)
        reference_path = resolve_depth(args.reference_dir, name)
        reference = np.load(reference_path).astype(np.float64)
        if prediction.shape != reference.shape:
            raise ValueError(f"{name}: prediction is {prediction.shape} but the reference is {reference.shape}")
        K = np.asarray(view["K"], dtype=np.float64)

        # A rasterized reference leaves 0 where no point landed. Those pixels carry no
        # constraint, and with a sparse reference they are almost all of the image.
        valid = reference > 0
        if not valid.any():
            raise ValueError(f"{name}: the reference has no valid pixels to fit against")
        ray_length = ray_length_grid(K, *reference.shape)
        target = reference if args.quantity == "ray" else reference / ray_length

        a, b = fit_affine(prediction[valid], target[valid])
        aligned = a * prediction + b
        # Convert only when the fit and the output disagree; the fit runs in the quantity the
        # prediction is affine in, the output in the quantity the score is read in.
        if args.emit == args.quantity:
            emitted = aligned
        elif args.emit == "ray":
            emitted = aligned * ray_length
        else:
            emitted = aligned / ray_length
        # Named as the reference names it, so the output folder substitutes for the reference.
        np.save(args.out_dir / reference_path.name, emitted.astype(np.float32))
        stats.append(
            {
                "name": name,
                "scale": a,
                "shift": b,
                "fit_pixels": int(valid.sum()),
                "median_residual": float(np.median(np.abs(aligned[valid] - target[valid]))),
            }
        )
        pixels = reference.size

    residuals = [s["median_residual"] for s in stats]
    scales = [s["scale"] for s in stats]
    coverage = np.mean([s["fit_pixels"] for s in stats]) / pixels
    print(
        f"{len(stats)} frames, fit in {args.quantity}, emitted {args.emit}, fit coverage {coverage:.1%}, "
        f"median residual {np.median(residuals):.5f} (max {np.max(residuals):.5f}), "
        f"per-frame scale {np.min(scales):.4f}..{np.max(scales):.4f}"
    )
    if args.report:
        args.report.write_text(json.dumps(stats, indent=2))
    if args.manifest_out:
        for entry in views:
            entry["depth_path"] = str(resolve_depth(args.out_dir, entry["name"]).resolve())
            # The manifest is what a reader trusts for the convention, so it records what was
            # emitted rather than what was fitted.
            entry["depth_convention"] = args.emit
        args.manifest_out.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
