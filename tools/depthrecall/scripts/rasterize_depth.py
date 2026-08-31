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

"""Z-buffer a point cloud into depth maps, to calibrate the recall metric.

This is not a renderer and is not meant to score anything. It exists to provide the two rungs
that tell you what a recall number means before any reconstruction is involved:

- **Ceiling.** Rasterize the ground-truth scan itself and score the scan against it. Recall
  should be near 1, and whatever it falls short by is the metric's own floor on this scene:
  self-occlusion (a point whose pixel is won by a nearer point) plus the discretization of a
  point cloud into pixels. Reporting a method's recall without knowing this number leaves the
  reader unable to tell a reconstruction error from a metric artefact.
- **Floor.** Rasterize the sparse COLMAP points. Almost every pixel is empty, so recall should
  be near zero. If it is not, something is wrong with the pairing rather than impressive.

Points are splatted to the single pixel containing them, keeping the nearest, so a surface
sampled more sparsely than the pixel grid leaves holes -- which is honest for a ceiling rung:
those holes are exactly the "no surface" failures the metric counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthrecall.dtu import read_dtu_views, scan_to_normalized  # noqa: E402
from depthrecall.io_cameras import View, read_colmap_views, read_manifest_views, write_manifest_views  # noqa: E402
from depthrecall.io_points import apply_alignment, load_alignment, read_ply  # noqa: E402
from depthrecall.tnt import crop_volume_mask  # noqa: E402


def rasterize(
    view: View,
    points: np.ndarray,
    width: int,
    height: int,
    convention: str = "ray",
    dilate: int = 0,
) -> np.ndarray:
    """Return an (H, W) depth map of `points` seen from `view`; 0 where no point landed."""
    cam = points @ view.R.T + view.t
    z = cam[:, 2]
    front = z > 0
    cam, z = cam[front], z[front]

    u = view.K[0, 0] * cam[:, 0] / z + view.K[0, 2] - 0.5
    v = view.K[1, 1] * cam[:, 1] / z + view.K[1, 2] - 0.5
    ui = np.floor(u + 0.5).astype(np.int64)
    vi = np.floor(v + 0.5).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi = ui[inside], vi[inside]
    depth_values = np.linalg.norm(cam[inside], axis=1) if convention == "ray" else z[inside]

    buffer = np.full(height * width, np.inf)
    flat = vi * width + ui
    # np.minimum.at is the scatter-min that keeps the nearest point per pixel; a plain
    # assignment would keep whichever point happened to come last in the file.
    np.minimum.at(buffer, flat, depth_values)

    if dilate > 0:
        # Fill single-pixel gaps between sparsely sampled points, for a cloud coarser than the
        # pixel grid. Off by default: it invents surface where none was measured.
        for _ in range(dilate):
            grid = buffer.reshape(height, width)
            padded = np.full((height + 2, width + 2), np.inf)
            padded[1:-1, 1:-1] = grid
            stacked = np.stack(
                [padded[a : a + height, b : b + width] for a in range(3) for b in range(3)],
                axis=0,
            )
            buffer = np.where(np.isfinite(grid), grid, np.min(stacked, axis=0)).ravel()

    buffer[~np.isfinite(buffer)] = 0.0
    return buffer.reshape(height, width).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dtu-cameras", type=Path, help="DTU cameras.npz")
    group.add_argument("--colmap", type=Path, help="COLMAP sparse/0 directory")
    group.add_argument("--manifest", type=Path, help="Existing manifest JSON to take cameras from")

    parser.add_argument("--ply", type=Path, required=True, help="Point cloud to rasterize")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=None, help="Output width (required for --dtu-cameras)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--convention", choices=["ray", "z"], default="ray")
    parser.add_argument(
        "--space",
        choices=["normalized", "scan"],
        default="normalized",
        help="DTU frame to place cameras and points in (default: normalized, as models render)",
    )
    parser.add_argument(
        "--ply-frame",
        choices=["scan", "world"],
        default="scan",
        help=(
            "Frame the PLY is in: 'scan' for a DTU reference scan in millimetres (converted to "
            "--space), or 'world' for a cloud already in the camera frame, such as the "
            "normalized sparse COLMAP points shipped with each scan"
        ),
    )
    parser.add_argument("--dilate", type=int, default=0, help="Fill gaps by N rounds of 3x3 min-dilation")
    parser.add_argument("--limit-views", type=int, default=None, help="Only rasterize the first N views")
    parser.add_argument(
        "--alignment",
        type=Path,
        default=None,
        help=(
            "4x4 transform (.npy or .txt) placing the PLY in the cameras' frame. The general "
            "form of --ply-frame, for datasets whose alignment is not derivable from the "
            "cameras, e.g. Tanks and Temples (see depthrecall.tnt)"
        ),
    )
    parser.add_argument(
        "--crop-json",
        type=Path,
        default=None,
        help="Tanks and Temples SelectionPolygonVolume; crops the PLY *before* --alignment",
    )
    args = parser.parse_args(argv)

    depth_dir = args.out_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    points = read_ply(args.ply).xyz
    if args.crop_json:
        # Cropping is defined in the scan's own coordinates, so it happens before --alignment.
        mask = crop_volume_mask(points, args.crop_json)
        print(f"crop volume keeps {int(mask.sum())} of {len(points)} points ({mask.mean():.1%})")
        points = points[mask]
    if args.alignment:
        points = apply_alignment(points, load_alignment(args.alignment))
    if args.dtu_cameras:
        if args.width is None or args.height is None:
            raise ValueError("--width and --height are required with --dtu-cameras")
        # The depth maps do not exist yet, so take the cameras alone.
        views = read_dtu_views(
            args.dtu_cameras,
            None,
            space=args.space,
            extension=".npy",
            image_size=(args.width, args.height),
        )
        if args.space == "normalized" and args.ply_frame == "scan":
            points = apply_alignment(points, scan_to_normalized(args.dtu_cameras))
        elif args.space == "scan" and args.ply_frame == "world":
            points = apply_alignment(points, np.linalg.inv(scan_to_normalized(args.dtu_cameras)))
        width, height = args.width, args.height
    elif args.colmap:
        # The depth maps do not exist yet, so take the cameras alone.
        views = read_colmap_views(args.colmap, None, extension=".npy")
        width = args.width or views[0].width
        height = args.height or views[0].height
    else:
        views = read_manifest_views(args.manifest)
        width = args.width or views[0].width
        height = args.height or views[0].height

    if args.limit_views:
        views = views[: args.limit_views]

    written: list[View] = []
    for view in views:
        depth = rasterize(view, points, width, height, convention=args.convention, dilate=args.dilate)
        path = depth_dir / f"{view.name}.npy"
        np.save(path, depth)
        written.append(
            View(
                name=view.name,
                width=width,
                height=height,
                K=view.K,
                R=view.R,
                t=view.t,
                depth_path=Path("depths") / f"{view.name}.npy",
                depth_convention=args.convention,
            )
        )
        filled = float((depth > 0).mean())
        print(f"{view.name}: {filled * 100:.2f}% of pixels have a point")

    write_manifest_views(written, args.out_dir / "manifest.json")
    print(f"Wrote {len(written)} depth maps and a manifest to {args.out_dir}")


if __name__ == "__main__":
    main()
