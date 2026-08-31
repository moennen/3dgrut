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

"""Command-line entry point for depthrecall."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .dtu import above_ground_plane_mask, read_dtu_views, read_ground_plane, scan_to_normalized
from .io_cameras import read_colmap_views, read_manifest_views, write_manifest_views
from .io_points import load_alignment, read_ply
from .metric import MetricConfig, evaluate
from .tnt import crop_volume_mask


def _comma_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recall of ground-truth scan points against rendered depth maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  depthrecall --colmap sparse/0 --depths out/depths --ply gt.ply \\
              --taus 0.5,1,2,5,10 --depth-convention ray -o metrics.json

  depthrecall --dtu-cameras scan24/cameras.npz --depths out/depths \\
              --ply /data/dtu_eval/Points/stl/stl024_total.ply --dtu-space scan \\
              --taus 0.5,1,2,5,10 --depth-convention ray -o scan24.json
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--colmap", type=Path, help="Path to COLMAP sparse/0 directory")
    group.add_argument("--manifest", type=Path, help="Path to camera/depth manifest JSON")
    group.add_argument(
        "--dtu-cameras",
        type=Path,
        help="DTU cameras.npz; also supplies the scan-to-normalized alignment unless --dtu-space scan",
    )

    parser.add_argument(
        "--depths",
        type=Path,
        default=None,
        help="Directory containing depth maps; required unless --manifest, which carries its own paths",
    )
    parser.add_argument(
        "--depth-ext", type=str, default=None, help="Depth file extension (e.g. .npy); auto-detected if omitted"
    )
    parser.add_argument("--ply", type=Path, required=True, help="Ground-truth point cloud PLY file")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output metrics JSON")

    parser.add_argument(
        "--taus",
        type=_comma_floats,
        required=True,
        help="Comma-separated list of absolute distance thresholds (same units as depth)",
    )
    parser.add_argument(
        "--tau-units",
        choices=["depth", "gt"],
        default="depth",
        help=(
            "Units --taus are given in. 'depth' (default) is the units of the depth maps. 'gt' "
            "is the ground truth's own units -- DTU millimetres, TnT metres -- converted using "
            "the alignment's scale, which is the only way to quote a benchmark's published "
            "threshold when the render frame is scaled"
        ),
    )
    parser.add_argument(
        "--depth-convention",
        choices=["ray", "z"],
        required=True,
        help="How depth maps encode distance: Euclidean ray distance or camera z-depth",
    )
    parser.add_argument(
        "--sampling",
        choices=["nearest", "bilinear"],
        default="nearest",
        help="How to read the depth map at a projected point (default: nearest)",
    )
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale factor applied to raw depth values")
    parser.add_argument("--max-valid-depth", type=float, default=None, help="Treat depth values above this as missing")
    parser.add_argument("--invalid-value", type=float, default=None, help="Treat this depth value as missing")
    parser.add_argument(
        "--visibility-manifest",
        type=Path,
        default=None,
        help=(
            "Manifest of GT z-buffer depth maps used to exclude self-occluded scan points. "
            "Required for a meaningful TnT recall; create it with scripts/rasterize_depth.py."
        ),
    )
    parser.add_argument(
        "--visibility-tolerance",
        type=float,
        default=1e-3,
        help="Relative GT-depth agreement required by --visibility-manifest",
    )

    parser.add_argument(
        "--alignment", type=Path, default=None, help="4x4 alignment matrix (.txt/.npy) transforming GT to world"
    )
    parser.add_argument(
        "--crop-json",
        type=Path,
        default=None,
        help=(
            "Tanks and Temples SelectionPolygonVolume (<scene>.json). Crops the GT to the "
            "officially scored region, in scan coordinates, before --alignment"
        ),
    )
    parser.add_argument(
        "--dtu-plane",
        type=Path,
        default=None,
        help=(
            "DTU Plane<scan>.mat (or 4 coefficients as .txt/.npy). Culls the GT below the "
            "ground plane, as the official completeness measure does. Required with "
            "--dtu-cameras unless --no-gt-mask is given"
        ),
    )
    parser.add_argument(
        "--no-gt-mask",
        action="store_true",
        help=(
            "Score the full ground-truth cloud, skipping the benchmark's own GT-side cull. "
            "Must be stated explicitly, because whether the cull is applied moves the number "
            "by several points and cannot be recovered from the output otherwise"
        ),
    )
    parser.add_argument("--invert-alignment", action="store_true", help="Invert the provided alignment matrix")
    parser.add_argument(
        "--dtu-space",
        choices=["normalized", "scan"],
        default="normalized",
        help=(
            "World frame of the DTU depth maps: 'normalized' (the unit-sphere frame models "
            "train in; the scan is transformed into it and taus are in normalized units) or "
            "'scan' (DTU millimetres, so taus are in mm)"
        ),
    )
    parser.add_argument(
        "--dtu-image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Resolution the DTU intrinsics were authored for, if the depth maps are downscaled",
    )

    parser.add_argument(
        "--downsample-voxel",
        type=float,
        default=None,
        help="Voxel size for downsampling GT points before evaluation",
    )

    parser.add_argument(
        "--view-stride",
        type=int,
        default=1,
        help="Evaluate every Nth view (with --view-offset, selects a held-out subset)",
    )
    parser.add_argument("--view-offset", type=int, default=0, help="Index offset for --view-stride")

    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Write resolved camera/depth manifest to this path for reuse",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.taus != sorted(set(args.taus)):
        raise ValueError("--taus must be unique and sorted")
    taus = np.asarray(sorted(set(args.taus)), dtype=np.float64)

    # A manifest names a depth file per view, so a --depths alongside it is not merely redundant:
    # it would be ignored, and a caller pointing it at another folder would score the manifest's
    # depths while believing they had scored that folder's.
    if args.manifest and args.depths is not None:
        raise ValueError(
            f"--manifest carries a depth path per view, so --depths {args.depths} would be ignored. "
            "Drop it, or point --manifest at a manifest describing the depths you mean."
        )
    if not args.manifest and args.depths is None:
        raise ValueError("--depths is required with --colmap and --dtu-cameras")

    dtu_alignment = None
    if args.colmap:
        views = read_colmap_views(args.colmap, args.depths, extension=args.depth_ext)
    elif args.dtu_cameras:
        views = read_dtu_views(
            args.dtu_cameras,
            args.depths,
            space=args.dtu_space,
            extension=args.depth_ext,
            image_size=tuple(args.dtu_image_size) if args.dtu_image_size else None,
        )
        # In the normalized frame the scan has to be brought along; in the scan frame it is
        # already there. Deriving this here rather than asking the user for a matrix is what
        # keeps the direction from being guessed: the transform is ~325x, so the wrong way
        # round gives a recall of zero with no other symptom.
        if args.dtu_space == "normalized":
            dtu_alignment = scan_to_normalized(args.dtu_cameras)
    else:
        views = read_manifest_views(args.manifest)

    # Sort by name so the view order -- and therefore any stride subset -- does not depend on
    # the order COLMAP happened to store its images in.
    views = sorted(views, key=lambda v: v.name)
    if args.view_stride > 1 or args.view_offset:
        views = views[args.view_offset :: args.view_stride]
        if not views:
            raise ValueError("--view-stride/--view-offset selected no views")

    if args.write_manifest:
        write_manifest_views(views, args.write_manifest)

    visibility_depths = None
    if args.visibility_manifest is not None:
        visibility_views = {view.name: view for view in read_manifest_views(args.visibility_manifest)}
        visibility_depths = {}
        for view in views:
            visibility = visibility_views.get(view.name)
            if visibility is None:
                raise ValueError(f"--visibility-manifest has no view named {view.name}")
            if (
                visibility.width != view.width
                or visibility.height != view.height
                or not np.allclose(visibility.K, view.K)
            ):
                raise ValueError(f"Visibility camera for {view.name} does not match the scored camera")
            if not np.allclose(visibility.R, view.R) or not np.allclose(visibility.t, view.t):
                raise ValueError(f"Visibility pose for {view.name} does not match the scored camera")
            visibility_depths[view.name] = visibility.depth_path

    if args.ply.suffix == ".npy":
        gt_points = np.load(args.ply).astype(np.float64)
        if gt_points.ndim != 2 or gt_points.shape[1] != 3:
            raise ValueError(f"--ply .npy must hold an (N, 3) array, got {gt_points.shape}")
    else:
        gt_points = read_ply(args.ply).xyz

    # GT-side masks, all in the scan's own coordinates and so applied before the alignment.
    # These are the culls the official *completeness* measures apply to the ground truth, which
    # is the direction this recall corresponds to. Reconstruction-side masks -- DTU's ObsMask --
    # have no analogue: they exist to avoid judging a reconstruction where the scanner never
    # observed, and every point iterated over here is by construction an observed scan point.
    gt_masks: list[tuple[str, np.ndarray]] = []
    if args.crop_json is not None:
        gt_masks.append(("crop volume", crop_volume_mask(gt_points, args.crop_json)))
    if args.dtu_plane is not None:
        plane = read_ground_plane(args.dtu_plane)
        gt_masks.append(("ground plane", above_ground_plane_mask(gt_points, plane)))

    if args.no_gt_mask:
        if gt_masks:
            raise ValueError("--no-gt-mask conflicts with the GT mask given by --crop-json/--dtu-plane")
        print("GT mask: none (--no-gt-mask); scoring the full cloud, not the officially scored region")
    elif not gt_masks:
        # Only DTU is detectable from the arguments, so only DTU can be held to this. TnT has
        # no marker to key off; its crop is documented instead.
        if args.dtu_cameras:
            raise ValueError(
                "DTU's official completeness culls the GT below the ground plane, which drops "
                "38.5% of scan24's reference cloud and moves recall@1mm from 0.639 to 0.679. "
                "Pass --dtu-plane <dtu_eval/ObsMask/Plane{scan}.mat>, or --no-gt-mask to score "
                "the full cloud deliberately."
            )
    applied_masks: list[str] = []
    for name, mask in gt_masks:
        print(f"{name} keeps {int(mask.sum())} of {len(gt_points)} GT points ({mask.mean():.1%})")
        applied_masks.append(f"{name} ({mask.mean():.4f} kept)")
        gt_points = gt_points[mask]

    alignment = dtu_alignment
    if args.alignment is not None:
        if alignment is not None:
            raise ValueError("--alignment conflicts with the alignment implied by --dtu-cameras")
        alignment = load_alignment(args.alignment, invert=args.invert_alignment)

    if args.tau_units == "gt":
        if alignment is None:
            raise ValueError("--tau-units gt needs an alignment to convert with (--alignment or --dtu-cameras)")
        # The alignment maps GT into the depth maps' frame, so its scale is exactly the factor
        # between a distance there and one here. Requiring a uniform scale is not a limitation:
        # a non-uniform one would make "a threshold in GT units" direction-dependent and so
        # meaningless, which is worth refusing rather than approximating with a mean.
        row_norms = np.linalg.norm(alignment[:3, :3], axis=1)
        if row_norms.max() - row_norms.min() > 1e-6 * row_norms.max():
            raise ValueError(
                f"--tau-units gt needs a uniform scale, but the alignment's row norms are {row_norms}. "
                "Convert the thresholds yourself and pass --tau-units depth."
            )
        gt_to_depth_scale = float(row_norms.mean())
        taus = taus * gt_to_depth_scale
        print(f"--tau-units gt: scaling thresholds by {gt_to_depth_scale:.6g} -> {np.array2string(taus, precision=6)}")

    config = MetricConfig(
        taus=taus,
        depth_convention=args.depth_convention,
        max_valid_depth=args.max_valid_depth,
        invalid_value=args.invalid_value,
        depth_scale=args.depth_scale,
        sampling=args.sampling,
        visibility_depths=visibility_depths,
        visibility_tolerance=args.visibility_tolerance,
    )

    result = evaluate(
        views=views,
        gt_points=gt_points,
        config=config,
        alignment=alignment,
        downsample_voxel=args.downsample_voxel,
        gt_masks=applied_masks,
    )

    result.save_json(args.out)
    print(f"Wrote metrics to {args.out}")

    data = result.asdict()
    in_frustum = data["in_frustum_pairs"]
    no_surface = data["no_surface_pairs"]
    with_surface = in_frustum - no_surface
    print(f"points={data['gt_points']}  views={data['views_count']}")
    print(
        f"in-frustum pairs={in_frustum}  with surface={with_surface} "
        f"({with_surface / max(in_frustum, 1):.1%})  no surface={no_surface} "
        f"({no_surface / max(in_frustum, 1):.1%}, counted as failures)"
    )
    # recall is a share of *all* in-frustum pairs while the signed diagnostics are shares of
    # only those pairs that had a surface to compare against, so the three do not sum to one.
    # Printing them on one line without saying so reads as if they did, which understates how
    # much of a sparse result is simply missing geometry.
    print("  tau: recall (of in-frustum pairs) | too_near, too_far (of pairs with a surface)")
    for tau, rec, near, far in zip(data["taus"], data["recall"], data["too_near"], data["too_far"]):
        print(f"  {tau:g}: {rec:.6f} | {near:.4f}, {far:.4f}")


if __name__ == "__main__":
    main()
