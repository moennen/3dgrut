"""Evaluate a predicted mesh/point cloud with standard DTU and Tanks and Temples metrics.

DTU: pass the official ObsMask for predicted-surface accuracy and Plane file for reference-side
completeness. TnT: pass its SelectionPolygonVolume as --crop-json, which crops both surfaces.
Meshes are sampled uniformly by triangle area; point clouds use their vertices directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthrecall.dtu import above_ground_plane_mask, observed_volume_mask, read_ground_plane  # noqa: E402
from depthrecall.io_points import apply_alignment, load_alignment, read_ply  # noqa: E402
from depthrecall.surface import evaluate_surface  # noqa: E402
from depthrecall.tnt import crop_volume_mask  # noqa: E402


def read_surface(path: Path, samples: int) -> np.ndarray:
    """Uniformly sample a triangle mesh, or use a point cloud unchanged."""
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles):
            return np.asarray(mesh.sample_points_uniformly(number_of_points=samples).points, dtype=np.float64)
    except ImportError:
        pass
    return read_ply(path).xyz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--taus", required=True, help="Comma-separated thresholds in the evaluated frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--pred-alignment", type=Path)
    parser.add_argument("--gt-alignment", type=Path)
    parser.add_argument("--dtu-obsmask", type=Path)
    parser.add_argument("--dtu-plane", type=Path)
    parser.add_argument("--crop-json", type=Path)
    args = parser.parse_args()
    taus = np.asarray([float(value) for value in args.taus.split(",")], dtype=np.float64)
    predicted, reference = read_surface(args.pred, args.samples), read_surface(args.gt, args.samples)
    masks = []
    # Benchmark masks live in scan coordinates, so apply them before optional transforms into
    # the common evaluation frame.
    if args.dtu_obsmask:
        mask = observed_volume_mask(predicted, args.dtu_obsmask)
        predicted, masks = predicted[mask], ["dtu observation mask"]
    if args.dtu_plane:
        mask = above_ground_plane_mask(reference, read_ground_plane(args.dtu_plane))
        reference, masks = reference[mask], masks + ["dtu ground plane"]
    if args.crop_json:
        predicted, reference = (
            predicted[crop_volume_mask(predicted, args.crop_json)],
            reference[crop_volume_mask(reference, args.crop_json)],
        )
        masks.append("tnt crop volume (both surfaces)")
    if args.pred_alignment:
        predicted = apply_alignment(predicted, load_alignment(args.pred_alignment))
    if args.gt_alignment:
        reference = apply_alignment(reference, load_alignment(args.gt_alignment))
    result = evaluate_surface(predicted, reference, taus).asdict(taus)
    result.update({"pred_samples": len(predicted), "gt_samples": len(reference), "masks": masks})
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
