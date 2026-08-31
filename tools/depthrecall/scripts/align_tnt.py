"""Fit the official Tanks and Temples scan-to-render registration from an exported manifest.

The output transforms official scan points into the world frame of a 3dgrut depth export.  It
uses camera-centre correspondences, not ICP, so the registration is determined before seeing a
predicted mesh.  The companion ``--inverse-out`` is useful for mesh evaluation in scan space.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthrecall.io_cameras import read_manifest_views  # noqa: E402
from depthrecall.tnt import gt_to_render_alignment  # noqa: E402


def log_indices_from_manifest(path: Path) -> np.ndarray:
    """Read TnT's zero-based log indices from exporter-preserved image names."""
    entries = json.loads(path.read_text())["views"]
    indices = []
    for entry in entries:
        source = entry.get("source_image")
        match = re.fullmatch(r"(\d+)(?:\.[A-Za-z0-9]+)?", str(source or ""))
        if match is None:
            raise ValueError(
                f"{path}: view {entry.get('name')!r} has no numeric source_image. "
                "Re-export with the current scripts/export_depth_maps.py, or use a manifest ordered "
                "exactly like the official TnT log."
            )
        # TnT image files are 000001.jpg... while its .log starts at zero.
        indices.append(int(match.group(1)) - 1)
    return np.asarray(indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest from scripts/export_depth_maps.py")
    parser.add_argument("--trans", type=Path, required=True, help="<scene>_trans.txt")
    parser.add_argument("--log", type=Path, required=True, help="<scene>_COLMAP_SfM.log")
    parser.add_argument("--out", type=Path, required=True, help="Write scan-to-render 4x4 .npy")
    parser.add_argument("--inverse-out", type=Path, help="Optionally write render-to-scan 4x4 .npy")
    parser.add_argument("--report", type=Path, help="Write registration residual metadata as JSON")
    args = parser.parse_args()

    views = read_manifest_views(args.manifest)
    indices = log_indices_from_manifest(args.manifest)
    alignment = gt_to_render_alignment(
        args.trans, args.log, np.stack([view.cam_center() for view in views]), log_indices=indices
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, alignment.matrix)
    if args.inverse_out:
        args.inverse_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.inverse_out, np.linalg.inv(alignment.matrix))
    report = {
        "scan_to_render": str(args.out),
        "inverse": str(args.inverse_out) if args.inverse_out else None,
        "scale": alignment.scale,
        "residual_median": alignment.residual_median,
        "residual_max": alignment.residual_max,
        "n_correspondences": alignment.n_correspondences,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    warning = alignment.warn_if_coarse(np.array([0.003, 0.005, 0.01, 0.025]))
    if warning:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
