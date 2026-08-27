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

"""Render every report variant for a scene, one subprocess per variant.

Two things this has to get right, both of which would otherwise produce a figure that looks
fine and compares nothing:

* **One process per variant.** `enable_normals`, `primitive_type` and `enable_depth_variance`
  are compiled into `lib3dgut_cc` and a process holds exactly one binary, so variants that
  differ in those cannot share a process.
* **One colour range per scene.** The range is taken from the *baseline* run and passed to
  every other variant, so a depth panel that looks different is different. Fitting each panel
  to its own range makes any two variants look unlike each other regardless of accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Variant name -> label used in the deck. The first entry must be the baseline: it establishes
# the shared depth colour range for the scene.
#
# `pd01_gaussian_nogate` is the ungated ordinal term. The first sweep named the *gated* variant
# `pd01_gaussian` and only later made ungated the default, so the run directories carry the old
# names; mapping here rather than renaming directories keeps the figures traceable to the runs
# that produced the numbers.
VARIANTS = (
    ("gaussian", "baseline"),
    ("pd01_gaussian_nogate", "pd01"),
    ("dn05_gaussian", "dn05"),
    ("pd01_gaussian_dn", "pd01_dn"),
    ("dvrel001_gaussian", "dvrel001"),
    ("pd01_dvrel001_gaussian", "pd01_dvrel001"),
)


def find_checkpoints(search_root: Path, scene: str) -> dict[str, Path]:
    """Newest `ckpt_7000.pt` per variant for `scene`, keyed by variant name."""
    found: dict[str, Path] = {}
    for run_dir in sorted(search_root.glob("*/abl_*")):
        match = re.match(rf"abl_(.+?)_{re.escape(scene)}$", run_dir.name)
        if not match:
            continue
        checkpoints = sorted(run_dir.glob("*/ours_7000/ckpt_7000.pt"))
        if checkpoints:
            found[match.group(1)] = checkpoints[-1]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--search-root", default="/tmp/abl_pd")
    parser.add_argument("--dataset-root", default="/mnt/data/nerf_datasets/ob3d/OB3D_colmap")
    parser.add_argument("--out-root", default="/tmp/report_fig")
    parser.add_argument("--frames", type=int, nargs="+", default=[4])
    args = parser.parse_args()

    checkpoints = find_checkpoints(Path(args.search_root), args.scene)
    missing = [name for name, _ in VARIANTS if name not in checkpoints]
    if missing:
        raise SystemExit(f"no checkpoint for {missing} in {args.search_root} (scene {args.scene})")

    out_dir = Path(args.out_root) / args.scene
    depth_range: list[str] = []
    for name, tag in VARIANTS:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/report/render_views.py"),
            "--checkpoint",
            str(checkpoints[name]),
            "--out-dir",
            str(out_dir),
            "--tag",
            tag,
            "--scene-path",
            f"{args.dataset_root}/{args.scene}",
            "--frames",
            *[str(f) for f in args.frames],
        ]
        print(f"[{args.scene}] {tag:14s} <- {checkpoints[name].parent.parent.parent.name}", flush=True)
        subprocess.run(command + depth_range, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)

        if not depth_range:
            summary = json.loads((out_dir / f"{tag}_frames.json").read_text())
            lo, hi = summary[str(args.frames[0])]["depth_range"]
            depth_range = ["--depth-range", str(lo), str(hi)]
            print(f"[{args.scene}] shared depth range {lo:.3f} .. {hi:.3f}", flush=True)


if __name__ == "__main__":
    main()
