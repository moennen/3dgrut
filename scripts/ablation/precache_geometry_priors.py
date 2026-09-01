#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precompute the MoGe-3 and C-RADIOv4 target caches used by the 30k ablation.

Run this once per dataset preset before launching parallel workers with ``--cache-root``.  The
cache is namespaced by suite and scene because benchmark image names repeat between scenes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

from evaluate_depth_models import selected_scenes  # noqa: E402

from threedgrut.datasets.dataset_colmap import ColmapDataset


def scene_root(args, suite: str, scene: str) -> Path:
    if suite == "ob3d":
        return args.ob3d_root / scene
    if suite == "dtu":
        return args.dtu_root / scene
    return args.tnt_reconstruction_root / "TrainingSet" / scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-scale", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--ob3d-root", type=Path, required=True)
    parser.add_argument("--dtu-root", type=Path, required=True)
    parser.add_argument("--tnt-reconstruction-root", type=Path, required=True)
    parser.add_argument("--ob3d-scenes", default=None)
    parser.add_argument("--dtu-scenes", default=None)
    parser.add_argument("--tnt-scenes", default=None)
    parser.add_argument("--suites", default="ob3d,dtu,tnt")
    parser.add_argument("--moge3-model", required=True)
    parser.add_argument("--radio-model", default="c-radio_v4-h")
    parser.add_argument("--feature-dim", type=int, default=48)
    parser.add_argument("--feature-stride", type=int, default=16)
    parser.add_argument("--fit-samples", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = selected_scenes(
        args.dataset_scale, {"ob3d": args.ob3d_scenes, "dtu": args.dtu_scenes, "tnt": args.tnt_scenes}
    )
    suites = tuple(value.strip() for value in args.suites.split(",") if value.strip())
    unknown = set(suites) - set(selected)
    if unknown:
        raise SystemExit(f"Unknown suites: {sorted(unknown)}")
    for suite in suites:
        for scene in selected[suite]:
            root = scene_root(args, suite, scene)
            cache = args.cache_root / suite / scene
            print(f"[{suite}/{scene}] -> {cache}", flush=True)
            # Instantiate the same train split and cache configurations as the ablation. In
            # particular PCA is fitted *only* on train views; fitting it on every image would
            # let the frozen feature target observe held-out NVS views before evaluation.
            ColmapDataset(
                str(root),
                split="train",
                pseudo_depth={
                    "enabled": True,
                    "backend": "moge3",
                    "model": args.moge3_model,
                    "cache_dir": str(cache / "pseudo_depth"),
                    "align_to_sparse_points": True,
                },
                image_features={
                    "enabled": True,
                    "backend": "nvradio4",
                    "model": args.radio_model,
                    "output_dim": args.feature_dim,
                    "feature_stride": args.feature_stride,
                    "cache_dir": str(cache / "image_features"),
                    "projector": "pca",
                    "fit_samples": args.fit_samples,
                    "seed": args.seed,
                },
            )


if __name__ == "__main__":
    main()
