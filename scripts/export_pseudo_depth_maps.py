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

"""Run a monocular pseudo-depth backend over a folder of images, for `tools/depthrecall`.

The counterpart of `export_depth_maps.py` for priors rather than checkpoints: it lets a
monocular prior be scored on the same ladder as a trained method. It drives the same backend
classes training uses, so the prior measured here is the prior consumed there.

A relative prior (DA3MONO, DAv2) is affine to the truth up to a per-frame scale and shift and
scores essentially zero unscored -- run the output through
`tools/depthrecall/scripts/align_affine.py` before reading anything into it.

    PYTHONPATH=/mnt/oss/Depth-Anything-3/src:/mnt/oss/da3deps CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python scripts/export_pseudo_depth_maps.py \
        --images .../scan24/images --out-dir /tmp/da3 --backend depth_anything_3 \
        --model depth-anything/DA3MONO-LARGE
"""

import argparse
import os

import numpy as np
from PIL import Image

from threedgrut.datasets.pseudo_depth import BACKENDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Directory of source images")
    parser.add_argument("--out-dir", required=True, help="Where to write one .npy per image")
    parser.add_argument("--backend", default="depth_anything_3", choices=sorted(BACKENDS))
    parser.add_argument("--model", default="depth-anything/DA3MONO-LARGE")
    parser.add_argument(
        "--process-res",
        type=int,
        default=None,
        help=(
            "Longest side the backend preprocesses to. None keeps its default, which is what "
            "training gets. Raising it is not free: on DTU, DA3 at native 1554px is a *worse* "
            "prior than at its default 504px"
        ),
    )
    args = parser.parse_args()

    predictor_kwargs = {"model_id": args.model}
    if args.process_res is not None:
        predictor_kwargs["process_res"] = args.process_res
    predictor = BACKENDS[args.backend](**predictor_kwargs)
    print(f"backend={args.backend} model={args.model} quantity={BACKENDS[args.backend].QUANTITY}")

    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith((".png", ".jpg", ".jpeg")))
    if not names:
        raise FileNotFoundError(f"No images in {args.images}")

    for index, name in enumerate(names):
        image = np.asarray(Image.open(os.path.join(args.images, name)).convert("RGB"))
        depth = predictor.predict(image)
        height, width = image.shape[:2]
        if depth.shape != (height, width):
            # The intrinsics describe the source pixel grid, so the prediction has to be back on
            # it before it can be projected against anything. Bilinear cannot invent detail, so
            # the backend's own working resolution stays a confound to report, not one to hide.
            depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.BILINEAR), dtype=np.float32)
        np.save(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".npy"), depth.astype(np.float32))
        if index % 10 == 0:
            print(f"{index + 1}/{len(names)} {name}", flush=True)

    print(f"Wrote {len(names)} depth maps to {args.out_dir}")


if __name__ == "__main__":
    main()
