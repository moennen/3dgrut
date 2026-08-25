# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-compile the native kernel variants a set of configurations needs.

Kernels are selected by compile-time defines, so every configuration that flips one
(normals on/off, surfel vs ellipsoid, feature dimensions, ...) builds its own binary. Two
consequences motivate this tool:

  * compilation time would otherwise land inside the first training step and pollute any
    wall-clock measurement;
  * 3dgrt and the playground still share one generated Slang header, so their variants
    must not be built concurrently.

Building every variant up front, sequentially, removes both problems: the timed runs then
only hit warm caches, and a parallel sweep never compiles at all.

Example:

    python -m threedgrut.utils.build_variants \\
        --config-name apps/colmap_3dgut.yaml \\
        --overrides render.enable_normals=false \\
        --overrides render.enable_normals=true
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from threedgrut.utils.logger import logger

# Resolvers are registered by the training entry points; a bare compose() needs them too.
if not OmegaConf.has_resolver("int_list"):
    OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


def compose_config(config_name: str, overrides: Sequence[str], config_path: str) -> DictConfig:
    """Compose a config exactly the way the hydra entry points do."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def build_variant(conf: DictConfig) -> str:
    """Compile the native extension selected by `conf` and return the method it built."""
    method = conf.render.method
    if method == "3dgut":
        from threedgut_tracer.setup_3dgut import setup_3dgut

        setup_3dgut(conf)
    elif method == "3dgrt":
        from threedgrt_tracer.setup_3dgrt import setup_3dgrt

        setup_3dgrt(conf)
    else:
        raise ValueError(f"Unknown rendering method: {method}")
    return method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config-name",
        required=True,
        help="Config name resolved against --config-path, e.g. apps/colmap_3dgut.yaml",
    )
    parser.add_argument(
        "--config-path",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "configs"),
        help="Directory holding the hydra configs (defaults to the repository configs/).",
    )
    parser.add_argument(
        "--overrides",
        action="append",
        default=[],
        metavar="A=B[,C=D]",
        help="One variant per occurrence; comma-separated within an occurrence. "
        "Pass no occurrence to build the config as-is.",
    )
    parser.add_argument(
        "--build-one",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: build a single variant in this process
    )
    args = parser.parse_args(argv)

    variants = [[part for part in o.split(",") if part] for o in args.overrides] or [[]]

    if args.build_one:
        if len(variants) != 1:
            parser.error("--build-one expects exactly one --overrides occurrence")
        conf = compose_config(args.config_name, variants[0], args.config_path)
        build_variant(conf)
        return 0

    failures = 0
    for index, overrides in enumerate(variants, start=1):
        label = " ".join(overrides) or "<config defaults>"
        logger.info(f"[{index}/{len(variants)}] building variant: {label}")
        started = time.perf_counter()
        # Each variant gets a fresh process: the native modules keep a stable name so that
        # `TORCH_EXTENSION_NAME` stays valid, which means pybind refuses to register a
        # second variant of the same module in a process that already loaded one.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "threedgrut.utils.build_variants",
                "--build-one",
                "--config-name",
                args.config_name,
                "--config-path",
                args.config_path,
                "--overrides",
                ",".join(overrides),
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            failures += 1
            logger.error(f"[{index}/{len(variants)}] failed after {elapsed:.1f}s:")
            logger.error((completed.stderr or completed.stdout).strip()[-2000:])
            continue
        logger.info(f"[{index}/{len(variants)}] built in {elapsed:.1f}s")

    if failures:
        logger.error(f"{failures}/{len(variants)} variant(s) failed to build")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
