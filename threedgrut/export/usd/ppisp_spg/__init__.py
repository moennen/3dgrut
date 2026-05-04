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

"""
PPISP SPG shader assets for USD RenderProduct post-processing.

Provides loader for the three SPG sidecar files (Slang shader, Lua launcher,
USDA definition) that must be packaged alongside the exported USDZ.
"""

import logging
from pathlib import Path
from typing import List

from threedgrut.export.usd.stage_utils import NamedSerialized

log = logging.getLogger(__name__)

_SPG_DIR = Path(__file__).parent
_SPG_STATIC_FILES = [
    "ppisp_usd_spg.slang",
    "ppisp_usd_spg.slang.lua",
    "ppisp_usd_spg.slang.usda",
]
_SPG_DYN_SLANG_FILES = [
    "ppisp_usd_spg_dyn.slang",
    "ppisp_usd_spg_dyn.slang.lua",
    "ppisp_usd_spg_dyn.slang.usda",
]
_SPG_DYN_CUDA_FILES = [
    "ppisp_usd_spg_dyn.cu",
    "ppisp_usd_spg_dyn.cu.lua",
    "ppisp_usd_spg_dyn.cu.usda",
]


def _load_files(filenames) -> List[NamedSerialized]:
    result: List[NamedSerialized] = []
    for filename in filenames:
        path = _SPG_DIR / filename
        if path.exists():
            result.append(NamedSerialized(filename=filename, serialized=path.read_bytes()))
            log.debug(f"Loaded PPISP SPG sidecar: {filename}")
        else:
            log.warning(f"PPISP SPG sidecar not found: {path}")
    return result


def get_ppisp_spg_files() -> List[NamedSerialized]:
    """Load static-parameter PPISP SPG sidecar files (controller-free path)."""
    return _load_files(_SPG_STATIC_FILES)


def get_ppisp_spg_dyn_files(backend: str | None = None) -> List[NamedSerialized]:
    """Load controller-aware PPISP SPG sidecar files for ``backend``.

    ``backend`` is ``"cuda"`` (default; mirrors the controller-side default
    and works around a slang-resource-reflection bug we hit in current Kit)
    or ``"slang"``.

    Both variants accept the same parameter schema and read the controller
    output texture identically -- the choice is purely about which SPG
    plugin executes the per-pixel filter.
    """
    backend = (backend or "cuda").lower()
    if backend == "cuda":
        return _load_files(_SPG_DYN_CUDA_FILES)
    if backend == "slang":
        return _load_files(_SPG_DYN_SLANG_FILES)
    raise ValueError(
        f"Unknown ppisp backend {backend!r}; expected 'cuda' or 'slang'"
    )
