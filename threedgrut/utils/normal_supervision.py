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

"""Configuration checks shared by both backends before a normal loss is allowed to run.

Deliberately free of heavy imports so that either tracer's setup can call it.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def normal_supervision_requested(conf) -> bool:
    """Whether any configured loss reads the rendered normal buffer."""
    return bool(OmegaConf.select(conf, "loss.use_depth_normal", default=False))


def check_normals_are_rendered(conf) -> None:
    """Reject a normal loss on a build that renders no normals.

    With `render.enable_normals=false` the tracers do not return an empty buffer: they
    substitute a *constant* placeholder normal for backwards compatibility. A loss built on
    that trains against a constant, which costs throughput, moves the geometry, and reports
    a perfectly plausible-looking loss curve while supervising nothing. Both backends
    therefore refuse the combination instead of running it.
    """
    if not normal_supervision_requested(conf):
        return
    if OmegaConf.select(conf, "render.enable_normals", default=False):
        return

    raise ValueError(
        "loss.use_depth_normal is set but render.enable_normals is false, so the tracer "
        "returns a constant placeholder instead of a rendered normal and the term would "
        "supervise against that constant. Set render.enable_normals=true."
    )
