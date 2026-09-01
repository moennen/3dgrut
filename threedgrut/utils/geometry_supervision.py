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

"""Configuration checks shared by both backends before a geometry loss is allowed to run.

Each check here exists because the combination it rejects would otherwise *train*, at full
cost, while supervising nothing -- a silently wrong run rather than a failed one.

Deliberately free of heavy imports so that either tracer's setup can call it.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def normal_supervision_requested(conf) -> bool:
    """Whether any configured loss reads the rendered normal buffer."""
    return bool(
        OmegaConf.select(conf, "loss.use_depth_normal", default=False)
        or OmegaConf.select(conf, "loss.use_normal_variance", default=False)
    )


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


def check_depth_variance_is_rendered(conf) -> None:
    """Reject a depth-variance loss where the second-moment buffer is absent.

    Two ways it can be, with the same consequence. `render.enable_depth_variance=false`
    compiles the accumulator out and the tracer returns an *empty* tensor, and 3DGRT does not
    implement the moment at all. Either way the term has nothing to read; unlike the normal's
    constant placeholder an empty buffer at least cannot masquerade as data, but it sums to a
    perfectly respectable `0.0`, which trains at full cost and supervises nothing.
    """
    if not OmegaConf.select(conf, "loss.use_depth_variance", default=False):
        return

    method = OmegaConf.select(conf, "render.method", default="3dgut")
    if method != "3dgut":
        raise ValueError(
            f"loss.use_depth_variance is set with render.method={method}, which does not render "
            "the depth second moment. The term is 3DGUT-only; use render.method=3dgut."
        )

    if not OmegaConf.select(conf, "render.enable_depth_variance", default=False):
        raise ValueError(
            "loss.use_depth_variance is set but render.enable_depth_variance is false, so the "
            "tracer returns an empty second-moment buffer and the term would be a silent zero. "
            "Set render.enable_depth_variance=true."
        )


def check_flatness_applies(conf) -> None:
    """Reject the flatness penalty on a primitive that is already flat.

    Surfel kernels overwrite `scale.z` with 1e-6 on fetch and never accumulate a gradient
    into it (`gaussianParticles.slang`), so the stored third scale is dead storage. A
    penalty on the smallest axis would therefore select that dead component for most
    particles and spend the whole term shrinking a number the renderer never reads --
    costing throughput and reporting a falling loss curve for no effect on the geometry.
    Flatness is already guaranteed for these primitives, so asking for it is a mistake
    worth surfacing rather than absorbing.
    """
    if not OmegaConf.select(conf, "loss.use_scale_flatten", default=False):
        return
    if OmegaConf.select(conf, "render.primitive_type", default="instances") != "trisurfel":
        return

    raise ValueError(
        "loss.use_scale_flatten is set with render.primitive_type=trisurfel, which the "
        "kernel already flattens by forcing scale.z and dropping its gradient. The term "
        "would only shrink an unused parameter. Use it with the ellipsoid primitives."
    )
