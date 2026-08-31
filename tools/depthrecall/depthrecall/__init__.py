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

"""Recall of ground-truth scan points against rendered depth maps.

A reconstruction is scored by projecting every ground-truth point into every view and
comparing the point's range against the rendered depth at that pixel. A pair passes at
threshold ``tau`` when the two agree to within ``tau``; recall is the fraction of pairs
that pass, as a curve over ``tau``.

The metric deliberately involves no mesh, so it carries none of the extraction
parameters (voxel size, tessellation density, cull radius, sampling scheme) that make
mesh Chamfer and F-score numbers incomparable between implementations. It is a relative
measure: ground-truth points occluded from a view fail regardless of reconstruction
quality, which is a fixed property of the (scene, cameras, scan) triple and therefore
biases every method scored over the same triple identically.

Nothing here imports the renderer that produced the depth maps, so the same operator
scores any method that can write a folder of depth maps.
"""

from .metric import MetricConfig, evaluate  # noqa: F401

__all__ = ["MetricConfig", "evaluate"]
