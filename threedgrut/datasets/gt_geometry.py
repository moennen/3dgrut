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

"""Reading of ground-truth depth and normal maps that accompany a COLMAP scene.

Conventions, as documented by the OB3D `conversion.json`:

  * depth is the **Euclidean distance along the camera ray**, not the distance to the
    image plane, expressed in the same units as the COLMAP poses;
  * normals are **world-space XYZ** unit vectors;
  * ground truth must be resampled with **nearest neighbour** only.

Two properties of real files drive the implementation. Channel naming is not uniform even
inside a single dataset -- OB3D stores depth as either `V` or `B,G,R` and normals as either
`X,Y,Z` or `B,G,R` depending on the scene -- so channels are negotiated rather than
assumed. And missing geometry (sky) is encoded as a large sentinel instead of a NaN, which
silently poisons any average it reaches, so it is turned into an explicit validity mask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import Imath
    import OpenEXR
except ImportError:  # Optional: only needed when EXR ground truth is actually loaded.
    OpenEXR = None
    Imath = None

# Rendered scenes mark "no surface along this ray" with a large finite value rather than a
# NaN. Anything at or above this threshold is treated as missing.
DEPTH_SENTINEL_THRESHOLD = 1e9

# Channel names to try, in order of preference. A single-channel depth may legitimately be
# called anything, but a three-channel map must be matched exactly: `B,G,R` sorted
# alphabetically would silently transpose x and z.
_DEPTH_CHANNEL_CANDIDATES: tuple[tuple[str, ...], ...] = (("V",), ("R",), ("Y",), ("Z",), ("A",))
_NORMAL_CHANNEL_CANDIDATES: tuple[tuple[str, ...], ...] = (("X", "Y", "Z"), ("R", "G", "B"))

_GT_EXTENSIONS = (".exr", ".npy", ".png", ".tiff", ".tif")

# Conventions this reader implements; a scene that declares anything else is rejected
# rather than silently mis-evaluated.
_EXPECTED_CONVENTIONS = {
    "depth_convention": "Euclidean ray distance",
    "normal_convention": "World-space XYZ",
}


def _select_channels(available: Iterable[str], candidates: Sequence[tuple[str, ...]], count: int, path: str):
    available = set(available)
    for candidate in candidates:
        if all(name in available for name in candidate):
            return candidate
    if count == 1 and len(available) == 1:
        # An unrecognized name is unambiguous when it is the only one present.
        return tuple(available)
    raise ValueError(
        f"Cannot identify {count} ground-truth channel(s) in {path}: found {sorted(available)}, "
        f"expected one of {[list(c) for c in candidates]}"
    )


def _read_exr(path: str, channels: int) -> np.ndarray:
    """Read an EXR as an HWC float32 array, negotiating the channel names."""
    if OpenEXR is None or Imath is None:
        raise ImportError(f"Reading {path} requires the OpenEXR package; install it with `pip install OpenEXR`.")
    candidates = _DEPTH_CHANNEL_CANDIDATES if channels == 1 else _NORMAL_CHANNEL_CANDIDATES
    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        names = _select_channels(header["channels"], candidates, channels, path)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        planes = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width) for name in names
        ]
    finally:
        exr.close()
    return np.stack(planes, axis=-1)


def read_gt_map(path: str, channels: int) -> np.ndarray:
    """Read a depth (`channels=1`) or normal (`channels=3`) map as HWC float32."""
    suffix = Path(path).suffix.lower()
    if suffix == ".exr":
        array = _read_exr(path, channels)
    elif suffix == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path))
        if channels == 1 and array.ndim == 3:
            array = array[..., 0]

    if np.issubdtype(array.dtype, np.integer):
        if channels == 1:
            # There is no way to recover metric depth from quantized values without knowing
            # the scale it was divided by; guessing one would corrupt every depth metric.
            raise ValueError(
                f"Integer depth is ambiguous at {path}; provide float EXR/NPY or convert it "
                "with an explicit scene-unit scale."
            )
        # Encoded normal maps store [-1, 1] mapped onto the full integer range.
        array = 2.0 * array.astype(np.float32) / np.iinfo(array.dtype).max - 1.0

    array = np.asarray(array, dtype=np.float32)
    if channels == 1 and array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] != channels:
        raise ValueError(f"Expected a {channels}-channel ground-truth map at {path}, got shape {array.shape}")
    return array


def find_gt_paths(image_paths: Sequence[str], root: str, folder: str, suffix: str) -> list[Optional[str]]:
    """Locate a ground-truth file per image, matching `<root>/<folder>/<stem>_<suffix>.<ext>`.

    The RGB stem loses a trailing `_rgb` first, so `images/00000_rgb.png` pairs with
    `depths/00000_depth.exr`. Entries stay `None` when no candidate exists, which lets the
    caller distinguish a partially annotated scene from an unannotated one.
    """
    paths: list[Optional[str]] = []
    for image_path in image_paths:
        stem = Path(str(image_path)).stem
        stem = stem[: -len("_rgb")] if stem.endswith("_rgb") else stem
        found = None
        for extension in _GT_EXTENSIONS:
            candidate = os.path.join(root, folder, f"{stem}_{suffix}{extension}")
            if os.path.isfile(candidate):
                found = candidate
                break
        paths.append(found)
    return paths


def validate_scene_conventions(root: str) -> None:
    """Reject a scene whose `conversion.json` declares conventions this reader cannot honor.

    Absent metadata is accepted: most COLMAP scenes carry none.
    """
    metadata_path = os.path.join(root, "conversion.json")
    if not os.path.isfile(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return  # Unreadable metadata is not evidence of a different convention.
    if not isinstance(metadata, dict):
        return
    for key, expected in _EXPECTED_CONVENTIONS.items():
        declared = metadata.get(key)
        if isinstance(declared, str) and expected.lower() not in declared.lower():
            raise ValueError(
                f"{metadata_path} declares {key}={declared!r}, but the ground-truth reader "
                f"implements {expected!r}. Convert the scene or extend the reader."
            )


def depth_validity(depth: np.ndarray) -> np.ndarray:
    """Mask of pixels holding a usable depth, excluding the sky sentinel and non-finite values."""
    return np.isfinite(depth) & (np.abs(depth) < DEPTH_SENTINEL_THRESHOLD) & (depth > 0.0)


def resize_gt_map(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resample to `(height, width)`.

    Interpolation is deliberately not smooth: averaging across a depth discontinuity
    invents a surface that exists in neither the scene nor the reference, and averaging in
    the sky sentinel would drag whole neighbourhoods to ~1e10.
    """
    source_height, source_width = array.shape[:2]
    if (source_height, source_width) == (height, width):
        return array
    rows = np.minimum((np.arange(height) + 0.5) * source_height / height, source_height - 1).astype(np.int64)
    columns = np.minimum((np.arange(width) + 0.5) * source_width / width, source_width - 1).astype(np.int64)
    return array[rows[:, None], columns[None, :]]


def similarity_scale(transform: np.ndarray, tolerance: float = 1e-4) -> float:
    """Uniform scale of a similarity transform, verifying that it really is uniform."""
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if scales.min() <= 0.0 or (scales.max() - scales.min()) > tolerance * scales.max():
        raise ValueError(f"World normalization is not a similarity transform; column scales are {scales}")
    return float(scales.mean())


def transform_gt_depth(depth: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Rescale metric depth into a normalized world, leaving the sentinel recognizable.

    Ray distance scales with the transform, so skipping this leaves the reference in the
    original units while the renderer works in normalized ones.
    """
    scaled = depth * similarity_scale(transform)
    return np.where(depth_validity(depth), scaled, depth)


def transform_gt_normal(normal: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Rotate world-space normals into a normalized world.

    Only the rotation matters: a uniform scale cannot change a direction, so applying the
    linear block and renormalizing is equivalent to applying the rotation alone.
    """
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    rotated = normal.astype(np.float64) @ linear.T
    return normalize_gt_normal(rotated.astype(np.float32))


def normalize_gt_normal(normal: np.ndarray) -> np.ndarray:
    """Scale normals to unit length, leaving degenerate (zero) entries at zero.

    A zero-length normal marks a pixel with no surface; forcing it to unit length would
    fabricate an orientation for the sky.
    """
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-8)
