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

"""Depth-map loaders.  All return (depth, valid) as numpy float64/bool arrays."""

import warnings
from pathlib import Path

import numpy as np


class DepthLoadError(Exception):
    pass


def load_depth(path: str | Path, depth_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Load a depth map and return (depth, valid_mask).

    Validity is determined from finite, positive values; callers can further restrict with
    ``max_valid_depth`` or an explicit sentinel value.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    loaders = {
        ".npy": _load_npy,
        ".npz": _load_npz,
        ".pt": _load_pt,
        ".pth": _load_pt,
        ".png": _load_png,
        ".exr": _load_exr,
    }
    loader = loaders.get(suffix)
    if loader is None:
        raise DepthLoadError(f"No loader for extension {suffix}; known: {sorted(loaders)}")
    depth = loader(path)
    if depth_scale != 1.0:
        depth = depth * depth_scale
    valid = np.isfinite(depth) & (depth > 0)
    return depth.astype(np.float64, copy=False), valid


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    if "depth" in data:
        return data["depth"]
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr
    raise DepthLoadError(f"npz {path} has no recognisable 2D depth array")


def _load_pt(path: Path) -> np.ndarray:
    try:
        import torch
    except ImportError as e:
        raise DepthLoadError(".pt depth requires torch") from e
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(tensor, dict):
        if "depth" in tensor:
            tensor = tensor["depth"]
        else:
            # take the first 2D tensor
            for v in tensor.values():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    tensor = v
                    break
    if not isinstance(tensor, torch.Tensor):
        raise DepthLoadError(f"Unrecognised content in {path}: {type(tensor)}")
    return tensor.detach().cpu().numpy()


def _load_png(path: Path) -> np.ndarray:
    """Load a 16-bit PNG depth map. Uses cv2 if available, else Pillow."""
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise DepthLoadError(f"cv2 could not read {path}")
        return img.astype(np.float64)
    except ImportError:
        pass
    try:
        from PIL import Image

        img = Image.open(path)
        return np.array(img, dtype=np.float64)
    except ImportError as e:
        raise DepthLoadError("16-bit PNG depth requires opencv-python or pillow") from e


def _load_exr(path: Path) -> np.ndarray:
    """Load an EXR depth map. Tries OpenEXR, then imageio, then opencv."""
    try:
        import Imath
        import OpenEXR

        f = OpenEXR.InputFile(str(path))
        header = f.header()
        dw = header["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        channel = next(iter(header["channels"]))
        half_type = Imath.PixelType(Imath.PixelType.HALF)
        buf = f.channel(channel, half_type)
        arr = np.frombuffer(buf, dtype=np.float16).reshape(size[1], size[0])
        return arr.astype(np.float64)
    except ImportError:
        pass
    try:
        import imageio.v3 as iio

        return iio.imread(path).astype(np.float64)
    except ImportError:
        pass
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise DepthLoadError(f"cv2 could not read EXR {path}")
        return img.astype(np.float64)
    except ImportError as e:
        raise DepthLoadError("EXR depth requires OpenEXR, imageio, or opencv-python") from e


def has_no_surface(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """True wherever there is no valid surface to compare against."""
    return ~valid


def set_invalid_value(depth: np.ndarray, invalid_value: float | None) -> tuple[np.ndarray, np.ndarray]:
    """If ``invalid_value`` is not None, treat pixels with that value as invalid."""
    if invalid_value is None:
        return depth, np.isfinite(depth) & (depth > 0)
    valid = np.isfinite(depth) & (depth != invalid_value)
    return depth, valid
