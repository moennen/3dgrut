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

"""Camera readers: COLMAP sparse/0 and a JSON manifest fallback."""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class View:
    """A single view's intrinsics and extrinsics, with the depth map path it owns."""

    name: str
    width: int
    height: int
    K: np.ndarray = field(repr=False)  # (3, 3) float64
    R: np.ndarray = field(repr=False)  # (3, 3) float64
    t: np.ndarray = field(repr=False)  # (3,) float64, world-to-camera: X_cam = R @ X_w + t
    depth_path: Path
    depth_convention: str | None = None  # optional hint: "ray" or "z"

    def __post_init__(self):
        assert self.K.shape == (3, 3), self.K.shape
        assert self.R.shape == (3, 3), self.R.shape
        assert self.t.shape == (3,), self.t.shape

    def cam_center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def pose_matrix(self) -> np.ndarray:
        """Return the 4x4 camera-to-world matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R.T
        T[:3, 3] = self.cam_center()
        return T


# COLMAP camera model parameter counts and intrinsic decomposition helpers.
_CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

# COLMAP's binary format stores the model as this id; the order is the enum's, not alphabetical.
_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def _parse_colcam_params(model: str, params: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, dist_coeffs) from COLMAP camera parameters.

    K maps the *original* (possibly distorted) pixels. We currently only support models
    that are effectively pinhole after ignoring distortion, because depth maps are
    typically already undistorted and distortion-aware reprojection would need a depth
    value per pixel. A warning is raised if distortion is non-zero and not ignored.
    """
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.zeros(0)
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.zeros(0)
    if model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy, k = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.array([k, 0.0, 0.0, 0.0, 0.0])
    if model in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy, k1, k2 = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.array([k1, k2, 0.0, 0.0, 0.0])
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return K, np.array([k1, k2, p1, p2, 0.0])
    raise ValueError(f"Unsupported COLMAP camera model for direct use: {model}")


def _read_colmap_cameras(path: Path) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    """Read cameras.txt or cameras.bin, return {camera_id: (model, K, dist)}."""
    if (path / "cameras.bin").exists():
        return _read_cameras_bin(path / "cameras.bin")
    if (path / "cameras.txt").exists():
        return _read_cameras_txt(path / "cameras.txt")
    raise FileNotFoundError(f"No cameras.bin or cameras.txt found in {path}")


def _read_cameras_txt(path: Path) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    out: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        n_expected = _CAMERA_MODELS.get(model)
        if n_expected is None:
            raise ValueError(f"Unknown COLMAP camera model: {model}")
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]: parameters start at index 4.
        params = [float(x) for x in parts[4 : 4 + n_expected]]
        if len(params) != n_expected:
            raise ValueError(f"Camera {cam_id} ({model}) expects {n_expected} params, got {len(params)}")
        K, dist = _parse_colcam_params(model, params)
        out[cam_id] = (model, K, dist, width, height)
    return out


def _read_cameras_bin(path: Path) -> dict[int, tuple[str, np.ndarray, np.ndarray, int, int]]:
    out: dict[int, tuple[str, np.ndarray, np.ndarray, int, int]] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            # The model is a numeric id, not a length-prefixed name: the record is
            # (camera_id, model_id, width, height) as "<iiQQ", then num_params doubles.
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            if model_id not in _CAMERA_MODEL_NAMES:
                raise ValueError(f"Unknown COLMAP camera model id {model_id} in {path}")
            model = _CAMERA_MODEL_NAMES[model_id]
            n_expected = _CAMERA_MODELS[model]
            params = struct.unpack(f"<{n_expected}d", f.read(8 * n_expected))
            K, dist = _parse_colcam_params(model, list(params))
            out[cam_id] = (model, K, dist, int(width), int(height))
    return out


def _read_colmap_images(path: Path) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    """Return list of (camera_id, R, t, name) from images.txt or images.bin."""
    if (path / "images.bin").exists():
        return _read_images_bin(path / "images.bin")
    if (path / "images.txt").exists():
        return _read_images_txt(path / "images.txt")
    raise FileNotFoundError(f"No images.bin or images.txt found in {path}")


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to rotation matrix."""
    return np.array(
        [
            [
                1 - 2 * (qvec[2] ** 2 + qvec[3] ** 2),
                2 * (qvec[1] * qvec[2] - qvec[0] * qvec[3]),
                2 * (qvec[1] * qvec[3] + qvec[0] * qvec[2]),
            ],
            [
                2 * (qvec[1] * qvec[2] + qvec[0] * qvec[3]),
                1 - 2 * (qvec[1] ** 2 + qvec[3] ** 2),
                2 * (qvec[2] * qvec[3] - qvec[0] * qvec[1]),
            ],
            [
                2 * (qvec[1] * qvec[3] - qvec[0] * qvec[2]),
                2 * (qvec[2] * qvec[3] + qvec[0] * qvec[1]),
                1 - 2 * (qvec[1] ** 2 + qvec[2] ** 2),
            ],
        ]
    )


def _read_images_txt(path: Path) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    out = []
    # Each image occupies two lines: the pose, then its 2D observations. Neither index
    # parity nor blank-filtering finds the pose lines reliably -- a leading comment block
    # shifts the parity, and an image with no observations has a *blank* second line, so
    # dropping blanks shifts it back. Pose lines are instead identified structurally: a pose
    # line has exactly 10 fields, while an observation line has 3 per point and so can never
    # have 10.
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        cam_id = int(parts[8])
        name = parts[9]
        R = _qvec2rotmat(qvec)
        out.append((cam_id, R, tvec, name))
    return out


def _read_images_bin(path: Path) -> list[tuple[int, np.ndarray, np.ndarray, str]]:
    out = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # The name is null-terminated, *not* length-prefixed. Reading a 4-byte length here
            # consumes the first four characters of the name and then seeks by whatever they
            # happen to spell, which desynchronises the whole file.
            name_bytes = bytearray()
            while True:
                char = f.read(1)
                if char in (b"\x00", b""):
                    break
                name_bytes += char
            name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Each point2D is x, y (doubles) and a point3D_id (int64): 24 bytes, not 16.
            f.seek(24 * num_points2d, 1)
            R = _qvec2rotmat(np.array(qvec))
            out.append((cam_id, R, np.array(tvec), name))
    return out


def read_colmap_views(sparse_dir: str | Path, depth_dir: str | Path | None, extension: str | None = None) -> list[View]:
    """Build Views from a COLMAP sparse/0 directory, matching image names to depth maps.

    Depth maps are searched in ``depth_dir`` by image stem. If ``extension`` is None the
    first file whose stem matches is used; otherwise only that extension is tried. Pass
    ``depth_dir=None`` for the cameras alone, which is what a rasterizer needs: it is about to
    *create* the depth maps, so requiring them first is unsatisfiable.

    Views come back sorted by image name. COLMAP's own order is by image id, which is neither
    the name order nor stable across reconstructions, and callers that pair these cameras with
    an external per-image list -- a TnT ``.log``, say -- would otherwise be matching by an
    order that looks fine and is wrong.
    """
    sparse_dir = Path(sparse_dir)
    depth_dir = Path(depth_dir) if depth_dir is not None else None
    cameras = _read_colmap_cameras(sparse_dir)
    images = sorted(_read_colmap_images(sparse_dir), key=lambda row: row[3])
    views: list[View] = []
    for cam_id, R, t, name in images:
        model, K, _dist, width, height = cameras[cam_id]
        stem = Path(name).stem
        if depth_dir is None:
            depth_path = Path(f"{stem}{extension or '.npy'}")
        elif extension is not None:
            depth_path = depth_dir / f"{stem}{extension}"
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth map for view {name}: {depth_path}")
        else:
            candidates = list(depth_dir.glob(f"{stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"Missing depth map for view {name} in {depth_dir}")
            depth_path = candidates[0]
        # COLMAP R,t is world-to-camera; View keeps that convention.
        views.append(
            View(
                name=name,
                width=width,
                height=height,
                K=K.astype(np.float64),
                R=R,
                t=t,
                depth_path=depth_path,
            )
        )
    if not views:
        raise ValueError(f"No images found in COLMAP reconstruction {sparse_dir}")
    return views


def read_manifest_views(manifest_path: str | Path) -> list[View]:
    """Read a JSON manifest. Schema:

    {
      "views": [
        {"name": "0000", "width": 1554, "height": 1162,
         "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
         "R": [[...]], "t": [tx, ty, tz],
         "depth_path": "depths/0000.npy",
         "depth_convention": "ray"}
      ]
    }

    ``R`` and ``t`` follow the same convention as COLMAP: X_cam = R @ X_w + t.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    views = []
    for entry in manifest["views"]:
        # Relative depth paths resolve against the manifest, not the working directory, so a
        # manifest stays valid when it is read from somewhere else.
        depth_path = Path(entry["depth_path"])
        if not depth_path.is_absolute():
            depth_path = manifest_path.parent / depth_path
        views.append(
            View(
                name=entry["name"],
                width=entry.get("width", 0),
                height=entry.get("height", 0),
                K=np.asarray(entry["K"], dtype=np.float64),
                R=np.asarray(entry["R"], dtype=np.float64),
                t=np.asarray(entry["t"], dtype=np.float64),
                depth_path=depth_path,
                depth_convention=entry.get("depth_convention"),
            )
        )
    return views


def write_manifest_views(views: list[View], path: str | Path) -> None:
    """Write a list of Views to a manifest JSON file."""
    payload: dict[str, Any] = {"views": []}
    for v in views:
        payload["views"].append(
            {
                "name": v.name,
                "width": v.width,
                "height": v.height,
                "K": v.K.tolist(),
                "R": v.R.tolist(),
                "t": v.t.tolist(),
                "depth_path": str(v.depth_path),
                "depth_convention": v.depth_convention,
            }
        )
    Path(path).write_text(json.dumps(payload, indent=2))


def viewset_hash(views: list[View]) -> str:
    """Stable hash of the view set for reproducibility tracking."""
    import hashlib

    h = hashlib.sha256()
    for v in sorted(views, key=lambda x: x.name):
        h.update(v.name.encode())
        h.update(v.R.tobytes())
        h.update(v.t.tobytes())
        h.update(v.K.tobytes())
    return h.hexdigest()[:16]
