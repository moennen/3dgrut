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

"""Minimal PLY reader and point-cloud downsampling."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray  # (N, 3) float64
    count: int

    def __post_init__(self):
        if self.xyz.shape != (self.count, 3):
            raise ValueError(f"PointCloud shape mismatch: xyz {self.xyz.shape}, count {self.count}")


_PLY_DTYPES: dict[str, np.dtype] = {
    "char": np.dtype("int8"),
    "uchar": np.dtype("uint8"),
    "short": np.dtype("int16"),
    "ushort": np.dtype("uint16"),
    "int": np.dtype("int32"),
    "uint": np.dtype("uint32"),
    "float": np.dtype("float32"),
    "double": np.dtype("float64"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "int16": np.dtype("int16"),
    "uint16": np.dtype("uint16"),
    "int32": np.dtype("int32"),
    "uint32": np.dtype("uint32"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


def _parse_header(data: bytes) -> tuple[str, int, int, np.dtype, list[tuple[str, np.dtype]], int]:
    """Return (format, vertex_count, vertex_bytesize, vertex_dtype, vertex_props, header_end)."""
    marker = data.find(b"end_header")
    if marker < 0:
        raise ValueError("PLY file missing end_header marker")
    # The official DTU reference scans terminate header lines with CRLF, so the newline after
    # the marker cannot be assumed to be a bare "\n" -- searching for "end_header\n" rejects
    # those files outright. Skip whatever line terminator follows.
    header_end = marker + len(b"end_header")
    while header_end < len(data) and data[header_end : header_end + 1] in (b"\r", b"\n"):
        header_end += 1
    header = data[:header_end].decode("ascii")

    fmt_match = re.search(r"^format\s+(ascii|binary_little_endian)\s+1\.0\s*$", header, re.MULTILINE)
    if not fmt_match:
        raise ValueError("PLY file must be ascii or binary_little_endian format 1.0")
    fmt = fmt_match.group(1)

    elements: list[tuple[str, int]] = []
    vertex_props: list[tuple[str, np.dtype]] = []
    current_element: str | None = None
    for line in header.splitlines():
        tokens = line.strip().split()
        if len(tokens) == 0 or tokens[0] == "comment":
            continue
        if tokens[0] == "element":
            current_element = tokens[1]
            elements.append((current_element, int(tokens[2])))
            continue
        if tokens[0] == "property":
            if current_element != "vertex":
                continue
            # "property <type> <name>", or "property list <count-type> <item-type> <name>".
            if tokens[1] == "list":
                raise ValueError("List properties on the vertex element are not supported")
            dtype_str, name = tokens[1], tokens[2]
            dtype = _PLY_DTYPES.get(dtype_str)
            if dtype is None:
                raise ValueError(f"Unsupported PLY property type: {dtype_str}")
            vertex_props.append((name, dtype))
            continue

    if not elements or elements[0][0] != "vertex":
        raise ValueError("PLY file has no vertex element")
    vertex_count = elements[0][1]

    dtype_desc = [(name, dtype) for name, dtype in vertex_props]
    vertex_dtype = np.dtype(dtype_desc)

    return fmt, vertex_count, vertex_dtype.itemsize, vertex_dtype, vertex_props, header_end


def _find_xyz_indices(prop_names: list[str]) -> tuple[int, int, int]:
    lowered = [p.lower() for p in prop_names]
    try:
        return lowered.index("x"), lowered.index("y"), lowered.index("z")
    except ValueError as e:
        raise ValueError("PLY vertex element must contain x, y, z properties") from e


def read_ply(path: str | Path) -> PointCloud:
    """Read a PLY file, keeping only vertex positions.

    Supports both ASCII and binary_little_endian. Vertex-list properties and face elements
    are skipped; faces are ignored because the GT cloud is points-only.
    """
    path = Path(path)
    data = path.read_bytes()
    fmt, vertex_count, vertex_bytesize, vertex_dtype, vertex_props, header_end = _parse_header(data)
    prop_names = [name for name, _ in vertex_props]
    ix, iy, iz = _find_xyz_indices(prop_names)

    if fmt == "ascii":
        body = data[header_end:].decode("ascii")
        rows = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            rows.append([float(tokens[ix]), float(tokens[iy]), float(tokens[iz])])
            if len(rows) == vertex_count:
                break
        xyz = np.asarray(rows, dtype=np.float64)
    else:
        if len(data) < header_end + vertex_count * vertex_bytesize:
            raise ValueError("PLY binary body truncated")
        arr = np.frombuffer(data[header_end : header_end + vertex_count * vertex_bytesize], dtype=vertex_dtype)
        xyz = np.column_stack([arr[prop_names[i]].astype(np.float64) for i in (ix, iy, iz)])

    return PointCloud(xyz=xyz, count=xyz.shape[0])


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return one representative point per occupied voxel.

    ``points`` is (N, 3); returned array is (M, 3) with M <= N. The representative is the
    first point in the original order falling in each voxel.
    """
    if points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def identity_alignment() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def load_alignment(path: str | Path, invert: bool = False) -> np.ndarray:
    """Load a 4x4 alignment matrix from a .txt or .npy file.

    Text files are parsed as space-separated 4x4 matrices. .npy is loaded directly.
    If ``invert`` is True, the inverse is returned.
    """
    path = Path(path)
    if path.suffix == ".npy":
        matrix = np.load(path).astype(np.float64)
    else:
        matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Alignment matrix must be 4x4, got {matrix.shape}")
    return np.linalg.inv(matrix) if invert else matrix


def apply_alignment(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 similarity/SE(3) to (N, 3) points."""
    if points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = (homogeneous @ matrix.T)[:, :3]
    return transformed


class Normalizer:
    """Convenience object that applies optional GT alignment and downsampling.

    Alignment comes *first* so that the voxel size is expressed in the same units as the
    thresholds and the depth maps. Downsampling in the source units instead makes the option
    silently useless whenever the alignment carries a scale: a DTU voxel of 0.003 means 3
    microns in the scan's millimetres but 1 mm in the normalized frame the taus live in, and
    the former leaves a 5-million-point cloud untouched while looking like it worked.
    """

    def __init__(
        self,
        points: np.ndarray,
        alignment: np.ndarray | None = None,
        downsample_voxel: float | None = None,
    ):
        if alignment is not None:
            points = apply_alignment(points, alignment)
        if downsample_voxel is not None and downsample_voxel > 0:
            points = voxel_downsample(points, downsample_voxel)
        self.points = points
        self.count = points.shape[0]

    def hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(self.points.tobytes())
        h.update(f"{self.count}".encode())
        return h.hexdigest()[:16]
