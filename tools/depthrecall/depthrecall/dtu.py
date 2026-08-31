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

"""DTU support: cameras from ``cameras.npz`` and the scan-to-world alignment.

DTU as redistributed for neural reconstruction (IDR/NeuS/2DGS and this repo's copy) stores,
per image ``i``:

- ``world_mat_i``: the 3x4 projection ``K [R | t]``, padded to 4x4, mapping *scan* coordinates
  (DTU millimetres) to pixels.
- ``scale_mat_i``: the similarity mapping *normalized* coordinates (a unit sphere, which is
  what models are trained in) to scan coordinates. Identical for every image of a scan.

Two things follow, and both are easy to get backwards. First, a model trained in normalized
space renders depth in normalized units, so the scan must be multiplied by ``inv(scale_mat)``
-- not ``scale_mat`` -- to be comparable; the scale is ~325, so the wrong direction is not a
subtle error but it *is* a silent one. Second, ``world_mat`` projects scan coordinates, so
using it as-is places the cameras in millimetres while the depth maps are normalized. Both
directions are offered explicitly rather than inferred.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_cameras import View


def rq_decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition: return (R, Q) with R upper-triangular and matrix = R @ Q.

    numpy has no ``rq``, and pulling in scipy or cv2 for one 3x3 decomposition is not worth
    the dependency; this is QR of a doubly-flipped transpose.
    """
    flipped = np.flipud(matrix)
    Q, R = np.linalg.qr(flipped.T)
    R = np.flipud(R.T)
    return R[:, ::-1], Q.T[::-1, :]


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a 3x4 projection into (K, R, t) with K[2, 2] = 1 and positive focal lengths.

    ``R``/``t`` are world-to-camera, matching COLMAP's convention: X_cam = R @ X_world + t.
    The factorization is verified against the input, because an RQ decomposition always
    returns *something* -- the sign conventions are what go wrong, and a K with a negative
    focal length or a left-handed R reprojects to plausible-looking pixels.
    """
    P = np.asarray(P, dtype=np.float64)[:3, :4]

    # A projection matrix is homogeneous, so its overall scale is arbitrary and has to be
    # fixed before anything metric can be read out of it: the third row of K [R | t] is
    # (R_2, t_2) with R_2 a unit vector, so dividing by its norm makes t the real translation.
    # Skipping this yields a K whose focal length looks fine and a t that is off by the
    # scale of the matrix -- which is how a 325x error hides inside a valid-looking camera.
    row_norm = np.linalg.norm(P[2, :3])
    if row_norm == 0:
        raise ValueError("Degenerate projection matrix: zero viewing axis")
    P = P / row_norm

    # With K's diagonal positive and R a rotation, det(K [R]) > 0. If the file's sign
    # convention gives the opposite, the valid choice is -P (identical as a projection), which
    # is also the one that puts visible points at positive depth.
    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    K, R = rq_decompose(P[:, :3])
    # Fix the signs: K's diagonal must be positive, absorbed into R.
    signs = np.diag(np.sign(np.diag(K)))
    K, R = K @ signs, signs @ R
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])

    if not np.allclose(R @ R.T, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0:
        raise ValueError("Projection matrix decomposition did not yield a rotation")
    reconstructed = K @ np.concatenate([R, t[:, None]], axis=1)
    if not np.allclose(reconstructed, P, rtol=1e-6, atol=1e-6 * np.abs(P).max()):
        raise ValueError("Projection matrix decomposition failed to reproduce its input")
    return K, R, t


def scan_to_normalized(cameras_npz: str | Path) -> np.ndarray:
    """Return the 4x4 mapping DTU scan coordinates (mm) to normalized model coordinates.

    This is ``inv(scale_mat)``; pass it as ``--alignment`` when scoring depth rendered by a
    model trained on the normalized scene.
    """
    npz = np.load(str(cameras_npz))
    scale_mats = [npz[key] for key in npz.files if key.startswith("scale_mat_") and "inv" not in key]
    if not scale_mats:
        raise ValueError(f"No scale_mat_* entries in {cameras_npz}")
    reference = np.asarray(scale_mats[0], dtype=np.float64)
    # Every image of a scan shares one scale matrix. If that ever stops holding, the scan has
    # no single alignment and silently taking the first would be wrong.
    for matrix in scale_mats[1:]:
        if not np.allclose(np.asarray(matrix, dtype=np.float64), reference, rtol=1e-6, atol=1e-6):
            raise ValueError(f"scale_mat entries in {cameras_npz} disagree; the scan has no single alignment")
    if reference.shape != (4, 4):
        raise ValueError(f"DTU scale_mat must be 4x4, got {reference.shape}")
    return np.linalg.inv(reference)


def read_ground_plane(path: str | Path) -> np.ndarray:
    """Read a DTU ``Plane<scan>.mat`` (or a 4-coefficient .npy/.txt) as ``[a, b, c, d]``.

    ``.mat`` needs scipy, which is an optional dependency here; the same coefficients can be
    supplied as plain text to keep the package numpy-only.
    """
    path = Path(path)
    if path.suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"Reading {path} needs scipy. Either install it, or pass the four plane "
                "coefficients as a .txt/.npy file."
            ) from exc
        plane = np.asarray(loadmat(str(path))["P"], dtype=np.float64).reshape(-1)
    elif path.suffix == ".npy":
        plane = np.load(path).astype(np.float64).reshape(-1)
    else:
        plane = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if plane.shape != (4,):
        raise ValueError(f"Expected 4 plane coefficients in {path}, got {plane.shape}")
    return plane


def above_ground_plane_mask(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Mask of the scan points on the object side of DTU's ground plane.

    This is the *ground-truth-side* cull that the official evaluation applies to ``stl`` before
    measuring completeness -- the direction this package's recall corresponds to. It is not
    ``ObsMask``: that one filters the *reconstruction*, to avoid penalising points where the
    scanner never observed, and has no analogue here because every point we iterate over is a
    scan point and so was observed by construction.

    Points are in scan millimetres, so this is applied before any alignment. On scan24 it drops
    38.5% of the reference cloud (the table), which is enough to dominate the score: leaving it
    in raises the ceiling from 0.639 to 0.679 at tau = 1 mm.
    """
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=-1)
    return (plane.reshape(1, 4) * homogeneous).sum(-1) > 0


def observed_volume_mask(points: np.ndarray, path: str | Path) -> np.ndarray:
    """DTU's official reconstruction-side ``ObsMask`` lookup for accuracy.

    The observation volume is not a GT crop: it removes predicted samples in regions the scanner
    could not observe, exactly as the reference DTU accuracy direction does.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("DTU ObsMask files need scipy") from exc
    data = loadmat(str(path))
    volume = np.asarray(data["ObsMask"], dtype=bool)
    bounds = np.asarray(data["BB"], dtype=np.float64)
    resolution = float(np.asarray(data["Res"]).reshape(-1)[0])
    if bounds.shape != (2, 3):
        # MATLAB stores the two bounding-box corners by row in the official evaluator.
        raise ValueError(f"DTU ObsMask BB must be (2, 3), got {bounds.shape}")
    grid = np.rint((np.asarray(points, dtype=np.float64) - bounds[0]) / resolution).astype(np.int64)
    inside = np.all((grid >= 0) & (grid < np.asarray(volume.shape)), axis=1)
    result = np.zeros(len(points), dtype=bool)
    result[inside] = volume[grid[inside, 0], grid[inside, 1], grid[inside, 2]]
    return result


def read_dtu_views(
    cameras_npz: str | Path,
    depth_dir: str | Path | None = None,
    *,
    space: str = "normalized",
    extension: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[View]:
    """Build views from a DTU ``cameras.npz``.

    ``space`` selects the world frame the returned extrinsics are expressed in:

    - ``"normalized"`` (default): the unit-sphere frame models train and render in. Scan
      points must then be transformed by :func:`scan_to_normalized`.
    - ``"scan"``: DTU millimetres, for depth maps rendered directly in scan coordinates. Scan
      points are then used as they are.

    ``image_size`` is the (width, height) the intrinsics were authored for. When omitted it is
    taken from the depth maps, which is correct only if they are rendered at full resolution;
    a mismatch scales the focal length and shifts the principal point, so pass it explicitly
    for downscaled renders.
    """
    if space not in ("normalized", "scan"):
        raise ValueError(f"space must be 'normalized' or 'scan', got {space}")
    if depth_dir is None and image_size is None:
        raise ValueError("image_size is required when no depth directory is given")

    npz = np.load(str(cameras_npz))
    depth_dir = Path(depth_dir) if depth_dir is not None else None
    indices = sorted(int(key.split("_")[-1]) for key in npz.files if key.startswith("world_mat_") and "inv" not in key)
    if not indices:
        raise ValueError(f"No world_mat_* entries in {cameras_npz}")

    scale_mat = np.asarray(npz[f"scale_mat_{indices[0]}"], dtype=np.float64)
    views: list[View] = []
    for index in indices:
        P = np.asarray(npz[f"world_mat_{index}"], dtype=np.float64)
        if space == "normalized":
            # Composing the projection with scale_mat re-expresses the same camera in the
            # normalized frame, rather than converting R and t by hand afterwards.
            P = P @ scale_mat
        K, R, t = decompose_projection(P)

        name = f"{index:04d}"
        if depth_dir is None:
            # Cameras only, for a caller that is about to *write* the depth maps.
            depth_path = Path(f"{name}{extension or '.npy'}")
        elif extension is not None:
            depth_path = depth_dir / f"{name}{extension}"
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth map for DTU view {name}: {depth_path}")
        else:
            candidates = sorted(depth_dir.glob(f"{name}.*"))
            if not candidates:
                raise FileNotFoundError(f"Missing depth map for DTU view {name} in {depth_dir}")
            depth_path = candidates[0]

        if image_size is not None:
            width, height = image_size
        else:
            from .io_depth import load_depth

            depth, _ = load_depth(depth_path)
            height, width = depth.shape

        views.append(
            View(
                name=name,
                width=int(width),
                height=int(height),
                K=K,
                R=R,
                t=t,
                depth_path=depth_path,
            )
        )
    return views
