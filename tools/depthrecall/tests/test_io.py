# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the readers.

The COLMAP binary parsers and the PLY parser are hand-rolled to keep the package free of
heavy dependencies, which means a silent misparse is the main risk here: a transposed
rotation or a swapped quaternion order still yields a valid-looking camera.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from depthrecall.io_cameras import View, read_colmap_views, read_manifest_views, write_manifest_views
from depthrecall.io_depth import load_depth
from depthrecall.io_points import Normalizer, read_ply, voxel_downsample


def write_cameras_txt(path: Path) -> None:
    path.write_text(
        "# Camera list\n" "1 PINHOLE 1554 1162 1150.5 1150.5 777.0 581.0\n" "2 SIMPLE_PINHOLE 100 100 50.0 50.0 50.0\n"
    )


def write_images_txt(path: Path, qvec, tvec) -> None:
    q = " ".join(str(x) for x in qvec)
    t = " ".join(str(x) for x in tvec)
    path.write_text(f"# Image list\n1 {q} {t} 1 0000.png\n\n")


def write_cameras_bin(path: Path) -> None:
    """Write cameras.bin exactly as COLMAP does: `<iiQQ` then the model's params.

    The model is a numeric *id* (1 = PINHOLE), not a length-prefixed name. These writers
    previously mirrored the reader's own mistaken layout, so the round trip below passed while
    both sides were wrong and no real COLMAP file could be read at all. Writers in a test for
    an external format have to follow the format's spec, not the reader under test.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 1554, 1162))
        f.write(struct.pack("<4d", 1150.5, 1150.5, 777.0, 581.0))


def write_images_bin(path: Path, qvec, tvec, n_points2d: int = 3, name: bytes = b"0000.png") -> None:
    """Write images.bin as COLMAP does: null-terminated name, 24 bytes per point2D."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<4d", *qvec))
        f.write(struct.pack("<3d", *tvec))
        f.write(struct.pack("<I", 1))
        f.write(name + b"\x00")
        f.write(struct.pack("<Q", n_points2d))
        for i in range(n_points2d):
            f.write(struct.pack("<2d", float(i), float(i)))
            f.write(struct.pack("<q", -1))


def test_colmap_txt_and_bin_agree(tmp_path: Path) -> None:
    """The two COLMAP encodings must produce identical cameras.

    They are parsed by completely separate code paths, so agreement between them is the
    cheapest available check that the binary offsets (the null-terminated name and the
    24-byte-per-point2D skip) are right -- provided the writer follows COLMAP and not the
    reader. Verified against `threedgrut.datasets.utils.read_colmap_extrinsics_binary` on
    real Barn and room reconstructions (410 and 311 images, exact agreement).
    """
    qvec = (0.9238795, 0.3826834, 0.0, 0.0)
    tvec = (0.1, -0.2, 3.0)

    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    write_cameras_txt(txt_dir / "cameras.txt")
    write_images_txt(txt_dir / "images.txt", qvec, tvec)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_cameras_bin(bin_dir / "cameras.bin")
    write_images_bin(bin_dir / "images.bin", qvec, tvec)

    depths = tmp_path / "depths"
    depths.mkdir()
    np.save(depths / "0000.npy", np.ones((10, 10)))

    from_txt = read_colmap_views(txt_dir, depths)
    from_bin = read_colmap_views(bin_dir, depths)

    assert len(from_txt) == len(from_bin) == 1
    np.testing.assert_allclose(from_txt[0].R, from_bin[0].R)
    np.testing.assert_allclose(from_txt[0].t, from_bin[0].t)
    np.testing.assert_allclose(from_txt[0].K, from_bin[0].K)


def test_colmap_rotation_is_world_to_camera(tmp_path: Path) -> None:
    """R must be world-to-camera and orthonormal, with the camera centre at -R^T t.

    Transposing R here would still give a plausible reconstruction of the scene from a
    mirrored viewpoint, and every depth residual would be wrong without any error.
    """
    # 90 degrees about x: (w, x, y, z) with w = cos(45deg), x = sin(45deg)
    qvec = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
    tvec = (0.0, 0.0, 5.0)
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    write_cameras_txt(sparse / "cameras.txt")
    write_images_txt(sparse / "images.txt", qvec, tvec)
    depths = tmp_path / "depths"
    depths.mkdir()
    np.save(depths / "0000.npy", np.ones((10, 10)))

    view = read_colmap_views(sparse, depths)[0]
    np.testing.assert_allclose(view.R @ view.R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(view.R) == pytest.approx(1.0)
    # A 90-degree rotation about x maps world +y to camera -z (or +z), never to +y.
    assert abs(view.R[1, 1]) < 1e-9
    np.testing.assert_allclose(view.cam_center(), -view.R.T @ view.t)


def test_manifest_round_trip(tmp_path: Path) -> None:
    view = View(
        name="0000",
        width=100,
        height=80,
        K=np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]]),
        R=np.eye(3),
        t=np.array([0.0, 0.0, 1.0]),
        depth_path=tmp_path / "0000.npy",
    )
    path = tmp_path / "manifest.json"
    write_manifest_views([view], path)
    (restored,) = read_manifest_views(path)
    assert restored.name == view.name
    np.testing.assert_allclose(restored.K, view.K)
    np.testing.assert_allclose(restored.R, view.R)
    np.testing.assert_allclose(restored.t, view.t)
    assert json.loads(path.read_text())["views"][0]["width"] == 100


def test_missing_depth_map_is_an_error_not_a_dropped_view(tmp_path: Path) -> None:
    """Silently skipping a view would change the pair population between two runs and
    make their recall numbers incomparable without any visible sign."""
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    write_cameras_txt(sparse / "cameras.txt")
    write_images_txt(sparse / "images.txt", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    empty = tmp_path / "depths"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_colmap_views(sparse, empty)


def ply_header(count: int, fmt: str, extra: str = "") -> str:
    return (
        f"ply\nformat {fmt} 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{extra}"
        "end_header\n"
    )


def test_ply_ascii_and_binary_agree(tmp_path: Path) -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25], [0.0, 0.0, 0.0]], dtype=np.float32)

    ascii_path = tmp_path / "a.ply"
    body = "\n".join(f"{x} {y} {z}" for x, y, z in points)
    ascii_path.write_text(ply_header(len(points), "ascii") + body + "\n")

    bin_path = tmp_path / "b.ply"
    with open(bin_path, "wb") as f:
        f.write(ply_header(len(points), "binary_little_endian").encode())
        f.write(points.astype("<f4").tobytes())

    np.testing.assert_allclose(read_ply(ascii_path).xyz, points)
    np.testing.assert_allclose(read_ply(bin_path).xyz, points)


def test_ply_with_extra_properties_reads_only_xyz(tmp_path: Path) -> None:
    """DTU and T&T clouds carry normals and colours; the strides must still be right."""
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rows = np.zeros(2, dtype=dtype)
    rows["x"] = [1.0, 4.0]
    rows["y"] = [2.0, 5.0]
    rows["z"] = [3.0, 6.0]
    rows["nx"] = [0.5, 0.5]
    rows["r"] = [255, 0]

    path = tmp_path / "c.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty uchar r\nproperty uchar g\nproperty uchar b\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rows.tobytes())

    np.testing.assert_allclose(read_ply(path).xyz, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_downsample_voxel_is_in_evaluation_units_not_source_units() -> None:
    """The voxel size must mean the same thing as tau, i.e. apply after the alignment.

    Downsampling first makes the option silently inert whenever the alignment carries a scale:
    with DTU's 325x normalization, a 0.001 voxel means 0.3 microns in the scan's millimetres
    and leaves the cloud untouched, while reporting that it downsampled.
    """
    scale = 325.0
    alignment = np.diag([1 / scale, 1 / scale, 1 / scale, 1.0])
    # Two points 0.5 apart in normalized units, i.e. 162.5 apart in source units.
    points = np.array([[0.0, 0.0, 0.0], [0.5 * scale, 0.0, 0.0]])

    # A voxel well below the normalized separation keeps both points; one well above merges
    # them. Read in source units, the second voxel would span 10 of 162.5 units and merge
    # nothing, so this pins the order rather than just the behaviour.
    assert Normalizer(points, alignment, downsample_voxel=0.01).count == 2
    assert Normalizer(points, alignment, downsample_voxel=10.0).count == 1


def test_ply_with_crlf_header_reads(tmp_path: Path) -> None:
    """The official DTU reference scans (stl*_total.ply) use CRLF line endings.

    Locating the body by searching for "end_header\\n" skips past the "\\r" and rejects those
    files outright, which is how the entire DTU ground truth was unreadable.
    """
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]], dtype=np.float32)
    header = (
        "ply\r\nformat binary_little_endian 1.0\r\n"
        "element vertex 2\r\n"
        "property float x\r\nproperty float y\r\nproperty float z\r\n"
        "end_header\r\n"
    )
    path = tmp_path / "crlf.ply"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(points.astype("<f4").tobytes())

    np.testing.assert_allclose(read_ply(path).xyz, points)


def test_voxel_downsample_keeps_one_point_per_cell() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]])
    kept = voxel_downsample(points, voxel_size=0.1)
    assert kept.shape[0] == 2
    # Deterministic: first point in original order wins its cell.
    np.testing.assert_allclose(kept[0], [0.0, 0.0, 0.0])


def test_depth_loaders_agree_across_formats(tmp_path: Path) -> None:
    """A 16-bit PNG with its scale must load to the same values as the float array."""
    depth = np.array([[1.5, 2.0], [0.0, 3.25]], dtype=np.float32)

    npy = tmp_path / "d.npy"
    np.save(npy, depth)
    from_npy, valid_npy = load_depth(npy)
    np.testing.assert_allclose(from_npy, depth)
    # Zero means "no surface" and must not be reported as valid.
    assert not valid_npy[1, 0]
    assert valid_npy[0, 0]

    npz = tmp_path / "d.npz"
    np.savez(npz, depth=depth)
    from_npz, _ = load_depth(npz)
    np.testing.assert_allclose(from_npz, depth)

    png = tmp_path / "d.png"
    quantised = (depth * 1000).astype(np.uint16)
    try:
        import cv2

        cv2.imwrite(str(png), quantised)
    except ImportError:
        pytest.skip("no cv2 for PNG writing")
    from_png, _ = load_depth(png, depth_scale=1e-3)
    np.testing.assert_allclose(from_png, depth, atol=1e-6)
