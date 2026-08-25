# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
from PIL import Image

from threedgrut.datasets import gt_geometry

FLOAT_CHANNEL = None


def write_exr(path, arrays: dict) -> str:
    import Imath
    import OpenEXR

    height, width = next(iter(arrays.values())).shape
    header = OpenEXR.Header(width, height)
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {name: Imath.Channel(float_type) for name in arrays}
    output = OpenEXR.OutputFile(str(path), header)
    output.writePixels({name: np.ascontiguousarray(a, dtype=np.float32).tobytes() for name, a in arrays.items()})
    output.close()
    return str(path)


# --- channel negotiation -------------------------------------------------------------
# OB3D stores depth as either `V` or `B,G,R` and normals as either `X,Y,Z` or `B,G,R`
# depending on the scene, so a reader that assumes one naming fails on most of the dataset.


@pytest.mark.parametrize("channel", ["V", "R", "Y", "Z", "A"])
def test_single_channel_depth_names_are_all_accepted(tmp_path, channel) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = write_exr(tmp_path / f"d_{channel}.exr", {channel: values})

    depth = gt_geometry.read_gt_map(path, 1)

    assert depth.shape == (2, 3, 1)
    np.testing.assert_array_equal(depth[..., 0], values)


def test_depth_accepts_an_unknown_name_when_it_is_the_only_channel(tmp_path) -> None:
    values = np.full((2, 2), 3.0, dtype=np.float32)
    path = write_exr(tmp_path / "d.exr", {"depth": values})

    np.testing.assert_array_equal(gt_geometry.read_gt_map(path, 1)[..., 0], values)


def test_depth_rejects_an_ambiguous_multi_channel_file(tmp_path) -> None:
    zeros = np.zeros((2, 2), dtype=np.float32)
    path = write_exr(tmp_path / "d.exr", {"foo": zeros, "bar": zeros})

    with pytest.raises(ValueError, match="Cannot identify 1 ground-truth channel"):
        gt_geometry.read_gt_map(path, 1)


def test_bgr_depth_reads_the_red_channel(tmp_path) -> None:
    """The 7 OB3D scenes storing depth as B,G,R replicate the value across all three."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = write_exr(tmp_path / "d.exr", {"B": values, "G": values, "R": values})

    np.testing.assert_array_equal(gt_geometry.read_gt_map(path, 1)[..., 0], values)


def test_xyz_and_bgr_normals_yield_the_same_vectors(tmp_path) -> None:
    """`X,Y,Z` and `R,G,B` name the same axes in that order; conflating them transposes x and z."""
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0]], dtype=np.float32)
    z = np.array([[0.0, 0.0]], dtype=np.float32)

    xyz = gt_geometry.read_gt_map(write_exr(tmp_path / "n1.exr", {"X": x, "Y": y, "Z": z}), 3)
    bgr = gt_geometry.read_gt_map(write_exr(tmp_path / "n2.exr", {"R": x, "G": y, "B": z}), 3)

    np.testing.assert_array_equal(xyz, bgr)
    np.testing.assert_array_equal(xyz[0, 0], [1.0, 0.0, 0.0])


def test_normals_are_not_matched_by_alphabetical_fallback(tmp_path) -> None:
    zeros = np.zeros((1, 1), dtype=np.float32)
    path = write_exr(tmp_path / "n.exr", {"A": zeros, "B": zeros, "C": zeros})

    with pytest.raises(ValueError, match="Cannot identify 3 ground-truth channel"):
        gt_geometry.read_gt_map(path, 3)


# --- other container formats ---------------------------------------------------------


def test_npy_depth_round_trips(tmp_path) -> None:
    values = np.array([[1.5, 2.5]], dtype=np.float32)
    path = tmp_path / "d.npy"
    np.save(path, values)

    np.testing.assert_array_equal(gt_geometry.read_gt_map(str(path), 1)[..., 0], values)


def test_integer_depth_is_rejected_rather_than_guessed(tmp_path) -> None:
    """A quantized depth has no recoverable scale; assuming one corrupts every metric."""
    path = tmp_path / "d.png"
    Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.uint16)).save(path)

    with pytest.raises(ValueError, match="ambiguous"):
        gt_geometry.read_gt_map(str(path), 1)


def test_integer_normals_are_decoded_from_the_unit_range(tmp_path) -> None:
    path = tmp_path / "n.png"
    Image.fromarray(np.array([[[255, 128, 0]]], dtype=np.uint8), mode="RGB").save(path)

    normal = gt_geometry.read_gt_map(str(path), 3)

    np.testing.assert_allclose(normal[0, 0], [1.0, 128.0 / 255.0 * 2 - 1, -1.0], atol=1e-6)


def test_wrong_channel_count_is_reported(tmp_path) -> None:
    path = tmp_path / "n.npy"
    np.save(path, np.zeros((2, 2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected a 3-channel"):
        gt_geometry.read_gt_map(str(path), 3)


# --- sky sentinel --------------------------------------------------------------------


def test_validity_excludes_the_sentinel_and_non_finite_values() -> None:
    depth = np.array([1.0, 1e10, np.nan, np.inf, 0.0, -1.0, 1e8], dtype=np.float32)

    np.testing.assert_array_equal(
        gt_geometry.depth_validity(depth),
        [True, False, False, False, False, False, True],
    )


# --- resampling ----------------------------------------------------------------------


def test_resize_is_a_no_op_at_the_requested_size() -> None:
    array = np.random.default_rng(0).random((4, 6, 1)).astype(np.float32)
    assert gt_geometry.resize_gt_map(array, 4, 6) is array


def test_resize_never_invents_intermediate_values() -> None:
    """Nearest neighbour keeps the sky sentinel from bleeding into its neighbours."""
    depth = np.array([[1.0, 1e10], [1.0, 1e10]], dtype=np.float32)[..., None]

    resized = gt_geometry.resize_gt_map(depth, 4, 4)

    assert resized.shape == (4, 4, 1)
    assert set(np.unique(resized)) <= {1.0, 1e10}


def test_resize_preserves_channels_and_orientation() -> None:
    array = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    resized = gt_geometry.resize_gt_map(array, 4, 4)

    assert resized.shape == (4, 4, 3)
    np.testing.assert_array_equal(resized[0, 0], array[0, 0])
    np.testing.assert_array_equal(resized[-1, -1], array[-1, -1])


# --- world normalization -------------------------------------------------------------
# The renderer works in the normalized world, so metric ground truth has to follow it.


def similarity(scale: float, axis=(0.0, 0.0, 1.0), angle: float = 0.7, translation=(1.0, 2.0, 3.0)):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = np.asarray(translation) * scale
    return transform, rotation


def test_similarity_scale_recovers_the_uniform_factor() -> None:
    transform, _ = similarity(0.25)
    assert gt_geometry.similarity_scale(transform) == pytest.approx(0.25)


def test_non_uniform_scaling_is_rejected() -> None:
    transform = np.diag([1.0, 2.0, 3.0, 1.0])
    with pytest.raises(ValueError, match="not a similarity"):
        gt_geometry.similarity_scale(transform)


def test_depth_is_rescaled_but_the_sentinel_is_left_recognizable() -> None:
    transform, _ = similarity(0.25)
    depth = np.array([[4.0, 1e10]], dtype=np.float32)[..., None]

    scaled = gt_geometry.transform_gt_depth(depth, transform)

    assert scaled[0, 0, 0] == pytest.approx(1.0)
    # Scaling the sentinel would push it below the validity threshold and turn the sky
    # into a surface 2.5e9 units away.
    assert not gt_geometry.depth_validity(scaled)[0, 1, 0]


def test_normals_are_rotated_by_the_transform() -> None:
    transform, rotation = similarity(0.25)
    normal = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32)

    rotated = gt_geometry.transform_gt_normal(normal, transform)

    np.testing.assert_allclose(rotated[0, 0], rotation @ np.array([0.0, 0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(rotated[0, 1], rotation @ np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_normal_rotation_is_scale_invariant() -> None:
    """A similarity's scale cannot change a direction, so it must not affect the result."""
    normal = np.array([[[0.3, -0.5, 0.81]]], dtype=np.float32)
    small, _ = similarity(0.01)
    large, _ = similarity(100.0)

    np.testing.assert_allclose(
        gt_geometry.transform_gt_normal(normal, small),
        gt_geometry.transform_gt_normal(normal, large),
        atol=1e-6,
    )


def test_transformed_normals_are_unit_length() -> None:
    rng = np.random.default_rng(0)
    normal = gt_geometry.normalize_gt_normal(rng.normal(size=(8, 8, 3)).astype(np.float32))
    transform, _ = similarity(3.0, axis=(1.0, 2.0, 3.0), angle=1.1)

    rotated = gt_geometry.transform_gt_normal(normal, transform)

    np.testing.assert_allclose(np.linalg.norm(rotated, axis=-1), 1.0, atol=1e-6)


def test_missing_normals_stay_zero_instead_of_gaining_an_orientation() -> None:
    normal = np.zeros((1, 1, 3), dtype=np.float32)
    transform, _ = similarity(2.0)

    np.testing.assert_array_equal(gt_geometry.transform_gt_normal(normal, transform), np.zeros((1, 1, 3)))
    np.testing.assert_array_equal(gt_geometry.normalize_gt_normal(normal), np.zeros((1, 1, 3)))


# --- path discovery ------------------------------------------------------------------


def test_paths_pair_rgb_images_with_their_ground_truth(tmp_path) -> None:
    os.makedirs(tmp_path / "depths")
    (tmp_path / "depths" / "00000_depth.exr").touch()
    (tmp_path / "depths" / "00001_depth.exr").touch()
    images = ["/somewhere/images/00000_rgb.png", "/somewhere/images/00001_rgb.png"]

    paths = gt_geometry.find_gt_paths(images, str(tmp_path), "depths", "depth")

    assert [os.path.basename(p) for p in paths] == ["00000_depth.exr", "00001_depth.exr"]


def test_a_stem_without_the_rgb_suffix_still_matches(tmp_path) -> None:
    os.makedirs(tmp_path / "depths")
    (tmp_path / "depths" / "frame7_depth.npy").touch()

    paths = gt_geometry.find_gt_paths(["/x/frame7.jpg"], str(tmp_path), "depths", "depth")

    assert os.path.basename(paths[0]) == "frame7_depth.npy"


def test_a_frame_without_ground_truth_stays_none(tmp_path) -> None:
    os.makedirs(tmp_path / "depths")
    (tmp_path / "depths" / "00000_depth.exr").touch()
    images = ["/x/00000_rgb.png", "/x/00001_rgb.png"]

    paths = gt_geometry.find_gt_paths(images, str(tmp_path), "depths", "depth")

    assert paths[1] is None


def test_exr_wins_over_other_extensions(tmp_path) -> None:
    os.makedirs(tmp_path / "depths")
    (tmp_path / "depths" / "a_depth.png").touch()
    (tmp_path / "depths" / "a_depth.exr").touch()

    paths = gt_geometry.find_gt_paths(["/x/a_rgb.png"], str(tmp_path), "depths", "depth")

    assert paths[0].endswith(".exr")


# --- declared conventions ------------------------------------------------------------


def test_a_scene_without_metadata_is_accepted(tmp_path) -> None:
    gt_geometry.validate_scene_conventions(str(tmp_path))


def test_matching_conventions_are_accepted(tmp_path) -> None:
    (tmp_path / "conversion.json").write_text(
        json.dumps(
            {
                "depth_convention": "Euclidean ray distance inherited from OB3D",
                "normal_convention": "World-space XYZ inherited from OB3D",
            }
        )
    )
    gt_geometry.validate_scene_conventions(str(tmp_path))


def test_a_different_depth_convention_is_refused(tmp_path) -> None:
    """Silently evaluating plane-distance depth as ray distance would look plausible."""
    (tmp_path / "conversion.json").write_text(json.dumps({"depth_convention": "distance to image plane"}))

    with pytest.raises(ValueError, match="depth_convention"):
        gt_geometry.validate_scene_conventions(str(tmp_path))


def test_unreadable_metadata_is_not_treated_as_a_mismatch(tmp_path) -> None:
    (tmp_path / "conversion.json").write_text("{not json")
    gt_geometry.validate_scene_conventions(str(tmp_path))
