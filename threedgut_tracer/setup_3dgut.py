# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os

import torch

from threedgrut.utils import jit


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf):

    build_dir = torch.utils.cpp_extension._get_build_directory("lib3dgut_cc", verbose=True)

    include_paths = []
    prefix = os.path.dirname(__file__)
    include_paths.append(os.path.join(prefix, "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "include"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies"))
    include_paths.append(os.path.join(prefix, "..", "thirdparty", "tiny-cuda-nn", "dependencies", "fmt", "include"))
    include_paths.append(build_dir)

    # Compiler options.

    def to_cpp_bool(value):
        return "true" if value else "false"

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha * ut_alpha * (ut_d + ut_kappa))

    def _get_particle_feature_dim(conf):
        """Per-particle feature vector dimension (stored on each Gaussian). For nht, = nht_features.dim."""
        feature_type = conf.model.feature_type.lower()
        if feature_type == "sh":
            sh_degree = conf.model.progressive_training.max_n_features
            return 3 * ((sh_degree + 1) ** 2)
        elif feature_type == "nht":
            return conf.model.nht_features.dim
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def _get_num_interpolation_points(conf):
        """NHT: 1=center, 4=barycentric (tetrahedra or trisurfel). Per-point dim = dim // this."""
        feature_type = conf.model.feature_type.lower()
        if feature_type != "nht":
            return 1
        if _get_interpolation_support(conf) == 0:
            return 1
        return 4  # barycentric: tetrahedra (4 verts) or trisurfel (2 coplanar triangles, 4 verts)

    def _get_feature_activation_type(conf):
        """Feature activation: 0=none, 1=siren (sin), 2=sincos (sin+cos), 3=relu. From nht_features.activation."""
        feature_type = conf.model.feature_type.lower()
        if feature_type != "nht":
            return 0
        v = getattr(conf.model.nht_features, "activation", None)
        if v is None:
            return 0
        t = getattr(v, "type", "none")
        if isinstance(t, str):
            t = t.lower()
        if t == "none":
            return 0
        if t == "siren":
            return 1
        if t == "sincos":
            return 2
        if t == "relu":
            return 3
        raise ValueError(f"Unknown nht_features.activation.type: {t}")

    def _get_feature_activation_num_frequencies(conf):
        """Number of frequency bands for feature activation. 1 when type=none or relu."""
        act_type = _get_feature_activation_type(conf)
        if act_type == 0 or act_type == 3:  # none or relu
            return 1
        v = getattr(conf.model.nht_features, "activation", None)
        return int(getattr(v, "num_frequencies", 1)) if v else 1

    def _get_interp_point_feature_dim(conf):
        """Per-interpolation-point feature dim before activation: dim // num_interpolation_points."""
        feature_type = conf.model.feature_type.lower()
        if feature_type != "nht":
            return 3
        dim = conf.model.nht_features.dim
        num_points = _get_num_interpolation_points(conf)
        return dim // num_points

    def _get_ray_feature_dim(conf):
        """Per-ray feature dimension (decoder input): interp_point_dim * num_frequencies when activation on, else interp_point_dim."""
        feature_type = conf.model.feature_type.lower()
        if feature_type == "sh":
            return 3  # RGB output
        elif feature_type == "nht":
            interp_point_dim = _get_interp_point_feature_dim(conf)
            num_freq = _get_feature_activation_num_frequencies(conf)
            return interp_point_dim * num_freq
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def _get_feature_transform_type(conf):
        """Get feature transform type: 0=SH, 1=nht"""
        feature_type = conf.model.feature_type.lower()
        if feature_type == "sh":
            return 0
        elif feature_type == "nht":
            return 1
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def _get_interpolation_type(conf):
        """NHT: 0=barycentric (or center), 1=bezier. Merged with support."""
        feature_type = conf.model.feature_type.lower()
        if feature_type != "nht":
            return 0
        v = getattr(conf.model.nht_features, "interpolation_type", "none").lower()
        if v == "none":
            return 0
        if v == "barycentric":
            return 0
        if v == "bezier":
            return 1
        raise ValueError(f"Unknown nht_features.interpolation_type: {v}")

    def _get_interpolation_support(conf):
        """NHT: 0=center, 1=tetrahedra (gaussian), 2=triangle (trisurfel). From merged interpolation_type."""
        feature_type = conf.model.feature_type.lower()
        if feature_type != "nht":
            return 0
        v = getattr(conf.model.nht_features, "interpolation_type", "none").lower()
        if v == "none":
            return 0
        if v == "barycentric":
            primitive = getattr(conf.render, "primitive_type", "instances")
            return 2 if primitive == "trisurfel" else 1
        raise ValueError(f"Unknown nht_features.interpolation_type: {v}")

    defines = [
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={to_cpp_bool(conf.render.enable_normals)}",
        f"-DGAUSSIAN_PARTICLE_SURFEL={to_cpp_bool(conf.render.primitive_type=='trisurfel')}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={to_cpp_bool(conf.render.enable_hitcounts)}",
        # Feature-based radiance dimensions
        f"-DPARTICLE_FEATURE_DIM={_get_particle_feature_dim(conf)}",
        f"-DRAY_FEATURE_DIM={_get_ray_feature_dim(conf)}",
        f"-DFEATURE_TRANSFORM_TYPE={_get_feature_transform_type(conf)}",
        f"-DFEATURE_INTERPOLATION_TYPE={_get_interpolation_type(conf)}",
        f"-DFEATURE_INTERPOLATION_SUPPORT={_get_interpolation_support(conf)}",
        f"-DFEATURE_ACTIVATION_TYPE={_get_feature_activation_type(conf)}",
        f"-DFEATURE_ACTIVATION_NUM_FREQUENCIES={_get_feature_activation_num_frequencies(conf)}",
        f"-DINTERP_POINT_FEATURE_DIM={_get_interp_point_feature_dim(conf)}",
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        f"-DFEATURE_OUTPUT_HALF={to_cpp_bool(getattr(conf.render.splat, 'feature_output_half', False))}",
        # -- Unscented Transform --
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={conf.render.splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={to_cpp_bool(conf.render.splat.ut_require_all_sigma_points_valid)}",
        # -- Culling --
        f"-DGAUSSIAN_RECT_BOUNDING={to_cpp_bool(conf.render.splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={to_cpp_bool(conf.render.splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={to_cpp_bool(conf.render.splat.tile_based_culling)}",
    ]

    cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    cuda_cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-use_fast_math",
        "-O3",
        *defines,
    ]

    # List of sources.
    source_files = [
        "src/splatRaster.cpp",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels
    # TODO: do not overwrite files, use config hash to register the needed version
    import importlib
    import subprocess

    slang_mod = importlib.import_module("slangtorch")
    slang_dir = os.path.dirname(slang_mod.__file__)

    slang_build_env = os.environ
    slang_build_env["PATH"] += ";" if os.name == "nt" else ":"
    slang_build_env["PATH"] += os.path.join(slang_dir, "bin")
    slang_build_inc_dir = os.path.join(os.path.dirname(__file__), "include", "3dgut")

    slang_out_path = os.path.join(build_dir, "threedgutSlang.cuh")
    slang_tmp_path = slang_out_path + ".tmp"
    subprocess.check_call(
        [
            "slangc",
            "-target",
            "cuda",
            "-I",
            os.path.join(os.path.dirname(__file__), "include"),
            "-I",
            os.path.join(os.path.dirname(__file__), "..", "threedgrt_tracer", "include"),
            "-line-directive-mode",
            "none",
            "-matrix-layout-row-major",  # NB : this is required for cuda target
            "-Wno-41018",
            "-O2",
            *defines,
            f"{os.path.join(slang_build_inc_dir,'threedgut.slang')}",
            "-o",
            slang_tmp_path,
        ],
        env=slang_build_env,
    )
    # Only overwrite if content changed: preserves timestamp and avoids spurious ninja rebuilds
    import shutil
    if not os.path.exists(slang_out_path) or open(slang_tmp_path, "rb").read() != open(slang_out_path, "rb").read():
        shutil.move(slang_tmp_path, slang_out_path)
    else:
        os.remove(slang_tmp_path)

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    return jit.load(
        name="lib3dgut_cc",
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )
