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

import glob
import hashlib
import math
import os
import platform
import sys
from typing import Iterable, Sequence

import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

# Escape hatch: set to 1 to force Slang regeneration even when the stamp is current.
_FORCE_SLANG_REBUILD_ENV = "THREEDGRUT_FORCE_SLANG_REBUILD"


def variant_digest(defines: Sequence[str], extra: Sequence[str] = ()) -> str:
    """Short stable hash identifying one compiled kernel variant.

    The digest covers every compile-time define that selects a code path (normals on/off,
    surfel vs ellipsoid, feature dimensions, ...). Keying build artifacts by it keeps
    variants from sharing a build directory, which would otherwise silently reuse a stale
    binary when the configuration changes, or corrupt each other when two configurations
    are built concurrently.
    """
    payload = "\n".join([*sorted(defines), *sorted(extra)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def variant_build_directory(
    name: str,
    defines: Sequence[str],
    extra: Sequence[str] = (),
    label: str = "",
    verbose: bool = True,
) -> str:
    """Return a per-variant build directory nested under torch's extension directory.

    The extension *name* is left untouched, so the loaded module keeps its import name and
    `TORCH_EXTENSION_NAME` stays valid; only the directory holding the objects, the shared
    library and torch's JIT lock file is specialized. Deriving the directory from the full
    flag set rather than from a hand-picked subset means a newly introduced define cannot
    be forgotten here and silently share a build with its opposite.

    `label` is a purely cosmetic prefix to keep the directories recognizable on disk.
    """
    root = torch.utils.cpp_extension._get_build_directory(name, verbose=verbose)
    digest = variant_digest(defines, extra)
    path = os.path.join(root, f"{label}-{digest}" if label else digest)
    os.makedirs(path, exist_ok=True)
    return path


def _slang_source_fingerprint(kernel_files: Iterable[str], include_paths: Iterable[str]) -> str:
    """Hash every Slang source that could take part in the compilation.

    The entry files `#include` other `.slang` modules, so hashing only the entry points
    would miss edits to the included ones and hand back a stale kernel. Hashing all
    reachable `.slang` files is cheap (a few hundred KB) and avoids that trap.
    """
    paths = set(kernel_files)
    for include_path in include_paths:
        paths.update(glob.glob(os.path.join(include_path, "**", "*.slang"), recursive=True))

    digest = hashlib.sha1()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        try:
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except OSError:
            # A missing file is part of the fingerprint too: if it reappears the hash changes.
            digest.update(b"<missing>")
    return digest.hexdigest()


def compile_slang_kernel(
    kernel_files: list[str],
    output_file: str,
    defines: list[str],
    include_paths: list[str],
) -> str:
    """Generate CUDA from Slang, skipping the work when the output is already current.

    `output_file` should live in a per-variant directory (see `variant_build_directory`):
    the generated code depends on `defines`, so a shared path would make two variants
    overwrite each other's header.
    """
    import importlib
    import subprocess

    # Skip the (not exactly cheap) slangc invocation when nothing that feeds it changed.
    stamp_file = output_file + ".stamp"
    stamp = "\n".join(
        [
            _slang_source_fingerprint(kernel_files, include_paths),
            *sorted(defines),
            *sorted(include_paths),
            *sorted(kernel_files),
        ]
    )
    force = os.environ.get(_FORCE_SLANG_REBUILD_ENV, "0") == "1"
    if not force and os.path.isfile(output_file) and os.path.isfile(stamp_file):
        with open(stamp_file, "r", encoding="utf-8") as handle:
            if handle.read() == stamp:
                return output_file

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    slang_build_env = os.environ.copy()
    slang_build_env["PATH"] += ";" if os.name == "nt" else ":"

    try:
        slang_mod = importlib.import_module("slangtorch")
        slang_build_env["PATH"] += os.path.join(os.path.dirname(slang_mod.__file__), "bin")
    except ImportError:
        print("Slangtorch not found, assuming slangc is in the path")

    subprocess.check_call(
        [
            "slangc",
            "-target",
            "cuda",
            *(arg for path in include_paths for arg in ("-I", path)),
            "-line-directive-mode",
            "none",
            "-matrix-layout-row-major",  # NB : this is required for cuda target
            "-O2",
            *defines,
            *kernel_files,
            "-o",
            output_file,
        ],
        env=slang_build_env,
    )

    with open(stamp_file, "w", encoding="utf-8") as handle:
        handle.write(stamp)

    return output_file


def load(
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    with_cuda=True,
    verbose=True,
    *args,
    **kwargs,
):

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == "nt":

        def find_cl_path():
            import glob

            for arch in [" (x86)", ""]:
                for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                    paths = sorted(
                        glob.glob(
                            r"C:\Program Files\%s\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64"
                            % (arch, edition)
                        ),
                        reverse=True,
                    )
                    if paths:
                        return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

    elif os.name == "posix":
        pass

    # Compiler flags.
    cflags = [
        "-DNVDR_TORCH",
    ]
    # Add Windows-specific flags
    if os.name == "nt":
        cflags.append("/DNOMINMAX")

    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags = [
        "-DNVDR_TORCH",
        "-std=c++17",
        "--extended-lambda",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fno-strict-aliasing",
        "-diag-suppress=1444",
        "-diag-suppress=3287",
        # "-Wdeprecated-declarations",
    ]
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    # Linker options.
    if os.name == "posix":
        _cuda_arch = f"{platform.machine()}-linux"
        ldflags = [
            # NOTE: ad-hoc fix for CUDA 12.8.1
            f"-L{os.path.join(CUDA_HOME, 'lib', 'stubs')}",
            f"-L{os.path.join(CUDA_HOME, 'targets', _cuda_arch, 'lib')}",
            f"-L{os.path.join(CUDA_HOME, 'targets', _cuda_arch, 'lib', 'stubs')}",
            "-lcuda",
            "-lnvrtc",
        ]
    elif os.name == "nt":
        ldflags = [
            "cuda.lib",
            "advapi32.lib",
            "nvrtc.lib",
        ]
    if extra_ldflags is not None:
        ldflags += extra_ldflags

    # Include paths.
    include_paths = []

    # Add special CUDA include paths
    if os.path.isdir(os.path.join(CUDA_HOME, "targets")):
        for arch in os.listdir(os.path.join(CUDA_HOME, "targets")):
            if os.path.isdir(p := os.path.join(CUDA_HOME, "targets", arch, "include")):
                include_paths.append(p)

    if extra_include_paths is not None:
        include_paths += extra_include_paths

    # Load
    module = torch.utils.cpp_extension.load(
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        with_cuda=with_cuda,
        verbose=verbose,
        *args,
        **kwargs,
    )

    # Explicitly register module in sys.modules for compatibility with pybind11 3.x
    # In pybind11 3.0+, exec_module() no longer auto-registers modules in sys.modules,
    # which breaks subsequent `import module_name` statements.
    # This is safe for pybind11 2.x as well (no-op since same object is already registered).
    module_name = kwargs.get("name")
    if module_name is not None:
        sys.modules[module_name] = module

    return module
