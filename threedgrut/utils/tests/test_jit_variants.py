# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import pytest

from threedgrut.utils import jit


def test_variant_digest_is_stable_and_order_insensitive() -> None:
    defines = ["-DA=1", "-DB=false"]
    assert jit.variant_digest(defines) == jit.variant_digest(defines)
    assert jit.variant_digest(defines) == jit.variant_digest(list(reversed(defines)))


def test_variant_digest_separates_code_paths() -> None:
    """A flag that selects a code path must not share a digest with its opposite."""
    on = ["-DGAUSSIAN_PARTICLE_ENABLE_NORMAL=true", "-DGAUSSIAN_PARTICLE_SURFEL=false"]
    off = ["-DGAUSSIAN_PARTICLE_ENABLE_NORMAL=false", "-DGAUSSIAN_PARTICLE_SURFEL=false"]
    surfel = ["-DGAUSSIAN_PARTICLE_ENABLE_NORMAL=false", "-DGAUSSIAN_PARTICLE_SURFEL=true"]

    digests = {jit.variant_digest(on), jit.variant_digest(off), jit.variant_digest(surfel)}
    assert len(digests) == 3


def test_variant_digest_accounts_for_extra_flags() -> None:
    defines = ["-DA=1"]
    assert jit.variant_digest(defines) != jit.variant_digest(defines, extra=["-DSLANG_CUDA_ENABLE_HALF=1"])


def test_variant_build_directory_is_distinct_per_variant(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))

    on = jit.variant_build_directory("lib_test_cc", ["-DENABLE=true"], verbose=False)
    off = jit.variant_build_directory("lib_test_cc", ["-DENABLE=false"], verbose=False)

    assert on != off
    assert os.path.isdir(on) and os.path.isdir(off)
    # Both nest under the same extension root, so the module import name is unaffected.
    assert os.path.dirname(on) == os.path.dirname(off)


def test_variant_build_directory_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    defines = ["-DENABLE=true"]

    first = jit.variant_build_directory("lib_test_cc", defines, verbose=False)
    second = jit.variant_build_directory("lib_test_cc", defines, verbose=False)

    assert first == second


class _RecordingSlangc:
    """Stand-in for `subprocess.check_call` that records invocations and writes the output."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, env=None):  # noqa: ANN001 - mirrors subprocess.check_call
        self.calls.append(list(argv))
        output_file = argv[argv.index("-o") + 1]
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("// generated\n")


@pytest.fixture()
def slang_sandbox(monkeypatch, tmp_path):
    """A fake slangc plus a minimal Slang source tree."""
    recorder = _RecordingSlangc()
    monkeypatch.setattr(subprocess, "check_call", recorder)
    monkeypatch.delenv("THREEDGRUT_FORCE_SLANG_REBUILD", raising=False)

    include_dir = tmp_path / "include"
    include_dir.mkdir()
    entry = include_dir / "entry.slang"
    entry.write_text("// entry\n")
    included = include_dir / "models" / "shared.slang"
    included.parent.mkdir()
    included.write_text("// shared\n")

    def compile(defines=("-DA=1",), output=None):
        return jit.compile_slang_kernel(
            kernel_files=[str(entry)],
            output_file=str(output or (tmp_path / "out" / "generated.cuh")),
            defines=list(defines),
            include_paths=[str(include_dir)],
        )

    return recorder, compile, included


def test_slang_compilation_is_skipped_when_current(slang_sandbox) -> None:
    recorder, compile, _ = slang_sandbox

    compile()
    compile()

    assert len(recorder.calls) == 1, "an unchanged input should not re-run slangc"


def test_slang_recompiles_when_defines_change(slang_sandbox) -> None:
    recorder, compile, _ = slang_sandbox

    compile(defines=["-DA=1"])
    compile(defines=["-DA=2"])

    assert len(recorder.calls) == 2


def test_slang_recompiles_when_an_included_source_changes(slang_sandbox) -> None:
    """Editing an `#include`d module must invalidate the stamp, or the kernel goes stale."""
    recorder, compile, included = slang_sandbox

    compile()
    included.write_text("// shared, edited\n")
    compile()

    assert len(recorder.calls) == 2


def test_slang_force_rebuild_env_overrides_the_stamp(slang_sandbox, monkeypatch) -> None:
    recorder, compile, _ = slang_sandbox

    compile()
    monkeypatch.setenv("THREEDGRUT_FORCE_SLANG_REBUILD", "1")
    compile()

    assert len(recorder.calls) == 2


def test_slang_output_directory_is_created(slang_sandbox, tmp_path) -> None:
    _, compile, _ = slang_sandbox

    output = tmp_path / "nested" / "deeper" / "generated.cuh"
    assert compile(output=output) == str(output)
    assert output.is_file()
