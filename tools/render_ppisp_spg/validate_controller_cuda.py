#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical sanity check for ppisp_controller.cu.

Mirrors ``validate_controller.py`` (which exercises the slang variant via
slangpy) but dispatches the CUDA kernel that SPG/Kit actually executes.

Two modes:

* No ``--checkpoint``: build a torch ``_PPISPController`` with random
  weights (matching ``ppisp._PPISPController``'s architecture) and a
  synthetic HDR image. Pure numerical sanity check.

* ``--checkpoint runs/.../ckpt_last.pt``: load the trained PPISP module
  from the checkpoint and validate every per-camera controller against
  the gaussian-rendered HDR for one frame per camera, mirroring the
  controller-drift section of ``validate_trained.py`` but on CUDA.

Either way the harness:
  1. Computes the PyTorch reference forward pass (9 floats).
  2. Builds a small standalone CUDA runner from
     ``_cuda_controller_runner.cu`` (which #include's ppisp_controller.cu),
     using ``nvcc`` so the same texture/surface object setup SPG uses
     is exercised.
  3. Pipes the inputs through binary files, reads back the 9-float
     output, and compares to the torch reference.

Requires only ``torch``, ``numpy`` and a working ``nvcc``. No cupy /
slangpy dependency.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_writer_helpers():
    """Return ``(EXPECTED_SIZES, flatten_controller_weights)``.

    Synthetic mode wants to run without booting the heavy ``threedgrut``
    package (it pulls CUDA optimizers etc.), so we import the writer
    module directly. Checkpoint mode needs the real package, so the
    direct import is only used as a fallback.
    """
    # Fast path: real package available (checkpoint mode requires it).
    try:
        from threedgrut.export.usd.writers.ppisp_controller_writer import (  # noqa: F401
            EXPECTED_SIZES, flatten_controller_weights,
        )
        return EXPECTED_SIZES, flatten_controller_weights
    except Exception:
        pass

    # Standalone path: load the writer file directly via spec loader,
    # stubbing only the parts of threedgrut that the writer touches.
    import importlib.util as _ilu
    import types as _types
    import dataclasses as _dc

    for _pkg in (
        "threedgrut",
        "threedgrut.export",
        "threedgrut.export.usd",
        "threedgrut.export.usd.writers",
    ):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = _types.ModuleType(_pkg)

    _stage_utils_stub = _types.ModuleType("threedgrut.export.usd.stage_utils")

    @_dc.dataclass
    class _NamedSerialized:
        filename: str
        serialized: bytes

    _stage_utils_stub.NamedSerialized = _NamedSerialized
    sys.modules["threedgrut.export.usd.stage_utils"] = _stage_utils_stub

    _ppisp_spg_stub = _types.ModuleType("threedgrut.export.usd.ppisp_spg")
    _ppisp_spg_stub._SPG_DIR = (
        Path(__file__).resolve().parents[2] / "threedgrut/export/usd/ppisp_spg"
    )
    sys.modules["threedgrut.export.usd.ppisp_spg"] = _ppisp_spg_stub

    _writer_path = (
        Path(__file__).resolve().parents[2]
        / "threedgrut/export/usd/writers/ppisp_controller_writer.py"
    )
    _spec = _ilu.spec_from_file_location(
        "threedgrut.export.usd.writers.ppisp_controller_writer", str(_writer_path)
    )
    _writer_mod = _ilu.module_from_spec(_spec)
    sys.modules["threedgrut.export.usd.writers.ppisp_controller_writer"] = _writer_mod
    _spec.loader.exec_module(_writer_mod)
    return _writer_mod.EXPECTED_SIZES, _writer_mod.flatten_controller_weights


EXPECTED_SIZES, flatten_controller_weights = _load_writer_helpers()


logger = logging.getLogger("validate_controller_cuda")


REPO_ROOT     = Path(__file__).resolve().parents[2]
CONTROLLER_CU = REPO_ROOT / "threedgrut/export/usd/ppisp_spg/ppisp_controller.cu"
RUNNER_CU     = Path(__file__).resolve().parent / "_cuda_controller_runner.cu"


# ---------------------------------------------------------------------------
# Synthetic torch controller (lets the harness run without ppisp installed).
# ---------------------------------------------------------------------------


def _make_test_controller(seed: int = 0):
    import torch
    from torch import nn

    class _Controller(nn.Module):
        def __init__(self):
            super().__init__()
            cfd = EXPECTED_SIZES["cnn_feature_dim"]
            grid = (EXPECTED_SIZES["pool_grid_h"], EXPECTED_SIZES["pool_grid_w"])
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1),
                nn.MaxPool2d(EXPECTED_SIZES["input_downsampling"],
                             stride=EXPECTED_SIZES["input_downsampling"]),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, cfd, kernel_size=1),
                nn.AdaptiveAvgPool2d(grid),
                nn.Flatten(),
            )
            in_dim = cfd * grid[0] * grid[1] + 1
            hd = EXPECTED_SIZES["mlp_hidden_dim"]
            self.mlp_trunk = nn.Sequential(
                nn.Linear(in_dim, hd), nn.ReLU(inplace=True),
                nn.Linear(hd, hd),     nn.ReLU(inplace=True),
                nn.Linear(hd, hd),     nn.ReLU(inplace=True),
            )
            self.exposure_head = nn.Linear(hd, 1)
            self.color_head = nn.Linear(hd, EXPECTED_SIZES["color_params_per_frame"])

        def forward(self, rgb, prior_exposure):
            features = self.cnn_encoder(rgb.permute(2, 0, 1).unsqueeze(0).detach())
            features = torch.cat([features.squeeze(0), prior_exposure], dim=0)
            hidden = self.mlp_trunk(features)
            return self.exposure_head(hidden).squeeze(-1), self.color_head(hidden)

    torch.manual_seed(seed)
    ctrl = _Controller().eval()
    with torch.no_grad():
        for p in ctrl.parameters():
            p.normal_(0.0, 0.01)
    return ctrl


def _torch_reference(ctrl, hdr_image: np.ndarray, prior_exposure: float) -> np.ndarray:
    import torch
    device = next(ctrl.parameters()).device
    rgb = torch.from_numpy(hdr_image).float().to(device)
    pe = torch.tensor([prior_exposure], dtype=torch.float32, device=device)
    with torch.no_grad():
        exposure, color = ctrl(rgb, pe)
    return np.concatenate([
        np.array([float(exposure)], dtype=np.float32),
        color.detach().cpu().numpy().astype(np.float32),
    ])


# ---------------------------------------------------------------------------
# CUDA runner: build once via nvcc, then drive it per-frame.
# ---------------------------------------------------------------------------


def _find_nvcc() -> str:
    candidates: List[str] = []
    if "CUDA_HOME" in os.environ:
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    if shutil.which("nvcc"):
        candidates.append(shutil.which("nvcc"))
    # Mamba/conda envs and standard CUDA install paths.
    candidates += [
        "/usr/local/cuda/bin/nvcc",
        str(Path(sys.executable).parent / "nvcc"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise RuntimeError(
        "could not locate nvcc; set CUDA_HOME or put nvcc on PATH"
    )


def _build_runner(work: Path, *, sm_arch: str = "sm_75", verbose: bool = False) -> Path:
    """Compile _cuda_controller_runner.cu -> binary in ``work``."""
    nvcc = _find_nvcc()
    out = work / "controller_runner"
    cmd = [
        nvcc, "-O2", "-std=c++17",
        f"-arch={sm_arch}",
        # The runner #includes "ppisp_controller.cu" -- give nvcc the
        # path so the unqualified include resolves.
        f"-I{CONTROLLER_CU.parent}",
        str(RUNNER_CU),
        "-o", str(out),
    ]
    logger.info("Building CUDA runner: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"nvcc failed (exit {proc.returncode})")
    if verbose:
        sys.stderr.write(proc.stderr)
    return out


def _run_cuda_controller(
    runner: Path,
    work: Path,
    tag: str,
    hdr_image: np.ndarray,
    weights: np.ndarray,
    prior_exposure: float,
) -> np.ndarray:
    h, w = hdr_image.shape[:2]
    if hdr_image.shape[2] == 3:
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., :3] = hdr_image
        rgba[..., 3] = 1.0
    else:
        rgba = hdr_image.astype(np.float32, copy=False)

    hdr_path     = work / f"{tag}.hdr.bin"
    weights_path = work / f"{tag}.weights.bin"
    out_path     = work / f"{tag}.out.bin"

    np.ascontiguousarray(rgba.astype(np.float32, copy=False)).tofile(hdr_path)
    np.ascontiguousarray(weights.astype(np.float32, copy=False).reshape(-1)).tofile(weights_path)

    cmd = [str(runner), str(w), str(h), str(prior_exposure),
           str(hdr_path), str(weights_path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"controller_runner failed (exit {proc.returncode})")

    return np.fromfile(out_path, dtype=np.float32, count=9)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _validate_synthetic(args, work: Path, runner: Path) -> int:
    ctrl = _make_test_controller(args.seed)
    rng = np.random.default_rng(args.seed)
    hdr = (rng.random((args.height, args.width, 3), dtype=np.float32) * 0.8 + 0.1)

    expected = _torch_reference(ctrl, hdr, args.prior)
    weights  = flatten_controller_weights(ctrl)
    actual   = _run_cuda_controller(runner, work, "synth", hdr, weights, args.prior)

    diff = np.abs(actual - expected)
    print(f"[synthetic seed={args.seed} {args.width}x{args.height}]")
    print(f"  reference: {expected}")
    print(f"  cuda:      {actual}")
    print(f"  abs diff:  {diff}")
    print(f"  max abs diff: {diff.max():.6g} (tol={args.tol})")
    return 0 if diff.max() <= args.tol else 1


def _validate_checkpoint(args, work: Path, runner: Path) -> int:
    """Mirror validate_trained.py's controller-drift section, but CUDA-side."""
    import torch
    from threedgrut.render import Renderer  # noqa: E402

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --checkpoint mode.")

    renderer = Renderer.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        path=str(args.data_path) if args.data_path else "",
        out_dir=str(work / "_renderer_unused"),
        save_gt=False,
        computes_extra_metrics=False,
    )
    model = renderer.model
    pp = renderer.post_processing
    if pp is None or type(pp).__name__ != "PPISP":
        raise SystemExit("Checkpoint has no PPISP post-processing module.")
    if not getattr(pp.config, "use_controller", False):
        raise SystemExit("PPISP was trained without a controller; nothing to validate.")
    controllers = pp.controllers

    val_dataset = renderer.dataset
    val_dataloader = renderer.dataloader

    cam_names = (
        list(val_dataset.get_camera_names())
        if hasattr(val_dataset, "get_camera_names") else ["cam_0"]
    )

    seen_cams: set[int] = set()
    rows: List[Tuple[int, str, np.ndarray, np.ndarray, float]] = []
    max_diff = 0.0

    print(f"[checkpoint {args.checkpoint}]")
    print(f"  cameras: {len(cam_names)}, max-frames-per-camera={args.max_frames}")

    for frame_idx, batch in enumerate(val_dataloader):
        cam_idx = (val_dataset.get_camera_idx(frame_idx)
                   if hasattr(val_dataset, "get_camera_idx") else 0)

        if args.max_frames is not None and cam_idx in seen_cams:
            continue

        gpu_batch = val_dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            outputs = model(gpu_batch)
            hdr_t = outputs["pred_rgb"][0]
        if hdr_t.abs().max().item() < 1e-6:
            gt = gpu_batch.rgb_gt
            hdr_t = gt[0] if gt.dim() == 4 else gt
        hdr_np = hdr_t.detach().cpu().numpy().astype(np.float32)

        ctrl = controllers[cam_idx]
        torch_out = _torch_reference(ctrl, hdr_np, prior_exposure=0.0)
        weights   = flatten_controller_weights(ctrl)
        cuda_out  = _run_cuda_controller(runner, work, f"cam{cam_idx}_f{frame_idx}",
                                         hdr_np, weights, prior_exposure=0.0)
        diff      = np.abs(cuda_out - torch_out)
        rows.append((cam_idx, cam_names[cam_idx] if cam_idx < len(cam_names) else f"cam_{cam_idx}",
                     torch_out, cuda_out, float(diff.max())))
        max_diff = max(max_diff, float(diff.max()))
        seen_cams.add(cam_idx)
        if args.max_frames is not None and len(seen_cams) >= args.max_frames:
            break

    if not rows:
        raise SystemExit("No validation frames produced.")

    # Print one row per camera.
    print(f"  {'cam':>4s} {'name':<24s} {'max|Δ|':>10s}  torch[0..3] -> cuda[0..3]")
    for cam_idx, name, torch_out, cuda_out, mxd in rows:
        print(f"  {cam_idx:4d} {name:<24s} {mxd:10.4g}  "
              f"[{torch_out[0]:+.4f},{torch_out[1]:+.4f},{torch_out[2]:+.4f}] -> "
              f"[{cuda_out[0]:+.4f},{cuda_out[1]:+.4f},{cuda_out[2]:+.4f}]")
    print(f"  overall max abs diff = {max_diff:.6g} (tol={args.tol})")
    return 0 if max_diff <= args.tol else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1.0e-3,
                        help="abs tol per output element")
    parser.add_argument("--cu", type=Path, default=CONTROLLER_CU,
                        help="Path to ppisp_controller.cu (defaults to in-repo file)")
    parser.add_argument("--sm-arch", type=str, default="sm_75",
                        help="-arch flag for nvcc (default sm_75)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="If set, validate per-camera controllers from this "
                             "checkpoint against gaussian-rendered HDR.")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Override the dataset path stored in the checkpoint.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="In --checkpoint mode, max distinct cameras to test.")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the working directory after the run.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    work = Path(tempfile.mkdtemp(prefix="ppisp_cuda_val_"))
    try:
        runner = _build_runner(work, sm_arch=args.sm_arch, verbose=args.verbose >= 2)
        if args.checkpoint is not None:
            return _validate_checkpoint(args, work, runner)
        return _validate_synthetic(args, work, runner)
    finally:
        if args.keep_tmp:
            print(f"  (kept tmp at {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
