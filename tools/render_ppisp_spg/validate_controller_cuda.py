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
import dataclasses
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


REPO_ROOT      = Path(__file__).resolve().parents[2]
SPG_DIR        = REPO_ROOT / "threedgrut/export/usd/ppisp_spg"
CONTROLLER_CU  = SPG_DIR / "ppisp_controller.cu"
RUNNER_CU      = Path(__file__).resolve().parent / "_cuda_controller_runner.cu"
PIPELINE_RUNNER_CU = Path(__file__).resolve().parent / "_cuda_controller_pipeline_runner.cu"


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


def _build_runner(
    work: Path,
    src: Path,
    name: str,
    *,
    sm_arch: str = "sm_75",
    verbose: bool = False,
) -> Path:
    """Compile a controller runner .cu file to a binary in ``work``."""
    nvcc = _find_nvcc()
    out = work / name
    cmd = [
        nvcc, "-O2", "-std=c++17",
        f"-arch={sm_arch}",
        # Runners #include the kernel .cu files unqualified.
        f"-I{SPG_DIR}",
        str(src),
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


def _hdr_to_rgba_bytes(work: Path, tag: str, hdr_image: np.ndarray) -> tuple[Path, int, int]:
    h, w = hdr_image.shape[:2]
    if hdr_image.shape[2] == 3:
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[..., :3] = hdr_image
        rgba[..., 3] = 1.0
    else:
        rgba = hdr_image.astype(np.float32, copy=False)
    hdr_path = work / f"{tag}.hdr.bin"
    np.ascontiguousarray(rgba.astype(np.float32, copy=False)).tofile(hdr_path)
    return hdr_path, w, h


def _run_cuda_controller(
    runner: Path,
    work: Path,
    tag: str,
    hdr_image: np.ndarray,
    weights: np.ndarray,
    prior_exposure: float,
) -> np.ndarray:
    """Run the single-kernel CUDA controller."""
    hdr_path, w, h = _hdr_to_rgba_bytes(work, tag, hdr_image)
    weights_path   = work / f"{tag}.weights.bin"
    out_path       = work / f"{tag}.out.bin"
    np.ascontiguousarray(weights.astype(np.float32, copy=False).reshape(-1)).tofile(weights_path)

    cmd = [str(runner), str(w), str(h), str(prior_exposure),
           str(hdr_path), str(weights_path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"controller_runner failed (exit {proc.returncode})")

    return np.fromfile(out_path, dtype=np.float32, count=9)


def _run_cuda_pipeline(
    runner: Path,
    work: Path,
    tag: str,
    hdr_image: np.ndarray,
    weights: np.ndarray,
    prior_exposure: float,
    warmup: int,
    iters: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run the 2-stage CUDA pipeline (pixel CNN + pool/MLP) at full input
    resolution. Returns (output[9], timings_dict_ms).
    """
    hdr_path, w, h = _hdr_to_rgba_bytes(work, tag, hdr_image)
    weights_path   = work / f"{tag}.weights.bin"
    out_path       = work / f"{tag}.out.bin"
    np.ascontiguousarray(weights.astype(np.float32, copy=False).reshape(-1)).tofile(weights_path)

    cmd = [str(runner), str(w), str(h), str(prior_exposure),
           str(hdr_path), str(weights_path), str(out_path),
           str(warmup), str(iters)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"pipeline_runner failed (exit {proc.returncode})")

    timings: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        # Lines look like "pixel_cnn : 0.123 ms (avg over 5)"
        if ":" in line and "ms" in line:
            stage, rest = line.split(":", 1)
            try:
                ms = float(rest.strip().split()[0])
                timings[stage.strip()] = ms
            except (ValueError, IndexError):
                pass
    out = np.fromfile(out_path, dtype=np.float32, count=9)
    return out, timings


# ---------------------------------------------------------------------------
# Torch timing (full-resolution forward pass)
# ---------------------------------------------------------------------------


def _torch_time_controller(ctrl, hdr_image: np.ndarray, prior_exposure: float,
                           warmup: int, iters: int) -> tuple[np.ndarray, float]:
    """Run the torch controller on GPU, time it via cudaEvents.

    Returns (output[9], avg_ms_per_call).
    """
    import torch
    if not torch.cuda.is_available():
        # Fall back to CPU timing.
        out = _torch_reference(ctrl, hdr_image, prior_exposure)
        return out, float("nan")
    ctrl = ctrl.to("cuda")
    rgb = torch.from_numpy(hdr_image).float().to("cuda")
    pe = torch.tensor([prior_exposure], dtype=torch.float32, device="cuda")
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = ctrl(rgb, pe)
        torch.cuda.synchronize()
        ev0.record()
        for _ in range(iters):
            exposure, color = ctrl(rgb, pe)
        ev1.record()
        torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / max(1, iters)
    out = np.concatenate([
        np.array([float(exposure)], dtype=np.float32),
        color.detach().cpu().numpy().astype(np.float32),
    ])
    return out, ms


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64))
    mse = float((diff * diff).mean())
    if mse <= 0.0:
        return float("inf")
    import math
    return 10.0 * math.log10((peak * peak) / mse)


def _validate_synthetic(args, work: Path, runner: Path,
                        pipeline_runner: Path | None) -> int:
    ctrl = _make_test_controller(args.seed)
    rng = np.random.default_rng(args.seed)
    hdr = (rng.random((args.height, args.width, 3), dtype=np.float32) * 0.8 + 0.1)
    weights  = flatten_controller_weights(ctrl)

    expected = _torch_reference(ctrl, hdr, args.prior)

    print(f"[synthetic seed={args.seed} {args.width}x{args.height}, prior={args.prior}]")

    # ----- single-kernel CUDA -----
    actual_single = _run_cuda_controller(runner, work, "synth", hdr, weights, args.prior)
    diff_single = np.abs(actual_single - expected)
    print(f"  single-kernel CUDA vs torch:")
    print(f"    max abs diff: {diff_single.max():.6g}  PSNR: {_psnr(actual_single, expected):.2f} dB")

    # ----- pipeline CUDA (full-res, 2 nodes) -----
    if pipeline_runner is not None:
        actual_pipe, timings = _run_cuda_pipeline(
            pipeline_runner, work, "synth_pipe",
            hdr, weights, args.prior,
            args.warmup, args.iters,
        )
        diff_pipe = np.abs(actual_pipe - expected)
        _, torch_ms = _torch_time_controller(
            ctrl, hdr, args.prior, args.warmup, args.iters,
        )

        print(f"  pipeline CUDA (full-res, 2 nodes) vs torch:")
        print(f"    max abs diff: {diff_pipe.max():.6g}  PSNR: {_psnr(actual_pipe, expected):.2f} dB")
        print(f"  timings (avg over {args.iters} iters):")
        for k in ("pixel_cnn", "pool_mlp", "total"):
            if k in timings:
                print(f"    cuda {k:<10s} {timings[k]:8.3f} ms")
        print(f"    torch (GPU)        {torch_ms:8.3f} ms")
        if torch_ms == torch_ms and "total" in timings and timings["total"] > 0:
            print(f"    speedup vs torch:   {torch_ms / timings['total']:.2f}x")

        return 0 if diff_pipe.max() <= args.tol else 1

    return 0 if diff_single.max() <= args.tol else 1


def _validate_checkpoint(args, work: Path, runner: Path,
                         pipeline_runner: Path | None) -> int:
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

    @dataclasses.dataclass
    class _Row:
        cam_idx: int
        name: str
        hw: tuple[int, int]
        max_diff_single: float
        psnr_single: float
        max_diff_pipe: float | None = None
        psnr_pipe: float | None = None
        cuda_total_ms: float | None = None
        cuda_cnn_ms: float | None = None
        cuda_mlp_ms: float | None = None
        torch_ms: float | None = None

    rows: List[_Row] = []

    print(f"[checkpoint {args.checkpoint}]")
    print(f"  cameras: {len(cam_names)}, max-frames-per-camera={args.max_frames}")
    if pipeline_runner is not None:
        print(f"  pipeline mode: warmup={args.warmup}, iters={args.iters}")

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
        H, W = hdr_np.shape[:2]

        ctrl = controllers[cam_idx]
        torch_out = _torch_reference(ctrl, hdr_np, prior_exposure=0.0)
        weights   = flatten_controller_weights(ctrl)

        # Single-kernel CUDA (correctness baseline).
        cuda_out  = _run_cuda_controller(runner, work, f"cam{cam_idx}_f{frame_idx}",
                                         hdr_np, weights, prior_exposure=0.0)
        diff_single = np.abs(cuda_out - torch_out)
        row = _Row(
            cam_idx=cam_idx,
            name=cam_names[cam_idx] if cam_idx < len(cam_names) else f"cam_{cam_idx}",
            hw=(H, W),
            max_diff_single=float(diff_single.max()),
            psnr_single=_psnr(cuda_out, torch_out),
        )

        # 2-stage pipeline + timing.
        if pipeline_runner is not None:
            cuda_pipe_out, timings = _run_cuda_pipeline(
                pipeline_runner, work, f"pipe_cam{cam_idx}_f{frame_idx}",
                hdr_np, weights, prior_exposure=0.0,
                warmup=args.warmup, iters=args.iters,
            )
            row.max_diff_pipe = float(np.abs(cuda_pipe_out - torch_out).max())
            row.psnr_pipe     = _psnr(cuda_pipe_out, torch_out)
            row.cuda_total_ms = timings.get("total")
            row.cuda_cnn_ms   = timings.get("pixel_cnn")
            row.cuda_mlp_ms   = timings.get("pool_mlp")
            _, row.torch_ms   = _torch_time_controller(
                ctrl, hdr_np, 0.0, args.warmup, args.iters)

        rows.append(row)
        seen_cams.add(cam_idx)
        if args.max_frames is not None and len(seen_cams) >= args.max_frames:
            break

    if not rows:
        raise SystemExit("No validation frames produced.")

    # Correctness summary.
    print()
    print(f"  {'cam':>4s} {'name':<24s} {'HxW':>11s} {'single max|Δ|':>14s} {'PSNR single':>12s}")
    for r in rows:
        hw = f"{r.hw[0]}x{r.hw[1]}"
        print(f"  {r.cam_idx:4d} {r.name:<24s} {hw:>11s} {r.max_diff_single:14.4g} {r.psnr_single:12.2f}")

    if pipeline_runner is not None and any(r.max_diff_pipe is not None for r in rows):
        print()
        print(f"  pipeline timing + correctness (avg over {args.iters} iters):")
        print(f"  {'cam':>4s} {'name':<24s} {'pipe max|Δ|':>12s} {'PSNR pipe':>10s}  "
              f"{'cnn':>8s} {'mlp':>8s} {'total':>8s}  "
              f"{'torch':>8s}  {'speedup':>8s}")
        for r in rows:
            if r.max_diff_pipe is None:
                continue
            sp = (r.torch_ms / r.cuda_total_ms) if (r.cuda_total_ms and r.torch_ms) else float("nan")
            print(
                f"  {r.cam_idx:4d} {r.name:<24s} "
                f"{r.max_diff_pipe:12.4g} {r.psnr_pipe:10.2f}  "
                f"{r.cuda_cnn_ms:8.3f} {r.cuda_mlp_ms:8.3f} {r.cuda_total_ms:8.3f}  "
                f"{r.torch_ms:8.3f}  {sp:7.2f}x"
            )

    overall_max = max(r.max_diff_single for r in rows)
    print(f"  overall single-kernel max abs diff = {overall_max:.6g} (tol={args.tol})")
    return 0 if overall_max <= args.tol else 1


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
    parser.add_argument("--pipeline", action="store_true",
                        help="Also exercise the 2-kernel pipeline (pixel CNN + "
                             "pool/MLP, full input resolution) and report per-stage "
                             "timing + PSNR vs torch.")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Pipeline warm-up iterations before timing.")
    parser.add_argument("--iters", type=int, default=10,
                        help="Pipeline iterations to time-average.")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the working directory after the run.")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    work = Path(tempfile.mkdtemp(prefix="ppisp_cuda_val_"))
    try:
        runner = _build_runner(work, RUNNER_CU, "controller_runner",
                               sm_arch=args.sm_arch, verbose=args.verbose >= 2)
        pipeline_runner = None
        if args.pipeline:
            pipeline_runner = _build_runner(
                work, PIPELINE_RUNNER_CU, "controller_pipeline_runner",
                sm_arch=args.sm_arch, verbose=args.verbose >= 2,
            )
        if args.checkpoint is not None:
            return _validate_checkpoint(args, work, runner, pipeline_runner)
        return _validate_synthetic(args, work, runner, pipeline_runner)
    finally:
        if args.keep_tmp:
            print(f"  (kept tmp at {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
