# Development guide

## Verification

The virtual environment must be on `PATH`: tracer tests invoke `slangc` in a subprocess.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
.venv/bin/python -m black --check --line-length 120 .
.venv/bin/python -m isort --check-only --profile black --line-length 120 .
```

Run focused tests while iterating, then the full suite before a feature commit. Avoid formatting
vendored code under `thirdparty/`.

## Geometry conventions

- Rendered depth is Euclidean ray distance, not camera z-depth.
- Rendered normals face the camera. They are alpha-premultiplied; normalize non-zero vectors
  before use.
- With normals disabled, the tracer emits a constant placeholder. Any normal-dependent loss
  must reject that configuration.
- A process loads one compile-time tracer variant. Run configurations with different compile
  flags in separate processes.

## Evaluation

Use the geometry evaluation commands documented in `docs/geometry-reconstruction.md`. Keep
experiment-specific observations in run records and issue/PR discussion rather than this file.
