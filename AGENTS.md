# Working notes

## Verifying

The venv must be on `PATH`, not just used as the interpreter. `threedgrt_tracer`'s
pipeline-consistency test compiles Slang in a subprocess that resolves `slangc` from `PATH`;
without it, that test fails with a `FileNotFoundError` that has nothing to do with the change
under test.

```bash
cd /mnt/oss/3dgrut-bernardin
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
# ~11 min, currently 295 passed, 1 skipped

.venv/bin/python -m black --line-length 120 . && .venv/bin/python -m isort --profile black --line-length 120 .
```

`black`/`isort` are not on `PATH` either; invoke them as `.venv/bin/python -m`.

## Ablation

OB3D scenes live at `/mnt/data/nerf_datasets/ob3d/OB3D_colmap`. A 7k cell is ~80s on one GPU,
so a sweep can be split across GPUs 0 and 1 by scene.

```bash
.venv/bin/python scripts/ablation/run_ob3d.py \
  --dataset-root /mnt/data/nerf_datasets/ob3d/OB3D_colmap \
  --out-dir /tmp/abl --scenes sponza --variants gaussian trisurfel dn05_trisurfel
.venv/bin/python scripts/ablation/report.py /tmp/abl/results.jsonl
```

Variants are defined in `scripts/ablation/run_ob3d.py`. Anything gated on an iteration count
needs its gate lowered for a 7k sweep -- `depth_normal_from_iter` defaults to 7000, sized for a
30k run, and would otherwise switch on exactly as the sweep ends and report a false null.

Judge geometry changes on `n_gain` (the rendered normal against a view-direction control that
uses no geometry) rather than the raw angle: the baseline showed a plausible-looking normal
buffer that loses to that control.

## Conventions

- Rendered depth is Euclidean ray distance, not z-depth, so a plane is not constant depth.
  The reference implementations in `/mnt/oss/blob-to-spoke` use z-depth; do not port formulas
  across without converting.
- Normals are oriented towards the camera, and the tracers return alpha-premultiplied normals
  whose magnitude carries ray coverage, so consumers must normalize and treat a zero-length
  normal as "no surface" rather than a direction.
- `render.enable_normals=false` makes the tracers return a *constant* placeholder normal, not
  an empty buffer, so any new consumer of `pred_normals` must reject that combination rather
  than silently training against a constant.
- Losses running every iteration should stay on device: no `.item()`/`int()`/`bool()` on
  intermediate tensors, and handle empty masks with a clamped division rather than a Python
  branch, so the training loop never stalls on a host sync.

## Documentation

Ongoing geometry-supervision work is written up in `docs/normal-supervision.md`, including
measurements and the reasoning that was wrong. Keep the record of superseded predictions.
