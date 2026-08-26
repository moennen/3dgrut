# Working notes

## Verifying

The venv must be on `PATH`, not just used as the interpreter. `threedgrt_tracer`'s
pipeline-consistency test compiles Slang in a subprocess that resolves `slangc` from `PATH`;
without it, that test fails with a `FileNotFoundError` that has nothing to do with the change
under test.

```bash
cd /mnt/oss/3dgrut-bernardin
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
# ~8 min, currently 302 passed, 1 skipped

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
- The rendered normal is the particle's local **z** axis for *both* primitives, not the
  shortest axis and not the ellipsoid surface normal. `canonicalRayNormal` in
  `gaussianParticles.slang` is templated on `Surfel`, never branches on it, and ignores the
  `scale` it is passed; the true ray-ellipsoid normal in `gaussianParticles.cuh` is dead on
  this path. Any term that reasons about particle shape must target z. Pinned by
  `threedgut_tracer/tests/test_normal_axis.py`.
- When porting a loss from a reference implementation, check that the quantity it names means
  the same thing here. PGSR's min-scale penalty is correct there because its normal *is* the
  shortest axis; transplanted unchanged it regressed normals by 30 degrees.
- Run-to-run noise on OB3D at 7k is ~0.04-0.13 dB PSNR, ~0.1-1.2 deg normal and ~0.002-0.004
  `abs_rel`, measured over `seed_initialization` 1-4. Effects of a few tenths of a dB need
  repeated seeds; normal effects of several degrees do not. Scatter across a *hyperparameter*
  sweep is not a noise estimate, and reading it as one produced a wrong "free of PSNR cost"
  claim. Repeat with `--override seed_initialization=N` on the ablation harness.
- A process holds exactly **one** compiled `lib3dgut_cc`, cached in `load_3dgut_plugin`, so
  the first config to render decides the binary and every later config silently reuses it.
  Combined with `enable_normals=false` returning a *constant* placeholder normal, this made
  `test_normal_axis` "pass" against a normals-off binary. `load_3dgut_plugin` now raises on a
  variant mismatch instead; a test needing a new variant (a new `-D`) must render in a child
  process, as `threedgut_tracer/tests/test_depth_variance.py` does. Adding a define also
  changes every variant's flag hash, so the next run rebuilds all of them.
- OB3D scenes fail differently, so a scene average hides the effect a geometry term has.
  Sponza and emerald carry floaters (0.2-13% of pixels) that per-ray spread detects well
  (AUC 0.68-0.90); lone-monk has essentially none (0.01%) yet 17.7% `delta1` failures, its
  depth being wrong in an opaque, confidently-placed way that no ray-concentration term can
  reach. Judge a term on the scenes exhibiting the failure it targets, and say which those are.
- Before adding a buffer to justify a loss, check it against signals already rendered. The
  relative gradient of the existing depth gets within 0.02-0.10 AUC of the new variance buffer
  at spotting bad depth, which reframes the variance term as an optimisation target rather than
  a diagnostic advance. `scripts/ablation/depth_variance_diagnostic.py` runs this comparison.
- Losses running every iteration should stay on device: no `.item()`/`int()`/`bool()` on
  intermediate tensors, and handle empty masks with a clamped division rather than a Python
  branch, so the training loop never stalls on a host sync.

## Documentation

Ongoing geometry-supervision work is written up in `docs/normal-supervision.md`, including
measurements and the reasoning that was wrong. Keep the record of superseded predictions.
