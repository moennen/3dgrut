# Working notes

## Verifying

The venv must be on `PATH`, not just used as the interpreter. `threedgrt_tracer`'s
pipeline-consistency test compiles Slang in a subprocess that resolves `slangc` from `PATH`;
without it, that test fails with a `FileNotFoundError` that has nothing to do with the change
under test.

```bash
cd /mnt/oss/3dgrut-bernardin
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q
# ~9 min, currently 387 passed, 1 skipped

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
- 3DGUT has *three* backward compositing paths, and a forward accumulator added to the shared
  hit processing appears on all three, so each must be taught to differentiate it separately:
  hand-written CUDA `processHitBwd` (`k_buffer_size=0`, normals off), Slang
  `...BwdToRawParameters` (`k_buffer_size=0`, normals on) and Slang `...BwdToBuffer`
  (`k_buffer_size>0`). They are selected by *configuration*, so a gradient test that exercises
  only the default silently covers a third of the feature; parametrize over all three, as
  `test_depth_variance_gradient.py` does. The Slang backward replays hits back-to-front with
  `integratedDepth = lerp(integratedDepth, depth, alpha)`, which is easy to extend: an
  accumulator obeying the same recursion inverts the same way.
- A rendered buffer that is *empty* when its feature is compiled out will make a loss silently
  `0.0` rather than raise. Pair the flag with `mark_non_differentiable` so the config typo
  (loss on, buffer off) fails loudly -- `Tracer._dist_sq_differentiable`.
- `slangc` resolves the `static const bool` feature flags in `threedgut.slang` as ordinary
  names, so a flag referenced from a `[CudaDeviceExport]` entry point -- which sits *outside*
  the `namespace gaussianParticle` block -- must be qualified (`gaussianParticle.EnableFoo`)
  even though uses inside the namespace need no prefix. Unlike the `#if`-guarded CUDA, a
  disabled Slang flag still leaves its parameters in the generated signature.
- The C++ is not `clang-format` clean at HEAD, so do not run it over a touched file; match the
  surrounding alignment by hand.
- Test a loss by its *gradient*, at the granularity the renderer will apply it, not by its
  value. The depth-variance term detached the `acc` in `M2 - D^2/acc`, which drops the term
  completing the square and turns `dL/dw_i = (t_i - mu)^2` into `(t_i - mu)^2 - mu^2` -- an
  offset that grows as the squared depth and inverts the sign. Value tests, finite-difference
  tests against the renderer, and gradient tests on the *accumulators* all passed; it took
  backpropagating to per-hit weight and distance to see it. Suspect any `.detach()` justified
  by "that term diverges" -- check whether the numerator scales with the denominator first.
- A sharpening prior with no reference for *where* to sharpen will buy confidence instead of
  accuracy. Per-ray variance is zero for any Dirac at any distance, and in its pairwise form
  -- `M2 - D^2/acc == (1/2acc) sum_ij w_i w_j (t_i - t_j)^2`, the distortion loss -- the
  two-hit case is a double well in the near opacity, because `w_near * w_far` carries an
  `a(1-a)` peaking at one half. The near well is absorbing: once the front particle reaches
  alpha 1, transmittance zeroes the gradient to everything behind it for *every* loss, so the
  error is permanent. Measured as `wrong | tight` rising 170x while spread fell 6x. Report
  that pairing, not just the mean error, for any term of this family --
  `scripts/ablation/depth_variance_mechanism.py`. Prefer the pairwise form when reasoning
  about a compositing loss; the accumulator form hides the interaction between hits.
- "The minima are unchanged" is not an argument that behaviour is unchanged. Dividing that
  same variance by `mu^2` leaves both wells at exactly zero, so it was dismissed as a
  non-fix -- but it moved the *barrier* from 0.49 to 0.82, shrinking the bad basin from 51% of
  the axis to 18%, and that is what descent actually responds to. When judging a
  reparameterisation of a degenerate objective, locate the separatrix, not just the optima.
- Normalising a loss by a quantity it already contains can be free. `Var/mu^2` is
  `acc*M2/D^2 - 1`, so it needed no accumulator, no kernel and no backward work, while the
  mip-NeRF/2DGS `|t_i - t_j|` kernel it replaced would have needed a new accumulator on three
  backward paths to reach a strictly worse place. Check what the existing buffers can already
  express before extending the renderer.
- An offline agreement metric is not a proxy for a trained result, and counting *pairs* is not
  measuring what they teach. Gating the ordinal pseudo-depth loss to drop near-tied pairs lifts
  the prior's ordinal agreement with ground truth from 84% to 97%, which looked decisive; trained
  over 3 seeds it is neutral on two scenes and costs emerald-square its entire 10% depth gain.
  A large prior gap selects *long-range* pairs, the regime where a monocular prior drifts (one
  global affine scores `abs_rel` 0.068 against 0.011 per 16x16 patch), so the gate kept the
  prior's weakest structure. High agreement is also a warning for a one-sided loss: at 97% only
  3% of surviving pairs can produce any gradient. Sweep the knob rather than trusting the
  diagnostic that motivated it.
- A monocular depth prior is not usable as depth, only as *ordering*. `DepthAnythingV2` fitted
  per frame by the best possible affine still scores `abs_rel` 0.068 on sponza, worse than the
  7k model it would be teaching (0.058); per 16x16 patch it scores 0.011. It also emits
  *disparity* (correlation +0.978 with `1/z`, so reading it as distance is monotonically
  inverted) and *z-depth*, while this renderer's convention is Euclidean ray distance, which
  differ by 20% at the image corners. An ordinal loss is invariant to all three problems.
- Losses running every iteration should stay on device: no `.item()`/`int()`/`bool()` on
  intermediate tensors, and handle empty masks with a clamped division rather than a Python
  branch, so the training loop never stalls on a host sync.

## Documentation

Ongoing geometry-supervision work is written up in `docs/normal-supervision.md`, including
measurements and the reasoning that was wrong. Keep the record of superseded predictions.
