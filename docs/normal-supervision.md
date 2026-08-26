# Surfel geometry improvements

Working notes for the `nicolasm/surfel-geometry-improvments` branch. Records the plan, what
has landed, the measurements taken, and what is left.

## Goal

Improve the *surface* quality of the surfel and trisurfel primitives, not just their
photometric scores. Photometric metrics are largely indifferent to whether a surfel is
oriented correctly, so the objective is to supervise geometry directly, and the end
deliverable is a mesh: an extractable surface is the strongest evidence that the geometry is
real rather than merely photometrically adequate.

## Initial plan

**Prerequisites** — none of the terms below can be written or judged without these.

1. Load ground-truth depth and normals so geometry can be scored at all.
2. Emit real rendered normals from the forward pass (3DGUT was returning a placeholder).
3. Build an ablation harness and record a baseline, so any change can be judged.
4. Differentiate the rendered normal buffer, since a loss on it needs gradients.

**Geometry supervision terms** — the substance of the work, roughly in dependency order.

5. **Depth-normal consistency.** Pull the rendered normal towards the normal implied by the
   rendered depth. Self-supervised, needs no reference geometry.
6. **Pseudo-depth supervision.** Supervise rendered depth against depth from a monocular
   predictor, scale-invariantly since the prediction is only defined up to an affine
   transform. Supplies absolute geometry where multi-view constraints are weak — textureless
   regions and sparsely observed areas.
7. **Depth variance along the ray.** Penalise the spread of the per-ray weight distribution
   to concentrate it onto a single surface. This is 2DGS depth distortion, and it does double
   duty: it is a surface prior in its own right, and it is what makes expected depth
   approximate median depth, which term 5 depends on (see open question (a)).
8. **Multi-view consistency.** Warp a patch into a neighbouring view through the rendered
   depth and plane normal and penalise photometric and geometric disagreement, PGSR-style.
   The strongest geometric signal available, and the most expensive: it needs neighbour-view
   selection, occlusion handling, and a second render per iteration.
9. **Scale-z regularisation.** Drive the smallest scale axis towards zero so non-trisurfel
   primitives flatten into disks, giving surfel-like behaviour without a primitive change.

**Deliverable.**

10. **Mesh export.** Extract a surface from the trained representation — TSDF fusion of
    rendered depth, or Poisson reconstruction from oriented points.

Steps forced by measurement, not planned: the silent-gradient guard and the cross-pipeline
normal unification (both below) came out of what step 3 turned up.

The terms are deliberately not independent. 7 is a precondition for 5 being well-posed, 5 and
8 both constrain normals and may be redundant, and 10 is the acceptance test for all of them.
That is why the harness (step 3) came before any term: the ablation is the point.

## What has landed

Oldest first. Everything below is committed, formatted, and covered by tests; the full suite
is 264 passed / 1 skipped.

| Commit | Summary |
|---|---|
| `eceff307` | Surfel support in 3DGUT |
| `c1f7ec69` | NHT trisurfel interpolation |
| `61ff6906` | Key JIT build artifacts by compile-time variant |
| `411bd503` | Load ground-truth depth and normals from COLMAP scenes |
| `63e5fe56` | Emit real rendered normals from the 3DGUT forward pass |
| `7e45a5a4` | Report absolute depth and normal error against reference geometry |
| `ac05713c` | Report the no-geometry control alongside normal error |
| `fcce8dd4` | Add OB3D ablation harness |
| `844fdeb6` | Record the OB3D baseline sweep at 7k iterations |
| `6125688c` | Report floater rate, and identify what fails on emerald-square |
| `d3c7370b` | Differentiate the 3DGUT rendered normal buffer |
| `2e4fd4e0` | Reject 3DGRT normal supervision on a pipeline that cannot deliver it |
| `200fbe9b` | Make every 3DGRT pipeline report the same normal |

### Measurement infrastructure

`threedgrut/datasets/gt_geometry.py` loads reference depth and normals;
`threedgrut/utils/depth_normal_metrics.py` scores them. `scripts/ablation/run_ob3d.py`
sweeps variants over OB3D scenes and `report.py` renders the tables.

Two reporting decisions matter for reading any of the numbers:

- **A no-geometry control.** `n_control` is the error of pointing every normal straight back
  along the view ray, which uses no geometry whatsoever. `n_gain` is the control minus the
  rendered error. It exists because a raw normal angle looks respectable while being worse
  than nothing, which is exactly what the baseline found.
- **World-unit columns are not averaged meaningfully** across scenes, since the average is
  dominated by the physically largest scene. Rank on the scale-free `d_absrel` and
  `d_delta1`.

### Baseline (OB3D, 7k iterations, 4 shared scenes)

Full table in `scripts/ablation/baselines/ob3d_7k_dev4.md`.

| variant | psnr | d_absrel | d_delta1 | n_mean | n_control | n_gain | d_float |
|---|---|---|---|---|---|---|---|
| gaussian | **35.22** | **0.0873** | **0.890** | 56.9 | 40.1 | **-16.7** | **0.0069** |
| trisurfel | 33.72 | 0.0998 | 0.855 | 39.9 | 40.1 | +0.2 | 0.0344 |

Two findings drove the rest of the work:

- **Gaussian normals are beaten by the no-geometry control** (`n_gain` -16.7). Trisurfel only
  draws with it (+0.2). Neither buffer carries usable surface information yet, so normal
  supervision is the right target — but it also means the raw `n_mean` of 39.9 should not be
  read as "trisurfel normals are decent".
- **Trisurfel floaters are concentrated, not diffuse.** `d_float` 0.0344 vs 0.0069 is almost
  entirely emerald-square (0.1348 vs 0.0236); the other three scenes are at parity. One scene
  is doing the damage.

### 3DGUT normal backward (`d3c7370b`)

The rendered normal buffer now differentiates. The normal term had to be added at all three
`processHitBwd` call sites — the K-buffer renderer, the sorted renderer, and the
raw-parameters variant used by the K=0 path — since missing any one produces a pipeline that
trains silently wrong rather than failing.

Validated against finite differences across all 8 variants
(`threedgut_tracer/tests/test_normal_gradient.py`), which run in subprocesses because the
compiled extension is cached in a module global.

### 3DGRT: normal supervision is rejected, not silently ignored (`2e4fd4e0`)

Only `referenceSlangBwd` differentiates the normal buffer. `referenceBwd` — the default —
calls a hand-derived `processHitBwd` with no normal term, and `referenceB2FSlangBwd` is an
upstream stub with its `processHitBwd` commented out. Both render normals perfectly well
forward, so a normal-supervised run on either looks healthy while the term contributes
nothing.

Hand-deriving the missing backward was rejected as duplicated work: the Slang path already
gets it right by autodiff, so the derivation exists in an easier form. Instead
`check_normal_supervision_supported` refuses the combination when the tracer is built. It
keys on `loss.use_depth_normal` rather than `render.enable_normals`, because rendering normals
purely to score them is a legitimate forward-only use that must keep working everywhere.

The check trusts that flag to be the only route to normal supervision. A future normal term
under a different flag would reintroduce the silence, so it must reuse the flag or extend
the check.

### 3DGRT: one definition of "normal" (`200fbe9b`)

`reference` and `referenceSlang` were computing different quantities: 80° apart on average
for instances, 110° for trisurfel with a 147° median — close to a straight flip. Nothing
looked broken on screen, because normals look plausible whatever you fill them with.

The Slang model returns the flat-disk normal (the particle's third canonical axis carried
into world space, oriented to face the ray) and ignores its own `Surfel` template parameter
while doing so, which is the usual 2DGS-style choice and what 3DGUT also implements. The
hand-written CUDA `processHit` branched on `SurfelPrimitive` and got both branches wrong:

- **Ellipsoid branch:** multiplied the canonical intersection point elementwise by a rotated
  scale vector. That is not a normal transform — the correct map needs the inverse transpose,
  `R · (x / scale)`.
- **Surfel branch:** built `(0, 0, ±scaleRotated.z)`, a vector along *world* z, discarding the
  particle's orientation entirely and leaving the result unnormalized — which silently
  reweighted each hit by particle thickness, since normals are alpha-weighted before the
  buffer is normalized. The measured mean normal was `[0.0, 0.0, 0.327]`, x and y exactly
  zero, which is that bug visible in the output.

The CUDA branch was replaced by the Slang definition rather than reconciled with it.
`barycentricSurfelsOptix.cu` had its orientation test inverted too, and now matches; only the
orientation is shared there, since its normal comes from the precomputed surfel normal, a
better source than a derived disk normal.

**Result:** `reference` and `referenceSlang` now agree to floating-point noise (0.005° mean
for instances, 0.027° for trisurfel), and both sit the same 6.43° from 3DGUT. That residual is
rasterization vs ray tracing — different hit sets and alpha weights, not a different
definition; the tell is that the two pipelines are now equidistant from it.

Regression tests in `threedgrt_tracer/tests/test_normal_pipeline_consistency.py` were
confirmed to fail on the old kernel (175°/171° max, and mean |x| = 0.0000 for rotation
dependence) rather than assumed to.

> **Any 3DGRT normal metric recorded before `200fbe9b` is obsolete**, including the
> `n_mean` / `n_gain` columns in the 7k baseline above, which were produced by the default
> `reference` pipeline. The depth and photometric columns are unaffected.

## What remains

Map to the plan: prerequisites 1-4 are done; 5-10 are pending. Current state of each pending item:

| Plan item | Status | What is missing |
|---|---|---|
| 5. Depth-normal consistency | Landed, measured at 7k | 30k confirmation; `referenceSlang` throughput on 3DGRT |
| 6. Pseudo-depth supervision | Not started | Monocular depth predictor integration, scale-invariant loss |
| 7. Depth variance along the ray | Not started | Kernel accumulator for `w·t` and `w·t²`; backward pass |
| 8. Multi-view consistency | Not started | Patch warp, neighbour selection, occlusion handling, second render |
| 9. Scale-z regularisation | Landed, measured, **negative** | Nothing; it degrades normals monotonically and should stay off |
| 10. Mesh export | Not started | Surface extraction (TSDF or Poisson), Chamfer metric in ablation report |

### 1. The depth-normal consistency loss

The config fields have landed unset (`configs/base_gs.yaml`), so only the term itself is
missing:

```yaml
use_depth_normal: false
lambda_depth_normal: 0.0
depth_normal_from_iter: 7000
```

Intended shape, following the GGGS/PGSR reference implementation: unproject the rendered
depth, take cross products of central differences of neighbouring points to get a
depth-implied normal, and penalise `1 - cos` against the rendered normal, masked to valid
pixels, gated by `depth_normal_from_iter` and weighted by `lambda_depth_normal`. The one-pixel
border is invalid by construction.

Three questions are open before writing it.

**(a) Expected depth is the only depth available.** This codebase renders expected depth
only — `*depth += hitT * weight` in the 3DGRT kernel, normalised by accumulated opacity in
`expected_depth()`. There is no median depth buffer, whereas the reference implementation
uses median depth specifically.

Stated precisely, the concern is that expected depth is a weighted *mean* along the ray, so
wherever the weight distribution is spread or bimodal it returns a distance that no surface
occupies. At an occlusion boundary, a ray carrying weight `w_f` at a near surface and `w_b` at
a far one yields `(w_f·t_f + w_b·t_b) / (w_f + w_b)`, which lies between them; unprojecting it
places points on a phantom ramp bridging the two surfaces, and a finite-difference normal of
that ramp is neither surface's normal. Where the weight distribution is concentrated —
an opaque, well-fitted surface — expected and median depth coincide and the issue does not
arise, so the effect is confined to occlusion boundaries and semi-transparent regions.

That is the mechanism, but the claim that it matters *here* is unverified, and my earlier
phrasing overstated it. What is not known is what fraction of pixels are affected and whether
the resulting gradient bias is large enough to change training. It is cheap to measure:
derive normals from expected depth, compare against the rendered normals, and check whether
the disagreement concentrates on high depth-gradient pixels. Worth doing before either
accepting expected depth or paying for anything more elaborate.

If it does matter, the options in increasing cost are: mask pixels by local depth-gradient
magnitude or accumulated opacity; add a depth-distortion term (2DGS) to concentrate weights
so expected approaches median; or add a median depth buffer to both backends' kernels.

#### Measured (7k, `depth_derived_normal_metrics`, degrees)

`dn` is the depth-implied normal — the target the loss would pull towards — scored against
the reference normals. `dn_ref` is the same operator applied to the *reference* depth, so it
is the ceiling this construction can reach. `rendered` and `ctrl` are the existing rendered
normal error and the no-geometry view-direction control.

| scene | primitive | rendered | ctrl | dn | dn_ref | dn low-grad | dn high-grad |
|---|---|---|---|---|---|---|---|
| sponza | gaussian | 52.5 | 42.9 | 45.7 | 19.0 | 44.0 | 60.6 |
| sponza | trisurfel | 45.0 | 42.8 | 46.1 | 19.0 | 44.4 | 61.6 |
| emerald-square | gaussian | 66.0 | 50.3 | 62.3 | 29.1 | 60.8 | 75.2 |
| emerald-square | trisurfel | 49.2 | 50.3 | 64.5 | 29.2 | 63.1 | 77.2 |

Three conclusions, two of which contradict the plan above.

**The finite-difference operator is not the bottleneck.** Applied to the reference depth it
scores 19.0 and 29.1, against 45.7 and 62.3 from the rendered depth. The 27-35 degree gap is
the rendered depth's, so the target would improve substantially if the depth did. This is the
result that keeps the term worth having.

**The occlusion-boundary mechanism above is real but minor, so (a) is settled against
elaboration.** Restricting to the 90% of pixels with the smallest relative depth gradient
recovers 1.5-2 degrees of that 27-35 degree gap. The error is diffuse, not concentrated at
boundaries: the rendered depth is locally noisy everywhere, and it is that noise, amplified by
differentiation, that dominates. Note that the depth is accurate in the aggregate over the
same pixels (`abs_rel` 0.06 on sponza, `delta1` 0.95) — being right on average and being
locally smooth are different properties, and only the second one survives a derivative. A
median depth buffer would therefore not repay its kernel cost, and neither would a distortion
term aimed at this. Gradient masking is kept, because 2 degrees for a quantile is cheap, but
it is a trim rather than a fix.

**The target is worse than what trisurfel already has, which changes the loss's design.** For
trisurfel the rendered normal beats the depth-implied normal — 45.0 against 46.1 on sponza,
and 49.2 against 64.5 on emerald-square, a 15 degree deficit. A one-way pull of the rendered
normal towards the depth-implied one is therefore actively harmful for the primitive that
currently has the best normals, which is the opposite of the intended effect. It helps only
gaussians (52.5 to 45.7, 66.0 to 62.3), whose normals lose to the control anyway.

So the term cannot be justified as a normal *target*. It is worth having as a *mutual*
consistency constraint, with gradient flowing to both the depth and the normal, whose value is
on the depth side: the normal, being an explicit primitive orientation, acts as the smoothness
prior the noisy depth lacks. This is why the term works in 2DGS and PGSR, where it flattens and
aligns primitives rather than merely relabelling them. It also predicts that a large
`lambda_depth_normal` will degrade trisurfel normals, which the ablation should test rather
than assume.

**That last prediction was wrong, and the error is worth keeping.** See the sweep below:
trisurfel normals improve from 45.1 to 23.9 degrees on sponza and 49.2 to 37.0 on
emerald-square, monotonically in the weight until they plateau. The static measurement above
does not predict the training outcome, because it treats the target as fixed. Under
optimisation it is not: the depth-implied normal *itself* improves from 64.9 to 40.8 degrees on
emerald-square as the weight rises. Both sides move towards each other, so measuring where the
target sits at iteration 7000 of an unconstrained run says little about where the constrained
run ends up. The mutual-constraint reasoning was right; using the frozen target's quality to
forecast the outcome was not.

**(b) Invalid-pixel normalisation.** The reference does `masked_fill_(0).mean()`, averaging
over *all* pixels, so invalid ones dilute the loss rather than being excluded, and the
effective weight then drifts with the valid fraction. Dividing by the valid count is the
better default; matching the reference bit-for-bit is the argument against. Settled in favour
of the valid count, so a frame that is half sky carries the same weight per surface pixel as
one that is all surface.

**(c) 3DGRT requires `referenceSlang`,** which the startup assertion now enforces. Its
throughput relative to the default `reference` pipeline has not been measured, and it
determines whether normal supervision is affordable on 3DGRT at all.

#### As implemented

`threedgrut/utils/depth_normal_loss.py`, wired into `Trainer.get_losses` next to the other
regularisers, with `depth_normal_grad_percentile` added to the config.

- Gradient flows to **both** the depth and the normal. This follows from the measurement
  above: as a one-way normal target the term is harmful for trisurfel, and its value is on
  the depth side, where the normal supplies the local smoothness the depth lacks.
- No host synchronisation. The term runs every iteration, so it returns device tensors and
  handles the empty-mask case with a clamped division rather than reading a mask population
  back to branch on it. The gradient percentile is taken over a `+inf`-padded buffer with a
  rescaled quantile, which keeps it a fixed-shape kernel instead of a gather.
- A normal loss without `render.enable_normals` is now **rejected** by both backends
  (`threedgrut/utils/geometry_supervision.py`). This is not a cosmetic check: with normals
  disabled the tracers substitute a *constant* placeholder, so the term would supervise
  against a constant while producing a perfectly healthy-looking loss curve.
- The 7k sweep overrides `depth_normal_from_iter` to 3000. The config default of 7000 is
  sized for a 30k run and would switch the term on exactly as a 7k sweep ends, producing a
  null result that looks like a negative one.

#### Measured (7k, weight sweep, `dn*` variants in `scripts/ablation/run_ob3d.py`)

`n` is the rendered normal error against the reference, `gain` is that against the
view-direction control, `dn` is the depth-implied normal's own error, and `agree` is the
disagreement between the two that the term penalises directly.

sponza:

| lambda | primitive | psnr | d_absrel | n | gain | it/s |
|---|---|---|---|---|---|---|
| off | gaussian | 36.21 | 0.0598 | 52.2 | -9.4 | 103.6 |
| 0.005 | gaussian | 36.06 | 0.0516 | 30.9 | +12.0 | 88.6 |
| 0.05 | gaussian | 36.15 | 0.0478 | 25.5 | +17.3 | 87.8 |
| 0.2 | gaussian | 35.01 | 0.0515 | 25.0 | +17.8 | 87.7 |
| off | trisurfel | 36.07 | 0.0598 | 45.1 | -2.2 | 102.4 |
| 0.005 | trisurfel | 36.11 | 0.0517 | 30.1 | +12.8 | 88.9 |
| 0.05 | trisurfel | 36.10 | 0.0483 | 23.9 | +18.9 | 88.4 |
| 0.2 | trisurfel | 35.60 | 0.0542 | 23.9 | +19.0 | 86.2 |

emerald-square:

| lambda | primitive | psnr | d_absrel | n | gain | dn | agree |
|---|---|---|---|---|---|---|---|
| off | gaussian | 32.68 | 0.1395 | 64.8 | -14.4 | 63.1 | 81.2 |
| 0.005 | gaussian | 32.76 | 0.1333 | 44.2 | +6.1 | 51.2 | 36.9 |
| 0.05 | gaussian | 32.49 | 0.1315 | 40.9 | +9.4 | 43.5 | 16.7 |
| 0.2 | gaussian | 31.67 | 0.1395 | 41.1 | +9.2 | 42.7 | 11.5 |
| off | trisurfel | 32.80 | 0.1322 | 49.2 | +1.1 | 64.9 | 73.3 |
| 0.005 | trisurfel | 32.62 | 0.1222 | 37.2 | +13.1 | 50.4 | 41.7 |
| 0.05 | trisurfel | 32.25 | 0.1305 | 37.0 | +13.3 | 43.1 | 18.2 |
| 0.2 | trisurfel | 31.44 | 0.1324 | 37.3 | +13.0 | 40.8 | 12.0 |

This is the first change in this work to clear the control by a wide margin. The baseline's
finding was that both primitives' normals *lose* to a view-direction control, gaussians by 9-14
degrees; with the term at 0.05 they win by 9-19. Trisurfel goes from 45.1 to 23.9 on sponza,
which is the result the whole exercise was after.

Depth improves too, most at 0.05 on sponza (0.0598 to 0.0478, a 20% reduction in `abs_rel`)
and at 0.005 on emerald-square (0.1322 to 0.1222). So the term is not trading depth for
normals; below 0.05 it improves both.

The cost is appearance and throughput. PSNR is roughly free up to 0.05 on sponza (-0.06 for
trisurfel) but already -0.55 on emerald-square, and 0.2 costs 0.5-1.4 PSNR on every cell while
buying no further normal accuracy -- the normals plateau by 0.05, and past it the term is only
distorting radiance. Throughput is -15% (103.6 to 87.8 it/s), measured at 0.005 where the
primitive counts match, so that is the term's own cost rather than a densification difference.

`agree` falling to 11-18 degrees at the higher weights while `n` plateaus is worth noting: the
term keeps successfully minimising exactly what it optimises, after that has stopped
corresponding to accuracy. It is a consistency constraint, not a supervision signal, so past
the plateau the two buffers agree with each other about a geometry that is no better.

Default set to `lambda_depth_normal: 0.05`, still behind `use_depth_normal: false`. 0.05 rather
than 0.005 because normal accuracy is the objective here and it is worth 0.3-0.5 PSNR;
appearance-first configurations should prefer 0.005, which is nearly free on both.

### 2. Re-measure

- Re-run the OB3D sweep to replace the obsolete 3DGRT normal columns in the 7k baseline.
- Then judge the loss on `n_gain` and the depth metrics, not on whether it trains. Beating
  the no-geometry control is the bar, since the baseline shows a plausible-looking normal
  buffer can fail it.

### 3. Depth variance along the ray

Nothing landed. Worth promoting ahead of the depth-normal loss if open question (a) resolves
against expected depth, since concentrating the per-ray weights is what makes expected depth
a usable stand-in for median depth — it would turn a workaround into a term we want anyway.

Needs a per-ray weight-spread accumulator in both backends' forward and backward passes,
which is deeper kernel work than any of the other terms; everything else is a
post-render torch loss. The 2DGS distortion form needs the accumulated `w·t` and `w·t²`
moments, so a second accumulator alongside the existing depth one.

### 4. Pseudo-depth supervision

Nothing landed. Needs a monocular depth predictor and a cache — the dependency is heavier
than the loss, and whether it can be a hard dependency of this repo is an open question, so
the predictor should be run offline and its output loaded like any other reference channel.
`threedgrut/datasets/gt_geometry.py` already loads reference depth, so the loading path
exists; what it needs is a source that is not COLMAP ground truth.

Must be scale-invariant (fit per-image scale and shift before comparing, or supervise a
scale-free quantity). A naive L1 against a monocular prediction supervises the predictor's
arbitrary affine gauge and will fight the photometric term.

### 5. Multi-view consistency

Nothing landed. The most expensive item and the one most likely to be cut: it needs
neighbour-view selection, a patch warp through the rendered plane, occlusion masking, and a
second render per iteration. Worth deferring until 1 and 3 are measured, because the two
overlap — both constrain normals, and if depth-normal consistency plus depth distortion
clears the `n_gain` bar, this may not earn its cost.

### 6. Scale-z regularisation

Partly present, and the existing version is not the right term. `loss.use_scale` /
`lambda_scale` exist in `configs/base_gs.yaml` and `trainer.py` computes
`abs(get_scale()).mean()` — an isotropic shrink on all three axes, which penalises particle
*size*, not particle *flatness*. Driving all three axes down makes small round particles.

What is wanted is a penalty on the smallest axis only, `min(scale)` or `scale.z` in the
canonical frame, so the particle flattens into a disk while remaining free to grow in the
other two. Whether to extend `use_scale` with a mode or add a separate `use_scale_flatten` is
open; a separate flag is cleaner, since the two terms pull in different directions and
someone will want to ablate them independently.

Only meaningful for the ellipsoid primitives — trisurfel is already flat by construction, so
this is the term that lets the gaussian variant compete on geometry, and the baseline says it
is the variant with the geometry problem (`n_gain` -16.7).

#### Implemented as `use_scale_flatten`, and it does not work

`loss.use_scale_flatten` / `lambda_scale_flatten` penalise `min(scale)` per particle,
normalised by `scene_extent` so one weight transfers between scenes. The reference (PGSR, via
`blob-to-spoke/gaussian_wrapping/train.py`) uses weight 100 on an *unnormalised* min-scale
over visible particles; with the extent divided out, ~300 is the comparable setting here.

Rejected outright for trisurfel. `gaussianParticles.slang` overwrites `scale.z` with 1e-6 on
fetch and never accumulates a gradient into it, so the stored third scale is dead storage that
the renderer never reads — yet it is *not* small (measured aspect ratio 0.36), so `min` would
select it for most particles and the term would spend its entire budget shrinking a number
with no effect, showing a healthy falling loss curve throughout.

Measured on gaussians at 7k. `aspect` is the mean `min(scale)/max(scale)`, i.e. how flat the
particles actually became:

| lambda | sponza n | gain | aspect | emerald n | gain | aspect |
|---|---|---|---|---|---|---|
| off | 52.4 | -9.5 | 0.398 | 65.4 | -15.1 | 0.225 |
| 0.1 | 54.9 | -12.1 | 0.163 | 80.5 | -30.2 | 0.003 |
| 1 | 58.7 | -15.8 | 0.037 | 84.9 | -34.6 | 0.000 |
| 3 | 63.3 | -20.5 | 0.012 | 84.3 | -34.0 | 0.000 |
| 30 | 67.6 | -24.8 | 0.001 | 84.6 | -34.3 | 0.000 |
| 300 | 67.8 | -25.0 | 0.000 | 84.8 | -34.5 | 0.000 |

The term does exactly what it says — the aspect ratio collapses from 0.4 to 0.001 — and the
normals get monotonically *worse* at every weight on both scenes, 52 to 68 degrees on sponza
and 65 to 85 on emerald-square. Stacked on depth-normal consistency it is still a monotone
regression (25.2 to 31.9 on sponza, 42.6 to 46.5 on emerald-square), so it is not that
flattening needs a partner term. PSNR is untouched throughout (36.1-36.3 on sponza), which
rules out the runs simply being worse: this is a geometry-specific regression. The 0.1 and 1
cells were added after the fact precisely because 3 already collapsed the aspect ratio, and
without them the sweep could not distinguish "flattening hurts" from "over-flattening hurts".
It is the former.

The mechanism is that `min` is a hard selection on whichever axis is *currently* smallest, so
the term shrinks that axis and thereby freezes the orientation the particle happened to be
initialised with. Flattening makes a normal well-defined without making it correct, and a
thin disk pointing the wrong way is a worse normal than a round blob with no strong
orientation at all. Nothing in the term references the surface. This also explains why the
regression is much larger on emerald-square, where the particles start rounder (0.225 versus
0.398) and so have more arbitrary orientation to lock in.

So the premise recorded above — that this is "the term that lets the gaussian variant compete
on geometry" — is wrong, and the opposite is true. What fixes gaussian normals is
depth-normal consistency, which cut them from 52.4 to 25.2 while *raising* the aspect ratio
slightly: the useful direction is constraining orientation, not shape. Flatness is a
consequence of good geometry in 2DGS/PGSR, not a cause, and those methods pair it with
multi-view photometric constraints that pin orientation independently.

Kept in the tree, defaulting off, with the guard and tests. The measurement is only
reproducible if the term still exists, and it is ~10 lines. It should not be switched on.

Small caveat, recorded so it is not lost: combined with depth-normal consistency, flattening
does improve *depth* on emerald-square (`abs_rel` 0.1353 to 0.1260). If a later item wants
depth specifically, that is worth revisiting; it does not redeem the term for normals.

### 7. Mesh export

Nothing landed, and the existing tooling is unrelated: `threedgrut/export/` writes point
clouds (PLY) and Gaussian USD volumes, and `add_mesh_to_usdz` *packages* a mesh supplied from
outside rather than extracting one. There is no surface extraction from the trained
representation anywhere in the tree.

Two candidate routes, and the choice should follow from what the supervision work produces:
TSDF fusion of rendered depth over the training views (robust, resolution-limited, needs
depth to be metric and consistent — leans on terms 1/3/5), or Poisson reconstruction from
oriented particle centres (cheap, direct, and only as good as the normals — leans on terms 1
and 6). Poisson is the better first attempt since it consumes exactly what this branch has
been building and needs no new render pass.

This is also the acceptance test for the whole branch. Chamfer distance against OB3D
reference meshes belongs in the ablation report next to `n_gain`.

### 8. Open thread from the baseline

Emerald-square trisurfel floaters (13.5% vs 2.4% gaussian) are diagnosed but unfixed, and are
independent of the normal work. Depth variance (item 3) is the term most likely to fix it
incidentally, since a floater is a low-weight isolated blob and that is exactly what a
distortion penalty suppresses — worth re-checking `d_float` on that scene after item 3 rather
than attacking it directly.

## Verifying

The venv must be on `PATH`, not merely used as the interpreter: the pipeline-consistency test
compiles Slang in a subprocess that resolves `slangc` from `PATH`, and without it that test
fails with a `FileNotFoundError` unrelated to anything it is testing.

```bash
# Full suite (~8 min): 299 passed, 1 skipped
python -m pytest threedgrut threedgut_tracer threedgrt_tracer scripts -q

# The normal-specific tests
python -m pytest threedgut_tracer/tests/test_normal_gradient.py \
                 threedgrt_tracer/tests/test_normal_gradient_support.py \
                 threedgrt_tracer/tests/test_normal_pipeline_consistency.py \
                 threedgrut/utils/tests/test_depth_normal_loss.py \
                 threedgrut/utils/tests/test_geometry_supervision.py -q

black --check . && isort --check-only .   # line-length 120, configured in pyproject.toml
```
