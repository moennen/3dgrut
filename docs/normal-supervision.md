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
| 6. Pseudo-depth supervision | Landed as an *ordinal* loss, measured over 3 seeds | 30k confirmation. Use `use_pseudo_depth_order` at lambda 0.1: −10 to −11% depth `abs_rel` on all three scenes, the only term here to help all of them, for 0.2–0.6 dB PSNR on two. The scale-invariant *regression* loss the plan called for is superseded — a globally aligned prior is worse than the model it would teach |
| 7. Depth variance along the ray | Landed, measured over 4 seeds | 30k confirmation. Use `depth_variance_relative` at lambda 0.01; the absolute form is superseded and harms both scenes |
| 8. Multi-view consistency | Not started | Patch warp, neighbour selection, occlusion handling, second render |
| 9. Scale-z regularisation | Landed, measured over 4 seeds | 30k confirmation. Best normals so far combined with item 5, at a 0.2 dB PSNR cost |
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

#### Stage 0 landed: the forward accumulator, as a diagnostic only

`render.enable_depth_variance` (3DGUT, default off) accumulates `Σ w·t²` next to the depth's
`Σ w·t` and returns it as `pred_dist_sq`. Variance is formed in torch as
`Σw·t²/Σw − (Σw·t/Σw)²`; the raw moment is returned rather than the variance so the
normalisation stays where the accumulated opacity already lives.

The output is marked non-differentiable in the autograd forward, so a loss built on it raises
instead of silently receiving a zero gradient. Stage 0 is for measuring headroom — whether
`std/depth` actually correlates with depth error and with floater pixels — before paying for
the backward.

Deliberately **variance**, not the published `Σᵢⱼ wᵢwⱼ|tᵢ−tⱼ|` distortion: variance is
degree-2 and so directly measures the expected squared error of summarising the ray's weight
distribution by the depth that is reported, which is the quantity the depth-normal loss and
the depth metrics both consume. The L1 distortion form is gentler but answers a different
question.

#### Measured (7k checkpoints, `scripts/ablation/depth_variance_diagnostic.py`)

Does the spread actually mark the pixels we want to fix? Per pixel with reference depth and
enough accumulated opacity, relative spread `sqrt(Var)/depth` is scored as a *detector* of bad
depth, by AUC — the chance a bad pixel outranks a good one. Controls are the signals already
available without a new accumulator: ray transparency `1 - opacity`, and the relative gradient
of the depth we already render, which marks occlusion boundaries.

AUC for floater pixels (`pred < 0.5 * gt`), and for `delta1` failures:

| checkpoint | floater % | spread | transp | d-grad | δ1 % | spread | transp | d-grad |
|---|---|---|---|---|---|---|---|---|
| gaussian sponza | 0.35 | **0.884** | 0.596 | 0.848 | 3.8 | **0.927** | 0.691 | 0.844 |
| trisurfel sponza | 0.23 | **0.898** | 0.629 | 0.867 | 3.5 | **0.918** | 0.700 | 0.834 |
| dn05 trisurfel sponza | 0.34 | **0.837** | 0.479 | 0.735 | 2.7 | **0.868** | 0.588 | 0.752 |
| gaussian emerald | 2.4 | 0.684 | **0.752** | 0.693 | 13.7 | **0.785** | 0.583 | 0.716 |
| trisurfel emerald | 13.3 | **0.736** | 0.424 | 0.681 | 29.8 | **0.728** | 0.451 | 0.675 |
| gaussian lone-monk | 0.01 | **0.999** | 0.213 | 0.994 | 17.7 | 0.553 | **0.661** | 0.410 |

Rank correlation between spread and relative depth error is 0.35–0.58 across the six.

Four readings, the last three of which argue for restraint:

1. **Spread does see the floaters.** 0.84–0.90 on sponza, and floater pixels carry 1.9–9.4x
   the spread of the rest. Transparency alone is near or *below* chance, so this is not simply
   "the ray never became opaque".
2. **But most of that is visible without a new accumulator.** The relative depth gradient,
   computed from the depth already rendered, comes within 0.02–0.10 AUC of spread in every row
   and beats it once. The *incremental* information is modest. This does not sink the term —
   a depth-gradient penalty would be a bad *loss*, since it fights genuine occlusion
   boundaries, whereas variance is a legitimate objective — but it does mean the diagnostic
   value of the buffer is smaller than the first sponza number suggested.
3. **Spread is not a general depth-error detector.** On lone-monk the δ1 failures sit at 0.553,
   chance, and the depth gradient is *anti*-correlated at 0.410. That scene has 17.7% δ1
   failures and essentially no floaters (0.01%): its depth is wrong in an opaque,
   confidently-placed way that a concentrated ray cannot express. Penalising spread cannot
   reach that failure mode, so a scene-averaged result would dilute the effect towards nothing.
4. **The depth-normal loss already does part of the job.** `dn05` has a median spread of 0.014
   against the baseline's 0.041, a 3x reduction, having never been asked to concentrate
   anything. Item 7 stacked on top has correspondingly less left to take.

So stage 1 is worth building, but the expectation should be a small effect concentrated on
floater-heavy scenes (sponza, emerald), largely absent on lone-monk, and partly pre-empted by
the depth-normal term. Measure it there rather than as a four-scene average.

#### Two things this exposed, both recorded in `AGENTS.md`

- **The depth backward is real, and correct where it is smooth.** Finite differences against
  the analytic gradient of a depth loss agree to 0.01–0.02% on `density`, and to a median
  0.2–0.9% on `positions`/`scale`. A persistent tail (p90 ~8–30%, worst ~50–80%) does *not*
  shrink with the step size and the finite-difference estimate jumps non-monotonically across
  step sizes on exactly those entries — the signature of the hard culling and hit-acceptance
  thresholds, which make a finite difference meaningless there rather than the gradient wrong.
  Rotation receives no meaningful depth gradient, consistent with the normal being the local z
  axis regardless of shape.
- **A process holds one compiled binary.** Adding this variant made `test_normal_axis` fail,
  which looked like a regression in the normal path and was not: the first config to render
  decides the binary for the whole process, so the normals-off variance binary was serving the
  normal tests, which then measured the constant placeholder normal. Verified independently by
  rendering a single particle at identity rotation, where the normal is exactly `(0,0,-1)`,
  the local z axis, and is unchanged by which axis is shortest. `load_3dgut_plugin` now
  raises on a mismatch, and the new tests render in a child process.

#### Stage 1: the backward

Landed for the 3DGUT hand-written compositing path. The moment turns out to be *cheaper* to
differentiate than the depth: `t^2` is `gsqdist = dot(grds, grds)`, which the forward already
computes before taking the square root for `gdist`, so `d(t^2)/d(grds) = 2 * grds` with no
division by `gdist` and no degenerate case at zero distance. Its two contributions fold into
the same `galphaRayHitGrd` and `grdsRayHitGrd` the depth uses, and everything downstream of
those — scale, rotation, position, density — is shared unchanged. The whole backward is about
fifteen lines.

That sharing is also the risk, so the tests differentiate three losses rather than one: the
moment alone, the depth alone, and their sum. A backward that assigned to the shared gradient
instead of accumulating into it would pass the first two and fail the third. A further check
renders the depth-only gradient with the moment compiled *out*, in a nested subprocess, and
requires it to match the moment-enabled build — the regression that folding into shared
accumulators invites.

**All three backward paths carry it.** The renderer has three, selected by configuration rather
than named in the API: the hand-written CUDA `processHitBwd` (K=0, normals off) and two Slang
autodiff entry points, `...BwdToRawParameters` (K=0, normals on) and `...BwdToBuffer` (K>0).
The forward accumulator is shared, so a moment taught to only one of them would be *rendered*
everywhere and differentiable in one configuration — which is what the first cut of this shipped
as, with `Tracer._dist_sq_differentiable` withholding the gradient elsewhere so a loss raised
rather than training on a zero. That was not good enough, because normals-on is precisely the
configuration item 7 has to share with item 5.

The Slang extension turned out to be easy, for a reason worth recording: the Slang backward
replays hits **back-to-front**, where the depth accumulates as
`integratedDepth = lerp(integratedDepth, depth, alpha)` and therefore inverts exactly as
`(D - t*alpha) / (1 - alpha)`. The moment obeys the same recursion with `depth * depth`, so it
drops in beside the depth with no new algebra. `EnableHitDistanceSq` sits next to the existing
`EnableNormal` in `threedgut.slang`, and the accumulators are threaded through as possibly-null
pointers exactly like the normal — the pattern the codebase already had for an optional
differentiable accumulator.

`_dist_sq_differentiable` survives, now gating on the feature flag alone. That guard is still
worth having, and not for the reason it was written: with `enable_depth_variance` off the buffer
is *empty* rather than absent, and a loss on an empty tensor is silently `0.0`. Marking it
non-differentiable turns the config typo — variance loss on, variance buffer off — into a raise.

One cost note, unmeasured. The CUDA path is `#if`-guarded and compiles to nothing when the
moment is off; the Slang path is not, because `slangc` sees the flag as a `static const bool`
and the parameters stay in the signature. The generated CUDA for a disabled build is *not*
byte-identical to the pre-change version: two dead pointer arguments and one load of an
unconsumed stack local survive into it. Those should be eliminated by `nvcc`, but that was not
verified, so treat "free when disabled" as an assumption on the Slang path rather than a
measured fact.

#### Stage 2: the loss, and why it does not work

`depth_variance_loss` penalises `M2 - D^2/acc`, the opacity-weighted variance of the hit
distribution, normalised by the squared scene extent. It landed default-off behind
`loss.use_depth_variance`, guarded so that enabling it without `render.enable_depth_variance`,
or under 3DGRT, raises rather than training against an empty buffer.

**A gradient bug came first, and it is worth recording because it looked like a result.** The
`acc` in the denominator was detached, on the stated grounds that `dL/dacc` contained a
divergent `1/acc^2`. That reasoning was simply wrong -- `D` scales with `acc`, so the term is
`mu^2` and is bounded -- and detaching dropped exactly the term that completes the square:

    correct   dL/dw_i = (t_i - mu)^2            non-negative, translation-invariant
    detached  dL/dw_i = (t_i - mu)^2 - mu^2     negative offset growing as the squared depth

So it pushed opacity *up*, hardest on the most distant geometry: `-4` on a ray at `t=2` but
`-1600` at `t=40`, and `-10500` after sliding the same ray 100 units out. Accumulated opacity
pinned at 1.000 for every weight on both scenes, floaters went 0.018 -> 0.23 and depth bias
-4.2 -> -12.5 on emerald-square. Fixing it took lambda=10 on sponza from 14.6 dB to 30.8 dB.
Four tests in `test_depth_variance_loss.py` now backpropagate to *per-hit* weight and distance
and fail if the detach returns; accumulator-level gradients look reasonable under the bug,
which is why the original tests missed it.

**With the gradient correct, the term still fails, for a reason that is not fixable by
reweighting.** Measured with `scripts/ablation/depth_variance_mechanism.py`, which re-renders
trained checkpoints and asks where the removed spread went:

| | spread | neg_var | signed_err | abs_rel | floaters | tight | wrong given tight | opacity |
|---|---|---|---|---|---|---|---|---|
| sponza base | 0.0391 | 0.0000 | -0.010 | 0.042 | 0.003 | 0.227 | 0.0014 | 0.972 |
| sponza dv1 | 0.0135 | 0.0000 | -0.041 | 0.049 | 0.020 | 0.676 | 0.0402 | 0.973 |
| sponza dv10 | 0.0065 | 0.0000 | -0.116 | 0.120 | 0.059 | 0.883 | 0.2409 | 0.981 |
| emerald base | 0.1048 | 0.0000 | -0.046 | 0.107 | 0.018 | 0.071 | 0.0137 | 0.967 |
| emerald dv1 | 0.0066 | 0.0000 | -0.216 | 0.228 | 0.188 | 0.899 | 0.3377 | 0.985 |

The term does its job -- median relative spread falls 6x on sponza, 16x on emerald -- and it is
not an accumulator bug: the *unclamped* variance is negative on exactly zero rays, so the two
moments are consistent to float precision. Nor is it the transparency escape the docstring
warned about; mean accumulated opacity barely moves. What breaks is that confidence stops
implying correctness. The committed population grows 0.23 -> 0.88, while the probability that
a committed ray is *wrong* grows 0.0014 -> 0.24, a factor of 170. Those rays are wrong toward
the camera: mean signed error -0.38 on the confidently-wrong population at lambda=10.

The cause is that **nothing in the objective refers to the truth**. `M2 - D^2/acc` is zero for
*any* Dirac distribution at *any* distance -- zero for a ray committed to 2m and zero for the
same ray committed to 40m. It is pure sharpening; only the photometric loss says where.

The dynamics are clearest in the term's pairwise form, which is an identity (verified to ten
decimals for two, three and four hits):

    M2 - D^2/acc  ==  (1/(2*acc)) * sum_ij w_i w_j (t_i - t_j)^2

So every hit is pulled towards every other with strength `w_i w_j / acc` -- equivalently every
hit towards `mu` with strength `2 w_i`, the same gradient seen two ways. This is the mip-NeRF
360 distortion loss, squared rather than absolute and scaled by `1/acc`.

Read that way the failure is a two-line derivation. A ray with one floater and one surface has
exactly one pair, so with `w_near = a_near` and `w_far = a_far (1 - a_near)`:

    L = w_near * w_far * (dt)^2 / acc,   w_near * w_far = a_near (1 - a_near) * a_far

which peaks at `a_near = 1/2`. `L` is a **double well** in `a_near`, barrier at one half, and
*both* wells are `L = 0`: delete the floater, or promote it to full opacity and occlude the
true surface. The term cannot tell them apart; only which side of the barrier the ray starts
on decides. Measured sign flip is at `a_near ~ 0.47`, the half shifted by the `acc`
denominator.

What breaks the tie, and always the same way, is that one well is absorbing. At `a_near = 1`
the far surface's weight is `a_far * (1 - a_near) = 0`, so `dL/da_far` is exactly zero -- and so
is the gradient from every other term that reaches it through transmittance, the photometric
loss included. A ray that collapses onto the near surface can never recover. A ray that
collapses onto the far surface can, because an unoccluded near particle at `a = 0` is still
visible to the image loss. So each ray starting above the barrier becomes a permanent floater,
which is `floater_frac` 0.003 -> 0.059.

One consequence for the *implementation*, worth keeping: `t_i` is reached by two routes,
`+2 w_i t_i` through `M2` and `-2 mu w_i` through `D`, which must cancel to `2 w_i (t_i - mu)`.
Were `pred_dist` non-differentiable while `pred_dist_sq` was not, only the first would survive
-- positive for every hit, dragging the whole ray towards the camera irrespective of the mean.
That relative scaling, not the mere presence of each gradient, is what the `both` case in
`test_depth_variance_gradient.py` exists to check.

This also explains the sweep's headline failure, that no lambda transfers between scenes:
sponza's optimum is 0.01-0.1 and emerald is already losing 1.75 dB at 0.01. It is not the
world units -- the extent normalisation handles those -- it is that emerald's baseline spread
is 2.7x sponza's and only 7% of its rays are committed versus 23%, so far more of its rays sit
near the separatrix waiting to be locked onto the wrong surface.

**Conclusion on the absolute form: superseded.** In the window where it does no harm the depth
gain is at or barely above the noise floor (sponza's best is `abs_rel` 0.0577 -> 0.0539, against
a 0.002-0.004 noise band), and the depth-normal term beats it on sponza at every weight. It is
kept behind `depth_variance_relative: false` to preserve what was measured, and should not be
used.

**A prediction recorded here was wrong, twice over.** This section previously claimed the
normalised `std/depth` form "does *not* fix this, since any Dirac still scores zero". The
premise is right and the conclusion does not follow. See below.

#### Stage 3: the relative form, `Var/mu^2`

The obvious next candidate was the mip-NeRF 360 / 2DGS kernel, `sum_ij w_i w_j |t_i - t_j|`,
which is the established form and is implementable here -- the absolute pairwise sum telescopes
for sorted `t`,

    sum_ij w_i w_j |t_i - t_j| = 2 * sum_i w_i * (t_i * W_<i - T_<i)

where `W_<i` and `T_<i` are the running accumulated opacity and integrated depth already in the
payload, so it needs one accumulator and no new state. It was not built, because dividing by
the squared expected depth dominates it on every axis at a fraction of the cost:

| | `\|dt\|` kernel | `Var/mu^2` |
|---|---|---|
| renderer cost | new accumulator, three backward paths | none, same buffers |
| scale dependence | linear in `mu` | exactly flat |
| barrier in `a_near` | 0.49, unchanged | 0.82 |
| uniform fading reduces it | yes, degree one | no, degree zero |

`Var/mu^2 = acc*M2/D^2 - 1` is the squared coefficient of variation, computable from the three
buffers already rendered and already differentiated, so it is one line in the loss.

The third row is the one that was not anticipated, and it is why the earlier dismissal was
wrong. Dividing by `mu^2` charges specifically for the *near* collapse, because committing to a
near floater is what makes `mu` small. The degeneracy is untouched -- both wells are still
exactly zero, as predicted -- but the well's *symmetry* is not, and the symmetry is what decided
the outcome. The barrier moves from 0.487 to 0.818, so the basin that locks a ray onto a floater
falls from 51% of the axis to 18%. "Both minima are still zero" was a true statement about the
objective that said nothing about which minimum descent reaches.

Measured at *matched spread*, which is the only fair comparison since lambda does not carry
between the forms:

| scene, spread | form | wrong given tight | floaters | signed_err | abs_rel |
|---|---|---|---|---|---|
| sponza 0.039 | baseline | 0.0012 | 0.0027 | -0.011 | 0.0415 |
| sponza 0.0135 | absolute | 0.0402 | 0.0204 | -0.0405 | 0.0490 |
| sponza 0.0153 | relative | 0.0223 | 0.0178 | -0.0292 | 0.0390 |
| sponza 0.0065 | absolute | 0.2409 | 0.0587 | -0.1156 | 0.1202 |
| sponza 0.0073 | relative | 0.1353 | 0.0381 | -0.0755 | 0.0817 |
| emerald 0.118 | baseline | 0.0090 | 0.0254 | -0.0628 | 0.1112 |
| emerald 0.0155 | absolute | 0.2517 | 0.1156 | -0.1555 | 0.1736 |
| emerald 0.0196 | relative | 0.1269 | 0.0345 | -0.0153 | 0.1608 |

`wrong | tight` roughly halves at equal sharpening, floaters fall 3.4x on emerald and the
toward-camera bias 10x. The clearest single line is sponza at matched spread: the absolute form
takes `abs_rel` from 0.0415 to 0.0490 while the relative form takes it to 0.0390 -- degrading
where the other improves, at the same reduction in spread.

#### Measured over four seeds, which halves the claim

The first pass was a single seed and looked like a win on both scenes: emerald `abs_rel` 0.1448
-> 0.1370 alongside sponza 0.0562 -> 0.0515. Repeating seeds 1-4 keeps one and dissolves the
other. Emerald's baseline `abs_rel` has a seed spread of 0.0047, three times sponza's, and the
apparent gain sat inside it. The rule against reading one seed is written in `AGENTS.md`; it was
written after making this mistake and then repeated anyway.

| scene | variant | psnr | abs_rel | delta1 | floaters |
|---|---|---|---|---|---|
| sponza | baseline | 36.281 +/- 0.133 | 0.0580 +/- 0.0015 | 0.9489 +/- 0.0011 | 0.0028 +/- 0.0003 |
| sponza | rel 0.01 | 36.268 +/- 0.086 | **0.0523 +/- 0.0008** | **0.9558 +/- 0.0006** | 0.0030 +/- 0.0003 |
| sponza | rel 0.1 | 36.095 +/- 0.149 | **0.0499 +/- 0.0017** | **0.9566 +/- 0.0023** | 0.0052 +/- 0.0006 |
| emerald | baseline | 32.872 +/- 0.086 | 0.1393 +/- 0.0047 | 0.8304 +/- 0.0074 | 0.0237 +/- 0.0017 |
| emerald | rel 0.01 | 32.783 +/- 0.198 | 0.1359 +/- 0.0028 | 0.8343 +/- 0.0020 | 0.0269 +/- 0.0018 |
| emerald | rel 0.1 | 32.618 +/- 0.041 | 0.1394 +/- 0.0009 | 0.8380 +/- 0.0035 | 0.0280 +/- 0.0022 |

**Sponza: real.** At lambda 0.01, `abs_rel` -0.0057 (3.9x the baseline seed sd) and `delta1`
+0.0069 (6.3x) for a PSNR change of -0.013, inside noise. At 0.1 the depth gain grows to -0.0081
(5.5x) and 0.0077 (7.0x) and starts costing 0.19 dB, which is 1.4x sd and therefore real. That is
the first geometry gain in this document that comes with no photometric cost at all.

**Emerald: not established.** `abs_rel` -0.0034 at lambda 0.01 is inside noise, and at 0.1 it is
+0.0001, i.e. nothing. `delta1` improves by about one sd at both weights -- suggestive, not
established. PSNR costs 0.09 dB then 0.25 dB, the latter 2.9x sd and clearly real.

So the transferability claim needs splitting. The strong half holds: a *shared* lambda is now
safe, where the absolute form cost emerald 1.75 dB at the same 0.01 and 6.5 dB at 0.1. The weak
half does not: a shared lambda does not buy a *gain* on both scenes, it buys one on sponza and
approximately nothing on emerald for a small real photometric cost.

`d_cover` is flat, as degree zero requires, so the transparency caveat that hung over the
absolute form is gone rather than merely watched. Depth bias on emerald improves from -5.375 to
-4.081 on the first seed, reversing the absolute form's monotonic degradation to -12.5, though
that was not repeated across seeds and should be read as indicative.

What has *not* changed: the term is still an unanchored sharpener. Floaters rise measurably on
both scenes -- 7.4x sd on sponza at lambda 0.1 -- confirming the degeneracy is mitigated and not
removed, exactly as the barrier analysis predicted. Normals are still worse than the
view-direction control without the depth-normal term, and lambda = 1 already costs 1.3-1.9 dB,
so the usable window is narrow.

**Conclusion: worth having, default-off, on sponza-like scenes.** `depth_variance_relative=true`
at lambda 0.01 is the configuration to use if the term is used at all: a 10% relative reduction
in `abs_rel` for no measurable PSNR cost on one of two scenes, and no harm on the other. That is
a real if narrow result, and it is entirely due to the normalisation rather than to the moment
accumulator the two earlier stages were spent building. The anchor is still what is missing, and
pseudo-depth supervision (plan item 6, section 4 below) is still the thing that would supply it.

### 4. Pseudo-depth supervision

Landed as `loss.use_pseudo_depth_order`, but not in the form predicted, and the prediction is
worth keeping because measuring it is what redirected the work.

#### The original prediction, and why it was wrong

> Must be scale-invariant (fit per-image scale and shift before comparing, or supervise a
> scale-free quantity). A naive L1 against a monocular prediction supervises the predictor's
> arbitrary affine gauge and will fight the photometric term.

The diagnosis was right and the remedy was wrong. Fitting the per-image scale and shift *does*
remove the gauge, and the result is still not worth supervising. `DepthAnythingV2-Base` on
sponza, affine-fitted per frame against ground truth — the most favourable alignment possible,
since it uses the ground truth the loss would not have:

| Alignment | `abs_rel` |
| --- | --- |
| Global affine, one per frame | 0.0677 |
| Per-64×64-patch affine | 0.0158 |
| Per-16×16-patch affine | 0.0116 |
| *The 7k model being trained, for reference* | *0.0580* |

A globally aligned prior is **worse than the model it would be teaching** (0.068 vs 0.058), so
an aligned L1 would have spent a foundation-model dependency to make depth worse, and the
scale-invariance the plan called for would not have saved it. The patch numbers say why: the
prior's *local* structure is excellent and its *global* structure drifts, so the useful signal
is not in any single affine gauge and no amount of fitting one recovers it. This is the same
lesson as the PGSR min-scale port — the reference quantity has to mean here what it means
there, and "monocular depth" does not mean "depth up to one affine".

Two further properties, both cheap to get wrong:

- **The model emits disparity, not depth.** Correlation with `1/z` is +0.978; an affine fit in
  disparity reaches R² 0.956 against 0.821 in depth. Reading it as a distance is not merely
  mis-scaled, it is monotonically inverted.
- **It emits z-depth, while this renderer's convention is Euclidean ray distance.** The radial
  factor reaches 20% at the image corners, so the two are not interchangeable.

#### What landed instead: ordinal supervision

Since only the ordering survives, the loss reads only the ordering. For a pixel pair it asks
whether the render agrees with the prior about which is nearer, and penalises the rendered gap
only where they disagree. That is invariant to *any* increasing transform of the prior, not just
an affine one, which disposes of the alignment problem, the disparity/depth inversion (one sign
flip) and the z-vs-ray-distance question (irrelevant to ordering along a ray) at once. It is
also one-sided: a pair the render already orders correctly contributes exactly zero and no
gradient, so the term cannot fight geometry it agrees with.

`threedgrut/utils/pseudo_depth_loss.py`, with the prior cached per scene by
`threedgrut/datasets/pseudo_depth.py`.

The population this targets is real: on the pixels where the trained model fails `delta1`, the
prior is closer to ground truth **91%** of the time.

#### Measured (7k, 3 seeds, `pd*` variants, gaussian primitive)

`abs_rel`, mean ± stdev over `seed_initialization` 1–3, with the change against the `gaussian`
baseline:

| Variant | sponza | lone-monk | emerald-square |
| --- | --- | --- | --- |
| `gaussian` | 0.0571 ± 0.0009 | 0.0982 ± 0.0019 | 0.1368 ± 0.0010 |
| `pd01_gaussian` (λ=0.1) | **0.0506 ± 0.0008** (−11.4%) | **0.0876 ± 0.0010** (−10.8%) | **0.1228 ± 0.0033** (−10.3%) |
| `pd01_gaussian_gated` | 0.0508 ± 0.0010 (−11.2%) | 0.0873 ± 0.0012 (−11.0%) | 0.1378 ± 0.0045 (+0.7%) |

PSNR, same runs:

| Variant | sponza | lone-monk | emerald-square |
| --- | --- | --- | --- |
| `gaussian` | 36.29 ± 0.10 | 36.72 ± 0.04 | 32.90 ± 0.21 |
| `pd01_gaussian` | 36.41 ± 0.02 (+0.12) | 36.52 ± 0.15 (−0.20) | 32.29 ± 0.20 (−0.61) |

This is **the first geometry term in this document to improve depth on all three scenes**, by a
consistent 10–11% at 4–8σ. Section 3's terms had to be argued scene by scene; this one does not.
The cost is 0.2–0.6 dB PSNR on lone-monk and emerald-square, free on sponza — the same shape of
trade as depth-normal consistency, at roughly twice the depth gain.

The weight sweep at one seed, for where it breaks: λ=1 reaches −18% on lone-monk but costs
emerald 2.4 dB; λ=10 gives up most of the depth gain; λ=100 is catastrophic (emerald `abs_rel`
0.52, PSNR 16.4). The term is well-behaved only in a band, and 0.1 sits in it on all three
scenes.

Lone-monk deserves specific note. Section 3 recorded its depth as wrong "in an opaque,
confidently-placed way that no ray-concentration term can reach" — 17.7% `delta1` failures with
0.01% floaters. `dvrel1_gaussian` manages −1% there; the ordinal prior manages −11%. The
prediction that an external prior would reach the failure an internal consistency condition
cannot held.

#### The gate: predicted from an offline metric, refuted by training

This was the one substantive departure from the reference implementation in
`/mnt/oss/blob-to-spoke`, which normalises the prior's difference to ±1 regardless of magnitude
so that a near-tie counts as much as a confident ordering. Dropping near-ties looked clearly
right offline — measured on sponza against ground truth at 0.05 pair separation:

| Gate (fraction of disparity IQR) | Ordinal agreement with GT | Pairs kept |
| --- | --- | --- |
| 0 (the reference's behaviour) | 84% | 100% |
| 0.05 | 97% | 55% |
| 0.10 | 99% | 38% |

13 points of agreement for 45% of the pairs, and ungated "one pair in six pushes the wrong way".
Trained, it is neutral on sponza and lone-monk (within 0.0003 `abs_rel`, well inside seed noise)
and it costs emerald-square **its entire gain**: +0.7% gated against −10.3% ungated, consistent
across all three seeds (gated 0.1404/0.1404/0.1327, ungated 0.1264/0.1221/0.1199). The default is
now 0.

Two reasons, and both were visible in numbers already in this section:

- A large `|Δdisp|` is a *long-range* comparison, and long range is exactly where this prior
  drifts — one global affine scores 0.068 where per-16×16-patch scores 0.011. The gate therefore
  selects for the prior's weakest structure and discards the local structure that is its
  strongest. Emerald-square is the largest-extent scene of the three, which is why it is the one
  that exposes this.
- 97% agreement is not a benefit, it is a warning. The loss is one-sided, so a pair both sources
  already order the same way contributes nothing; raising agreement to 97% means only 3% of the
  surviving pairs can produce a gradient at all. The gate was selecting the pairs with the least
  to teach.

The methodological error is that the offline metric **counted pairs instead of asking what they
teach**, and pair-count agreement is not the quantity the loss integrates. This is the same
family of mistake as the `.detach()` in the depth-variance term: every check that was run passed,
and the check that mattered was a different one. Where that one needed the gradient rather than
the value, this one needed the trained result rather than the correlation.

Pairs are also formed by cropping rather than by `torch.roll`; that departure stands. Wrapping
pairs opposite image edges, which are unrelated in 3D, manufacturing disagreements the prior
never claimed.

#### The anchor hypothesis: confirmed in direction, still not enough

Every other term here is an internal consistency condition; this one brings external
information. Section 3 predicted that this is what the depth-variance family was missing: per-ray
variance is zero for a Dirac at *any* distance, so it can only ask a ray to commit, not say
where, and committing at the wrong distance is absorbing — the 170× rise in `wrong | tight`.

The first pass at this measured λ=1 for *both* terms — 10× the ordinal term's optimum and 100×
the relative-variance term's — and concluded the pairing was worse than the ordinal term alone.
That conclusion was an artefact of the weights. At each term's own optimum it reverses:

| Variant | sponza | lone-monk | emerald-square |
| --- | --- | --- | --- |
| `pd01_gaussian` alone | −10.9% | −9.7% | −7.6% |
| `dvrel001_gaussian` alone | −7.8% | +1.1% | −3.2% |
| `pd01_dvrel001_gaussian` | **−13.8%** | **−10.4%** | −6.8% |
| `pd1_dvrel1_gaussian` (both over-weighted) | −7.6% | −12.9% | +10.3% |

So the anchor does hold, mildly: at sane weights the pair beats the ordinal term alone on sponza
by 2.9 points and on lone-monk by 0.7, at a 0.1–0.2 dB PSNR cost, and loses slightly on
emerald-square. The gain is small and the sponza result is the only one clearly outside seed
noise, so this is a weak positive and not a reason to run both by default. The lesson is
methodological: **a null from a combination measured at the wrong weights is not a null about
the combination**, and the over-weighted cell is retained in the sweep to keep that visible.

#### Compounding with depth-normal consistency: sub-additive on depth, strongly additive on normals

This is the more interesting pairing, and it splits cleanly by channel.

Depth (`abs_rel` against the 3-seed baseline):

| Variant | sponza | lone-monk | emerald-square |
| --- | --- | --- | --- |
| `pd01_gaussian` alone | −10.9% | **−9.7%** | −7.6% |
| `dn05_gaussian` alone | **−17.0%** | −0.6% | −4.3% |
| `pd01_gaussian_dn` | −17.3% | −7.1% | −7.8% |

The two terms are **not additive on depth** — the combination lands at roughly the *maximum* of
the two, never the sum, and on lone-monk it is actually worse than the ordinal term alone
(−7.1% against −9.7%). They are complementary in *which scene* they fix rather than stacking on
the same one: depth-normal consistency owns sponza and is null on lone-monk, the ordinal prior is
the reverse. That is the same split the diagnostics predicted — sponza's error is geometric
inconsistency an internal condition can reach, lone-monk's is confidently-misplaced depth that
only external information can correct.

Normals are where the combination genuinely compounds:

| Variant | sponza | lone-monk | emerald-square |
| --- | --- | --- | --- |
| `pd01_gaussian` alone | 52.0° / n_gain **−9.2** | 52.1° / **−16.9** | 65.7° / **−15.4** |
| `dn05_gaussian` alone | 25.4° / +17.4 | 37.4° / **−2.1** | 41.3° / +9.0 |
| `pd01_gaussian_dn` | **24.0° / +18.8** | **28.3° / +6.9** | **30.5° / +19.8** |

The ordinal term **alone does nothing for normals** — 52.3° to 52.0° on sponza, and it still
loses to the view-direction control on all three scenes, which is the bar `n_gain` exists to
enforce. That is expected: it constrains depth ordering, and the rendered normal is the
particle's local z axis, which ordering does not touch.

But it improves normals *substantially through* the depth-normal term, which is the channel that
converts depth quality into normal quality. Adding it to `dn05` takes lone-monk from 37.4° to
28.3° and emerald-square from 41.3° to 30.5°. Most importantly, lone-monk's `n_gain` goes from
−2.1 to +6.9: `dn05` alone **loses to the view-direction control there**, and the pair is the
first configuration in this document whose normals beat that control on all three scenes.

The mechanism is worth stating because it generalises: depth-normal consistency ties the normal
to the depth *gradient*, so it can only be as good as the depth it is handed. A term that
improves depth without touching normals still improves normals if a consistency term is present
to carry it across. The ordinal prior is a *depth* term whose main value here turns out to be
what it does for *normals* once paired.

Cost: 0.5 dB PSNR on sponza, 0.2 on lone-monk, 0.9 on emerald-square — the largest photometric
bill of any combination measured here, and the reason this is not a default. `pd1_gaussian_dn`,
over-weighting the ordinal half, is worse on every axis except lone-monk depth.

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

#### Implemented as `use_scale_flatten`

`loss.use_scale_flatten` / `lambda_scale_flatten` penalise `scale.z` per particle, normalised
by `scene_extent` so one weight transfers between scenes.

**It has to be `scale.z`, not `min(scale)`, and getting this wrong inverts the result.** The
rendered normal is the local z axis: `canonicalRayNormal` in `gaussianParticles.slang` returns
`(0,0,1)` rotated into world space, and although it is templated on `Surfel` it never branches
on it and never reads the `scale` argument it is passed. So for an ellipsoid the normal is a
fixed body axis with no relation to the ellipsoid's shape. Penalising the *smallest* axis
therefore flattens the particle along an axis the normal does not track, leaving a thin disk
whose reported normal lies in its own plane. `test_normal_axis.py` pins this by rendering a
single particle with its shortest axis on x, y and z in turn and asserting the normal follows z
in all three cases.

The first version of this term used `min(scale)`, following PGSR (via
`blob-to-spoke/gaussian_wrapping/train.py`, weight 100 on an unnormalised min-scale over
visible particles). That is correct *there*, where the normal is defined as the shortest axis,
and wrong here. The measured cost of the mismatch, at matched weights:

| lambda | sponza `min` | sponza `z` | emerald `min` | emerald `z` |
|---|---|---|---|---|
| off | 52.4 | 52.4 | 65.4 | 65.4 |
| 0.1 | 54.9 | 46.4 | 80.5 | 50.1 |
| 1 | 58.7 | 44.8 | 84.9 | 49.4 |
| 30 | 67.6 | 44.6 | 84.6 | 49.3 |
| 300 | 67.8 | 44.6 | 84.8 | 49.2 |

Penalising the wrong axis degrades normals monotonically to near-perpendicular (84.8 degrees),
and penalising the right one improves them by 8 to 16 degrees. Both collapse the particle
equally well, so the loss curve looks healthy either way; nothing but the normal metric
distinguishes them. This is the third silent-failure mode in this work and the only one that
produced a confidently wrong *conclusion* rather than a null result — it was caught by the
observation that flattening an ellipsoid ought to approach a surfel and so ought to land
between the two primitives, which the `min` numbers flatly contradicted.

Weight sweep on gaussians at 7k, single seed. `z/xy` is the mean
`scale.z / max(scale.x, scale.y)`:

| lambda | sponza n | z/xy | emerald n | z/xy |
|---|---|---|---|---|
| off | 52.4 | 0.937 | 65.4 | 1.686 |
| 0.1 | 46.4 | 0.277 | 50.1 | 0.026 |
| 1 | 44.8 | 0.044 | 49.4 | 0.001 |
| 300 | 44.6 | 0.000 | 49.2 | 0.000 |

Weight 1 is the knee: it reaches `z/xy` 0.04, and nothing above it moves the normals by more
than 0.3 degrees. Default set to 1, still behind `use_scale_flatten: false`.

Note the baseline `z/xy` of 1.686 on emerald-square: for the unregularised gaussian, z is on
average the *longest* axis, so the reported normal points along the particle's long direction.
That is a large part of why baseline gaussian normals lose to a view-direction control.

#### What it actually buys, over 4 seeds

The weight sweep above is single-seed, which is fine for a 16-degree effect and useless for a
0.2 dB one. Repeated at lambda 1 with `seed_initialization` 1-4 (mean +- stdev):

| variant | sponza PSNR | sponza n | emerald PSNR | emerald n |
|---|---|---|---|---|
| gaussian | **36.26**±0.04 | 52.2±0.3 | 32.80±0.09 | 64.9±0.9 |
| trisurfel | 36.08±0.06 | **44.7**±0.1 | **32.89**±0.10 | **49.2**±0.1 |
| gaussian + sz | 36.04±0.08 | 45.0±0.2 | 32.82±0.09 | 49.2±0.1 |
| gaussian + dn | 36.00±0.11 | 25.2±0.4 | 32.41±0.12 | 41.9±1.2 |
| trisurfel + dn | 35.90±0.13 | 23.7±0.1 | 32.32±0.21 | 36.2±0.9 |
| gaussian + sz + dn | 35.88±0.11 | **23.4**±0.2 | 32.46±0.11 | **34.1**±0.2 |

Seed noise is 0.04-0.13 dB on PSNR and 0.1-1.2 degrees on normals, so normal effects of 8-16
degrees are unambiguous and everything at the 0.2 dB scale needs the repeats to interpret.

**The term converges the ellipsoid onto the surfel on both axes, not just on geometry.** It
matches surfel normals exactly (45.0 against 44.7; 49.2 against 49.2) — strong evidence it
does what it claims, since the primitives differ mainly in that the surfel kernel forces
`scale.z`, so driving `scale.z` to zero should reproduce it. But on sponza it also gives up
the ellipsoid's PSNR: -0.22 +- 0.07 against the gaussian baseline, landing at 36.04 against
the surfel's 36.08. There is no free lunch here; softly turning an ellipsoid into a surfel
gets surfel behaviour throughout.

Two claims made earlier in this work were wrong and are corrected here:

- "PSNR is free within noise" was inferred from the *across-weight* scatter (36.01-36.27)
  looking like noise. With seeds held fixed the per-cell sd is only 0.04-0.11, and the sponza
  cost is a real ~4 sigma regression. Across-weight scatter is not a noise estimate.
- The ellipsoid does not have a consistent PSNR advantage to preserve in the first place. It
  is +0.18 +- 0.05 over the surfel on sponza but -0.08 +- 0.10 on emerald-square: the sign
  flips, and the emerald gap is not significant. The advantage is a sponza-specific 0.18 dB.

Combined with depth-normal consistency it is still the best configuration measured, and the
two terms are complementary for a sharp reason: flattening makes z the particle's genuinely
thin axis, and depth-normal consistency is what rotates it to face the surface. Either alone
leaves half the job undone (25.2 and 45.0 separately against 23.4 together, on sponza).

Against the surfel primitive under the same loss the win is smaller than a single seed
suggested: -0.33 +- 0.17 degrees on sponza (2 sigma, call it a match) and -2.14 +- 0.65 on
emerald-square (3.3 sigma, real). So "matches the surfel on sponza, beats it by ~2 degrees on
emerald-square", not the 0.6/1.9 the first run showed. Depth slightly favours the surfel on
emerald-square (0.1277 against 0.1332).

Depth cost: alone it mildly *worsens* depth (`abs_rel` 0.0569 to 0.0602 on sponza); combined
with depth-normal consistency depth is neutral (0.0489 to 0.0490). It is a normals-only term
and should not be enabled to improve depth.

Rejected outright for trisurfel, where the kernel already forces `scale.z` to 1e-6 on fetch and
drops its gradient. The stored third scale is dead storage the renderer never reads, so the
term would shrink a number with no effect while showing a falling loss curve.

#### Follow-up this exposed: the ellipsoid normal is dead code

`gaussianParticles.cuh` (`processHitFwd`, the `else` branch of the `SurfelPrimitive`
condition) contains a real ray-ellipsoid surface normal — the gradient at the intersection
point, which for an anisotropic particle points along its genuinely thin direction. It is
not what runs. The Slang path is live, and it returns the z axis for both primitives with a
`TODO : unify the computation of normals` against it.

So an unregularised gaussian reports a normal that is an arbitrary body axis, which is the
mechanism behind the baseline's headline finding that gaussian normals lose to a
view-direction control. Two candidate fixes now exist: force z to be the meaningful axis
(this item, measured, works) or make the normal follow the geometry (the CUDA branch,
untested on this path). The second is more principled and needs no loss term at all, since
it would make the reported normal correct for a round particle rather than requiring the
particle to be flattened first. Worth measuring before adding further loss terms.

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
# Full suite (~9 min): 387 passed, 1 skipped
python -m pytest threedgrut threedgut_tracer threedgrt_tracer scripts -q

# The normal-specific tests
python -m pytest threedgut_tracer/tests/test_normal_gradient.py \
                 threedgrt_tracer/tests/test_normal_gradient_support.py \
                 threedgrt_tracer/tests/test_normal_pipeline_consistency.py \
                 threedgrut/utils/tests/test_depth_normal_loss.py \
                 threedgrut/utils/tests/test_geometry_supervision.py -q

# The pseudo-depth tests: the ordinal loss and the prior's on-disk cache
python -m pytest threedgrut/utils/tests/test_pseudo_depth_loss.py \
                 threedgrut/datasets/tests/test_pseudo_depth.py -q

black --check . && isort --check-only .   # line-length 120, configured in pyproject.toml
```
