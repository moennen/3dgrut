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
| 5. Depth-normal consistency | Not started | Loss term; decision on expected vs median depth, valid-pixel norm, and `referenceSlang` throughput |
| 6. Pseudo-depth supervision | Not started | Monocular depth predictor integration, scale-invariant loss |
| 7. Depth variance along the ray | Not started | Kernel accumulator for `w·t` and `w·t²`; backward pass |
| 8. Multi-view consistency | Not started | Patch warp, neighbour selection, occlusion handling, second render |
| 9. Scale-z regularisation | Mis-named | Current `use_scale` penalises size; need a flatness-only term |
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

**(b) Invalid-pixel normalisation.** The reference does `masked_fill_(0).mean()`, averaging
over *all* pixels, so invalid ones dilute the loss rather than being excluded, and the
effective weight then drifts with the valid fraction. Dividing by the valid count is the
better default; matching the reference bit-for-bit is the argument against.

**(c) 3DGRT requires `referenceSlang`,** which the startup assertion now enforces. Its
throughput relative to the default `reference` pipeline has not been measured, and it
determines whether normal supervision is affordable on 3DGRT at all.

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

```bash
# Full suite (~14 min): 264 passed, 1 skipped
python -m pytest threedgrut threedgut_tracer threedgrt_tracer scripts -q

# The normal-specific tests
python -m pytest threedgut_tracer/tests/test_normal_gradient.py \
                 threedgrt_tracer/tests/test_normal_gradient_support.py \
                 threedgrt_tracer/tests/test_normal_pipeline_consistency.py -q

black --check . && isort --check-only .   # line-length 120, configured in pyproject.toml
```
