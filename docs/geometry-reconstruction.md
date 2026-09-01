# Geometry reconstruction with Gaussian splatting

Gaussian splatting optimizes image appearance, not a unique surface. Alpha-composited Gaussian
centres can give excellent novel views while leaving depth, normals and extracted meshes
underconstrained. Geometry-oriented methods make a surface definition explicit, constrain it
across views, or replace a post-hoc mesh heuristic with a field that can be meshed.

## What the successful approaches add

| Contribution | Why it matters | Typical mechanism | 3dgrut status |
| --- | --- | --- | --- |
| Surface-aligned primitives | An unconstrained 3D ellipsoid has no stable tangent plane. | Flattened surfels/planar Gaussians; orient the normal axis consistently. | Trisurfels and rendered normal support exist. |
| Surface-consistent depth and normals | Blending Gaussian-centre depths does not describe the plane that rendered the splat. | Rasterize a ray--plane intersection / plane distance, then derive ray depth; retain alpha separately. | Trisurfel hit depth already uses its local ray--plane intersection; ellipsoid hit depth does not. Both still alpha-blend hit depths into an expected ray depth. |
| Local depth--normal consistency | Removes shapes that render well but cannot be a locally coherent surface. | Compare rendered normal with a normal unprojected from rendered depth; make the loss edge aware. | Implemented. |
| Ray-distribution sharpness | A mean depth can hide two separated, partially opaque surfaces on one ray. | Penalize distance, colour, and normal-direction spread; compare mean with median/quantile surface location. | Depth and normal variance exist. Appearance variance is available in 3DGUT (RGB for SH; pre-decoder feature variance for NHT); all are off by default. Median/quantile depth is missing. |
| Multi-view geometric consistency | A single view cannot resolve textureless, reflective, or repeated structure. | Reproject depth/normal patches and apply geometric and photometric consistency, commonly NCC. | Implemented: sparse pose affinity, depth-visibility gating, point/normal, raw-feature L2, and ZNCC terms. |
| External geometry priors | Gives a useful cue where photometric supervision is ambiguous. | Monocular depth/normal/point-map supervision, aligned to sparse SfM; use uncertainty/confidence to limit bad priors. | DA3 ordinal and sparse-aligned depth losses exist; MoGe-3 integration is next. |
| Confidence and ambiguity modelling | Prevents an uncertain pseudo-depth or low-parallax region from overriding multi-view evidence. | Learn or estimate per-pixel confidence from reprojection, prior agreement, opacity, and image structure. | Implemented deterministic, detached confidence from opacity and optional ray dispersion; multi-view also uses soft depth agreement. A learned uncertainty model remains future work. |
| A continuous extraction field | TSDF fusion of blended depth is convenient but is not a property of the Gaussian scene. | Opacity/occupancy/SDF field queried in 3D, then marching tetrahedra/cubes. | Missing. |
| Mesh-in-the-loop optimisation | A post-hoc mesh can discard geometry that the splats learned. | Differentiably extract and render a mesh during training; enforce two-way mesh--Gaussian agreement. | Missing. |

The first four rows capture the common planar/depth-rasterization line of work (for example,
[PGSR](https://arxiv.org/abs/2406.06521) and
[RaDe-GS](https://arxiv.org/abs/2406.01467)). The field route avoids depth fusion altogether:
[GOF](https://arxiv.org/abs/2404.10772) defines an opacity level set and extracts it with adaptive
marching tetrahedra. More recent systems strengthen this in complementary ways: [AmbiSuR](https://fictionarry.github.io/AmbiSuR-Proj/)
uses photometric disambiguation and prior-guided correction, [MILo](https://arxiv.org/abs/2506.24096)
optimizes a mesh alongside Gaussians, and [Blobs to Spokes](https://arxiv.org/abs/2604.07337)
uses oriented Gaussians, an occupancy formulation, and surface-aware densification.

## Mesh extraction baseline

The broadly used practical baseline is **rendered-depth TSDF fusion**:

1. Render a depth map from every training camera, rejecting invalid/low-opacity pixels.
2. Integrate each RGB-D frame with its calibrated pose into a TSDF volume. The shared implementation preserves source RGB as vertex colors in the exported PLY.
3. Extract the zero crossing as triangles and remove small disconnected components.

This is the extraction route used by AmbiSuR's public code. It is a good common baseline because
it is representation-agnostic and turns every checkpoint into a usable mesh. It must not be
described as a Gaussian-native surface: its result depends on view selection, depth convention,
voxel size, truncation, masks, and component filtering. Record all of those with a mesh.

### Proposed enhancement: pixel-footprint adaptive TSDF

The fixed voxel size is simple but spends the same memory on distant, textureless space as it
does on close, high-resolution observations. A compatible next baseline is a sparse **octree
TSDF**: allocate only cells near valid rendered depths, then recursively split a leaf when its
projected diagonal is larger than a target span (for example, one or two pixels) in any camera
that can see it. For a cell at depth `z` in a pinhole camera with focal length `f`, this is
equivalently a world-cell-width bound

`cell_width <= target_pixels * min_visible(z / f)`.

The minimum world-space pixel footprint is important: it preserves detail visible in the most
resolving view. The equivalent image-space rule is to split according to the **maximum projected
pixel footprint** across visible views. Using the maximum world-space footprint instead would
under-resolve surfaces seen closely by another camera.

Depth integration, opacity masks, ray-to-z conversion, source-RGB fusion, and DTU/TnT scoring
remain unchanged. The implementation requirements beyond the present Open3D `ScalableTSDFVolume`
are an octree/hash storage layer, narrow-band allocation around observations, visibility-aware
refinement, and crack-free adaptive extraction (adaptive dual marching cubes or Transvoxel-style
transition cells). Open3D's scalable volume is sparse in allocated blocks but still uses one
global voxel size. This would be a depth-based intermediate between the current fixed-voxel TSDF
and Gaussian-native pivot/Delaunay methods such as Blobs-to-Spokes.

## Visibility-aware multi-view supervision

The multi-view loss is deliberately independent of the Gaussian primitive: it applies to
ellipsoids and trisurfels alike, and uses rendered **Euclidean ray distance** throughout.
It has two stages.

1. At startup, it builds and persists a sparse directed camera graph at
   `<out_dir>/<experiment>/multiview_affinity.npz`. A candidate pair must jointly see the
   scene AABB centre/corners, exceed a minimum baseline relative to scene distance, and stay
   within the configured view-angle gate. The top `K` targets are retained and sampled in O(1)
   from a Vose alias table according to their overlap/parallax/angle score.
2. At each selected pair, source depth is unprojected into world space, projected into the
   target, and accepted only when both renders are opaque and target ray distance agrees with
   the projected world point. This runtime test is the occlusion/visibility decision; the
   affinity graph merely avoids wasting renders on implausible pairs.

The following losses can be enabled independently:

| Term | Configuration | Definition |
| --- | --- | --- |
| Point geometry | `geometric.lambda_point` | Robust L1 (Charbonnier) distance between the two visible world points, normalized by scene extent. |
| Normal geometry | `geometric.lambda_normal` | `1 - abs(dot(n_source, n_target))`, or signed dot with `signed_normals=true`. Normals must be rendered. |
| Raw feature L2 | `raw_feature_l2.lambda` and `source` | Channelwise mean squared error between source features and bilinearly reprojected target features. Source can be `rgb`, NHT `latent`, or decoded image features. |
| ZNCC | `zncc.lambda` and `source` | Patch zero-mean normalized cross correlation after reprojection; only patches with enough visible samples contribute. |

Minimal geometry-only example:

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
  loss.multiview.enabled=true \
  loss.multiview.geometric.lambda_point=0.05 \
  loss.multiview.geometric.lambda_normal=0.01 \
  render.enable_normals=true
```

For an NHT feature term, use `loss.multiview.raw_feature_l2.source=latent`; for a robust
photometric patch term, use `loss.multiview.zncc.lambda=0.05`. Start the loss after initial
geometry settles (`from_iter`) and use one target per source first; every additional target
adds a full target-view render. `target_batch_cache_size` only caches immutable target batch
inputs, not render outputs.

The initial reprojector intentionally supports global-shutter pinhole/OpenCV-pinhole cameras.
It rejects fisheye/F-theta and rolling-shutter target views rather than treating them as
pinhole cameras. Add their inverse camera models and per-pixel target-pose projection before
using the term on those captures. NCore's normal training sampler is random, so its paired
target path explicitly fetches the graph-selected frame instead of reusing that random sampler.

## Detached geometry confidence

`loss.confidence` turns rendered ambiguity into a reliability map for pseudo-depth and
multi-view supervision. It is explicitly detached: otherwise a model can reduce any weighted
loss by predicting low confidence instead of improving geometry. Opacity is always available;
it ramps from `min_opacity` to `full_opacity`. Optional depth, normal, and appearance dispersion
multiply that score by `exp(-weight * variance)`. The latter three must enable their corresponding
renderer buffers, and configuration fails early if they do not.

The same source/target structural weights are applied to every selected multi-view term. Inside
the hard visibility band, a soft target-depth agreement weight further distinguishes a near
match from a pixel just inside tolerance. Pseudo-depth L1 is weighted per pixel; ordinal pairs
use the geometric mean of their endpoint weights. Enable it without changing the baseline losses:

```bash
python train.py --config-name apps/colmap_3dgut.yaml \
  loss.confidence.enabled=true \
  loss.confidence.depth_variance_weight=1.0 \
  render.enable_depth_variance=true \
  loss.multiview.enabled=true \
  loss.multiview.geometric.lambda_point=0.05
```

Start with opacity-only confidence, then ablate one dispersion source at a time. The TensorBoard
metric `geometry/confidence_mean/train` is a guardrail: a collapse toward zero means the loss is
being starved, not that reconstruction became certain.

## Metrics to report

| Family | Metric | Direction | What it measures | Reporting rule |
| --- | --- | --- | --- | --- |
| Novel view | PSNR | higher | Pixelwise RGB fidelity | Report test views and colour/exposure protocol. |
| Novel view | SSIM | higher | Local luminance/contrast/structure similarity | Use the same implementation and crop for all runs. |
| Novel view | LPIPS | lower | Learned perceptual appearance distance | Name the backbone and colour range. |
| Rendered geometry diagnostic | Depth abs-rel, RMSE, delta-1 | lower, lower, higher | Per-view depth error where reference depth exists | No per-frame GT alignment unless the metric explicitly permits it; rendered depth here is ray distance. |
| Rendered geometry diagnostic | Mean angular normal error and normal gain vs. view direction | lower, higher | Normal fidelity and whether normals carry more information than camera-facing orientation | Normalize alpha-premultiplied normals and exclude zero coverage. |
| DTU mesh | Accuracy | lower | Mean predicted-surface to reference-scan distance | Apply the official observation mask to the prediction. |
| DTU mesh | Completeness | lower | Mean reference-scan to predicted-surface distance | Apply the official ground-plane mask to the reference. |
| DTU mesh | Overall | lower | `(accuracy + completeness) / 2` | Quote millimetres, sampling density, and both masks. |
| TnT mesh | Precision / recall / F-score at scene tau | higher | Fraction of predicted/reference samples within the official threshold, and their harmonic mean | Crop both surfaces with the scene volume, use the benchmark alignment, and state tau in scan metres. |
| General mesh | Symmetric Chamfer distance | lower | Mean bidirectional nearest-surface distance | Supplement, not replace, benchmark metrics; it is sampling-density sensitive. |
| Depth recall | Visible-scan recall at tau | higher | Whether rendered depths land on visible GT scan points | A depth diagnostic, not a mesh metric; use GT visibility maps on TnT. |

For DTU and Tanks and Temples, evaluate a sampled **surface**, not only mesh vertices: vertex
density is an algorithmic choice and can game a nearest-neighbour score. Preserve the alignment,
crop/mask, sampling radius, thresholds, and the signed error split next to every result. A method
with strong NVS scores but weak accuracy/completeness is a radiance-field success, not evidence
of faithful reconstruction.

## Recommended ablation order

Start from the TSDF baseline, then vary one causal group at a time: primitive/plane depth,
local depth--normal regularization, multi-view consistency, prior plus confidence gating, and
Gaussian-native field or mesh-in-the-loop extraction. Score each on NVS, rendered geometry, and
surface geometry; none is a substitute for another.
