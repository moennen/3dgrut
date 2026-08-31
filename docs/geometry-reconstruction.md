# Geometry reconstruction with Gaussian splatting

Gaussian splatting optimizes image appearance, not a unique surface. Alpha-composited Gaussian
centres can give excellent novel views while leaving depth, normals and extracted meshes
underconstrained. Geometry-oriented methods make a surface definition explicit, constrain it
across views, or replace a post-hoc mesh heuristic with a field that can be meshed.

## What the successful approaches add

| Contribution | Why it matters | Typical mechanism | 3dgrut status |
| --- | --- | --- | --- |
| Surface-aligned primitives | An unconstrained 3D ellipsoid has no stable tangent plane. | Flattened surfels/planar Gaussians; orient the normal axis consistently. | Trisurfels and rendered normal support exist. |
| Surface-consistent depth and normals | Blending Gaussian-centre depths does not describe the plane that rendered the splat. | Rasterize a ray--plane intersection / plane distance, then derive ray depth; retain alpha separately. | Ray-distance and normal buffers exist; plane-intersection depth does not. |
| Local depth--normal consistency | Removes shapes that render well but cannot be a locally coherent surface. | Compare rendered normal with a normal unprojected from rendered depth; make the loss edge aware. | Implemented. |
| Ray-distribution sharpness | A mean depth can hide two separated, partially opaque surfaces on one ray. | Penalize distance, colour, and normal-direction spread; compare mean with median/quantile surface location. | Depth variance exists but is off by default; colour/normal variance and median/quantile depth are missing. |
| Multi-view geometric consistency | A single view cannot resolve textureless, reflective, or repeated structure. | Reproject depth/normal patches and apply geometric and photometric consistency, commonly NCC. | Missing. |
| External geometry priors | Gives a useful cue where photometric supervision is ambiguous. | Monocular depth/normal/point-map supervision, aligned to sparse SfM; use uncertainty/confidence to limit bad priors. | DA3 ordinal and sparse-aligned depth losses exist; MoGe-3 integration is next. |
| Confidence and ambiguity modelling | Prevents an uncertain pseudo-depth or low-parallax region from overriding multi-view evidence. | Learn or estimate per-pixel confidence from reprojection, prior agreement, opacity, and image structure. | Missing unified confidence model. |
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
