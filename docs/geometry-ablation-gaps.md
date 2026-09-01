# Remaining geometry-ablation gaps

The current branch can compare Gaussian/trisurfel primitives, depth-normal consistency, ordinal
and aligned monocular depth supervision (DA3 and MoGe-3), TSDF meshes, rendered-depth diagnostics,
and DTU/TnT surface metrics. It also has an opacity-weighted **depth variance** along a ray
(`loss.use_depth_variance`), although it is disabled by default and currently requires the 3DGUT
second-moment buffer. A full state-of-the-art ablation still needs:

- Plane-intersection (unbiased) depth/normal rasterization, separate from alpha-blended expected
  ray depth, plus its gradient tests.
- Edge-aware local planar loss and multi-view reprojection/NCC consistency with visibility checks.
- Frozen compressed image-feature supervision is now available: cached DINOv2 or C-RADIOv4 maps,
  frozen PCA or nonlinear autoencoder compression, a direction-free NHT feature head, and an
  alpha-aware low-resolution cosine loss alongside RGB. Remaining work is DINOv3 support, feature
  pyramids, a robust cosine/Charbonnier residual, and sweeps of loss weight, target resolution,
  and RGB-only versus RGB-plus-feature supervision across seeds.
- Ray-distribution appearance and orientation regularizers: per-ray colour variance and
  normal-direction variance, with tests that distinguish a genuinely mixed ray from a sharp
  surface. The current depth-variance term only constrains ray distance.
- A robust surface-location renderer: opacity-weighted median/quantile depth (and, if useful,
  a corresponding quantile normal) as an alternative to expected depth. This needs a
  differentiable or suitably straight-through cumulative-opacity implementation and a direct
  comparison against mean-depth TSDF fusion and supervision.
- MoGe-3 normal and point-map supervision; the present adapter consumes depth only.
- A calibrated confidence model combining prior confidence, reprojection agreement, opacity, and
  image edges, with an ablation of confidence weighting versus hard masks.
- Gaussian-native occupancy/opacity/SDF fields and adaptive marching tetrahedra (GOF-like),
  including level-set and extraction-resolution sweeps.
- Mesh-in-the-loop optimisation (MILo-like): differentiable extraction, mesh rendering, and
  bidirectional mesh--Gaussian consistency.
- Surface-aware densification/pruning and oriented-normal shell closure (Blobs-to-Spokes-like).
- Benchmark-faithful mesh sampling/alignment drivers for every DTU/TnT scene, including saved
  masks, thresholds, point density, and registration residuals.
- Repeated-seed, per-scene sweeps crossing representation, supervision, extraction, and
  confidence choices; report NVS, rendered geometry, and mesh metrics together.
