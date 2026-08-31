# Remaining geometry-ablation gaps

The current branch can compare Gaussian/trisurfel primitives, depth-normal consistency, ordinal
and aligned monocular depth supervision (DA3 and MoGe-3), TSDF meshes, rendered-depth diagnostics,
and DTU/TnT surface metrics. A full state-of-the-art ablation still needs:

- Plane-intersection (unbiased) depth/normal rasterization, separate from alpha-blended expected
  ray depth, plus its gradient tests.
- Edge-aware local planar loss and multi-view reprojection/NCC consistency with visibility checks.
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
