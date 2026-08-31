# Depth-model evaluation

Smoke protocol: one view per scene at a 160 px maximum side; meshes use 1,000 samples.

## OB3D depth accuracy

| model | raw abs-rel | scale abs-rel | affine abs-rel |
| --- | ---: | ---: | ---: |
| dav2 | 0.987 | 0.164 | 0.067 |
| dav3 | 0.937 | 0.095 | 0.060 |
| moge3 | 0.228 | 0.070 | 0.071 |

## DTU scan24

Recall is visibility-corrected recall@5 mm; scale and affine are oracle scan-z-buffer fits.

| model | raw recall@5mm | scale recall@5mm | affine recall@5mm |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.017 | 0.179 |
| dav3 | 0.000 | 0.067 | 0.267 |
| moge3 | 0.000 | 0.295 | 0.303 |

DTU TSDF mesh Chamfer is `(accuracy + completeness) / 2` in millimetres; `—` is an empty mesh.

| model | raw Chamfer | scale Chamfer | affine Chamfer |
| --- | ---: | ---: | ---: |
| dav2 | — | — | 33.857 |
| dav3 | — | 43.169 | 32.251 |
| moge3 | — | 15.424 | 13.022 |

## Tanks and Temples Barn

Recall is visibility-corrected at the official 1 cm tolerance.

| model | raw recall@1cm | scale recall@1cm | affine recall@1cm |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.000 | 0.010 |
| dav3 | 0.001 | 0.005 | 0.001 |
| moge3 | 0.000 | 0.002 | 0.009 |

TnT mesh F1 is at the official 1 cm Barn tolerance; `—` denotes an empty mesh.

| model | raw F1 | scale F1 | affine F1 |
| --- | ---: | ---: | ---: |
| dav2 | 0.000 | 0.000 | 0.000 |
| dav3 | 0.000 | 0.000 | 0.000 |
| moge3 | 0.000 | 0.000 | 0.000 |

## Reading this run

- Raw DAv2/DA3 are intentionally uncalibrated relative outputs; their absolute scores are not comparable to a metric-depth claim.
- Per-frame scale/affine use a GT scan z-buffer and are explicitly oracle diagnostics, not deployable alignments.
- DTU uses its ground-plane completeness cull and predicted-surface observation mask; TnT uses its official crop plus a GT visibility z-buffer to remove scan self-occlusion.
- TSDF is the shared posed-depth path used by checkpoint extraction and this benchmark. One-view fusion is expected to be incomplete; mesh values here validate the path, not a competitive reconstruction setting.
