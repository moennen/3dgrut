# 30k geometry-improvement ablation

`scripts/ablation/run_geometry_ablation.py` is the reproducible experiment driver for the
geometry additions on this branch.  It trains every cell for 30,000 iterations by default,
scores rendered ray-depth/mesh geometry, and creates both `geometry-ablation.md` and
`geometry-ablation.pdf` in the output directory.

The default is the fixed one-third benchmark subset, not a random subset:

- OB3D: `archiviz-flat`, `classroom`, `lone-monk`, `san-miguel`
- DTU: `scan105`, `scan114`, `scan24`, `scan55`, `scan69`
- Tanks and Temples: `Barn`, `Ignatius`

Use `--dataset-scale full` to run all 12 / 15 / 6 supported scenes.  This is a large matrix:
13 variants over 11 reduced scenes (143 training cells), or 429 cells for the full set. A local
serial run can use `--resume` with one output directory. Parallel shards must use separate output
directories: JSONL appends are not a distributed database. Merge them afterwards with the supplied
merge script.

## Setup

The MoGe-3 and C-RADIO dependencies must be available before training.  The path below is the
Horde layout used by the existing benchmark documentation; alter only the local checkout paths.

```bash
cd /home/horde/dev/3dgrut
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/oss/MoGe:/mnt/oss/moge3deps:/mnt/oss/meshdeps:${PYTHONPATH:-}
export THREEDGRUT_RADIO_REPO=/mnt/oss/RADIO
test -f /mnt/oss/MoGe/checkpoints/moge-3-vitl/model.pt
.venv/bin/python -m pip install 'open3d>=0.18'
```

The runner defaults to `/mnt/data/nerf_datasets/...`.  For an uploaded Horde dataset under
`/home/horde/data`, pass every root explicitly:

```bash
DATA=/home/horde/data
ROOT_ARGS="--ob3d-root $DATA/ob3d/OB3D_colmap \
  --dtu-root $DATA/dtu_dataset/dtu --dtu-eval-root $DATA/dtu_dataset/dtu_eval \
  --tnt-root $DATA/tnt_dataset/tnt --tnt-reconstruction-root $DATA/tnt_dataset/tnt_gof"
```

## Run

```bash
.venv/bin/python scripts/ablation/run_geometry_ablation.py \
  --out-dir /home/horde/results/geometry-30k-reduced $ROOT_ARGS --resume
```

For a full-data run:

```bash
.venv/bin/python scripts/ablation/run_geometry_ablation.py \
  --out-dir /home/horde/results/geometry-30k-full --dataset-scale full $ROOT_ARGS --resume
```

Useful sharding examples (do not run overlapping cells concurrently against the same output
directory):

```bash
# One suite shard
.venv/bin/python scripts/ablation/run_geometry_ablation.py \
  --out-dir /home/horde/results/geometry-30k-reduced --suites dtu $ROOT_ARGS --resume

# One feature/full-stack shard
.venv/bin/python scripts/ablation/run_geometry_ablation.py \
  --out-dir /home/horde/results/geometry-30k-reduced \
  --variants radioc4_pca48,full_geometry $ROOT_ARGS --resume
```

`--dry-run` prints every training command without touching the datasets.  `--skip-score` is
only for debugging training: it deliberately leaves DTU/TnT metric cells unavailable.

## Matrix

Each condition begins from the same 3DGUT trisurfel control unless stated otherwise.  The matrix
isolates the added mechanism instead of combining every option with every other option:

- Gaussian and trisurfel references.
- Depth-normal consistency; relative ray-depth variance; ray normal variance; and RGB/NHT-latent
  appearance variance.
- Sparse-aligned MoGe-3 L1 depth supervision.
- Affinity-sampled, visibility-gated multi-view point, raw-RGB L2, and RGB-ZNCC losses.
- Detached opacity/depth/normal/appearance-dispersion confidence weighting.
- Direct NHT with frozen C-RADIOv4 (`c-radio_v4-h`) features, PCA-reduced to 48 dimensions.
- A compatible full stack, including all of the above and all three multi-view terms.

The score process is intentionally separate from training.  It exports Euclidean ray depths,
uses the shared `threedgrut.geometry.tsdf.fuse_depth_frames` extractor (including vertex RGB),
then uses visibility-aware recall and bounded exact surface queries.  DTU reports visible recall
at 5 mm and Chamfer in mm; TnT reports recall/F1 at its official scene threshold.

## Report regeneration

The driver regenerates the report after each invocation.  To regenerate it after merging or
copying JSONL shards:

```bash
.venv/bin/python scripts/ablation/report_geometry_ablation.py \
  /home/horde/results/geometry-30k-reduced/results.jsonl \
  --markdown /home/horde/results/geometry-30k-reduced/geometry-ablation.md \
  --pdf /home/horde/results/geometry-30k-reduced/geometry-ablation.pdf
```

The report only averages scenes that every reported variant completed in that suite. Failed or
unscored cells stay explicit in the Markdown tracker rather than being silently excluded.

## OSMO cloud execution

Use OSMO **independent tasks**, not multi-node TorchRun: every reconstruction cell uses one GPU,
while `(suite, variant)` shards do not communicate. The generator creates 39 independent tasks
for the full 13-condition matrix across the three suites; each task owns a unique object-store
prefix, checkpoints its run directory every ten minutes, and never concurrently appends another
task's JSONL file.

The image must contain this exact revision at `/workspace/3dgrut`, its `.venv`, Open3D, MoGe-3 at
`/opt/MoGe`, the checkpoint at `/opt/models/moge-3-vitl/model.pt`, and a pinned C-RADIO checkout
at `/opt/RADIO`. Build and push that image before submission; do not install these large
dependencies independently in every task.

First inspect your available pool and choose an appropriate one-GPU resource shape. Surface
evaluation needs substantially more host memory than reconstruction, so start at 16 CPUs, 128 GiB
RAM, and 300 GiB ephemeral storage per task.

```bash
osmo pool list
osmo resource list -p <pool> --mode free
```

Prewarm the two immutable image-derived caches once. This is optional but avoids repeating
MoGe-3 and C-RADIO inference in each variant task. `CACHE` must be an initially writable staging
directory; upload it as an object-store prefix after this completes.

```bash
DATA=/home/horde/data
CACHE=/home/horde/geometry-30k-cache
PYTHONPATH=/opt/MoGe:/opt/meshdeps THREEDGRUT_RADIO_REPO=/opt/RADIO \
  .venv/bin/python scripts/ablation/precache_geometry_priors.py \
  --cache-root "$CACHE" --dataset-scale reduced \
  --ob3d-root "$DATA/ob3d/OB3D_colmap" --dtu-root "$DATA/dtu_dataset/dtu" \
  --tnt-reconstruction-root "$DATA/tnt_dataset/tnt_gof" \
  --moge3-model /opt/models/moge-3-vitl/model.pt
```

Generate and validate the workflow. `--cache-url` is optional; omit it for a first functional
run, but expect each worker to populate a private cache. With a prewarmed object-store cache,
the URL must contain the `ob3d/`, `dtu/`, and `tnt/` scene subtrees produced above.

```bash
.venv/bin/python scripts/ablation/generate_osmo_geometry_ablation.py \
  --output /tmp/geometry-30k-osmo.yaml \
  --image registry.example.com/3dgrut:<git-sha> \
  --dataset-url s3://my-bucket/3dgrut-data \
  --cache-url s3://my-bucket/geometry-30k-cache \
  --output-url s3://my-bucket/geometry-30k/<git-sha> \
  --dataset-scale reduced --platform dgx-h100
osmo workflow validate /tmp/geometry-30k-osmo.yaml
osmo workflow submit /tmp/geometry-30k-osmo.yaml --pool <pool>
```

After the tasks complete, download/sync their `shards/` prefix locally and merge them. The merge
step rejects contradictory duplicate cells instead of silently selecting a retry, then writes the
single `results.jsonl`, Markdown tracker, and PDF deck.

```bash
.venv/bin/python scripts/ablation/merge_geometry_ablation_results.py \
  /local/download/geometry-30k/shards \
  --out-dir /local/download/geometry-30k/merged
```
