# Horde agent runbook: depth-model benchmark

This runbook is for an agent that is already running on a Horde instance. It makes no SSH,
bastion, or host-key configuration assumptions. Keep the repository, vendor dependencies, data,
and results below one writable work directory and retain the final result directory for collection.

The initial run is deliberately bounded to OB3D `emerald-square`, DTU `scan24`, and TnT `Barn`.
Do not copy complete dataset trees when only these scenes are evaluated.

## 1. Publish and fetch the exact branch

On the source workstation, publish the clean current branch. Commits that have not been pushed
cannot be fetched by the Horde agent.

```bash
cd /mnt/oss/3dgrut-bernardin
export BRANCH=nicolasm/surfel-geometry-improvments
git status --short                 # must be empty
git push internal "HEAD:refs/heads/$BRANCH"
git rev-parse HEAD                 # record this revision
```

On Horde, clone/fetch through the authenticated internal Git endpoint available to the agent:

```bash
export WORK=$HOME/3dgrut-depth-benchmark
export REPO=$WORK/3dgrut
export BRANCH=nicolasm/surfel-geometry-improvments
export INTERNAL_REPO_URL=ssh://git@gitlab-master.nvidia.com:12051/nrs/3dgrut-internal.git
mkdir -p "$WORK"
git clone --recursive "$INTERNAL_REPO_URL" "$REPO"
cd "$REPO"
git fetch internal "$BRANCH"
git checkout --track "internal/$BRANCH"
git submodule update --init --recursive
git status --short                 # must be empty
git rev-parse HEAD
```

## 2. Stage the minimal datasets

Reserve at least 30 GB. The minimal set is approximately 4.7 GB before environments, model
weights, generated depths, meshes, and reports.

| dataset content | source path | size |
| --- | --- | ---: |
| OB3D input/GT | `OB3D_colmap/emerald-square` | 2.1 GB |
| DTU input | `dtu/scan24` | 255 MB |
| DTU scan + masks | `stl024_total.ply`, `Plane24.mat`, `ObsMask24_10.mat` | 136 MB |
| TnT official Barn | `tnt/Barn` | 834 MB |
| TnT COLMAP Barn | `tnt_gof/TrainingSet/Barn` | 1.4 GB |

From the source host, use resumable `rsync` to the Horde-visible staging location. Replace
`<horde-data-destination>` with the destination provided to the Horde agent; do not transfer the
full 12 GB DTU evaluation tree.

```bash
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/ob3d/OB3D_colmap/emerald-square \
  <horde-data-destination>/ob3d/OB3D_colmap/
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/dtu_dataset/dtu/scan24 \
  <horde-data-destination>/dtu_dataset/dtu/
mkdir -p <horde-data-destination>/dtu_dataset/dtu_eval/{Points/stl,ObsMask}
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/dtu_dataset/dtu_eval/Points/stl/stl024_total.ply \
  <horde-data-destination>/dtu_dataset/dtu_eval/Points/stl/
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/dtu_dataset/dtu_eval/ObsMask/Plane24.mat \
  /mnt/data/nerf_datasets/dtu_dataset/dtu_eval/ObsMask/ObsMask24_10.mat \
  <horde-data-destination>/dtu_dataset/dtu_eval/ObsMask/
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/tnt_dataset/tnt/Barn \
  <horde-data-destination>/tnt_dataset/tnt/
rsync -aH --partial --info=progress2 /mnt/data/nerf_datasets/tnt_dataset/tnt_gof/TrainingSet/Barn \
  <horde-data-destination>/tnt_dataset/tnt_gof/TrainingSet/
```

On Horde, set `DATA` to this staging directory and verify it before downloading models:

```bash
export DATA=$WORK/data
test -f "$DATA/ob3d/OB3D_colmap/emerald-square/depths/00000_depth.exr"
test -f "$DATA/dtu_dataset/dtu/scan24/cameras.npz"
test -f "$DATA/dtu_dataset/dtu_eval/Points/stl/stl024_total.ply"
test -f "$DATA/dtu_dataset/dtu_eval/ObsMask/ObsMask24_10.mat"
test -f "$DATA/tnt_dataset/tnt/Barn/Barn.ply"
test -f "$DATA/tnt_dataset/tnt_gof/TrainingSet/Barn/sparse/0/images.bin"
```

## 3. Install 3dgrut and mesh support

```bash
cd "$REPO"
./scripts/create_venv_cuda.sh
./install_env_uv.sh
.venv/bin/python -m pip install -e '.[mesh]'
PATH="$REPO/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  threedgrut/geometry/tests/test_tsdf.py scripts/benchmark/tests/test_evaluate_depth_models.py -q
```

## 4. Download model runtimes

Keep DA3 and MoGe dependencies isolated from the 3dgrut venv.

```bash
export VENDOR=$WORK/vendor
mkdir -p "$VENDOR"
git clone https://github.com/microsoft/MoGe.git "$VENDOR/MoGe"
git -C "$VENDOR/MoGe" checkout 74fbce054ebed49800de42d0ad0e83495065719a
mkdir -p "$VENDOR/MoGe/checkpoints/moge-3-vitl" "$VENDOR/moge3deps"
.venv/bin/python -m huggingface_hub.commands.huggingface_cli download Ruicheng/moge-3-vitl model.pt \
  --local-dir "$VENDOR/MoGe/checkpoints/moge-3-vitl"
.venv/bin/python -m pip install --target "$VENDOR/moge3deps" --no-deps \
  'git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37' \
  'git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4' \
  'git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf'

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$VENDOR/Depth-Anything-3"
git -C "$VENDOR/Depth-Anything-3" checkout ed6989a23cd389e975ed9f7cbd7385396e6d867e
mkdir -p "$VENDOR/da3deps"
.venv/bin/python -m pip install --target "$VENDOR/da3deps" evo moviepy pycolmap
```

DAv2 and DA3 model weights download through their official loaders on first inference. Ensure
the Horde environment has any required Hugging Face credentials before the first benchmark run.

## 5. Run smoke, then full benchmark

```bash
cd "$REPO"
export PYTHONPATH="$VENDOR/MoGe:$VENDOR/moge3deps:$VENDOR/Depth-Anything-3/src:$VENDOR/da3deps"
export CUDA_VISIBLE_DEVICES=0
export OUT=$WORK/results/smoke
.venv/bin/python scripts/benchmark/evaluate_depth_models.py \
  --out-dir "$OUT" --models dav2,dav3,moge3 --max-frames 1 --max-image-side 160 \
  --mesh-samples 1000 --gt-voxel 10 \
  --moge3-model "$VENDOR/MoGe/checkpoints/moge-3-vitl/model.pt" \
  --ob3d-root "$DATA/ob3d/OB3D_colmap" \
  --dtu-root "$DATA/dtu_dataset/dtu" --dtu-eval-root "$DATA/dtu_dataset/dtu_eval" \
  --tnt-root "$DATA/tnt_dataset/tnt" --tnt-reconstruction-root "$DATA/tnt_dataset/tnt_gof"
test "$(wc -l < "$OUT/results.jsonl")" -eq 27
```

Only after the smoke succeeds, use the full protocol. Do not set `--max-frames`,
`--max-image-side`, `--mesh-samples`, or `--gt-voxel`; this preserves full resolution and the
two-million-sample surface metric default. The evaluator processes the exact nearest-neighbour
queries in 100,000-sample batches, which is suitable for a 64 GB host. Only use
`--surface-query-chunk-size` to reduce peak RAM further; it does not alter the metric.

```bash
export OUT=$WORK/results/full
mkdir -p "$OUT"
.venv/bin/python scripts/benchmark/evaluate_depth_models.py \
  --out-dir "$OUT" --models dav2,dav3,moge3 \
  --moge3-model "$VENDOR/MoGe/checkpoints/moge-3-vitl/model.pt" \
  --ob3d-root "$DATA/ob3d/OB3D_colmap" --ob3d-scenes emerald-square \
  --dtu-root "$DATA/dtu_dataset/dtu" --dtu-eval-root "$DATA/dtu_dataset/dtu_eval" --dtu-scenes scan24 \
  --tnt-root "$DATA/tnt_dataset/tnt" --tnt-reconstruction-root "$DATA/tnt_dataset/tnt_gof" --tnt-scenes Barn \
  2>&1 | tee "$OUT/run.log"
```

## 6. Generate and preserve the report

```bash
cd "$REPO"
.venv/bin/python scripts/benchmark/report_depth_models.py "$OUT/results.jsonl" \
  --markdown "$OUT/depth-model-evaluation.md" --pdf "$OUT/depth-model-evaluation.pdf" \
  --dtu-root "$DATA/dtu_dataset/dtu" \
  --note 'Full Horde run; see run.log, git_revision.txt, and nvidia-smi.txt.'
git rev-parse HEAD | tee "$OUT/git_revision.txt"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
```

Collect the complete `$OUT` directory: `results.jsonl`, `protocol.json`, `run.log`, the
Markdown/PDF reports, mesh files, and the revision/GPU records. Never mix smoke and full rows
in a single output directory.
