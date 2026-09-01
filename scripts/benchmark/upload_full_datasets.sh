#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Upload every locally available OB3D, DTU, and Tanks and Temples benchmark asset to Horde.
#
# The operation is additive: it never uses --delete.  Interrupted transfers can be re-run; rsync
# verifies and appends partial files instead of starting them again.  The destination layout is the
# one consumed by scripts/benchmark/evaluate_depth_models.py and the Horde benchmark runbook.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/benchmark/upload_full_datasets.sh [options]

Options:
  --source-root PATH   Local dataset root (default: /mnt/data/nerf_datasets)
  --remote HOST        Horde SSH target
                        (default: horde@nicolasm-gsplat.ov-agent-farm.svc.cluster.local)
  --proxy-jump HOST    SSH bastion target
                        (default: horde@bastion.horde-gke.nvidia.com:2222)
  --remote-data PATH   Destination data root (default: /home/horde/data)
  --dry-run            Print the remote layout and rsync changes without copying data
  -h, --help           Show this help

The full source set is about 61 GiB on the current workstation. Reserve at least 100 GiB on the
remote host for inputs plus aligned depths, visibility maps, meshes, and model caches.
EOF
}

source_root=/mnt/data/nerf_datasets
remote=horde@nicolasm-gsplat.ov-agent-farm.svc.cluster.local
proxy_jump=horde@bastion.horde-gke.nvidia.com:2222
remote_data=/home/horde/data
dry_run=false

while (($#)); do
    case "$1" in
        --source-root)
            source_root=$2
            shift 2
            ;;
        --remote)
            remote=$2
            shift 2
            ;;
        --proxy-jump)
            proxy_jump=$2
            shift 2
            ;;
        --remote-data)
            remote_data=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

sources=(
    "$source_root/ob3d/OB3D_colmap"
    "$source_root/dtu_dataset/dtu"
    "$source_root/dtu_dataset/dtu_eval"
    "$source_root/tnt_dataset/tnt"
    "$source_root/tnt_dataset/tnt_gof"
)
for source in "${sources[@]}"; do
    if [[ ! -d $source ]]; then
        printf 'Missing required source directory: %s\n' "$source" >&2
        exit 1
    fi
done

# StrictHostKeyChecking=no matches the Horde connection policy requested for this ephemeral host.
# Keep the command as an argv array for the control connection; rsync requires the equivalent
# transport string through -e.
ssh_options=(-o StrictHostKeyChecking=no -J "$proxy_jump")
rsync_transport="ssh -o StrictHostKeyChecking=no -J $proxy_jump"
rsync_options=(-aH --partial --append-verify --info=progress2 --human-readable -e "$rsync_transport")
if "$dry_run"; then
    rsync_options+=(--dry-run)
fi

printf 'Source footprint:\n'
du -sh "${sources[@]}"
printf '\nRemote free space:\n'
if ! "$dry_run"; then
    ssh "${ssh_options[@]}" "$remote" "df -h '$remote_data' 2>/dev/null || df -h \"\$HOME\""
fi

remote_dirs=(
    "$remote_data/ob3d/OB3D_colmap"
    "$remote_data/dtu_dataset/dtu"
    "$remote_data/dtu_dataset/dtu_eval"
    "$remote_data/tnt_dataset/tnt"
    "$remote_data/tnt_dataset/tnt_gof"
)
if "$dry_run"; then
    printf '\nWould create remote directories:\n'
    printf '  %s\n' "${remote_dirs[@]}"
else
    ssh "${ssh_options[@]}" "$remote" mkdir -p "${remote_dirs[@]}"
fi

for index in "${!sources[@]}"; do
    source=${sources[index]}/
    destination=$remote:${remote_dirs[index]}/
    printf '\nSyncing %s -> %s\n' "$source" "$destination"
    rsync "${rsync_options[@]}" "$source" "$destination"
done

if "$dry_run"; then
    printf '\nDry run complete; no data was copied.\n'
else
    printf '\nUpload complete. Verify the remote input layout with:\n'
    printf '  ssh -o StrictHostKeyChecking=no -J %q %q %q\n' "$proxy_jump" "$remote" "du -sh '$remote_data'/*"
fi
