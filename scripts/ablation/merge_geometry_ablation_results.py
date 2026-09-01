#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge isolated OSMO geometry-ablation shards and regenerate the single MD/PDF report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def result_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("results.jsonl")))
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("No results.jsonl files found")
    return files


def merge(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Reject contradictory duplicate cells rather than silently choosing one cloud retry."""
    rows, provenance = {}, []
    for path in result_files(paths):
        provenance.append(str(path))
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("suite"), row.get("scene"), row.get("variant"))
            previous = rows.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting duplicate cell {key} in {path}; choose one retry before merging.")
            rows[key] = row
    return [rows[key] for key in sorted(rows)], provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+", help="Shard directories or results.jsonl files")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance = merge(args.shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = args.out_dir / "results.jsonl"
    results.write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows))
    (args.out_dir / "merge-provenance.json").write_text(
        json.dumps({"inputs": provenance, "cells": len(rows)}, indent=2) + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("report_geometry_ablation.py")),
            str(results),
            "--markdown",
            str(args.out_dir / "geometry-ablation.md"),
            "--pdf",
            str(args.out_dir / "geometry-ablation.pdf"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
