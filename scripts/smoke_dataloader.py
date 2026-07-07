#!/usr/bin/env python3
"""Full-pass smoke test for the 10k train dataloader."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pxrd_cell_indexing.data.dataset import PXRDDatasetConfig, build_dataloader, load_sample_list

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_LMDB = (
    PROJECT_ROOT / ".." / ".." / "alex_aflow_oqmd_mp" / "datasets" / "pxrd_241113_train.lmdb"
).resolve()
DEFAULT_JSONL = PROJECT_ROOT / "data" / "processed" / "train10k_seed42.jsonl"


def run(args: argparse.Namespace) -> None:
    config = PXRDDatasetConfig(
        lmdb_path=args.lmdb_path,
        split="train",
        sample_list_path=args.sample_list,
        xrd_augment=False,
        strict=False,
    )
    metadata = load_sample_list(args.sample_list)
    loader = build_dataloader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    peak_counts: list[int] = []
    mismatch = 0
    start = time.perf_counter()

    for batch in loader:
        peak_counts.extend(batch["peak_num"].tolist())
        for sample_id, peak_num in zip(batch["sample_id"], batch["peak_num"].tolist()):
            record = next(r for r in metadata if r["lmdb_key"] == sample_id)
            if peak_num != record["peak_num_filtered"]:
                mismatch += 1

    elapsed = time.perf_counter() - start
    arr = np.asarray(peak_counts, dtype=np.int32)
    meta_arr = np.asarray([r["peak_num_filtered"] for r in metadata], dtype=np.int32)

    print(f"samples: {len(peak_counts)}")
    print(f"batches: {len(loader)} batch_size={args.batch_size} workers={args.num_workers}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"peak_num mismatch vs jsonl: {mismatch}")
    print(
        "loader peak stats:",
        {
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
        },
    )
    print(
        "jsonl peak stats:",
        {
            "min": int(meta_arr.min()),
            "median": float(np.median(meta_arr)),
            "p95": float(np.percentile(meta_arr, 95)),
            "max": int(meta_arr.max()),
        },
    )
    print("crystal_system counts:", dict(Counter(r["crystal_system"] for r in metadata)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test 10k dataloader full pass")
    parser.add_argument("--lmdb-path", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--sample-list", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
