#!/usr/bin/env python3
"""Compute gstar6 (reciprocal-metric Cholesky) normalization stats from train jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxrd_cell_indexing.data.normalization import (
    GStar6Normalizer,
    compute_gstar6_stats_from_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = PROJECT_ROOT / "data" / "processed" / "train100k_niggli_seed42.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "lattice_gstar6_stats_100k_niggli_seed42.json"
)


def load_records(path: Path, *, max_records: int | None = None, seed: int = 42) -> list[dict]:
    """Load jsonl records; optionally reservoir-sample to ``max_records``."""
    if max_records is None:
        records: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    import random

    rng = random.Random(seed)
    sample: list[dict] = []
    n_seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_seen += 1
            if len(sample) < max_records:
                sample.append(row)
            else:
                j = rng.randrange(n_seen)
                if j < max_records:
                    sample[j] = row
    return sample


def run(args: argparse.Namespace) -> dict:
    records = load_records(args.input_jsonl, max_records=args.max_records, seed=args.seed)
    stats = compute_gstar6_stats_from_records(records)
    normalizer = GStar6Normalizer.from_stats(stats)
    output = dict(normalizer.to_dict())
    output["source"] = str(args.input_jsonl)
    output["n_records"] = len(records)
    output["max_records"] = args.max_records
    output["representation"] = "gstar6"
    output["convention"] = args.convention
    output["pack_order"] = "logL11,logL22,logL33,L21,L31,L32"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"wrote {args.output_path}")
    print(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute gstar6 normalization stats")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--convention", type=str, default="niggli")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Reservoir-sample this many rows (for full-train stats without loading 6M).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
