#!/usr/bin/env python3
"""Compute matrix6 normalization stats from train jsonl (train split only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxrd_cell_indexing.data.normalization import (
    MatrixLatticeNormalizer,
    compute_matrix6_stats_from_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = PROJECT_ROOT / "data" / "processed" / "train100k_seed42.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "lattice_matrix6_stats_100k_seed42.json"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def run(args: argparse.Namespace) -> dict:
    records = load_records(args.input_jsonl)
    stats = compute_matrix6_stats_from_records(records)
    normalizer = MatrixLatticeNormalizer.from_stats(stats)
    output = dict(normalizer.to_dict())
    output["source"] = str(args.input_jsonl)
    output["n_records"] = len(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"wrote {args.output_path}")
    print(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute matrix6 normalization stats")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
