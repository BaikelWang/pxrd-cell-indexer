#!/usr/bin/env python3
"""Compute signal-floor baselines on valid1400 without training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.normalization import LatticeNormalizer
from pxrd_cell_indexing.data.dataset import load_sample_list
from pxrd_cell_indexing.eval import length_mape, lattice_mae, top1_lattice_match_proxy
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALID_JSONL = PROJECT_ROOT / "data" / "processed" / "valid1400_seed42.jsonl"
DEFAULT_TRAIN_JSONL = PROJECT_ROOT / "data" / "processed" / "train10k_seed42.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "data" / "processed" / "lattice_stats_seed42.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baseline_floor_seed42.json"


def _records_to_lattice_tensor(records: list[dict]) -> torch.Tensor:
    rows = [
        [
            record["lattice_a"],
            record["lattice_b"],
            record["lattice_c"],
            record["lattice_alpha"],
            record["lattice_beta"],
            record["lattice_gamma"],
        ]
        for record in records
    ]
    return torch.tensor(rows, dtype=torch.float32)


def run(args: argparse.Namespace) -> dict:
    valid_records = load_sample_list(args.valid_jsonl)
    train_records = load_sample_list(args.train_jsonl)
    normalizer = LatticeNormalizer.from_json(args.stats_path)

    cs_counts = Counter(record["crystal_system"] for record in valid_records)
    majority_cs = cs_counts.most_common(1)[0][0]
    majority_idx = CRYSTAL_SYSTEM_TO_IDX[majority_cs]
    majority_acc = cs_counts[majority_cs] / len(valid_records)

    valid_lattice = _records_to_lattice_tensor(valid_records)
    train_lattice = _records_to_lattice_tensor(train_records)
    mean_lattice = train_lattice.mean(dim=0, keepdim=True).repeat(valid_lattice.shape[0], 1)

    result = {
        "valid_size": len(valid_records),
        "train_size": len(train_records),
        "majority_crystal_system": majority_cs,
        "majority_crystal_system_accuracy": majority_acc,
        "mean_lattice_mae": lattice_mae(mean_lattice, valid_lattice),
        "mean_lattice_length_mape": length_mape(mean_lattice, valid_lattice),
        "mean_lattice_top1_match_proxy": top1_lattice_match_proxy(mean_lattice, valid_lattice),
        "crystal_system_counts_valid": dict(cs_counts),
        "mean_lattice_params": {
            name: float(mean_lattice[0, i].item())
            for i, name in enumerate(["a", "b", "c", "alpha", "beta", "gamma"])
        },
        "normalizer": normalizer.to_dict(),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute signal-floor baselines")
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
