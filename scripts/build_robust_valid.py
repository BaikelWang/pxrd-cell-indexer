#!/usr/bin/env python3
"""A4 (v3 §8.3 / v4 §7): build frozen robust-valid sets from valid1400.

For each named perturbation (zero/jitter/drop/impurity/mixed), reads the
original valid1400 spectra, applies a deterministic (fixed-seed) perturbation
per sample, and writes a self-contained LMDB + matching jsonl sample list
(same schema as the production valid1400 files) so the existing
``PXRDDataset``/eval scripts can read them unmodified with ``xrd_augment=False``.

V-clean is *not* regenerated here -- it is exactly the existing
valid1400 lmdb/jsonl (no perturbation).

Usage:
    python scripts/build_robust_valid.py \
        --base-jsonl data/processed/valid1400_niggli_seed42.jsonl \
        --base-lmdb /nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb \
        --out-dir data/processed/robust_valid \
        --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np

from pxrd_cell_indexing.data.robust_perturb import apply_named_perturbation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# name -> list of severities to sample uniformly per-record (v3 §8.3 ranges).
# "mixed" and "clean" ignore severity (fixed composite / no-op).
NAMED_SEVERITIES: dict[str, list[float]] = {
    "zero": [0.1, 0.2, 0.3],
    "jitter": [0.03, 0.05, 0.10],
    "drop": [1, 2, 4],
    "impurity": [1, 2, 4],
    "mixed": [0.0],
}
INTENSITY_MIN = 5.0


def build_one(
    name: str,
    records: list[dict],
    base_lmdb_path: Path,
    out_dir: Path,
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_lmdb_path = out_dir / f"robust_valid_{name}_seed{seed}.lmdb"
    out_jsonl_path = out_dir / f"robust_valid_{name}_seed{seed}.jsonl"
    if out_lmdb_path.exists():
        out_lmdb_path.unlink()

    severities = NAMED_SEVERITIES[name]
    src_env = lmdb.open(str(base_lmdb_path), subdir=False, readonly=True, lock=False)
    dst_env = lmdb.open(str(out_lmdb_path), subdir=False, map_size=1 << 30)

    out_records: list[dict] = []
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for i, record in enumerate(records):
            raw = src_txn.get(record["lmdb_key"].encode("ascii"))
            if raw is None:
                raise KeyError(f"missing lmdb key {record['lmdb_key']}")
            data = pickle.loads(gzip.decompress(raw))
            two_theta = np.asarray(data["pxrd_x"], dtype=np.float64)
            intensity = np.asarray(data["pxrd_y"], dtype=np.float64)
            # Match the production intensity>5 filter applied at load time
            # (peak_num_filtered in the source jsonl already reflects this).
            mask = intensity > INTENSITY_MIN
            two_theta, intensity = two_theta[mask], intensity[mask]

            rng = np.random.default_rng(seed + i)
            severity = severities[int(rng.integers(0, len(severities)))]
            two_theta_p, intensity_p = apply_named_perturbation(
                two_theta,
                intensity,
                name,  # type: ignore[arg-type]
                float(severity),
                rng,
                intensity_min=INTENSITY_MIN,
            )

            payload = gzip.compress(
                pickle.dumps({"pxrd_x": two_theta_p, "pxrd_y": intensity_p})
            )
            dst_txn.put(record["lmdb_key"].encode("ascii"), payload)

            new_record = dict(record)
            new_record["peak_num_filtered"] = int(two_theta_p.shape[0])
            new_record["peak_num_raw"] = int(two_theta_p.shape[0])
            if two_theta_p.shape[0] > 0:
                new_record["two_theta_min"] = float(two_theta_p.min())
                new_record["two_theta_max"] = float(two_theta_p.max())
            new_record["robust_perturb_name"] = name
            new_record["robust_perturb_severity"] = float(severity)
            out_records.append(new_record)

    src_env.close()
    dst_env.close()

    with out_jsonl_path.open("w", encoding="utf-8") as handle:
        for rec in out_records:
            handle.write(json.dumps(rec) + "\n")
    return out_jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/processed/valid1400_niggli_seed42.jsonl",
    )
    parser.add_argument(
        "--base-lmdb",
        type=Path,
        default=Path(
            "/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data/processed/robust_valid"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--names",
        type=str,
        default="zero,jitter,drop,impurity,mixed",
        help="Comma-separated subset of named perturbations to build.",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.base_jsonl.read_text().splitlines() if line.strip()]
    print(f"loaded {len(records)} base records from {args.base_jsonl}")

    for name in args.names.split(","):
        name = name.strip()
        out_path = build_one(name, records, args.base_lmdb, args.out_dir, args.seed)
        print(f"[{name}] wrote {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
