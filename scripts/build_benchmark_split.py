#!/usr/bin/env python3
"""Build R9/R10 small benchmark: train3500 + valid700, disjoint lmdb_keys.

Default source: reuse existing ``probefit3500_seed42.jsonl`` as train, and
sample a stratified valid700 from the train LMDB whose keys do not overlap
train3500 / overfit700.

Writes:
  data/processed/benchmark_train3500_seed42.jsonl
  data/processed/benchmark_valid700_seed42.jsonl
  data/processed/lattice_matrix6_stats_benchmark3500_seed42.json
  data/processed/benchmark_split_meta_seed42.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_cell_indexing.data.normalization import (  # noqa: E402
    MatrixLatticeNormalizer,
    compute_matrix6_stats_from_records,
)

CRYSTAL_SYSTEMS = [
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
    "trigonal",
    "monoclinic",
    "triclinic",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )


def _entry_from_key(env: lmdb.Environment, key: bytes) -> dict[str, Any]:
    raw = env.begin().get(key)
    if raw is None:
        raise KeyError(key)
    return pickle.loads(gzip.decompress(raw))


def _record_from_entry(global_idx: int, key: bytes, data: dict[str, Any], *, symprec: float) -> dict[str, Any] | None:
    pxrd_x = np.asarray(data["pxrd_x"], dtype=np.float64)
    pxrd_y = np.asarray(data["pxrd_y"], dtype=np.float64)
    atom_num = len(data["p_atom_type"])
    if atom_num >= 25:
        return None
    try:
        lattice = Lattice(data["p_lattice_matrix"])
        structure = Structure(
            lattice,
            data["p_atom_type"],
            data["p_atom_pos"],
            coords_are_cartesian=False,
        )
        crystal_system = SpacegroupAnalyzer(structure, symprec=symprec).get_crystal_system()
        abc = lattice.abc
        angles = lattice.angles
    except Exception:  # noqa: BLE001
        return None
    return {
        "global_idx": int(global_idx),
        "lmdb_key": key.decode("ascii"),
        "atom_num": int(atom_num),
        "peak_num_raw": int(pxrd_x.shape[0]),
        "peak_num_filtered": int((pxrd_y > 5).sum()),
        "two_theta_min": float(pxrd_x.min()) if pxrd_x.size else float("nan"),
        "two_theta_max": float(pxrd_x.max()) if pxrd_x.size else float("nan"),
        "crystal_system": crystal_system,
        "lattice_a": float(abc[0]),
        "lattice_b": float(abc[1]),
        "lattice_c": float(abc[2]),
        "lattice_alpha": float(angles[0]),
        "lattice_beta": float(angles[1]),
        "lattice_gamma": float(angles[2]),
        "symmetry_error": None,
        "label_convention": "primitive",
    }


def _sample_valid(
    *,
    lmdb_path: Path,
    blocked_keys: set[str],
    per_cs: int,
    seed: int,
    pool_size: int,
    symprec: float,
) -> list[dict[str, Any]]:
    env = _open_lmdb(lmdb_path)
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))[:pool_size]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx in tqdm(order, desc="scan for valid700"):
        key = keys[int(idx)]
        key_s = key.decode("ascii")
        if key_s in blocked_keys:
            continue
        data = _entry_from_key(env, key)
        rec = _record_from_entry(int(idx), key, data, symprec=symprec)
        if rec is None:
            continue
        cs = rec["crystal_system"]
        if len(buckets[cs]) < per_cs:
            buckets[cs].append(rec)
        if all(len(buckets[cs]) >= per_cs for cs in CRYSTAL_SYSTEMS):
            break
    missing = [cs for cs in CRYSTAL_SYSTEMS if len(buckets[cs]) < per_cs]
    if missing:
        raise RuntimeError(f"failed to fill valid700 for: {missing}")
    env.close()
    out: list[dict[str, Any]] = []
    for cs in CRYSTAL_SYSTEMS:
        out.extend(buckets[cs][:per_cs])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=ROOT / "data/processed/probefit3500_seed42.jsonl",
    )
    parser.add_argument(
        "--train-lmdb",
        type=Path,
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
    )
    parser.add_argument("--per-cs-valid", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=400_000)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()

    train_rows = _load_jsonl(args.train_source)
    for row in train_rows:
        row.setdefault("label_convention", "primitive")
    if len(train_rows) != 3500:
        raise SystemExit(f"expected 3500 train rows, got {len(train_rows)}")
    train_cs = Counter(r["crystal_system"] for r in train_rows)
    if any(train_cs[cs] != 500 for cs in CRYSTAL_SYSTEMS):
        raise SystemExit(f"train CS imbalance: {dict(train_cs)}")

    blocked = {r["lmdb_key"] for r in train_rows}
    # Also block overfit700 keys (subset of probefit, but keep explicit).
    overfit = ROOT / "data/processed/overfit700_seed42.jsonl"
    if overfit.exists():
        blocked |= {r["lmdb_key"] for r in _load_jsonl(overfit)}

    valid_rows = _sample_valid(
        lmdb_path=args.train_lmdb,
        blocked_keys=blocked,
        per_cs=args.per_cs_valid,
        seed=args.seed + 1,
        pool_size=args.pool_size,
        symprec=args.symprec,
    )
    valid_keys = {r["lmdb_key"] for r in valid_rows}
    overlap = blocked & valid_keys
    if overlap:
        raise SystemExit(f"train/valid key overlap: {len(overlap)}")

    out_dir: Path = args.out_dir
    train_path = out_dir / f"benchmark_train3500_seed{args.seed}.jsonl"
    valid_path = out_dir / f"benchmark_valid700_seed{args.seed}.jsonl"
    stats_path = out_dir / f"lattice_matrix6_stats_benchmark3500_seed{args.seed}.json"
    meta_path = out_dir / f"benchmark_split_meta_seed{args.seed}.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    stats = compute_matrix6_stats_from_records(train_rows)
    normalizer = MatrixLatticeNormalizer.from_stats(stats)
    stats_payload = dict(normalizer.to_dict())
    stats_payload["source"] = str(train_path)
    stats_payload["n_records"] = len(train_rows)
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    meta = {
        "seed": args.seed,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "stats_path": str(stats_path),
        "train_n": len(train_rows),
        "valid_n": len(valid_rows),
        "train_cs": dict(train_cs),
        "valid_cs": dict(Counter(r["crystal_system"] for r in valid_rows)),
        "overlap": 0,
        "label_convention": "primitive",
        "blocked_keys": len(blocked),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
