#!/usr/bin/env python3
"""M1.2 data investigation: 200k random pool stats + stratified 10k sample.

Read-only scan of pxrd_241113_train.lmdb. Produces:
  - data/processed/train10k_seed42.jsonl
  - data/processed/investigate_10k_stats_seed42.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

# Defaults aligned with D10 / plan
DEFAULT_SEED = 42
DEFAULT_POOL_SIZE = 200_000
DEFAULT_TARGET_SIZE = 10_000
DEFAULT_TRAIN_SIZE = 6_088_183
DEFAULT_LMDB = (
    Path(__file__).resolve().parents[1]
    / ".."
    / ".."
    / "alex_aflow_oqmd_mp"
    / "datasets"
    / "pxrd_241113_train.lmdb"
).resolve()
CRYSTAL_SYSTEMS = [
    "cubic",
    "tetragonal",
    "orthorhombic",
    "hexagonal",
    "trigonal",
    "monoclinic",
    "triclinic",
]

_WORKER_ENV: lmdb.Environment | None = None
_WORKER_KEYS: list[bytes] | None = None


@dataclass
class SampleRecord:
    global_idx: int
    lmdb_key: str
    atom_num: int
    peak_num_raw: int
    peak_num_filtered: int
    two_theta_min: float
    two_theta_max: float
    crystal_system: str | None = None
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    lattice_alpha: float | None = None
    lattice_beta: float | None = None
    lattice_gamma: float | None = None
    symmetry_error: str | None = None


def _init_worker(db_path: str, keys: list[bytes]) -> None:
    global _WORKER_ENV, _WORKER_KEYS
    _WORKER_ENV = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    _WORKER_KEYS = keys


def _load_entry(global_idx: int) -> dict[str, Any]:
    assert _WORKER_ENV is not None and _WORKER_KEYS is not None
    key = _WORKER_KEYS[global_idx]
    raw = _WORKER_ENV.begin().get(key)
    if raw is None:
        raise KeyError(f"missing key at index {global_idx}")
    data = pickle.loads(gzip.decompress(raw))
    return data


def _basic_stats(global_idx: int) -> SampleRecord:
    data = _load_entry(global_idx)
    pxrd_x = np.asarray(data["pxrd_x"], dtype=np.float64)
    pxrd_y = np.asarray(data["pxrd_y"], dtype=np.float64)
    atom_num = len(data["p_atom_type"])
    peak_num_raw = int(pxrd_x.shape[0])
    peak_num_filtered = int((pxrd_y > 5).sum())
    return SampleRecord(
        global_idx=global_idx,
        lmdb_key=_WORKER_KEYS[global_idx].decode("ascii"),  # type: ignore[union-attr]
        atom_num=atom_num,
        peak_num_raw=peak_num_raw,
        peak_num_filtered=peak_num_filtered,
        two_theta_min=float(pxrd_x.min()) if peak_num_raw else float("nan"),
        two_theta_max=float(pxrd_x.max()) if peak_num_raw else float("nan"),
    )


def _derive_crystal_system(record: SampleRecord, symprec: float) -> SampleRecord:
    if record.atom_num >= 25:
        return record
    try:
        data = _load_entry(record.global_idx)
        lattice = Lattice(data["p_lattice_matrix"])
        structure = Structure(
            lattice,
            data["p_atom_type"],
            data["p_atom_pos"],
            coords_are_cartesian=False,
        )
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        crystal_system = analyzer.get_crystal_system()
        abc = lattice.abc
        angles = lattice.angles
        record.crystal_system = crystal_system
        record.lattice_a, record.lattice_b, record.lattice_c = map(float, abc)
        record.lattice_alpha, record.lattice_beta, record.lattice_gamma = map(float, angles)
    except Exception as exc:  # noqa: BLE001 - count failures, do not abort pool
        record.symmetry_error = repr(exc)
    return record


def _percentiles(values: list[int], ps: list[int]) -> dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in ps}
    arr = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    out["median"] = float(np.median(arr))
    out["mean"] = float(arr.mean())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    return out


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _stratified_sample(
    candidates: list[SampleRecord],
    target_size: int,
    seed: int,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_cs: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in candidates:
        if rec.crystal_system is not None:
            by_cs[rec.crystal_system].append(rec)

    base = target_size // len(CRYSTAL_SYSTEMS)
    remainder = target_size % len(CRYSTAL_SYSTEMS)
    targets = {cs: base + (1 if i < remainder else 0) for i, cs in enumerate(CRYSTAL_SYSTEMS)}

    selected: list[SampleRecord] = []
    selected_ids: set[int] = set()
    shortfall = 0
    per_class_selected: dict[str, int] = {}

    for cs in CRYSTAL_SYSTEMS:
        pool = by_cs.get(cs, [])
        want = targets[cs]
        if len(pool) <= want:
            picked = pool
            shortfall += want - len(pool)
        else:
            idx = rng.choice(len(pool), size=want, replace=False)
            picked = [pool[i] for i in idx]
        per_class_selected[cs] = len(picked)
        for rec in picked:
            if rec.global_idx not in selected_ids:
                selected.append(rec)
                selected_ids.add(rec.global_idx)

    # Backfill from remaining candidates if any class was short
    if len(selected) < target_size:
        remaining = [r for r in candidates if r.global_idx not in selected_ids]
        need = target_size - len(selected)
        if len(remaining) < need:
            meta = {
                "targets_per_class": targets,
                "per_class_selected": per_class_selected,
                "shortfall_requested": shortfall,
                "final_size": len(selected),
                "warning": f"only {len(selected)} samples available (< {target_size})",
            }
            return selected, meta
        idx = rng.choice(len(remaining), size=need, replace=False)
        for i in idx:
            selected.append(remaining[i])
            selected_ids.add(remaining[i].global_idx)

    meta = {
        "targets_per_class": targets,
        "per_class_selected": per_class_selected,
        "shortfall_requested": shortfall,
        "final_size": len(selected),
    }
    return selected, meta


def _record_to_json(rec: SampleRecord) -> dict[str, Any]:
    d = asdict(rec)
    return d


def run(args: argparse.Namespace) -> dict[str, Any]:
    db_path = str(args.lmdb_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(db_path)

    env = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    env.close()

    train_size = len(keys)
    if train_size != args.train_size:
        print(f"warning: train_size arg={args.train_size}, actual={train_size}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    pool_indices = rng.choice(train_size, size=args.pool_size, replace=False).tolist()

    workers = min(args.workers, cpu_count(), len(pool_indices))
    print(f"Stage A: basic stats on pool_size={args.pool_size} with workers={workers}")

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(db_path, keys),
    ) as pool:
        pool_records = list(
            tqdm(
                pool.imap(_basic_stats, pool_indices, chunksize=256),
                total=len(pool_indices),
                desc="stage_a",
            )
        )

    atom_nums = [r.atom_num for r in pool_records]
    peaks_filtered = [r.peak_num_filtered for r in pool_records]
    peaks_raw = [r.peak_num_raw for r in pool_records]

    small_pool = [r for r in pool_records if r.atom_num < 25]
    print(
        f"Stage B: crystal system on atom_num<25 subset "
        f"(n={len(small_pool)}) symprec={args.symprec}"
    )

    derive_fn = partial(_derive_crystal_system, symprec=args.symprec)
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(db_path, keys),
    ) as pool:
        small_with_cs = list(
            tqdm(
                pool.imap(derive_fn, small_pool, chunksize=64),
                total=len(small_pool),
                desc="stage_b",
            )
        )

    symmetry_errors = sum(1 for r in small_with_cs if r.symmetry_error is not None)
    valid_candidates = [r for r in small_with_cs if r.crystal_system is not None]
    cs_counter_pool = Counter(r.crystal_system for r in valid_candidates)

    selected, sample_meta = _stratified_sample(
        valid_candidates,
        target_size=args.target_size,
        seed=args.seed,
    )
    cs_counter_10k = Counter(r.crystal_system for r in selected)
    peaks_10k = [r.peak_num_filtered for r in selected]

    stats: dict[str, Any] = {
        "seed": args.seed,
        "lmdb_path": db_path,
        "train_size": train_size,
        "pool_size": args.pool_size,
        "target_size": args.target_size,
        "symprec": args.symprec,
        "stage_a": {
            "atom_num": _percentiles(atom_nums, [50, 90, 95, 99]),
            "peak_num_filtered": _percentiles(peaks_filtered, [50, 90, 95, 99]),
            "peak_num_raw": _percentiles(peaks_raw, [50, 90, 95, 99]),
            "atom_num_lt25": int(sum(1 for n in atom_nums if n < 25)),
        },
        "stage_b": {
            "candidates_atom_num_lt25": len(small_pool),
            "symmetry_success": len(valid_candidates),
            "symmetry_fail": symmetry_errors,
            "crystal_system_counts_in_pool": _counter_to_dict(cs_counter_pool),
        },
        "sample_10k": {
            **sample_meta,
            "crystal_system_counts": _counter_to_dict(cs_counter_10k),
            "peak_num_filtered": _percentiles(peaks_10k, [50, 90, 95, 99]),
        },
        "max_peaks_candidates": _suggest_max_peaks(peaks_filtered, peaks_10k),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    size_label = f"{args.target_size // 1000}k" if args.target_size >= 1000 else str(args.target_size)
    jsonl_path = args.output_dir / f"train{size_label}_seed{args.seed}.jsonl"
    stats_path = args.output_dir / f"investigate_{size_label}_stats_seed{args.seed}.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(_record_to_json(rec), ensure_ascii=False) + "\n")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"wrote {jsonl_path} ({len(selected)} records)")
    print(f"wrote {stats_path}")
    return stats


def _suggest_max_peaks(pool_peaks: list[int], sample_peaks: list[int]) -> dict[str, Any]:
    pool_p = _percentiles(pool_peaks, [90, 95, 99])
    sample_p = _percentiles(sample_peaks, [90, 95, 99])
    return {
        "rationale": "cover ~95-99% of filtered peaks in pool/10k; trade-off vs attention O(N^2)",
        "pool_peak_num_filtered": pool_p,
        "sample10k_peak_num_filtered": sample_p,
        "candidates": [
            {
                "max_peaks": int(round(pool_p["p95"])),
                "covers_pool": "~95% (pool)",
                "note": "conservative for training stability",
            },
            {
                "max_peaks": int(round(pool_p["p99"])),
                "covers_pool": "~99% (pool)",
                "note": "balanced default",
            },
            {
                "max_peaks": int(pool_p["max"]),
                "covers_pool": "100% (pool max; may be heavy for attention)",
                "note": "no truncation",
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="M1.2: 10k stratified sample + peak stats")
    parser.add_argument("--lmdb-path", type=Path, default=DEFAULT_LMDB)
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "processed")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
