#!/usr/bin/env python3
"""Export full train LMDB → niggli jsonl (A3-G1 / 6M scale).

Protocol matches train100k construction:
  - keep samples with atom_num < 25 and a resolvable crystal_system
  - lattice labels from ``p_lattice_matrix`` → Niggli
  - crystal_system from SpacegroupAnalyzer(symprec=0.01) on the *original* cell

Usage:
    python scripts/export_train_full_niggli.py --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from collections import Counter
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

from pxrd_cell_indexing.data.canonical import canonicalize_lattice  # noqa: E402

DEFAULT_LMDB = Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb")
DEFAULT_OUT = ROOT / "data" / "processed" / "train_full_niggli_seed42.jsonl"

# Fork workers inherit these (avoid pickling 6M keys into each child).
_KEYS: list[bytes] = []
_DB_PATH: str = ""
_ENV: lmdb.Environment | None = None


def _init_worker() -> None:
    global _ENV
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _ENV = lmdb.open(
        _DB_PATH,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )


def _process(global_idx: int) -> dict[str, Any] | None:
    assert _ENV is not None
    key = _KEYS[global_idx]
    raw = _ENV.begin().get(key)
    if raw is None:
        return None
    data = pickle.loads(gzip.decompress(raw))
    atom_types = data["p_atom_type"]
    atom_num = len(atom_types)
    if atom_num >= 25:
        return None
    pxrd_x = np.asarray(data["pxrd_x"], dtype=np.float64)
    pxrd_y = np.asarray(data["pxrd_y"], dtype=np.float64)
    peak_num_raw = int(pxrd_x.shape[0])
    peak_num_filtered = int((pxrd_y > 5).sum())
    if peak_num_filtered < 1:
        return None
    try:
        lattice = Lattice(data["p_lattice_matrix"])
        structure = Structure(
            lattice,
            atom_types,
            data["p_atom_pos"],
            coords_are_cartesian=False,
        )
        crystal_system = SpacegroupAnalyzer(structure, symprec=0.01).get_crystal_system()
        if crystal_system is None:
            return None
        canon = canonicalize_lattice(data["p_lattice_matrix"], convention="niggli")
    except Exception:  # noqa: BLE001
        return None
    return {
        "global_idx": int(global_idx),
        "lmdb_key": key.decode("ascii"),
        "atom_num": int(atom_num),
        "peak_num_raw": peak_num_raw,
        "peak_num_filtered": peak_num_filtered,
        "two_theta_min": float(pxrd_x.min()),
        "two_theta_max": float(pxrd_x.max()),
        "crystal_system": str(crystal_system),
        "lattice_a": canon.a,
        "lattice_b": canon.b,
        "lattice_c": canon.c,
        "lattice_alpha": canon.alpha,
        "lattice_beta": canon.beta,
        "lattice_gamma": canon.gamma,
        "symmetry_error": None,
        "label_convention": "niggli",
        "label_source": "p_lattice_matrix+niggli",
    }


def main() -> None:
    global _KEYS, _DB_PATH
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmdb", type=Path, default=DEFAULT_LMDB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--chunksize", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="Debug: only first N indices")
    args = p.parse_args()

    if not args.lmdb.is_file():
        raise FileNotFoundError(args.lmdb)

    _DB_PATH = str(args.lmdb)
    env = lmdb.open(
        _DB_PATH,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )
    print("Listing LMDB keys...", flush=True)
    with env.begin() as txn:
        _KEYS = list(txn.cursor().iternext(values=False))
    env.close()
    n_total = len(_KEYS)
    indices = list(range(n_total if args.limit is None else min(args.limit, n_total)))
    print(f"LMDB entries={n_total} export={len(indices)} workers={args.workers}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".partial")
    cs_counts: Counter[str] = Counter()
    n_kept = 0
    n_done = 0

    ctx = mp.get_context("fork")
    with tmp.open("w", encoding="utf-8") as out, ctx.Pool(
        processes=args.workers, initializer=_init_worker
    ) as pool:
        for row in tqdm(
            pool.imap(_process, indices, chunksize=args.chunksize),
            total=len(indices),
            desc="export_full_niggli",
        ):
            n_done += 1
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            cs_counts[row["crystal_system"]] += 1
            n_kept += 1

    os.replace(tmp, args.output)
    meta = {
        "lmdb": str(args.lmdb),
        "output": str(args.output),
        "n_lmdb": n_total,
        "n_scanned": n_done,
        "n_kept": n_kept,
        "crystal_system_counts": dict(sorted(cs_counts.items())),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
