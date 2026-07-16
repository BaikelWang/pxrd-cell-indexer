#!/usr/bin/env python3
"""Rewrite JSONL lattice labels to a crystallographic canonical convention (R9).

Keeps ``lmdb_key`` and peak metadata unchanged; only replaces lattice_* and
adds ``label_convention``. Optionally recomputes crystal_system from the
canonical cell via SpacegroupAnalyzer (default: keep original CS field).
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lmdb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_cell_indexing.data.canonical import (  # noqa: E402
    CanonicalConvention,
    canonicalize_from_lmdb_entry,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--lmdb",
        type=Path,
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
    )
    parser.add_argument(
        "--convention",
        choices=["primitive", "reduced", "niggli"],
        default="niggli",
    )
    args = parser.parse_args()

    env = _open_lmdb(args.lmdb)
    rows_in = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: list[dict[str, Any]] = []
    missing = 0
    for row in tqdm(rows_in, desc=f"relabel->{args.convention}"):
        key = str(row["lmdb_key"]).encode("ascii")
        raw = env.begin().get(key)
        if raw is None:
            missing += 1
            continue
        entry = pickle.loads(gzip.decompress(raw))
        canon = canonicalize_from_lmdb_entry(
            entry, convention=args.convention  # type: ignore[arg-type]
        )
        new_row = dict(row)
        new_row["lattice_a"] = canon.a
        new_row["lattice_b"] = canon.b
        new_row["lattice_c"] = canon.c
        new_row["lattice_alpha"] = canon.alpha
        new_row["lattice_beta"] = canon.beta
        new_row["lattice_gamma"] = canon.gamma
        new_row["label_convention"] = args.convention
        new_row["label_source"] = "p_lattice_matrix+" + args.convention
        out.append(new_row)
    env.close()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input": str(args.input_jsonl),
                "output": str(args.output_jsonl),
                "convention": args.convention,
                "n_in": len(rows_in),
                "n_out": len(out),
                "missing": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
