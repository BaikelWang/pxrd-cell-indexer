#!/usr/bin/env python3
"""R9-A: audit primitive vs reduced vs Niggli label conventions vs stored PXRD.

Samples a stratified subset from train JSONL (default: benchmark_train3500),
reloads LMDB entries, and compares:
  - primitive params (current jsonl)
  - Structure.get_reduced_structure() lattice
  - Lattice.get_niggli_reduced_lattice()
  - soft peak Chamfer of theory lines vs stored 2θ

Writes a JSON report under results/beat_engine/r9/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_cell_indexing.data.canonical import (  # noqa: E402
    canonicalize_from_lmdb_entry,
    canonicalize_lattice,
    niggli_is_idempotent,
    params_close,
)
from pxrd_cell_indexing.model.fom import theoretical_two_theta  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _peak_chamfer(obs: np.ndarray, theory: np.ndarray) -> float:
    if obs.size == 0 or theory.size == 0:
        return float("nan")
    # one-way obs -> theory mean min |Δ2θ|
    d = np.abs(obs.reshape(-1, 1) - theory.reshape(1, -1))
    return float(d.min(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data/processed/benchmark_train3500_seed42.jsonl",
    )
    parser.add_argument(
        "--lmdb",
        type=Path,
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
    )
    parser.add_argument("--per-cs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/beat_engine/r9/canonical_audit.json",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.jsonl)
    by_cs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cs[row["crystal_system"]].append(row)
    rng = np.random.default_rng(args.seed)
    sample: list[dict[str, Any]] = []
    for cs, items in by_cs.items():
        order = rng.permutation(len(items))[: args.per_cs]
        sample.extend(items[int(i)] for i in order)

    env = _open_lmdb(args.lmdb)
    reports: list[dict[str, Any]] = []
    idempotent_ok = 0
    niggli_eq_reduced = 0
    chamfer = {"primitive": [], "reduced": [], "niggli": []}

    for row in tqdm(sample, desc="audit"):
        key = row["lmdb_key"].encode("ascii")
        raw = env.begin().get(key)
        if raw is None:
            continue
        entry = pickle.loads(gzip.decompress(raw))
        prim = canonicalize_from_lmdb_entry(entry, convention="primitive")
        red = canonicalize_from_lmdb_entry(entry, convention="reduced")
        nig = canonicalize_from_lmdb_entry(entry, convention="niggli")
        idem = niggli_is_idempotent(entry["p_lattice_matrix"])
        idempotent_ok += int(idem)
        eq = params_close(red.as_params6(), nig.as_params6(), length_tol=1e-3, angle_tol_deg=0.05)
        niggli_eq_reduced += int(eq)

        obs = np.asarray(entry["pxrd_x"], dtype=np.float64)
        inten = np.asarray(entry["pxrd_y"], dtype=np.float64)
        keep = inten > 5.0
        obs = obs[keep][:20]
        for name, lat in (("primitive", prim), ("reduced", red), ("niggli", nig)):
            theory = np.asarray(theoretical_two_theta(lat.as_params6())[:80], dtype=np.float64)
            chamfer[name].append(_peak_chamfer(obs, theory))

        reports.append(
            {
                "lmdb_key": row["lmdb_key"],
                "crystal_system": row["crystal_system"],
                "primitive": prim.as_params6(),
                "reduced": red.as_params6(),
                "niggli": nig.as_params6(),
                "niggli_idempotent": idem,
                "niggli_eq_reduced": eq,
            }
        )

    env.close()
    n = max(len(reports), 1)
    summary = {
        "n_samples": len(reports),
        "niggli_idempotent_rate": idempotent_ok / n,
        "niggli_eq_reduced_rate": niggli_eq_reduced / n,
        "mean_peak_chamfer_2theta": {
            k: float(np.nanmean(v)) if v else float("nan") for k, v in chamfer.items()
        },
        "median_peak_chamfer_2theta": {
            k: float(np.nanmedian(v)) if v else float("nan") for k, v in chamfer.items()
        },
        "recommendation": None,
        "samples": reports[:20],  # keep report small; full stats above
    }
    # Prefer Niggli if idempotent and peak fit not worse than reduced by >10% relative.
    red_c = summary["mean_peak_chamfer_2theta"]["reduced"]
    nig_c = summary["mean_peak_chamfer_2theta"]["niggli"]
    if summary["niggli_idempotent_rate"] >= 0.99 and (
        not np.isfinite(red_c) or nig_c <= red_c * 1.10 + 1e-6
    ):
        summary["recommendation"] = "niggli"
    else:
        summary["recommendation"] = "reduced"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
