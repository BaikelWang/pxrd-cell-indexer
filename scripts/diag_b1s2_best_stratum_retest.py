#!/usr/bin/env python3
"""Retest q-search on B1-S2 best stratum after pass-1 / offdiag fixes.

Subset: label CS == Niggli geom AND axial_ok==3 (same as P0/P0.2).
Reports recall@20 and empty-pool rate for orthorhombic + monoclinic.

Usage:
    python scripts/diag_b1s2_best_stratum_retest.py --n-per-label 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.canonical import canonicalize_lattice
from pxrd_cell_indexing.data.dataset import PeakFilterConfig, PXRDDatasetConfig, build_dataloader
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.search.qsearch import (
    DEFAULT_SEARCH_KWARGS,
    DEFAULT_WAVELENGTH_ANGSTROM,
    inverse_d2_from_two_theta_f64,
    search_crystal_system,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "diag_b1s2_best_stratum_retest.json"


def _niggli(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _geom_system(tn: list[float], tol_deg: float = 1.0) -> str:
    n90 = sum(abs(tn[k] - 90.0) <= tol_deg for k in (3, 4, 5))
    if n90 == 3:
        return "orthorhombic"
    if n90 == 2:
        return "monoclinic"
    return "triclinic"


def _axial_ok(gstar: np.ndarray, q: np.ndarray, tol: float = 1e-5) -> int:
    ok = 0
    for gii in (gstar[0, 0], gstar[1, 1], gstar[2, 2]):
        if any(float(np.min(np.abs(q - gii * h * h))) <= tol for h in (1, 2, 3, 4)):
            ok += 1
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/scale_100k_a3_g1_gstar6.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-label", type=int, default=20)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    ds = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(ds, batch_size=config.data.batch_size, num_workers=0, shuffle=False, pin_memory=False)

    labels = ("orthorhombic", "monoclinic")
    pool: dict[str, list] = {cs: [] for cs in labels}
    with torch.no_grad():
        for batch in loader:
            if all(len(pool[cs]) >= args.n_per_label for cs in labels):
                break
            for i in range(batch["lattice"].shape[0]):
                label = CRYSTAL_SYSTEMS[int(batch["crystal_system_idx"][i].item())]
                if label not in labels or len(pool[label]) >= args.n_per_label:
                    continue
                truth = batch["lattice"][i].cpu().numpy().tolist()
                tn = _niggli(truth)
                if _geom_system(tn) != label:
                    continue
                obs = np.asarray(slice_observed_two_theta(batch["pxrd_x"], batch["peak_num"], i), dtype=np.float64)
                obs = obs[np.isfinite(obs)]
                q = inverse_d2_from_two_theta_f64(obs)
                matrix = lattice_params_to_matrix(torch.tensor(truth, dtype=torch.float64)).numpy()
                gstar = np.linalg.inv(matrix @ matrix.T)
                if _axial_ok(gstar, q) < 3:
                    continue
                pool[label].append({"label": label, "truth_niggli": tn, "obs": obs})

    rows = []
    for label in labels:
        hits = 0
        empty = 0
        for j, row in enumerate(pool[label]):
            kw = dict(DEFAULT_SEARCH_KWARGS.get(label, {}))
            kw["pool_budget"] = max(100, int(kw.get("pool_budget", 30)))
            t0 = time.time()
            cands = search_crystal_system(
                row["obs"], label, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM, **kw
            )
            elapsed = time.time() - t0
            hit = None
            for r, c in enumerate(cands):
                if lattice_match_elementwise(
                    c.niggli_params6(), row["truth_niggli"], ltol=args.ltol, atol_deg=args.atol_deg
                ):
                    hit = r
                    break
            if hit is not None and hit < 20:
                hits += 1
            if len(cands) == 0:
                empty += 1
            print(
                f"{label[:4]} {j+1:02d}/{len(pool[label])} n={len(cands):3d} "
                f"hit={hit} wall={elapsed:.1f}s",
                flush=True,
            )
            rows.append(
                {
                    "label": label,
                    "n_cand": len(cands),
                    "hit_rank": hit,
                    "elapsed_s": elapsed,
                }
            )
        n = len(pool[label])
        print(
            f"=== {label}: recall@20={hits/n:.1%} empty={empty/n:.1%} n={n} ===",
            flush=True,
        )

    by = {cs: [r for r in rows if r["label"] == cs] for cs in labels}
    report = {
        "protocol": {"n_per_label": args.n_per_label, "ltol": args.ltol, "atol_deg": args.atol_deg},
        "overall": {
            "n": len(rows),
            "recall@20": float(np.mean([r["hit_rank"] is not None and r["hit_rank"] < 20 for r in rows])),
            "empty_rate": float(np.mean([r["n_cand"] == 0 for r in rows])),
        },
        "by_label": {
            cs: {
                "n": len(by[cs]),
                "recall@20": float(
                    np.mean([r["hit_rank"] is not None and r["hit_rank"] < 20 for r in by[cs]])
                )
                if by[cs]
                else None,
                "empty_rate": float(np.mean([r["n_cand"] == 0 for r in by[cs]])) if by[cs] else None,
            }
            for cs in labels
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))
    print(json.dumps(report["by_label"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
