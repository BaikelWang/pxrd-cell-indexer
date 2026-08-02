#!/usr/bin/env python3
"""Add the missing refinement stage: pool -> refine on peaks -> rank by coverage.

The phase4 pipeline emits raw seeds, local_mc cells and supcel derivatives, but
its `celref` stage produces zero rows, so supercell derivatives are never fitted
back to the pattern. This script inserts that step and measures what it buys.

Truth = primitive standard lattice (RealPXRD-solver works in primitive setting).
Hit   = find_mapping(ltol, atol) AND |det-1|<0.25 AND coverage >= cov_min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from pxrd_cell_indexing.data.mp100 import load_mp100_sample  # noqa: E402
from cell_refine import fom_stats, refine_cell  # noqa: E402
from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    PHASE4,
    parse_allcells,
    truth_cells,
)

LTOL, ATOL, DET_TOL = 0.05, 3.0, 0.25
COV_MIN = 0.8
TOPK = 20


def strict_match(pred, truth, ltol=LTOL, atol=ATOL) -> bool:
    try:
        r = Lattice.from_parameters(*pred).find_mapping(
            Lattice.from_parameters(*truth), ltol=ltol, atol=atol
        )
        if r is None:
            return False
        return abs(abs(float(np.linalg.det(r[2]))) - 1.0) < DET_TOL
    except Exception:
        return False


def _worker(payload: dict) -> dict:
    sid = payload["sid"]
    max_cells = payload["max_cells"]
    cov_min = payload["cov_min"]

    cif = CIF_DIR / f"{sid}.cif"
    allc = PHASE4 / sid / f"{sid.replace('-', '_')}.allcells"
    if not cif.exists() or not allc.exists():
        return {"sample_id": sid, "status": "missing"}

    t0 = time.perf_counter()
    obs = np.asarray(load_mp100_sample(cif).two_theta, dtype=np.float64)
    truth = truth_cells(cif)["prim"]

    cands = parse_allcells(allc)
    cands.sort(key=lambda c: -c["McM20"])
    # dedup on rounded params
    pool, seen = [], set()
    for c in cands:
        k = tuple(round(x, 2) for x in c["params"])
        if k in seen:
            continue
        seen.add(k)
        pool.append(c["params"])
        if max_cells and len(pool) >= max_cells:
            break

    # ---- baseline: pool as-is ----
    base = [(p, fom_stats(p, obs)) for p in pool]

    # ---- with refinement: keep whichever of (original, refined) scores better ----
    ref = []
    for p, st in base:
        r = refine_cell(p, obs)
        rp = r["params"]
        st_r = fom_stats(rp, obs)
        if (st_r["coverage"], st_r["m_n"]) >= (st["coverage"], st["m_n"]):
            ref.append((rp, st_r))
        else:
            ref.append((p, st))

    def report(cells, tag):
        # M_N ranking: coverage gates, M_N (which penalises N_poss) decides.
        ordered = sorted(
            cells, key=lambda x: (-(x[1]["coverage"] >= cov_min), -x[1]["m_n"])
        )
        best = ordered[0] if ordered else None
        rank = None
        for i, (p, st) in enumerate(ordered, 1):
            if st["coverage"] >= cov_min and strict_match(p, truth):
                rank = i
                break
        lib_hit = any(
            st["coverage"] >= cov_min and strict_match(p, truth) for p, st in cells
        )
        return {
            f"{tag}_best_cov": best[1]["coverage"] if best else 0.0,
            f"{tag}_best_m_n": best[1]["m_n"] if best else 0.0,
            f"{tag}_n_cov_ok": sum(1 for _, st in cells if st["coverage"] >= cov_min),
            f"{tag}_top1_hit": bool(rank == 1),
            f"{tag}_top20_hit": bool(rank is not None and rank <= TOPK),
            f"{tag}_lib_hit": bool(lib_hit),
            f"{tag}_first_hit_rank": rank,
            f"{tag}_top1_params": best[0] if best else None,
        }

    out = {
        "sample_id": sid,
        "status": "ok",
        "n_pool": len(pool),
        "truth_cov": fom_stats(truth, obs)["coverage"],
        "truth_m_n": fom_stats(truth, obs)["m_n"],
        "wall_s": None,
    }
    out.update(report(base, "base"))
    out.update(report(ref, "refined"))
    out["wall_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cells", type=int, default=0, help="0 = whole pool")
    ap.add_argument("--cov-min", type=float, default=COV_MIN)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/pool_refine_rank.json"
    )
    args = ap.parse_args()

    sids = sorted(
        p.name for p in PHASE4.iterdir() if p.is_dir() and p.name.startswith("mp-")
    )
    if args.limit:
        sids = sids[: args.limit]
    print(f"pool -> refine -> rank, {len(sids)} samples", flush=True)

    payloads = [
        {"sid": s, "max_cells": args.max_cells, "cov_min": args.cov_min} for s in sids
    ]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(sids)}", flush=True)
    rows.sort(key=lambda r: r["sample_id"])
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(sids)

    def rate(field):
        return sum(1 for r in ok if r.get(field)) / n

    summary = {
        "n": n,
        "cov_min": args.cov_min,
        "max_cells": args.max_cells or "all",
        "protocol": {
            "truth": "primitive standard lattice",
            "hit": f"find_mapping({LTOL},{ATOL}) AND |det-1|<{DET_TOL} AND cov>={args.cov_min}",
            "ranking": "coverage desc, then LSQ residual asc",
        },
        "truth_cov_ge_min": sum(1 for r in ok if r["truth_cov"] >= args.cov_min) / n,
        "baseline_no_refine": {
            "top1_hit": rate("base_top1_hit"),
            "top20_hit": rate("base_top20_hit"),
            "lib_hit": rate("base_lib_hit"),
            "median_best_cov": float(np.median([r["base_best_cov"] for r in ok])),
            "median_n_cov_ok": float(np.median([r["base_n_cov_ok"] for r in ok])),
        },
        "with_refinement": {
            "top1_hit": rate("refined_top1_hit"),
            "top20_hit": rate("refined_top20_hit"),
            "lib_hit": rate("refined_lib_hit"),
            "median_best_cov": float(np.median([r["refined_best_cov"] for r in ok])),
            "median_n_cov_ok": float(np.median([r["refined_n_cov_ok"] for r in ok])),
        },
        "median_wall_s": float(np.median([r["wall_s"] for r in ok])),
        "total_wall_s": time.perf_counter() - t0,
    }
    out = {"summary": summary, "per_sample": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
