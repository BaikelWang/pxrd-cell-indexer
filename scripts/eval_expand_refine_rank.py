#!/usr/bin/env python3
"""Full small pipeline: pool -> HNF expand -> refine on peaks -> rank by M_N.

Rationale
---------
* refinement fixes metric accuracy but cannot change |det|, so a subcell stays
  a subcell;
* HNF expansion fixes |det| but inherits (and multiplies) the base cell's error.

Neither works alone. This runs them in sequence and reports what each buys.

Truth = primitive standard lattice.
Hit   = find_mapping(0.05, 3) AND |det-1|<0.25 AND coverage >= cov_min.
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
from cell_refine import fom_stats, refine_cell, symmetrize_and_refine  # noqa: E402
from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    PHASE4,
    parse_allcells,
    truth_cells,
)
from test_subcell_to_primitive import hnf_matrices  # noqa: E402

LTOL, ATOL, DET_TOL = 0.05, 3.0, 0.25
TOPK = 20


def strict_match(pred, truth) -> bool:
    try:
        r = Lattice.from_parameters(*pred).find_mapping(
            Lattice.from_parameters(*truth), ltol=LTOL, atol=ATOL
        )
        if r is None:
            return False
        return abs(abs(float(np.linalg.det(r[2]))) - 1.0) < DET_TOL
    except Exception:
        return False


def cell_key(params) -> tuple:
    """Cheap dedup key: sorted lengths + sorted angles, coarsely rounded."""
    a, b, c, al, be, ga = params
    return (
        tuple(round(x, 1) for x in sorted((a, b, c))),
        tuple(round(x, 0) for x in sorted((al, be, ga))),
    )


def _worker(payload: dict) -> dict:
    sid = payload["sid"]
    cov_min = payload["cov_min"]
    max_bases = payload["max_bases"]
    max_index = payload["max_index"]

    cif = CIF_DIR / f"{sid}.cif"
    allc = PHASE4 / sid / f"{sid.replace('-', '_')}.allcells"
    if not cif.exists() or not allc.exists():
        return {"sample_id": sid, "status": "missing"}

    t0 = time.perf_counter()
    obs = np.asarray(load_mp100_sample(cif).two_theta, dtype=np.float64)
    truth = truth_cells(cif)["prim"]

    cands = parse_allcells(allc)
    pool, seen = [], set()
    for c in cands:
        k = cell_key(c["params"])
        if k in seen:
            continue
        seen.add(k)
        pool.append(c["params"])

    # ---------- stage 1: refine the pool itself ----------
    refined_pool = []
    for p in pool:
        st = fom_stats(p, obs)
        r = refine_cell(p, obs)
        st_r = fom_stats(r["params"], obs)
        if (st_r["n_indexed"], st_r["m_n"]) >= (st["n_indexed"], st["m_n"]):
            refined_pool.append((r["params"], st_r))
        else:
            refined_pool.append((p, st))
    t_refine1 = time.perf_counter() - t0

    def rank_and_report(cells, tag):
        ordered = sorted(
            cells, key=lambda x: (-(x[1]["coverage"] >= cov_min), -x[1]["m_n"])
        )
        rank = None
        for i, (p, st) in enumerate(ordered, 1):
            if st["coverage"] >= cov_min and strict_match(p, truth):
                rank = i
                break
        return {
            f"{tag}_n": len(cells),
            f"{tag}_best_cov": ordered[0][1]["coverage"] if ordered else 0.0,
            f"{tag}_top1_hit": bool(rank == 1),
            f"{tag}_top20_hit": bool(rank is not None and rank <= TOPK),
            f"{tag}_lib_hit": bool(rank is not None),
            f"{tag}_first_hit_rank": rank,
            f"{tag}_top1_params": ordered[0][0] if ordered else None,
        }

    out = {"sample_id": sid, "status": "ok", "n_pool": len(pool)}
    out.update(rank_and_report(refined_pool, "refine_only"))

    # ---------- stage 2: expand the best bases, then refine again ----------
    # a useful base indexes many lines accurately, even if it misses the rest
    bases = sorted(refined_pool, key=lambda x: (-x[1]["n_indexed"], -x[1]["m_n"]))
    bases = [p for p, _ in bases[:max_bases]]

    expanded, seen_e = [], set(cell_key(p) for p, _ in refined_pool)
    for p in bases:
        try:
            M = Lattice.from_parameters(*p).matrix
        except Exception:
            continue
        for n in range(2, max_index + 1):
            for H in hnf_matrices(n):
                try:
                    L = Lattice(np.asarray(H, dtype=float) @ M)
                    q = [L.a, L.b, L.c, L.alpha, L.beta, L.gamma]
                except Exception:
                    continue
                k = cell_key(q)
                if k in seen_e:
                    continue
                seen_e.add(k)
                expanded.append(q)
    t_expand = time.perf_counter() - t0 - t_refine1

    all_cells = list(refined_pool)
    for q in expanded:
        st = fom_stats(q, obs)
        # only pay for refinement where the cell already explains something
        if st["n_indexed"] >= 3:
            r = refine_cell(q, obs)
            st_r = fom_stats(r["params"], obs)
            if (st_r["n_indexed"], st_r["m_n"]) >= (st["n_indexed"], st["m_n"]):
                all_cells.append((r["params"], st_r))
                continue
        all_cells.append((q, st))

    out.update(rank_and_report(all_cells, "expand_refine"))

    # ---------- stage 3: symmetrise the shortlist, then re-rank ----------
    # A cell that is a fraction of a degree off an ideal symmetric lattice has
    # a wrecked M_N, so it never reaches the top on its own. Symmetrising is
    # too slow for the whole library, so it runs on a shortlist ordered by a
    # criterion that such a cell can still pass: number of lines indexed.
    shortlist = sorted(
        all_cells, key=lambda x: (-x[1]["n_indexed"], -x[1]["m_n"])
    )[: payload["max_symmetrise"]]
    keys = set()
    final = []
    for p, st in shortlist:
        r = symmetrize_and_refine(p, obs)
        final.append((r["params"], r["stats"]))
        keys.add(cell_key(p))
    for p, st in all_cells:
        if cell_key(p) not in keys:
            final.append((p, st))
    out.update(rank_and_report(final, "symmetrised"))

    out["n_expanded"] = len(expanded)
    out["t_refine1_s"] = t_refine1
    out["t_expand_s"] = t_expand
    out["wall_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-min", type=float, default=0.8)
    ap.add_argument("--max-bases", type=int, default=40)
    ap.add_argument("--max-index", type=int, default=6)
    ap.add_argument("--max-symmetrise", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/expand_refine_rank.json"
    )
    args = ap.parse_args()

    sids = sorted(
        p.name for p in PHASE4.iterdir() if p.is_dir() and p.name.startswith("mp-")
    )
    if args.limit:
        sids = sids[: args.limit]
    print(
        f"pool -> expand(index<={args.max_index}, {args.max_bases} bases) "
        f"-> refine -> rank M_N   |  {len(sids)} samples",
        flush=True,
    )

    payloads = [
        {
            "sid": s,
            "cov_min": args.cov_min,
            "max_bases": args.max_bases,
            "max_index": args.max_index,
            "max_symmetrise": args.max_symmetrise,
        }
        for s in sids
    ]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(sids)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["sample_id"])
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(sids)

    def rate(f):
        return sum(1 for r in ok if r.get(f)) / n

    summary = {
        "n": n,
        "cov_min": args.cov_min,
        "max_bases": args.max_bases,
        "max_index": args.max_index,
        "protocol": {
            "truth": "primitive standard lattice",
            "hit": f"find_mapping({LTOL},{ATOL}) AND |det-1|<{DET_TOL} AND cov>={args.cov_min}",
            "ranking": "coverage gate, then de Wolff M_N",
        },
        "refine_only": {
            "top1": rate("refine_only_top1_hit"),
            "top20": rate("refine_only_top20_hit"),
            "lib": rate("refine_only_lib_hit"),
        },
        "expand_refine": {
            "top1": rate("expand_refine_top1_hit"),
            "top20": rate("expand_refine_top20_hit"),
            "lib": rate("expand_refine_lib_hit"),
        },
        "symmetrised": {
            "top1": rate("symmetrised_top1_hit"),
            "top20": rate("symmetrised_top20_hit"),
            "lib": rate("symmetrised_lib_hit"),
        },
        "median_n_pool": float(np.median([r["n_pool"] for r in ok])),
        "median_n_expanded": float(np.median([r["n_expanded"] for r in ok])),
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
