#!/usr/bin/env python3
"""Can pool subcells be turned into the true PRIMITIVE cell by HNF supercells?

For each MP100 sample:
  pool = phase4 .allcells cells
  truth = primitive standard lattice from CIF
  search H in HNF(n), n <= max_index, over pool cells; early-exit on first
  strict hit (find_mapping ltol/atol AND |det-1|<0.25)

Also reports peak coverage of the recovered cell, i.e. whether the recovered
lattice actually indexes the observed pattern (sanity check on L4 tolerance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from pxrd_cell_indexing.data.mp100 import load_mp100_sample  # noqa: E402
from pxrd_cell_indexing.model.fom import theoretical_two_theta  # noqa: E402
from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    PHASE4,
    parse_allcells,
    truth_cells,
)

LTOL, ATOL, DET_TOL = 0.05, 3.0, 0.25
TT_TOL = 0.05  # degrees, for peak coverage
N_LINES = 20


@lru_cache(maxsize=64)
def hnf_matrices(n: int) -> tuple:
    """All Hermite normal form matrices with determinant n."""
    out = []
    for a in range(1, n + 1):
        if n % a:
            continue
        for d in range(1, n // a + 1):
            if (n // a) % d:
                continue
            f = n // (a * d)
            for b in range(d):
                for c in range(f):
                    for e in range(f):
                        out.append(((a, b, c), (0, d, e), (0, 0, f)))
    return tuple(out)


def peak_coverage(params, obs_tt: np.ndarray) -> float:
    """Fraction of observed lines matched by theoretical reflections."""
    try:
        th = theoretical_two_theta(params, two_theta_max=float(obs_tt.max()) + 1.0)
    except Exception:
        return 0.0
    if th.size == 0:
        return 0.0
    obs = np.sort(obs_tt)[:N_LINES]
    hit = 0
    for t in obs:
        if np.min(np.abs(th - t)) <= TT_TOL:
            hit += 1
    return hit / float(obs.size) if obs.size else 0.0


def _worker(payload: dict) -> dict:
    sid = payload["sid"]
    max_index = payload["max_index"]
    target = payload["target"]  # "prim" or "conv"
    ltol = payload.get("ltol", LTOL)
    atol = payload.get("atol", ATOL)

    cif = CIF_DIR / f"{sid}.cif"
    allc = PHASE4 / sid / f"{sid.replace('-', '_')}.allcells"
    if not cif.exists() or not allc.exists():
        return {"sample_id": sid, "status": "missing"}

    t0 = time.perf_counter()
    t = truth_cells(cif)
    truth = t[target]
    T = Lattice.from_parameters(*truth)
    Vt = float(T.volume)
    obs = np.asarray(load_mp100_sample(cif).two_theta, dtype=np.float64)

    cands = parse_allcells(allc)
    # Dedup bases by rounded params; only those that can reach Vt by index<=max_index
    bases, seen = [], set()
    for c in cands:
        p = c["params"]
        k = tuple(round(x, 1) for x in p)
        if k in seen:
            continue
        seen.add(k)
        try:
            lat = Lattice.from_parameters(*p)
        except Exception:
            continue
        v = float(lat.volume)
        if v <= 0 or v > Vt * 1.3:
            continue
        if v * max_index < Vt * 0.7:
            continue
        bases.append((lat, v))
    # Try bases whose n*V lands closest to Vt first
    bases.sort(key=lambda lv: min(abs(lv[1] * n - Vt) for n in range(1, max_index + 1)))

    n_tried = 0
    for lat, v in bases:
        for n in range(1, max_index + 1):
            vs = v * n
            if not (0.7 * Vt <= vs <= 1.3 * Vt):
                continue
            for H_t in hnf_matrices(n):
                n_tried += 1
                try:
                    L = Lattice(np.asarray(H_t, dtype=float) @ lat.matrix)
                    r = L.find_mapping(T, ltol=ltol, atol=atol)
                    if r is None:
                        continue
                    if abs(abs(float(np.linalg.det(r[2]))) - 1.0) >= DET_TOL:
                        continue
                    params = [L.a, L.b, L.c, L.alpha, L.beta, L.gamma]
                    return {
                        "sample_id": sid,
                        "status": "ok",
                        "system": t["system"],
                        "z_ratio": t["z_ratio"],
                        "recovered": True,
                        "index": n,
                        "n_tried": n_tried,
                        "n_bases": len(bases),
                        "truth_vol": Vt,
                        "rec_params": params,
                        "rec_cov": peak_coverage(params, obs),
                        "truth_cov": peak_coverage(truth, obs),
                        "wall_s": time.perf_counter() - t0,
                    }
                except Exception:
                    continue
    return {
        "sample_id": sid,
        "status": "ok",
        "system": t["system"],
        "z_ratio": t["z_ratio"],
        "recovered": False,
        "n_tried": n_tried,
        "n_bases": len(bases),
        "truth_vol": Vt,
        "truth_cov": peak_coverage(truth, obs),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("prim", "conv"), default="prim")
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--ltol", type=float, default=LTOL)
    ap.add_argument("--atol", type=float, default=ATOL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/subcell_to_primitive.json"
    )
    args = ap.parse_args()

    sids = sorted(
        p.name for p in PHASE4.iterdir() if p.is_dir() and p.name.startswith("mp-")
    )
    if args.limit:
        sids = sids[: args.limit]
    print(
        f"HNF expansion → {args.target} truth, index<={args.max_index}, "
        f"{len(sids)} samples",
        flush=True,
    )

    rows = []
    payloads = [
        {
            "sid": s,
            "max_index": args.max_index,
            "target": args.target,
            "ltol": args.ltol,
            "atol": args.atol,
        }
        for s in sids
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(sids)}", flush=True)
    rows.sort(key=lambda r: r["sample_id"])

    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(sids)
    rec = [r for r in ok if r.get("recovered")]
    idx_hist = Counter(r["index"] for r in rec)
    cov_ok = [r for r in rec if r.get("rec_cov", 0) >= 0.8]
    truth_cov = [r.get("truth_cov", 0) for r in ok]

    summary = {
        "n": n,
        "target_truth": args.target,
        "max_index": args.max_index,
        "protocol": {
            "L4": f"find_mapping ltol={args.ltol} atol={args.atol}",
            "strict": f"|det(scale)-1|<{DET_TOL}",
            "coverage": f"|2theta| within {TT_TOL} deg over first {N_LINES} lines",
        },
        "strict_recoverable": len(rec) / n,
        "strict_recoverable_and_indexes_peaks": len(cov_ok) / n,
        "index_hist": dict(sorted(idx_hist.items())),
        "median_rec_cov": (
            float(np.median([r["rec_cov"] for r in rec])) if rec else None
        ),
        "median_truth_cov": float(np.median(truth_cov)) if truth_cov else None,
        "truth_cov_ge_0.8": sum(1 for c in truth_cov if c >= 0.8) / n,
        "median_n_tried": float(np.median([r["n_tried"] for r in ok])) if ok else None,
        "median_wall_s": float(np.median([r["wall_s"] for r in ok])) if ok else None,
    }
    out = {"summary": summary, "per_sample": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
