#!/usr/bin/env python3
"""Does peak refinement rescue HNF-expanded cells?

Pipeline under test:  pool subcell → HNF supercell (index<=8) → refine on peaks

Reports, per sample, the peak coverage before/after refinement and whether the
refined cell still strictly matches the true primitive lattice.
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
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from pxrd_cell_indexing.data.mp100 import load_mp100_sample  # noqa: E402
from pxrd_cell_indexing.model.fom import theoretical_two_theta  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, PHASE4, truth_cells  # noqa: E402
from test_subcell_to_primitive import (  # noqa: E402
    DET_TOL,
    LTOL,
    ATOL,
    N_LINES,
    TT_TOL,
    peak_coverage,
)


def residuals(params, obs: np.ndarray) -> np.ndarray:
    """Distance from each observed line to its nearest theoretical reflection."""
    try:
        th = theoretical_two_theta(params, two_theta_max=float(obs.max()) + 1.0)
    except Exception:
        return np.full(obs.shape, 10.0)
    if th.size == 0:
        return np.full(obs.shape, 10.0)
    return np.array([np.min(np.abs(th - t)) for t in obs])


def refine(params, obs: np.ndarray, max_nfev: int = 200):
    """Least-squares refine 6 cell params against observed 2theta."""
    p0 = np.asarray(params, dtype=float)
    lo = np.concatenate([p0[:3] * 0.85, np.maximum(p0[3:] - 8.0, 20.0)])
    hi = np.concatenate([p0[:3] * 1.15, np.minimum(p0[3:] + 8.0, 160.0)])
    try:
        res = least_squares(
            residuals,
            p0,
            args=(obs,),
            bounds=(lo, hi),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
        )
        return list(res.x)
    except Exception:
        return list(p0)


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


def _worker(payload: dict) -> dict:
    sid = payload["sid"]
    rec = payload["rec"]
    cif = CIF_DIR / f"{sid}.cif"
    if not cif.exists() or not rec.get("recovered"):
        return {"sample_id": sid, "status": "skip"}

    t0 = time.perf_counter()
    t = truth_cells(cif)
    truth = t["prim"]
    obs_all = np.asarray(load_mp100_sample(cif).two_theta, dtype=np.float64)
    obs = np.sort(obs_all)[:N_LINES]

    p_before = rec["rec_params"]
    cov_before = peak_coverage(p_before, obs_all)
    p_after = refine(p_before, obs)
    cov_after = peak_coverage(p_after, obs_all)

    return {
        "sample_id": sid,
        "status": "ok",
        "system": rec.get("system"),
        "index": rec.get("index"),
        "cov_before": cov_before,
        "cov_after": cov_after,
        "strict_before": strict_match(p_before, truth),
        "strict_after": strict_match(p_after, truth),
        "params_after": p_after,
        "truth_cov": rec.get("truth_cov"),
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recovered-json",
        type=Path,
        default=PROJECT / "results/subcell_to_primitive.json",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/expand_then_refine.json"
    )
    args = ap.parse_args()

    src = json.loads(args.recovered_json.read_text())["per_sample"]
    if args.limit:
        src = src[: args.limit]
    payloads = [{"sid": r["sample_id"], "rec": r} for r in src]
    print(f"Refining {len(payloads)} recovered cells...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(payloads)}", flush=True)
    rows.sort(key=lambda r: r["sample_id"])
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(payloads)

    summary = {
        "n": n,
        "n_refined": len(ok),
        "cov_before_ge_0.8": sum(1 for r in ok if r["cov_before"] >= 0.8) / n,
        "cov_after_ge_0.8": sum(1 for r in ok if r["cov_after"] >= 0.8) / n,
        "median_cov_before": float(np.median([r["cov_before"] for r in ok])) if ok else None,
        "median_cov_after": float(np.median([r["cov_after"] for r in ok])) if ok else None,
        "strict_before": sum(1 for r in ok if r["strict_before"]) / n,
        "strict_after": sum(1 for r in ok if r["strict_after"]) / n,
        "strict_and_cov_after": sum(
            1 for r in ok if r["strict_after"] and r["cov_after"] >= 0.8
        )
        / n,
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
