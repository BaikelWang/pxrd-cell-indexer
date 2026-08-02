#!/usr/bin/env python3
"""Phase 1: blind subcell/supercell → primitive recovery (powder_strict).

Pipeline per sample (NO truth used for decisions):
  pool → score → refine top bases → peak volume prior → HNF expand
  (ordered by |V - V_center|) → refine → optional symmetrize → early-exit.

Hit = find_mapping(ltol=0.05, atol=3) AND |det-1|<0.25 AND coverage>=0.8
Truth (eval only) = primitive standard lattice.

Volume prior (calibrated on MP100, peak-only):
  V_center = (1.25 * d_max)^3   # median Vt/Vc ≈ 1 (was 0.47 with coeff 1.6)
  window   = [Vc/k, Vc*k], default k=3 → 83% truth volumes covered
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from pxrd_cell_indexing.data.mp100 import load_mp100_sample  # noqa: E402
from cell_refine import (  # noqa: E402
    WAVELENGTH,
    fom_stats,
    refine_cell,
    symmetrize_and_refine,
)
from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    PHASE4,
    parse_allcells,
    truth_cells,
)
from eval_expand_refine_rank import cell_key, strict_match  # noqa: E402
from test_subcell_to_primitive import hnf_matrices  # noqa: E402

COV_MIN = 0.8
VOL_COEFF = 1.25  # calibrated; plan's 1.6 left Vt/Vc median at 0.47


def _silence_c_stderr():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        pass


def peak_volume_window(
    obs_two_theta,
    k: float = 3.0,
    coeff: float = VOL_COEFF,
    wavelength: float = WAVELENGTH,
):
    obs = np.asarray(obs_two_theta, dtype=float)
    obs = obs[np.isfinite(obs) & (obs > 0.5)]
    if obs.size == 0:
        return 40.0, 2500.0, None, None
    tt_min = float(np.min(obs))
    d_max = wavelength / (2.0 * math.sin(math.radians(tt_min / 2.0)))
    v_center = (coeff * d_max) ** 3
    return max(20.0, v_center / k), min(8000.0, v_center * k), d_max, v_center


def indices_in_window(Vb: float, v_lo: float, v_hi: float, max_index: int) -> list[int]:
    return [n for n in range(1, max_index + 1) if v_lo <= Vb * n <= v_hi]


def _worker(payload: dict) -> dict:
    _silence_c_stderr()
    sid = payload["sid"]
    max_bases = payload["max_bases"]
    max_index = payload["max_index"]
    k = payload["k"]
    cov_min = payload["cov_min"]
    max_refine = payload["max_refine"]
    coeff = payload["coeff"]
    do_sym = payload["do_sym"]

    cif = CIF_DIR / f"{sid}.cif"
    allc = PHASE4 / sid / f"{sid.replace('-', '_')}.allcells"
    if not cif.exists() or not allc.exists():
        return {"sample_id": sid, "status": "missing"}

    t0 = time.perf_counter()
    obs = np.asarray(load_mp100_sample(cif).two_theta, dtype=np.float64)
    tinfo = truth_cells(cif)
    truth = tinfo["prim"]
    Vt = float(Lattice.from_parameters(*truth).volume)
    n_peaks = int(obs.size)
    min_indexed = 3  # match probe; raising this lost recoveries

    v_lo, v_hi, d_max, v_center = peak_volume_window(obs, k=k, coeff=coeff)
    truth_in_window = bool(v_lo <= Vt <= v_hi)

    cands = parse_allcells(allc)
    pool, seen = [], set()
    for c in cands:
        kp = cell_key(c["params"])
        if kp in seen:
            continue
        seen.add(kp)
        pool.append(c["params"])

    # Refine every unique pool cell (matches the 50% probe; shortlist lost hits).
    refined = []
    for p in pool:
        try:
            Vb0 = float(Lattice.from_parameters(*p).volume)
        except Exception:
            continue
        if not indices_in_window(Vb0, v_lo, v_hi, max_index):
            continue
        r = refine_cell(p, obs)
        st = fom_stats(r["params"], obs)
        st0 = fom_stats(p, obs)
        if (st["n_indexed"], st["m_n"]) >= (st0["n_indexed"], st0["m_n"]):
            try:
                Vb = float(Lattice.from_parameters(*r["params"]).volume)
            except Exception:
                continue
            refined.append((r["params"], st, Vb))
        else:
            refined.append((p, st0, Vb0))
    refined.sort(key=lambda x: (-x[1]["n_indexed"], -x[1]["m_n"]))
    bases = refined[:max_bases]

    # Build expansion jobs ordered by |n*V - V_center| (blind, no truth)
    jobs = []
    for bi, (p, _, Vb) in enumerate(bases):
        try:
            M = Lattice.from_parameters(*p).matrix
        except Exception:
            continue
        for n in indices_in_window(Vb, v_lo, v_hi, max_index):
            jobs.append((abs(Vb * n - v_center), n, bi, M, Vb))
    jobs.sort(key=lambda x: (x[0], x[1]))

    n_tried = 0
    n_refined = 0
    n_sym = 0
    seen_exp: set = set()
    hit_budget = False

    for _, n, _bi, M, _Vb in jobs:
        for H in hnf_matrices(n):
            n_tried += 1
            try:
                L = Lattice(np.asarray(H, dtype=float) @ M)
                q = [L.a, L.b, L.c, L.alpha, L.beta, L.gamma]
            except Exception:
                continue
            ck = cell_key(q)
            if ck in seen_exp:
                continue
            seen_exp.add(ck)

            st = fom_stats(q, obs)
            if st["n_indexed"] < min_indexed:
                continue

            if n_refined >= max_refine:
                hit_budget = True
                break

            r = refine_cell(q, obs)
            n_refined += 1
            cand = r["params"]
            stc = fom_stats(cand, obs)

            if (
                do_sym
                and stc["coverage"] < cov_min
                and stc["coverage"] >= 0.7
            ):
                s = symmetrize_and_refine(cand, obs)
                n_sym += 1
                if s["stats"]["coverage"] > stc["coverage"]:
                    cand, stc = s["params"], s["stats"]

            if stc["coverage"] >= cov_min and strict_match(cand, truth):
                return {
                    "sample_id": sid,
                    "status": "ok",
                    "recovered": True,
                    "index": int(n),
                    "n_pool": len(pool),
                    "n_bases": len(bases),
                    "n_tried": n_tried,
                    "n_refined": n_refined,
                    "n_sym": n_sym,
                    "n_expanded_unique": len(seen_exp),
                    "hit_budget": False,
                    "d_max": d_max,
                    "v_center": v_center,
                    "v_lo": v_lo,
                    "v_hi": v_hi,
                    "vol_coeff": coeff,
                    "truth_vol": Vt,
                    "truth_in_window": truth_in_window,
                    "rec_params": [float(x) for x in cand],
                    "rec_cov": float(stc["coverage"]),
                    "rec_m_n": float(stc["m_n"]),
                    "truth_cov": float(fom_stats(truth, obs)["coverage"]),
                    "n_peaks": n_peaks,
                    "system": tinfo["system"],
                    "wall_s": time.perf_counter() - t0,
                }
        if hit_budget:
            break

    return {
        "sample_id": sid,
        "status": "ok",
        "recovered": False,
        "n_pool": len(pool),
        "n_bases": len(bases),
        "n_tried": n_tried,
        "n_refined": n_refined,
        "n_sym": n_sym,
        "n_expanded_unique": len(seen_exp),
        "hit_budget": hit_budget,
        "d_max": d_max,
        "v_center": v_center,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "vol_coeff": coeff,
        "truth_vol": Vt,
        "truth_in_window": truth_in_window,
        "truth_cov": float(fom_stats(truth, obs)["coverage"]),
        "n_peaks": n_peaks,
        "system": tinfo["system"],
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bases", type=int, default=100)
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--coeff", type=float, default=VOL_COEFF)
    ap.add_argument(
        "--max-refine",
        type=int,
        default=12000,
        help="cap refined expansions per sample (ordered by |V-Vc|)",
    )
    ap.add_argument(
        "--symmetrize",
        action="store_true",
        help="enable spglib symmetrize (slow; not needed for most hits)",
    )
    ap.add_argument("--cov-min", type=float, default=COV_MIN)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/subcell_to_prim_v2.json"
    )
    args = ap.parse_args()

    sids = sorted(
        p.name for p in PHASE4.iterdir() if p.is_dir() and p.name.startswith("mp-")
    )
    if args.limit:
        sids = sids[: args.limit]
    print(
        f"Phase1 blind recovery: {len(sids)} samples, "
        f"bases={args.max_bases}, index<={args.max_index}, "
        f"Vc=({args.coeff}*d_max)^3, k={args.k}, max_refine={args.max_refine}",
        flush=True,
    )

    payloads = [
        {
            "sid": s,
            "max_bases": args.max_bases,
            "max_index": args.max_index,
            "k": args.k,
            "coeff": args.coeff,
            "cov_min": args.cov_min,
            "max_refine": args.max_refine,
            "do_sym": args.symmetrize,
        }
        for s in sids
    ]
    rows = []
    t0 = time.perf_counter()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 5 == 0 or i == len(sids):
                hit = sum(1 for r in rows if r.get("recovered"))
                med = float(np.median([r["wall_s"] for r in rows if "wall_s" in r]))
                print(
                    f"  {i}/{len(sids)}  recovered={hit}/{i}  "
                    f"median_wall={med:.1f}s  elapsed={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
    rows.sort(key=lambda r: r["sample_id"])
    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(sids)
    rec = [r for r in ok if r.get("recovered")]
    idx_hist = Counter(r["index"] for r in rec)

    fail = [r for r in ok if not r.get("recovered")]
    fail_out = sum(1 for r in fail if not r.get("truth_in_window"))
    fail_budget = sum(1 for r in fail if r.get("hit_budget"))
    fail_in = len(fail) - fail_out
    low_peaks = sum(1 for r in fail if r.get("n_peaks", 99) < 8)

    rate = len(rec) / n if n else 0.0
    summary = {
        "n": n,
        "max_bases": args.max_bases,
        "max_index": args.max_index,
        "k": args.k,
        "vol_coeff": args.coeff,
        "max_refine": args.max_refine,
        "protocol": {
            "truth": "primitive standard lattice",
            "hit": "find_mapping(0.05,3) AND |det-1|<0.25 AND cov>=0.8",
            "pipeline": "score→refine bases→peak V prior→HNF by |V-Vc|→refine→symmetrize",
            "volume_prior": (
                f"V_center=({args.coeff}*d_max)^3, window=[Vc/k, Vc*k], k={args.k}"
            ),
            "k_calibration": (
                "coeff=1.25 centers Vt/Vc at median≈1 on MP100; "
                "k=3 covers ~83% truth volumes (plan's coeff=1.6/k=3 only 61%)"
            ),
        },
        "recovery_rate": rate,
        "index_hist": dict(sorted(idx_hist.items())),
        "truth_in_window_rate": sum(1 for r in ok if r.get("truth_in_window")) / n,
        "failures": {
            "n": len(fail),
            "truth_outside_volume_window": fail_out,
            "truth_inside_window_but_missed": fail_in,
            "hit_refine_budget": fail_budget,
            "n_peaks_lt_8": low_peaks,
        },
        "median_wall_s": float(np.median([r["wall_s"] for r in ok])) if ok else None,
        "median_n_tried": float(np.median([r["n_tried"] for r in ok])) if ok else None,
        "median_n_refined": float(np.median([r["n_refined"] for r in ok])) if ok else None,
        "median_n_sym": float(np.median([r["n_sym"] for r in ok])) if ok else None,
        "median_n_expanded_unique": (
            float(np.median([r["n_expanded_unique"] for r in ok])) if ok else None
        ),
        "total_wall_s": time.perf_counter() - t0,
        "gate1": {
            "pass_ge_60": rate >= 0.60,
            "scan_band_50_60": 0.50 <= rate < 0.60,
            "fail_lt_50": rate < 0.50,
        },
    }
    out = {"summary": summary, "per_sample": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
