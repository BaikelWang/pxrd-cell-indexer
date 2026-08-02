#!/usr/bin/env python3
"""Compare two seed generators on MP100: native Mc accepted pool vs RealPXRD K=100.

IMPORTANT CAVEAT
----------------
Native McMaille does **not** dump raw Monte-Carlo blind samples. What we can
recover from ``.imp`` is the **accepted proposal list** after Rp / n_indexed
filtering + local MC + SUPCEL + CELREF. That is still the closest available
view of "what Mc's MC search puts into its candidate pool".

RealPXRD side: A2 xrd-only K=100 candidate lattices (pre-Mc).

Metrics are seed-level (pool composition), not ranking quality.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "third_party/McMaille/run_lab"))

from full_pipeline_mcmaille import parse_mcmaille_candidates  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

DEFAULT_RPXRD = (
    Path("/nanolab/users/wyx/archive/RealPXRD-Solver/实验/mp100_without_l_lattice")
    / "ablation_A2_xrd_only_tol_ladder_K100.json"
)
DEFAULT_MCM = PROJECT / "third_party/McMaille/run_lab/mp100_compare/original"


def vol6(p) -> float | None:
    try:
        return float(Lattice.from_parameters(*[float(x) for x in p[:6]]).volume)
    except Exception:
        return None


def near_orthog(p, tol=1.0) -> bool:
    a, b, g = p[3], p[4], p[5]
    return abs(a - 90) < tol and abs(b - 90) < tol and abs(g - 90) < tol


def near_hex(p, tol=1.0) -> bool:
    a, b, g = p[3], p[4], p[5]
    return abs(a - 90) < tol and abs(b - 90) < tol and abs(g - 120) < tol


def length_angle_err(pred, truth) -> tuple[float, float] | None:
    """Min relative length err + mean abs angle err over axis permutations."""
    try:
        pl = np.array(sorted(pred[:3]), dtype=float)
        tl = np.array(sorted(truth[:3]), dtype=float)
        pa = np.array(sorted(pred[3:]), dtype=float)
        ta = np.array(sorted(truth[3:]), dtype=float)
        rel = float(np.mean(np.abs(pl - tl) / np.maximum(tl, 1e-6)))
        ang = float(np.mean(np.abs(pa - ta)))
        return rel, ang
    except Exception:
        return None


def pool_stats(cands: list[list[float]], truth6: list[float], truth_vol: float) -> dict:
    n = len(cands)
    if n == 0:
        return {
            "n": 0,
            "lib_loose": False,
            "lib_strict": False,
            "n_loose": 0,
            "n_strict": 0,
            "best_det": None,
            "best_log_vol": None,
            "closest_rel_len": None,
            "closest_ang": None,
            "frac_orthog": None,
            "frac_hex": None,
            "vol_ratio_med": None,
            "log_vol_spread": None,
            "unique_vol_bins": 0,
        }

    flags = [l4(c, truth6) for c in cands]
    dets = [f[2] for f in flags if f[2] is not None]
    vols = [vol6(c) for c in cands]
    vols = [v for v in vols if v is not None and v > 0]
    log_vs = [abs(math.log(v / truth_vol)) for v in vols] if truth_vol > 0 else []
    errs = [length_angle_err(c, truth6) for c in cands]
    errs = [e for e in errs if e is not None]
    closest = min(errs, key=lambda e: e[0] + e[1] / 180.0) if errs else None

    # diversity: unique volume bins at factor-1.05
    bins = set()
    for v in vols:
        bins.add(round(math.log(v) / math.log(1.05)))

    return {
        "n": n,
        "lib_loose": any(f[0] for f in flags),
        "lib_strict": any(f[1] for f in flags),
        "n_loose": sum(f[0] for f in flags),
        "n_strict": sum(f[1] for f in flags),
        "best_det": min(dets) if dets else None,
        "best_log_vol": min(log_vs) if log_vs else None,
        "closest_rel_len": closest[0] if closest else None,
        "closest_ang": closest[1] if closest else None,
        "frac_orthog": sum(1 for c in cands if near_orthog(c)) / n,
        "frac_hex": sum(1 for c in cands if near_hex(c)) / n,
        "vol_ratio_med": (
            st.median([v / truth_vol for v in vols]) if vols and truth_vol > 0 else None
        ),
        "log_vol_spread": (
            st.pstdev([math.log(v) for v in vols]) if len(vols) >= 2 else 0.0
        ),
        "unique_vol_bins": len(bins),
    }


def load_rpxrd(path: Path) -> dict[str, list[list[float]]]:
    d = json.loads(path.read_text())
    out = {}
    for r in d["per_sample"]:
        out[r["sample_id"]] = [list(c)[:6] for c in r["candidate_lattices"][:100]]
    return out


def load_mc(path: Path) -> dict[str, list[list[float]]]:
    out = {}
    for d in sorted(path.iterdir()):
        if not d.is_dir() or not d.name.startswith("mp-"):
            continue
        stem = d.name.replace("-", "_")
        imp = d / f"{stem}.imp"
        if not imp.exists():
            out[d.name] = []
            continue
        cands = parse_mcmaille_candidates(imp.read_text(errors="replace"))
        out[d.name] = [
            [c["a"], c["b"], c["c"], c["alpha"], c["beta"], c["gamma"]] for c in cands
        ]
    return out


def work(sid: str, mc: list, rp: list) -> dict:
    t = truth_cells(CIF_DIR / f"{sid}.cif")
    row = {
        "sample_id": sid,
        "system": t["system"],
        "prim_vol": t["prim_vol"],
        "conv_vol": t["conv_vol"],
        "z_ratio": t["z_ratio"],
    }
    for tag, truth, vol in (
        ("prim", t["prim"], t["prim_vol"]),
        ("conv", t["conv"], t["conv_vol"]),
    ):
        row[f"mc_{tag}"] = pool_stats(mc, truth, vol)
        row[f"rp_{tag}"] = pool_stats(rp, truth, vol)
    return row


def rate(rows, key):
    return sum(1 for r in rows if r[key]) / len(rows)


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n_samples": n}
    for eng in ("mc", "rp"):
        for tag in ("prim", "conv"):
            pref = f"{eng}_{tag}"
            stats = [r[pref] for r in rows]
            out[pref] = {
                "mean_pool": sum(s["n"] for s in stats) / n,
                "median_pool": st.median(s["n"] for s in stats),
                "empty_rate": sum(1 for s in stats if s["n"] == 0) / n,
                "lib_loose": sum(1 for s in stats if s["lib_loose"]) / n,
                "lib_strict": sum(1 for s in stats if s["lib_strict"]) / n,
                "mean_n_strict": sum(s["n_strict"] for s in stats) / n,
                "best_det_median": med([s["best_det"] for s in stats]),
                "best_log_vol_median": med([s["best_log_vol"] for s in stats]),
                "closest_rel_len_median": med([s["closest_rel_len"] for s in stats]),
                "closest_ang_median": med([s["closest_ang"] for s in stats]),
                "mean_frac_orthog": st.mean(
                    s["frac_orthog"] for s in stats if s["frac_orthog"] is not None
                )
                if any(s["frac_orthog"] is not None for s in stats)
                else None,
                "mean_frac_hex": st.mean(
                    s["frac_hex"] for s in stats if s["frac_hex"] is not None
                )
                if any(s["frac_hex"] is not None for s in stats)
                else None,
                "vol_ratio_med_of_meds": med([s["vol_ratio_med"] for s in stats]),
                "mean_log_vol_spread": st.mean(
                    s["log_vol_spread"]
                    for s in stats
                    if s["log_vol_spread"] is not None
                ),
                "mean_unique_vol_bins": st.mean(s["unique_vol_bins"] for s in stats),
            }

    # complementarity on prim strict
    both = sum(
        1 for r in rows if r["mc_prim"]["lib_strict"] and r["rp_prim"]["lib_strict"]
    )
    mc_only = sum(
        1
        for r in rows
        if r["mc_prim"]["lib_strict"] and not r["rp_prim"]["lib_strict"]
    )
    rp_only = sum(
        1
        for r in rows
        if r["rp_prim"]["lib_strict"] and not r["mc_prim"]["lib_strict"]
    )
    neither = sum(
        1
        for r in rows
        if not r["mc_prim"]["lib_strict"] and not r["rp_prim"]["lib_strict"]
    )
    out["prim_strict_complement"] = {
        "both": both,
        "mc_only": mc_only,
        "rp_only": rp_only,
        "neither": neither,
        "union": both + mc_only + rp_only,
    }

    # by crystal system
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r["system"], []).append(r)
    out["by_system_prim_strict"] = {
        sys: {
            "n": len(rs),
            "mc": sum(1 for r in rs if r["mc_prim"]["lib_strict"]) / len(rs),
            "rp": sum(1 for r in rs if r["rp_prim"]["lib_strict"]) / len(rs),
        }
        for sys, rs in sorted(by_sys.items())
    }
    return out


def print_report(s: dict) -> None:
    print("\n=== Seed generator comparison (MP100) ===")
    print(
        "Mc side = accepted proposal list from .imp "
        "(NOT raw blind MC samples; those are never dumped)."
    )
    print("RPXRD side = A2 xrd-only K=100 candidate lattices.\n")

    print(f"{'metric':<28} {'Mc accepted':>14} {'RealPXRD K100':>14}")
    rows = [
        ("mean pool size", "mean_pool"),
        ("median pool size", "median_pool"),
        ("empty pool rate", "empty_rate"),
        ("prim lib loose", "lib_loose"),
        ("prim lib strict", "lib_strict"),
        ("mean #strict hits/pool", "mean_n_strict"),
        ("best |det| median", "best_det_median"),
        ("best |log V/Vtruth| med", "best_log_vol_median"),
        ("closest rel-len err med", "closest_rel_len_median"),
        ("closest angle err med", "closest_ang_median"),
        ("mean frac ~orthogonal", "mean_frac_orthog"),
        ("mean frac ~hexagonal", "mean_frac_hex"),
        ("median V/V_prim of pool", "vol_ratio_med_of_meds"),
        ("mean log-V spread", "mean_log_vol_spread"),
        ("mean unique vol bins", "mean_unique_vol_bins"),
    ]
    for label, key in rows:
        mc = s["mc_prim"][key]
        rp = s["rp_prim"][key]
        def fmt(v):
            if v is None:
                return "—"
            if isinstance(v, float):
                if "rate" in key or "lib" in key or "frac" in key:
                    return f"{v:.0%}"
                if abs(v) >= 10:
                    return f"{v:.1f}"
                return f"{v:.3f}"
            return str(v)
        print(f"{label:<28} {fmt(mc):>14} {fmt(rp):>14}")

    c = s["prim_strict_complement"]
    print("\n=== prim strict complementarity ===")
    print(
        f"  Mc only {c['mc_only']} | RPXRD only {c['rp_only']} | "
        f"both {c['both']} | neither {c['neither']} | union {c['union']}%"
    )

    print("\n=== prim strict by crystal system ===")
    print(f"{'system':<14} {'n':>4} {'Mc':>8} {'RPXRD':>8}")
    for sys, v in s["by_system_prim_strict"].items():
        print(f"{sys:<14} {v['n']:>4} {v['mc']:>8.0%} {v['rp']:>8.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rpxrd-json", type=Path, default=DEFAULT_RPXRD)
    ap.add_argument("--mc-dir", type=Path, default=DEFAULT_MCM)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rp = load_rpxrd(args.rpxrd_json)
    mc = load_mc(args.mc_dir)
    sids = sorted(set(rp) & set(mc))
    print(f"n common samples: {len(sids)}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, s, mc[s], rp[s]) for s in sids]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"{i}/{len(sids)}", flush=True)
    rows.sort(key=lambda r: r["sample_id"])
    summary = summarize(rows)
    print_report(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "caveat": (
                    "Mc side is accepted proposal list from .imp, "
                    "not raw blind Monte-Carlo samples."
                ),
                "rpxrd_json": str(args.rpxrd_json),
                "mc_dir": str(args.mc_dir),
                "summary": summary,
                "per_sample": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
