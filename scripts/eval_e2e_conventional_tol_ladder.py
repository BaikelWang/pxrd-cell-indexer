#!/usr/bin/env python3
"""End-to-end conventional-cell ladder for PXRD-indexer + McMaille vs JADE / McMaille.

Protocol
--------
- Truth: SpacegroupAnalyzer.get_conventional_standard_structure (CIF).
- Ours: take McMaille .allcells (ranked by McM20), treat each row as a
  *primitive-ish* prediction, convert to conventional via dummy-Structure +
  SpacegroupAnalyzer, then score against conventional truth.
- JADE9 / native McMaille: predictions already conventional; score as-is.

Metrics per L1–L4: mapping / strict × Top-1 / Top-20 / library.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import parse_allcells  # noqa: E402
from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    JADE_JSON,
    MCM_ORIG,
    mcm_ordered,
    truth_cells,
)

LADDER = (("L1", 0.30, 10.0), ("L2", 0.20, 8.0), ("L3", 0.10, 5.0), ("L4", 0.05, 3.0))
DET_TOL = 0.25
SYMPRECS = (0.01, 0.05, 0.1, 0.2)


def hit(pred, truth, ltol: float, atol: float) -> tuple[bool, bool, float | None]:
    try:
        r = Lattice.from_parameters(*pred).find_mapping(
            Lattice.from_parameters(*truth), ltol=ltol, atol=atol
        )
    except Exception:
        return False, False, None
    if r is None:
        return False, False, None
    det = abs(float(np.linalg.det(r[2])))
    return True, abs(det - 1.0) < DET_TOL, det


def to_conventional(params6: list[float]) -> list[float] | None:
    """Primitive/any setting → conventional standard lattice (dummy H structure)."""
    try:
        lat = Lattice.from_parameters(*[float(x) for x in params6])
    except Exception:
        return None
    s = Structure(lat, ["H"], [[0.0, 0.0, 0.0]])
    for sp in SYMPRECS:
        try:
            ana = SpacegroupAnalyzer(s, symprec=sp)
            conv = ana.get_conventional_standard_structure().lattice
            return [conv.a, conv.b, conv.c, conv.alpha, conv.beta, conv.gamma]
        except Exception:
            continue
    # Fallback: keep as-is (still scored; may fail strict vs conventional).
    return list(map(float, params6))


def score_pool(pool_conv: list[list[float]], truth_conv: list[float]) -> dict:
    out = {}
    for lv, ltol, atol in LADDER:
        flags_m, flags_s = [], []
        for c in pool_conv:
            m, s, _ = hit(c, truth_conv, ltol, atol)
            flags_m.append(m)
            flags_s.append(s)
        out[lv] = {
            "mapping": {
                "top1": bool(flags_m[:1] and flags_m[0]),
                "top20": any(flags_m[:20]),
                "library": any(flags_m),
            },
            "strict": {
                "top1": bool(flags_s[:1] and flags_s[0]),
                "top20": any(flags_s[:20]),
                "library": any(flags_s),
            },
        }
    return out


def worker_ours(job: tuple[str, str]) -> dict:
    sid, run_dir = job
    t = truth_cells(CIF_DIR / f"{sid}.cif")
    files = list((Path(run_dir) / sid).glob("*.allcells"))
    rows = parse_allcells(files[0]) if files else []
    ordered = sorted(rows, key=lambda r: -r["mcm20"])
    raw = [r["cell"] for r in ordered]
    # Convert every library row so library metrics are fair; Top-K uses prefix.
    conv_pool = []
    n_converted = 0
    for c in raw:
        cc = to_conventional(c)
        if cc is None:
            continue
        # Heuristic: conversion changed volume by >5% or params → count as converted
        if abs(Lattice.from_parameters(*cc).volume - Lattice.from_parameters(*c).volume) > 1e-3:
            n_converted += 1
        conv_pool.append(cc)
    return {
        "sample_id": sid,
        "n_raw": len(raw),
        "n_conv_pool": len(conv_pool),
        "n_vol_changed": n_converted,
        "direct_vs_conv": score_pool(raw, t["conv"]),
        "prim2conv_vs_conv": score_pool(conv_pool, t["conv"]),
        "raw_vs_prim": score_pool(raw, t["prim"]),
    }


def worker_jade(sid: str, pred) -> dict:
    t = truth_cells(CIF_DIR / f"{sid}.cif")
    pool = [pred] if pred else []
    return {"sample_id": sid, "n": len(pool), "vs_conv": score_pool(pool, t["conv"])}


def worker_mcm(sid: str) -> dict:
    t = truth_cells(CIF_DIR / f"{sid}.cif")
    pool = mcm_ordered(sid)  # already conventional-oriented suggestions
    return {"sample_id": sid, "n": len(pool), "vs_conv": score_pool(pool, t["conv"])}


def aggregate(rows: list[dict], key_path: tuple[str, ...]) -> dict:
    n = len(rows)
    rates = {}
    for lv, _, _ in LADDER:
        rates[lv] = {"mapping": {}, "strict": {}}
        for kind in ("mapping", "strict"):
            for metric in ("top1", "top20", "library"):
                hits = 0
                for r in rows:
                    node = r
                    for k in key_path:
                        node = node[k]
                    hits += int(node[lv][kind][metric])
                rates[lv][kind][metric] = hits / n
    return rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ours-run",
        action="append",
        default=[],
        help="label=path, e.g. K100=third_party/McMaille/run_lab/gate_nocap",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--out",
        default="results/flow_seedgen/pxrd_indexer_e2e_conventional_tol_ladder_mp100.json",
    )
    args = ap.parse_args()
    if not args.ours_run:
        args.ours_run = [
            "K100=third_party/McMaille/run_lab/gate_nocap",
            "K1000=third_party/McMaille/run_lab/k1000_native",
        ]

    jade = {r["sample_id"]: r for r in json.loads(JADE_JSON.read_text())["per_sample"]}
    sids = sorted(
        p.stem for p in CIF_DIR.glob("*.cif")
    )
    assert len(sids) == 100, len(sids)

    report: dict = {
        "protocol": {
            "truth": "conventional_standard from CIF (SGA symprec=0.01)",
            "ours_conversion": (
                "dummy H Structure + SpacegroupAnalyzer.get_conventional_standard_structure; "
                f"symprec try {list(SYMPRECS)}; fallback=raw cell"
            ),
            "mapping": "find_mapping(ltol, atol)",
            "strict": f"mapping AND |det(scale)-1|<{DET_TOL}",
            "ladder": {lv: {"ltol": lt, "atol_deg": at} for lv, lt, at in LADDER},
            "rank_ours": "McM20_desc",
            "jade": "jade-index Top-1 as-is (already conventional)",
            "mcmaille": "original .imp suggested + McM20 Top-20 as-is",
        },
        "engines": {},
    }

    # --- JADE ---
    print("JADE9 ...", flush=True)
    jade_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sid in sids:
            jr = jade.get(sid, {})
            pred = jr.get("pred") if jr.get("status") == "parsed" else None
            futs.append(ex.submit(worker_jade, sid, pred))
        for f in as_completed(futs):
            jade_rows.append(f.result())
    jade_rows.sort(key=lambda r: r["sample_id"])
    report["engines"]["JADE9"] = {
        "note": "Top-1 only; top20=top1; library=top1",
        "rates": aggregate(jade_rows, ("vs_conv",)),
    }

    # --- native McMaille ---
    print("McMaille original ...", flush=True)
    assert MCM_ORIG.exists(), MCM_ORIG
    mcm_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker_mcm, sid) for sid in sids]
        for i, f in enumerate(as_completed(futs), 1):
            mcm_rows.append(f.result())
            if i % 25 == 0:
                print(f"  mcm {i}/100", flush=True)
    mcm_rows.sort(key=lambda r: r["sample_id"])
    report["engines"]["McMaille_original"] = {
        "rates": aggregate(mcm_rows, ("vs_conv",)),
        "mean_n": sum(r["n"] for r in mcm_rows) / 100,
    }

    # --- Ours ---
    for spec in args.ours_run:
        label, path = spec.split("=", 1)
        run_dir = str((ROOT / path).resolve())
        print(f"Ours {label} ({path}) ...", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker_ours, (sid, run_dir)) for sid in sids]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 20 == 0:
                    print(f"  {label} {i}/100", flush=True)
        rows.sort(key=lambda r: r["sample_id"])
        report["engines"][f"PXRD-indexer_{label}_prim2conv"] = {
            "run_dir": path,
            "conversion": "prim→conventional via SGA",
            "mean_n_raw": sum(r["n_raw"] for r in rows) / 100,
            "mean_n_conv_pool": sum(r["n_conv_pool"] for r in rows) / 100,
            "rates": aggregate(rows, ("prim2conv_vs_conv",)),
        }
        report["engines"][f"PXRD-indexer_{label}_direct"] = {
            "run_dir": path,
            "conversion": "none (raw allcells vs conventional truth)",
            "rates": aggregate(rows, ("direct_vs_conv",)),
            "note": "diagnostic: |det| often 2/4 when answer is primitive",
        }
        report["engines"][f"PXRD-indexer_{label}_vs_prim"] = {
            "run_dir": path,
            "conversion": "none (raw vs primitive truth)",
            "rates": aggregate(rows, ("raw_vs_prim",)),
        }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}", flush=True)

    def pct(x):
        return f"{100 * x:.0f}%"

    print("\n===== Conventional truth · L1–L4 =====")
    for name, eng in report["engines"].items():
        if "direct" in name or "vs_prim" in name:
            continue
        print(f"\n--- {name} ---")
        print(f"{'lv':4s} {'map@1':>7s} {'map@20':>7s} {'str@1':>7s} {'str@20':>7s}")
        for lv, _, _ in LADDER:
            m = eng["rates"][lv]["mapping"]
            s = eng["rates"][lv]["strict"]
            print(
                f"{lv:4s} {pct(m['top1']):>7s} {pct(m['top20']):>7s} "
                f"{pct(s['top1']):>7s} {pct(s['top20']):>7s}"
            )


if __name__ == "__main__":
    main()
