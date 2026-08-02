#!/usr/bin/env python3
"""Stage-wise recall: does the correct cell enter the pool at each construction step?

Stages (in pipeline order):
  1. a2_base     — RealPXRD A2 Top-100 candidate_lattices
  2. seed        — .seed after crystal-system projection
  3. raw         — .allcells stage=1
  4. local_mc    — .allcells stage=2
  5. celref      — .allcells stage=3  (phase5 has this alive)
  6. supcel      — .allcells stage=4
  7. allcells    — full .allcells union

For each stage report:
  lib_loose / lib_strict vs conventional AND primitive truth
  (|det|≈1 via |det(scale)-1|<0.25)

Also report first-entry stage (where the first strict/loose hit appears).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
CIF_DIR = PROJECT / "data/MP-100samples-benchmark"
PHASE5 = PROJECT / "third_party/McMaille/run_lab/mp100_seeded_phase5"
A2_JSON = Path(
    "/nanolab/users/wyx/archive/RealPXRD-Solver/实验/"
    "mp100_without_l_lattice/ablation_A2_xrd_only_tol_ladder_K100.json"
)

LTOL, ATOL, DET_TOL = 0.05, 3.0, 0.25

ALLCELLS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)

STAGE_ORDER = ["a2_base", "seed", "raw", "local_mc", "celref", "supcel", "allcells"]
STAGE_NUM = {"raw": 1, "local_mc": 2, "celref": 3, "supcel": 4}


def truth_cells(cif: Path) -> dict:
    s = Structure.from_file(cif)
    ana = SpacegroupAnalyzer(s, symprec=0.01)
    conv = ana.get_conventional_standard_structure().lattice
    prim = ana.get_primitive_standard_structure().lattice
    return {
        "conv": [conv.a, conv.b, conv.c, conv.alpha, conv.beta, conv.gamma],
        "prim": [prim.a, prim.b, prim.c, prim.alpha, prim.beta, prim.gamma],
        "system": ana.get_crystal_system(),
    }


def l4(pred, truth) -> tuple[bool, bool, float | None]:
    if pred is None:
        return False, False, None
    try:
        r = Lattice.from_parameters(*pred).find_mapping(
            Lattice.from_parameters(*truth), ltol=LTOL, atol=ATOL
        )
        if r is None:
            return False, False, None
        det = abs(float(np.linalg.det(r[2])))
        return True, abs(det - 1.0) < DET_TOL, det
    except Exception:
        return False, False, None


def pool_stats(pool: list[list[float]], truth: list[float]) -> dict:
    best_det = None
    n_loose = n_strict = 0
    for p in pool:
        lo, st, det = l4(p, truth)
        if lo:
            n_loose += 1
            if best_det is None or det < best_det:
                best_det = det
        if st:
            n_strict += 1
    return {
        "n": len(pool),
        "lib_loose": n_loose > 0,
        "lib_strict": n_strict > 0,
        "n_loose": n_loose,
        "n_strict": n_strict,
        "best_det": best_det,
    }


def parse_seed(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    out = []
    for line in lines[1:]:  # first line = count
        parts = line.split()
        if len(parts) >= 6:
            try:
                out.append([float(x) for x in parts[:6]])
            except ValueError:
                continue
    return out


def parse_allcells_by_stage(path: Path) -> dict[str, list[list[float]]]:
    buckets = {k: [] for k in ("raw", "local_mc", "celref", "supcel", "allcells")}
    if not path.exists():
        return buckets
    for line in path.read_text(errors="replace").splitlines():
        m = ALLCELLS_ROW.match(line)
        if not m:
            continue
        g = m.groups()
        stage = int(g[2])
        params = [float(g[i]) for i in range(7, 13)]
        buckets["allcells"].append(params)
        for name, num in STAGE_NUM.items():
            if stage == num:
                buckets[name].append(params)
                break
    return buckets


def first_entry(stage_hits: dict[str, bool]) -> str | None:
    for name in STAGE_ORDER:
        if name == "allcells":
            continue
        if stage_hits.get(name):
            return name
    return None


def eval_sid(sid: str, a2_lattices: list[list[float]]) -> dict:
    cif = CIF_DIR / f"{sid}.cif"
    if not cif.exists():
        return {"sample_id": sid, "status": "missing_cif"}
    t = truth_cells(cif)
    stem = sid.replace("-", "_")
    seed = parse_seed(PHASE5 / sid / f"{stem}.seed")
    by_stage = parse_allcells_by_stage(PHASE5 / sid / f"{stem}.allcells")

    pools = {
        "a2_base": a2_lattices,
        "seed": seed,
        "raw": by_stage["raw"],
        "local_mc": by_stage["local_mc"],
        "celref": by_stage["celref"],
        "supcel": by_stage["supcel"],
        "allcells": by_stage["allcells"],
    }

    row = {
        "sample_id": sid,
        "status": "ok",
        "system": t["system"],
        "stages": {},
    }
    for tag in ("conv", "prim"):
        hits_loose, hits_strict = {}, {}
        for name, pool in pools.items():
            st = pool_stats(pool, t[tag])
            row["stages"].setdefault(name, {})[tag] = st
            hits_loose[name] = st["lib_loose"]
            hits_strict[name] = st["lib_strict"]
        row[f"first_loose_{tag}"] = first_entry(hits_loose)
        row[f"first_strict_{tag}"] = first_entry(hits_strict)
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {
        "n": n,
        "protocol": {
            "loose": f"find_mapping ltol={LTOL} atol={ATOL}deg",
            "strict": f"loose AND |det(scale)-1|<{DET_TOL}",
            "a2_json": str(A2_JSON),
            "phase_dir": str(PHASE5),
            "stages": STAGE_ORDER,
        },
        "rates": {},
        "first_entry": {},
        "best_det_median": {},
        "pool_size_median": {},
    }
    for name in STAGE_ORDER:
        block = {}
        for tag in ("conv", "prim"):
            for mode in ("loose", "strict"):
                key = f"lib_{mode}"
                rate = sum(r["stages"][name][tag][key] for r in rows) / n
                block[f"{tag}_{mode}"] = rate
            dets = [
                r["stages"][name][tag]["best_det"]
                for r in rows
                if r["stages"][name][tag]["best_det"] is not None
            ]
            block[f"{tag}_best_det_median"] = (
                float(np.median(dets)) if dets else None
            )
            block[f"{tag}_n_with_any_loose"] = len(dets)
        sizes = [r["stages"][name]["conv"]["n"] for r in rows]
        block["n_median"] = float(np.median(sizes))
        block["n_mean"] = float(np.mean(sizes))
        out["rates"][name] = block
        out["pool_size_median"][name] = block["n_median"]

    from collections import Counter

    for tag in ("conv", "prim"):
        for mode in ("loose", "strict"):
            c = Counter(r[f"first_{mode}_{tag}"] or "NONE" for r in rows)
            out["first_entry"][f"{tag}_{mode}"] = dict(c.most_common())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "results/stagewise_prim_recall_mp100.json",
    )
    args = ap.parse_args()

    a2 = json.loads(A2_JSON.read_text())
    a2_map = {r["sample_id"]: r["candidate_lattices"] for r in a2["per_sample"]}
    sids = sorted(p.name for p in PHASE5.iterdir() if p.is_dir())
    missing = [s for s in sids if s not in a2_map]
    if missing:
        raise SystemExit(f"A2 missing {len(missing)} samples e.g. {missing[:5]}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_sid, sid, a2_map[sid]) for sid in sids]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"done {i}/{len(sids)}", flush=True)

    rows = [r for r in rows if r.get("status") == "ok"]
    rows.sort(key=lambda r: r["sample_id"])
    summary = summarize(rows)

    # pretty print main table
    print(f"\nn={summary['n']}")
    print(
        f"{'stage':<12} {'n_med':>7} "
        f"{'conv_loose':>10} {'conv_strict':>11} "
        f"{'prim_loose':>10} {'prim_strict':>11} "
        f"{'prim_det_med':>12}"
    )
    for name in STAGE_ORDER:
        b = summary["rates"][name]
        det = b["prim_best_det_median"]
        det_s = f"{det:.2f}" if det is not None else "—"
        print(
            f"{name:<12} {b['n_median']:7.0f} "
            f"{b['conv_loose']:10.0%} {b['conv_strict']:11.0%} "
            f"{b['prim_loose']:10.0%} {b['prim_strict']:11.0%} "
            f"{det_s:>12}"
        )

    print("\n=== first-entry stage (where hit first appears) ===")
    for key, counts in summary["first_entry"].items():
        print(f"{key}: {counts}")

    payload = {"summary": summary, "per_sample": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
