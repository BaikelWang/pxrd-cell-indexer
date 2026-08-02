#!/usr/bin/env python3
"""Replay the native McMaille post-seed pipeline on our flow seed pool.

The native path is destructive by design: a candidate that fails the Rp / peak
count gate is dropped, each surviving candidate collapses to a single library
row, SUPCEL overwrites that row, and Rmax tightens as better solutions appear.
Our seeded path keeps every stage as a separate row instead. This replays the
native policy offline so the two can be compared on the same candidates.

All rates are L4-strict against the primitive-standard truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import STAGE_NAME, parse_allcells  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

# From the .dat: Rmin, Rmax, Rmaxref = 0.05 0.15 0.50
NATIVE_RMAX = 0.15
NATIVE_RMAXREF = 0.50


def collapse_to_one_row(chain: dict[int, dict]) -> dict | None:
    """Native keeps one cell per accepted candidate: the most refined survivor."""
    for stage in ("celref", "local_mc", "supcel", "raw"):
        if stage in chain:
            return chain[stage]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="third_party/McMaille/run_lab/sym_tight")
    ap.add_argument("--out", default="results/flow_seedgen/native_policy_sim.json")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    policies = [
        "ours_nondestructive",
        "collapse_one_row",
        "collapse_gate_rmaxref",
        "collapse_gate_rmax",
        "collapse_gate_rmax_progressive",
    ]
    top1 = defaultdict(int)
    lib = defaultdict(int)
    kept_rows = defaultdict(list)
    gate_loss = defaultdict(int)
    hit_rp = []
    miss_rp = []
    n = 0

    for d in sorted(run_dir.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        files = list(d.glob("*.allcells")) if d.is_dir() else []
        if not (cif.exists() and files):
            continue
        rows = parse_allcells(files[0])
        if not rows:
            continue
        tcell = truth_cells(cif)["prim"]
        for r in rows:
            r["hit"] = l4(r["cell"], tcell)[1]
            r["rp_ok"] = np.isfinite(r["rp"])
        n += 1

        # group the multi-stage rows back into one chain per originating seed
        chains: dict[int, dict[str, dict]] = defaultdict(dict)
        for r in rows:
            name = STAGE_NAME.get(r["stage"])
            if name:
                chains[r["seed_src"]].setdefault(name, r)

        for ch in chains.values():
            raw = ch.get("raw")
            if raw is not None and raw["rp_ok"]:
                (hit_rp if raw["hit"] else miss_rp).append(raw["rp"])

        def score(subset):
            if not subset:
                return None
            finite = [r for r in subset if np.isfinite(r["mcm20"])]
            return max(finite, key=lambda r: r["mcm20"]) if finite else None

        def record(name, subset):
            kept_rows[name].append(len(subset))
            if any(r["hit"] for r in subset):
                lib[name] += 1
            best = score(subset)
            if best is not None and best["hit"]:
                top1[name] += 1

        record("ours_nondestructive", rows)

        collapsed = [c for c in (collapse_to_one_row(ch) for ch in chains.values()) if c]
        record("collapse_one_row", collapsed)

        for name, thr in (
            ("collapse_gate_rmaxref", NATIVE_RMAXREF),
            ("collapse_gate_rmax", NATIVE_RMAX),
        ):
            subset = []
            for sid_, ch in chains.items():
                raw = ch.get("raw")
                if raw is None or not raw["rp_ok"] or raw["rp"] > thr:
                    continue
                row = collapse_to_one_row(ch)
                if row:
                    subset.append(row)
            record(name, subset)
            if any(r["hit"] for r in rows) and not any(r["hit"] for r in subset):
                gate_loss[name] += 1

        # progressive tightening: Rmax shrinks 5% each time a better Rp appears
        subset = []
        rmax = NATIVE_RMAX
        for sid_ in sorted(chains):
            ch = chains[sid_]
            raw = ch.get("raw")
            if raw is None or not raw["rp_ok"] or raw["rp"] > rmax:
                continue
            row = collapse_to_one_row(ch)
            if row:
                subset.append(row)
            if rmax > 0.2:
                rmax -= rmax * 0.05
        record("collapse_gate_rmax_progressive", subset)
        if any(r["hit"] for r in rows) and not any(r["hit"] for r in subset):
            gate_loss["collapse_gate_rmax_progressive"] += 1

    out = {
        "run_dir": args.run_dir,
        "n_samples": n,
        "native_gate": {"Rmax": NATIVE_RMAX, "Rmaxref": NATIVE_RMAXREF},
        "policies": {
            p: {
                "top1_strict": top1[p] / n,
                "library_strict": lib[p] / n,
                "median_pool_size": float(np.median(kept_rows[p])) if kept_rows[p] else 0,
                "samples_gate_killed_only_hit": gate_loss.get(p, 0) / n,
            }
            for p in policies
        },
        "raw_seed_rp": {
            "hits": {
                "n": len(hit_rp),
                "median": float(np.median(hit_rp)) if hit_rp else None,
                "frac_below_Rmax": float(np.mean(np.array(hit_rp) <= NATIVE_RMAX)) if hit_rp else None,
                "frac_below_Rmaxref": float(np.mean(np.array(hit_rp) <= NATIVE_RMAXREF)) if hit_rp else None,
            },
            "misses": {
                "n": len(miss_rp),
                "median": float(np.median(miss_rp)) if miss_rp else None,
                "frac_below_Rmax": float(np.mean(np.array(miss_rp) <= NATIVE_RMAX)) if miss_rp else None,
                "frac_below_Rmaxref": float(np.mean(np.array(miss_rp) <= NATIVE_RMAXREF)) if miss_rp else None,
            },
        },
    }
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
