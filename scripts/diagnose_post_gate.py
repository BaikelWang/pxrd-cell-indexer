#!/usr/bin/env python3
"""Where do correct seeds die: at the gate, or after it?

The refinement pushed 51 of 67 recoverable answers through the Rp and
n-indexed conditions, yet the native library only holds 35%. Two things were
invisible to the earlier autopsy: the gate's third condition RMAX2 (positional
quality among indexed lines only, never stored in the library) and CALCUL1's
NHKL>NDAT10 early return, which rejects a cell without scoring it at all.

The Fortran now logs a verdict per seed, so this joins that log against truth and
splits the loss into a gate part and a post-gate part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import parse_allcells  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

NIND = 3
RMAX_GATE = 0.15
RMAX2_GATE = 0.15


def parse_seedgate(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 14:
            continue
        try:
            out.append(
                {
                    "iseed": int(p[0]),
                    "ifi": int(p[1]),
                    "rmax": float(p[2]),
                    "rmax2": float(p[3]),
                    "llhkl": int(p[4]),
                    "ndat": int(p[5]),
                    "novf": int(p[6]),
                    "ipass": int(p[7]),
                    "cell": [float(x) for x in p[8:14]],
                }
            )
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tally = {
        "pool_has_answer": 0,
        "answer_passes_gate": 0,
        "answer_in_library": 0,
        "killed_novf": 0,
        "killed_rmax": 0,
        "killed_rmax2": 0,
        "killed_ipen": 0,
        "killed_rmax2_only": 0,
        "lost_after_gate": 0,
    }
    n_samples = 0
    rmax2_of_hits: list[float] = []
    per_sample = []

    for sub in sorted(Path(args.run_dir).iterdir()):
        if not sub.is_dir():
            continue
        sg = next(sub.glob("*.seedgate"), None)
        ac = next(sub.glob("*.allcells"), None)
        cif = CIF_DIR / f"{sub.name}.cif"
        if sg is None or not cif.exists():
            continue
        try:
            prim = truth_cells(cif)["prim"]
        except Exception:
            continue
        n_samples += 1

        seeds = parse_seedgate(sg)
        hits = [s for s in seeds if l4(s["cell"], prim)[1]]
        rec = {"id": sub.name, "n_seeds": len(seeds), "n_hit_seeds": len(hits)}
        if not hits:
            per_sample.append(rec)
            continue
        tally["pool_has_answer"] += 1
        rmax2_of_hits += [h["rmax2"] for h in hits if h["novf"] == 0]

        passed = [h for h in hits if h["ipass"] == 1]
        rec["n_hit_passed_gate"] = len(passed)
        if passed:
            tally["answer_passes_gate"] += 1
        else:
            # Attribute the kill to every condition the best hit violated.

            if all(h["novf"] == 1 for h in hits):
                tally["killed_novf"] += 1
            if all(h["novf"] == 1 or h["rmax"] >= RMAX_GATE for h in hits):
                tally["killed_rmax"] += 1
            if all(h["novf"] == 1 or h["rmax2"] >= RMAX2_GATE for h in hits):
                tally["killed_rmax2"] += 1
            if all(h["novf"] == 1 or (h["ndat"] - h["llhkl"]) > NIND for h in hits):
                tally["killed_ipen"] += 1
            # Seeds that cleared Rp and coverage yet still died: RMAX2 alone.
            if any(
                h["novf"] == 0
                and h["rmax"] < RMAX_GATE
                and (h["ndat"] - h["llhkl"]) <= NIND
                and h["rmax2"] >= RMAX2_GATE
                for h in hits
            ):
                tally["killed_rmax2_only"] += 1

        in_lib = False
        if ac is not None:
            in_lib = any(l4(r["cell"], prim)[1] for r in parse_allcells(ac))
        rec["answer_in_library"] = in_lib
        if in_lib:
            tally["answer_in_library"] += 1
        elif passed:
            tally["lost_after_gate"] += 1
        per_sample.append(rec)

    arr = np.array(rmax2_of_hits) if rmax2_of_hits else np.array([np.nan])
    summary = {
        "run_dir": args.run_dir,
        "n_samples": n_samples,
        **tally,
        "rmax2_of_correct_seeds": {
            "median": float(np.nanmedian(arr)),
            "frac_below_gate": float(np.mean(arr < RMAX2_GATE)),
        },
        "per_sample": per_sample,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
