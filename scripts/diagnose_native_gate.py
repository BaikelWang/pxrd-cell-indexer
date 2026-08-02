#!/usr/bin/env python3
"""Is McMaille's native gate rejecting wrong cells, or correct-but-imprecise ones?

The native per-candidate path admits a cell only if Rp < Rmax (0.15). Running the
faithful replica on flow seeds empties 29/100 samples, so the question is whether
the discarded candidates were geometrically wrong or merely under-refined. We read
the ungated multi-stage RAW rows, mark the L4-strict hits, and look at where their
Rp sits relative to the gate.
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

from diagnose_mcmaille_value import cell_err, parse_allcells  # noqa: E402
from refine_metric_lsq import read_dat_peaks  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

GATE = 0.15
NIND = 3  # McMaille .dat: at most this many observed lines may stay unindexed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_sample = []
    for sub in sorted(Path(args.run_dir).iterdir()):
        if not sub.is_dir():
            continue
        cells = next(sub.glob("*.allcells"), None)
        cif = CIF_DIR / f"{sub.name}.cif"
        if cells is None or not cif.exists():
            continue
        try:
            prim = truth_cells(cif)["prim"]
        except Exception:
            continue

        # RAW rows only: these are the seeds as they enter the gate.
        raw = [r for r in parse_allcells(cells) if r["stage"] == 1]
        if not raw:
            continue
        hits = [r for r in raw if l4(r["cell"], prim)[1]]
        # The native gate is a conjunction; Rp alone understates how tight it is.
        # NIND=3 means at most 3 of the NDAT observed lines may stay unindexed.
        ndat = len(read_dat_peaks(next(sub.glob("*.dat")))[1]) if any(sub.glob("*.dat")) else 0
        need = max(ndat - NIND, 0)
        rec = {
            "id": sub.name,
            "n_raw": len(raw),
            "n_hit": len(hits),
            "ndat": ndat,
            "best_hit_rp": min((r["rp"] for r in hits), default=None),
            "best_any_rp": min(r["rp"] for r in raw),
            "best_hit_nidx": max((r["n_indexed"] for r in hits), default=0),
            "hit_passes_rp": any(r["rp"] < GATE for r in hits),
            "hit_passes_nidx": any(r["n_indexed"] >= need for r in hits),
            "hit_passes_both": any(r["rp"] < GATE and r["n_indexed"] >= need for r in hits),
        }
        if hits:
            best = min(hits, key=lambda r: r["rp"])
            le, ae = cell_err(best["cell"], prim)
            rec["best_hit_len_relerr"] = None if np.isnan(le) else le
            rec["best_hit_ang_err"] = None if np.isnan(ae) else ae
        per_sample.append(rec)

    with_hit = [s for s in per_sample if s["n_hit"]]
    hit_rp = np.array([s["best_hit_rp"] for s in with_hit])
    # A sample survives the gate only if *some* candidate passes, correct or not.
    survives = [s for s in per_sample if s["best_any_rp"] < GATE]
    saved = [s for s in with_hit if s["best_hit_rp"] < GATE]
    errs = np.array([s["best_hit_len_relerr"] for s in with_hit if s.get("best_hit_len_relerr")])

    summary = {
        "gate_Rp": GATE,
        "n_samples": len(per_sample),
        "n_samples_pool_has_answer": len(with_hit),
        "n_samples_answer_passes_gate": len(saved),
        "n_samples_anything_passes_gate": len(survives),
        "frac_correct_answers_killed_by_gate": (
            1 - len(saved) / len(with_hit) if with_hit else None
        ),
        "gate_breakdown_on_samples_whose_pool_has_answer": {
            "passes_Rp_only": sum(1 for s in with_hit if s["hit_passes_rp"]),
            "passes_n_indexed_only": sum(1 for s in with_hit if s["hit_passes_nidx"]),
            "passes_both": sum(1 for s in with_hit if s["hit_passes_both"]),
            "killed_by_n_indexed_despite_Rp": sum(
                1 for s in with_hit if s["hit_passes_rp"] and not s["hit_passes_nidx"]
            ),
            "median_ndat": float(np.median([s["ndat"] for s in with_hit])),
            "median_best_hit_n_indexed": float(
                np.median([s["best_hit_nidx"] for s in with_hit])
            ),
        },
        "rp_of_correct_answers": {
            "median": float(np.nanmedian(hit_rp)),
            "p10": float(np.nanpercentile(hit_rp, 10)),
            "p90": float(np.nanpercentile(hit_rp, 90)),
            "frac_below_gate": float((hit_rp < GATE).mean()),
        },
        "geometric_error_of_correct_answers": {
            "len_relerr_median": float(np.median(errs)),
            "len_relerr_p90": float(np.percentile(errs, 90)),
        },
        "per_sample": per_sample,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    trimmed = {k: v for k, v in summary.items() if k != "per_sample"}
    print(json.dumps(trimmed, indent=2))


if __name__ == "__main__":
    main()
