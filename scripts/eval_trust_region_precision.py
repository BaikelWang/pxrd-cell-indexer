#!/usr/bin/env python3
"""Does trust-region-constrained CELREF actually make correct candidates better?

Step 2 of the proposed pipeline aims at precision, not pool membership, so
measure the setting-invariant parameter error before and after refinement,
restricted to refinements that stay inside a trust region.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import STAGE_NAME, cell_err, parse_allcells  # noqa: E402
from eval_proposed_pipeline import drift  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

RUN = ROOT / "third_party/McMaille/run_lab/mp100_reseed_flow6m_k100_noproj"
TAUS = [0.005, 0.01, 0.02, 0.05, 1.0]


def main() -> None:
    pairs = {t: [] for t in TAUS}
    pool_rows = pool_hits = 0

    for d in sorted(RUN.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        files = list(d.glob("*.allcells")) if d.is_dir() else []
        if not (cif.exists() and files):
            continue
        rows = parse_allcells(files[0])
        tcell = truth_cells(cif)["prim"]
        for r in rows:
            r["hit"] = l4(r["cell"], tcell)[1]

        raw = [r for r in rows if STAGE_NAME.get(r["stage"]) == "raw"]
        pool_rows += len(raw)
        pool_hits += sum(r["hit"] for r in raw)

        celref = {}
        for r in rows:
            if STAGE_NAME.get(r["stage"]) == "celref":
                celref.setdefault(r["seed_src"], r)

        for r in raw:
            c = celref.get(r["seed_src"])
            if c is None or not r["hit"]:
                continue
            before = cell_err(r["cell"], tcell)
            d_ = drift(c["cell"], r["cell"])
            for t in TAUS:
                if d_ > t:
                    pairs[t].append((before, before))  # refinement rejected
                elif c["hit"]:
                    pairs[t].append((before, cell_err(c["cell"], tcell)))
                else:
                    pairs[t].append((before, (float("nan"), float("nan"))))

    out = {"ungated_pool_precision": pool_hits / max(pool_rows, 1), "by_tau": {}}
    for t in TAUS:
        b = np.array([p[0][0] for p in pairs[t]])
        a = np.array([p[1][0] for p in pairs[t]])
        ba = np.array([p[0][1] for p in pairs[t]])
        aa = np.array([p[1][1] for p in pairs[t]])
        ok = np.isfinite(a)
        out["by_tau"][str(t)] = {
            "n": int(len(b)),
            "frac_refinement_accepted": float((a != b).mean()),
            "frac_lost": float((~ok).mean()),
            "len_relerr_before": float(np.median(b[ok])),
            "len_relerr_after": float(np.median(a[ok])),
            "ang_err_before": float(np.median(ba[ok])),
            "ang_err_after": float(np.median(aa[ok])),
        }
    Path(ROOT / "results/flow_seedgen/trust_region_precision.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
