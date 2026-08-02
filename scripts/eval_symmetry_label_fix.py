#!/usr/bin/env python3
"""Estimate how much of McM20 comes back once seeds carry a crystal-system label.

McMaille computes XFOM = 1/Rp * 100/CNCALC * X2 * X3, where X3 is a crystal
system bonus. Our seeds are all injected with IFI=6 (triclinic) so X3 == 1 for
every candidate. Detect the real system from the cell metric and re-apply X3
offline; no Fortran change needed to get the estimate.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import STAGE_NAME, parse_allcells  # noqa: E402
from eval_proposed_pipeline import cluster  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

RUN = ROOT / "third_party/McMaille/run_lab/mp100_reseed_flow6m_k100_noproj"

# IFI codes and the X3 bonus McMaille applies to each, from the XFOM block.
X3 = {"cubic": 6.0, "hexagonal": 4.0, "tetragonal": 4.0, "orthorhombic": 2.0,
      "monoclinic": 1.0, "triclinic": 1.0, "rhombohedral": 6.0}


def crystal_system(cell, ltol: float = 0.02, atol: float = 1.5) -> str:
    a, b, c, al, be, ga = cell
    eq = lambda x, y: abs(x - y) / max(abs(y), 1e-9) < ltol  # noqa: E731
    ang = lambda x, v: abs(x - v) < atol  # noqa: E731
    n90 = sum(ang(x, 90.0) for x in (al, be, ga))
    if eq(a, b) and eq(b, c):
        if n90 == 3:
            return "cubic"
        if abs(al - be) < atol and abs(be - ga) < atol:
            return "rhombohedral"
    if n90 == 3:
        if eq(a, b) or eq(b, c) or eq(a, c):
            return "tetragonal"
        return "orthorhombic"
    if (eq(a, b) and ang(al, 90) and ang(be, 90) and ang(ga, 120)) or (
        eq(b, c) and ang(be, 90) and ang(ga, 90) and ang(al, 120)
    ):
        return "hexagonal"
    if n90 == 2:
        return "monoclinic"
    return "triclinic"


def main() -> None:
    top1: Counter[str] = Counter()
    sys_of_hits: Counter[str] = Counter()
    sys_of_all: Counter[str] = Counter()
    n = 0

    for d in sorted(RUN.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        files = list(d.glob("*.allcells")) if d.is_dir() else []
        if not (cif.exists() and files):
            continue
        tcell = truth_cells(cif)["prim"]
        raw = [r for r in parse_allcells(files[0]) if STAGE_NAME.get(r["stage"]) == "raw"]
        if not raw:
            continue
        n += 1

        flags = np.array([l4(r["cell"], tcell)[1] for r in raw])
        xfom = np.array([r["mcm20"] for r in raw])          # X3 == 1 in this run
        systems = [crystal_system(r["cell"]) for r in raw]
        x3 = np.array([X3[s] for s in systems])
        cells = np.array([r["cell"] for r in raw])

        for s, f in zip(systems, flags, strict=False):
            sys_of_all[s] += 1
            if f:
                sys_of_hits[s] += 1

        size = np.zeros(len(raw))
        for c in cluster(cells):
            for i in c:
                size[i] = len(c)

        def pick(score, name):
            if flags[int(np.argmax(score))]:
                top1[name] += 1

        zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
        pick(xfom, "mcm20_broken_X3=1")
        pick(xfom * x3, "mcm20_with_X3_restored")
        pick(size, "cluster_size")
        pick(zs(xfom * x3) + zs(size), "cluster + mcm20_X3_restored")

    out = {
        "n_samples": n,
        "top1": {k: v / n for k, v in top1.most_common()},
        "system_mix_all_candidates": {
            k: v / sum(sys_of_all.values()) for k, v in sys_of_all.most_common()
        },
        "hit_rate_within_system": {
            k: sys_of_hits[k] / v for k, v in sys_of_all.most_common()
        },
    }
    Path(ROOT / "results/flow_seedgen/symmetry_label_fix.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
