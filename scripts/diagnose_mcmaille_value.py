#!/usr/bin/env python3
"""Quantify what each McMaille stage actually contributes on top of flow seeds.

Answers three questions against L4 vs primitive-standard truth:
  G1  which stage's candidates poison the McM20 Top-1 ranking?
  G2  does CELREF make an already-correct candidate more accurate?
  G3  does any non-RAW stage rescue a sample the seed pool missed?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pymatgen.core import Lattice  # noqa: E402
from remeasure_l4_prim_vs_conv import ATOL, CIF_DIR, LTOL, l4, truth_cells  # noqa: E402

STAGE_NAME = {1: "raw", 2: "local_mc", 3: "celref", 4: "supcel"}
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+" + r"([-\d.eE+]+)\s+" * 3 + r"([-\d.eE+]+)")


def parse_allcells(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 14:
            continue
        try:
            rows.append(
                {
                    "idx": int(parts[0]),
                    "seed_src": int(parts[1]),
                    "stage": int(parts[2]),
                    "n_indexed": int(parts[3]),
                    "mcm20": float(parts[4]),
                    "volume": float(parts[5]),
                    "rp": float(parts[6]),
                    "cell": [float(x) for x in parts[7:13]],
                }
            )
        except ValueError:
            continue
    return rows


def cell_err(a: list[float], b: list[float]) -> tuple[float, float]:
    """Setting-invariant error: align ``a`` onto ``b`` via find_mapping first.

    Comparing a,b,c,alpha,... elementwise is meaningless when the candidate is a
    valid but differently-oriented setting of the same lattice, which is exactly
    what an L4 hit allows. Returns (max relative length error, max angle error).
    """
    r = Lattice.from_parameters(*a).find_mapping(
        Lattice.from_parameters(*b), ltol=LTOL, atol=ATOL
    )
    if r is None:
        return float("nan"), float("nan")
    aligned = r[0].parameters
    len_err = max(abs(aligned[i] - b[i]) / max(b[i], 1e-9) for i in range(3))
    ang_err = max(abs(aligned[i] - b[i]) for i in range(3, 6))
    return len_err, ang_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="third_party/McMaille/run_lab/mp100_reseed_flow6m_k100_noproj",
    )
    ap.add_argument("--out", default="results/flow_seedgen/mcmaille_value.json")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    truth = {}
    for d in sorted(run_dir.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        if d.is_dir() and cif.exists():
            truth[d.name] = truth_cells(cif)["prim"]

    lib = defaultdict(int)          # stage -> samples with >=1 strict hit
    top1 = defaultdict(int)         # ranking policy -> samples whose McM20-best is a hit
    rescue = {"local_mc": 0, "celref": 0, "supcel": 0}
    refine_pairs = []               # (before_err, after_err) for strict raw seeds
    refine_flip = {"hit_to_miss": 0, "miss_to_hit": 0, "kept": 0}
    n = 0

    for sid, tcell in sorted(truth.items()):
        d = run_dir / sid
        if not d.is_dir():
            continue
        files = list(d.glob("*.allcells"))
        if not files:
            continue
        rows = parse_allcells(files[0])
        if not rows:
            continue
        n += 1

        for r in rows:
            r["hit"] = l4(r["cell"], tcell)[1]

        by_stage = defaultdict(list)
        for r in rows:
            by_stage[STAGE_NAME.get(r["stage"], "?")].append(r)

        for name, rs in by_stage.items():
            if any(r["hit"] for r in rs):
                lib[name] += 1
        if any(r["hit"] for r in rows):
            lib["all"] += 1

        # --- G1: McM20 Top-1 under different candidate subsets ---
        def best_hit(subset: list[dict], key: str = "mcm20") -> bool:
            if not subset:
                return False
            return max(subset, key=lambda r: r[key])["hit"]

        if best_hit(rows):
            top1["mcm20_all_stages"] += 1
        if best_hit(by_stage["raw"]):
            top1["mcm20_raw_only"] += 1
        if best_hit(by_stage["raw"] + by_stage["celref"]):
            top1["mcm20_raw_celref"] += 1
        if best_hit(by_stage["raw"] + by_stage["local_mc"] + by_stage["celref"]):
            top1["mcm20_no_supcel"] += 1
        nonsup = [r for r in rows if STAGE_NAME.get(r["stage"]) != "supcel"]
        if nonsup and min(nonsup, key=lambda r: r["rp"])["hit"]:
            top1["rp_no_supcel"] += 1

        # --- G3: rescue = stage found a hit the raw pool did not have ---
        raw_hit = any(r["hit"] for r in by_stage["raw"])
        if not raw_hit:
            for name in ("local_mc", "celref", "supcel"):
                if any(r["hit"] for r in by_stage[name]):
                    rescue[name] += 1

        # --- G2: CELREF precision on paired seeds ---
        celref_by_src = {}
        for r in by_stage["celref"]:
            celref_by_src.setdefault(r["seed_src"], r)
        for r in by_stage["raw"]:
            c = celref_by_src.get(r["seed_src"])
            if c is None:
                continue
            if r["hit"] and c["hit"]:
                refine_flip["kept"] += 1
                refine_pairs.append((cell_err(r["cell"], tcell), cell_err(c["cell"], tcell)))
            elif r["hit"] and not c["hit"]:
                refine_flip["hit_to_miss"] += 1
            elif not r["hit"] and c["hit"]:
                refine_flip["miss_to_hit"] += 1

    before_len = np.array([p[0][0] for p in refine_pairs])
    after_len = np.array([p[1][0] for p in refine_pairs])
    before_ang = np.array([p[0][1] for p in refine_pairs])
    after_ang = np.array([p[1][1] for p in refine_pairs])

    out = {
        "run_dir": args.run_dir,
        "n_samples": n,
        "library_strict_by_stage": {k: v / n for k, v in lib.items()},
        "top1_strict_by_policy": {k: v / n for k, v in top1.items()},
        "rescue_beyond_raw": {k: v / n for k, v in rescue.items()},
        "celref_pairs": refine_flip,
        "celref_precision": {
            "n_pairs": len(refine_pairs),
            "len_relerr_before_median": float(np.median(before_len)) if len(before_len) else None,
            "len_relerr_after_median": float(np.median(after_len)) if len(after_len) else None,
            "ang_err_before_median": float(np.median(before_ang)) if len(before_ang) else None,
            "ang_err_after_median": float(np.median(after_ang)) if len(after_ang) else None,
            "frac_improved_len": float((after_len < before_len).mean()) if len(before_len) else None,
        },
    }
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
