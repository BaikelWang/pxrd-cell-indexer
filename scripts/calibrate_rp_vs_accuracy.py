#!/usr/bin/env python3
"""How accurate must a seed be before McMaille's Rp gate accepts it?

Perturb the ground-truth primitive cell by a known amount and read back the Rp
McMaille assigns. This converts the native gate (Rp <= 0.15) into a statement
about cell accuracy, which can then be compared against how accurate the flow
seeds actually are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_LAB = ROOT / "third_party/McMaille/run_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RUN_LAB))

from diagnose_mcmaille_value import STAGE_NAME, parse_allcells  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, truth_cells  # noqa: E402
from run_mp100_reseed_nn import HKL_FILES, symmetrize_seed  # noqa: E402

SRC_RUN = RUN_LAB / "mp100_seeded_phase5"
LEVELS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]


def run_one(job):
    sid, out_dir, seeds, timeout = job
    work = out_dir / sid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    stem = sid.replace("-", "_")
    dat = SRC_RUN / sid / f"{stem}.dat"
    if not dat.exists():
        return None
    shutil.copy(dat, work / f"{stem}.dat")
    lines = [str(len(seeds))]
    for (a, b, c, al, be, ga), ifi in seeds:
        lines.append(f"{a:.6f} {b:.6f} {c:.6f} {al:.6f} {be:.6f} {ga:.6f} {ifi:d}")
    (work / f"{stem}.seed").write_text("\n".join(lines) + "\n")
    for h in HKL_FILES:
        shutil.copy(RUN_LAB / h, work / h)
    shutil.copy(RUN_LAB / "mcmaille_seeded", work / "mc")
    (work / "mc").chmod(0o755)
    try:
        subprocess.run(["./mc", stem], cwd=work, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    ac = work / f"{stem}.allcells"
    if not ac.exists():
        return None
    out = {}
    for r in parse_allcells(ac):
        if STAGE_NAME.get(r["stage"]) != "raw":
            continue
        idx = r["seed_src"] - 1
        if 0 <= idx < len(LEVELS) and idx not in out:
            out[idx] = {"rp": r["rp"], "n_indexed": r["n_indexed"]}
    return {"sample_id": sid, "levels": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN_LAB / "rp_calibration"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="results/flow_seedgen/rp_vs_accuracy.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    jobs = []
    for d in sorted(SRC_RUN.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        if not (d.is_dir() and cif.exists()):
            continue
        prim = np.asarray(truth_cells(cif)["prim"], dtype=float)
        seeds = []
        for lvl in LEVELS:
            p = prim.copy()
            if lvl > 0:
                p[:3] *= 1.0 + rng.normal(0, lvl, 3)
                p[3:] += rng.normal(0, lvl * 90.0, 3)
            seeds.append(symmetrize_seed(p.tolist()))
        jobs.append((d.name, out_dir, seeds, args.timeout))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                results.append(r)
            if i % 25 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    table = {}
    for i, lvl in enumerate(LEVELS):
        rps = [
            r["levels"][i]["rp"]
            for r in results
            if i in r["levels"] and np.isfinite(r["levels"][i]["rp"])
        ]
        nis = [r["levels"][i]["n_indexed"] for r in results if i in r["levels"]]
        if not rps:
            continue
        a = np.array(rps)
        table[f"{lvl:.3f}"] = {
            "n": len(a),
            "median_rp": float(np.median(a)),
            "frac_pass_Rmax_0.15": float((a <= 0.15).mean()),
            "frac_pass_Rmaxref_0.50": float((a <= 0.50).mean()),
            "median_n_indexed": float(np.median(nis)) if nis else None,
        }
    summary = {"n_samples": len(results), "perturbation_vs_rp": table}
    Path(ROOT / args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
