#!/usr/bin/env python3
"""Feed the ground-truth primitive cell to seeded McMaille as the only seed.

If the native Rp / peak-count gate is a usable filter on our data, the true cell
must comfortably clear it. Whatever Rp McMaille assigns to the exact answer is
the ceiling any seed could reach, so it bounds what the native pipeline could
ever accept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_LAB = ROOT / "third_party/McMaille/run_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RUN_LAB))

from diagnose_mcmaille_value import STAGE_NAME, parse_allcells  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, truth_cells  # noqa: E402
from run_mp100_reseed_nn import HKL_FILES, symmetrize_seed  # noqa: E402

SRC_RUN = RUN_LAB / "mp100_seeded_phase5"


def run_one(job):
    sid, out_dir, cell, ifi, timeout = job
    work = out_dir / sid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    stem = sid.replace("-", "_")
    dat = SRC_RUN / sid / f"{stem}.dat"
    if not dat.exists():
        return {"sample_id": sid, "error": "missing dat"}
    shutil.copy(dat, work / f"{stem}.dat")
    a, b, c, al, be, ga = cell
    (work / f"{stem}.seed").write_text(
        f"1\n{a:.6f} {b:.6f} {c:.6f} {al:.6f} {be:.6f} {ga:.6f} {ifi:d}\n"
    )
    for h in HKL_FILES:
        shutil.copy(RUN_LAB / h, work / h)
    shutil.copy(RUN_LAB / "mcmaille_seeded", work / "mc")
    (work / "mc").chmod(0o755)
    try:
        subprocess.run(
            ["./mc", stem], cwd=work, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"sample_id": sid, "error": "timeout"}
    ac = work / f"{stem}.allcells"
    if not ac.exists():
        return {"sample_id": sid, "error": "no allcells"}
    rows = parse_allcells(ac)
    raw = [r for r in rows if STAGE_NAME.get(r["stage"]) == "raw"]
    if not raw:
        return {"sample_id": sid, "error": "no raw row"}
    r = raw[0]
    return {
        "sample_id": sid,
        "ifi": ifi,
        "rp": r["rp"],
        "n_indexed": r["n_indexed"],
        "mcm20": r["mcm20"],
        "volume": r["volume"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN_LAB / "truth_seed_probe"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json", default="results/flow_seedgen/truth_seed_probe.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for d in sorted(SRC_RUN.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        if not (d.is_dir() and cif.exists()):
            continue
        prim = truth_cells(cif)["prim"]
        cell, ifi = symmetrize_seed(prim)
        jobs.append((d.name, out_dir, cell, ifi, args.timeout))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    ok = [r for r in results if "error" not in r and r["rp"] == r["rp"]]
    import numpy as np

    rp = np.array([r["rp"] for r in ok])
    ni = np.array([r["n_indexed"] for r in ok])
    summary = {
        "n_jobs": len(jobs),
        "n_ok": len(ok),
        "truth_cell_rp": {
            "median": float(np.median(rp)),
            "p25": float(np.percentile(rp, 25)),
            "p75": float(np.percentile(rp, 75)),
            "min": float(rp.min()),
            "max": float(rp.max()),
            "frac_below_Rmax_0.15": float((rp <= 0.15).mean()),
            "frac_below_Rmaxref_0.50": float((rp <= 0.50).mean()),
        },
        "truth_cell_n_indexed": {
            "median": float(np.median(ni)),
            "min": int(ni.min()),
            "max": int(ni.max()),
        },
        "per_sample": sorted(ok, key=lambda r: r["rp"]),
    }
    Path(ROOT / args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.json).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
