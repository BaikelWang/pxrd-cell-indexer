#!/usr/bin/env python3
"""Layer-by-layer attribution of the CNRS orthorhombic regression.

Ours loses to native McMaille only on orthorhombic (lib 9.7% vs 16.1%), and
``lib == top20`` there, so the correct cell never enters the pool at all. This
walks the pipeline stage by stage to find where it is lost:

  L0 raw flow seeds (K from pool json)
  L1 after ``_seed_physically_valid``
  L2 after ``build_seed_rows`` (symmetrize + keep_original)
  L3 final ``.allcells`` pool out of seeded McMaille

For every sample it also records the closest seed to truth, so a miss can be
told apart from a near-miss that the L4 tolerance rejects.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RUN_LAB = ROOT / "third_party" / "McMaille" / "run_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RUN_LAB))

from remeasure_l4_prim_vs_conv import l4, parse_allcells, truth_cells  # noqa: E402
from run_mp100_reseed_nn import (  # noqa: E402
    _seed_physically_valid,
    build_seed_rows,
)


def cell_dist(pred, truth) -> dict:
    """Best permutation-aligned relative error between two 6-parameter cells."""
    import itertools

    p = np.asarray(pred, float)
    t = np.asarray(truth, float)
    best = None
    for perm in itertools.permutations(range(3)):
        pl = p[list(perm)]
        pa = p[3:][list(perm)]
        dl = np.abs(pl - t[:3]) / np.maximum(t[:3], 1e-9)
        da = np.abs(pa - t[3:])
        score = float(dl.max())
        if best is None or score < best["len_relerr"]:
            best = {
                "len_relerr": score,
                "ang_abserr": float(da.max()),
                "perm": list(perm),
                "pred_sorted": [float(x) for x in pl],
            }
    return best


def pool_hit(cells, truth) -> dict:
    strict = [l4(c, truth)[1] for c in cells]
    loose = [l4(c, truth)[0] for c in cells]
    first = next((i for i, v in enumerate(strict, 1) if v), None)
    return {
        "n": len(cells),
        "strict": any(strict),
        "loose": any(loose),
        "first_strict": first,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/flow_seedgen/cnrs_e2e_k100")
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    pool = json.loads((run / f"pool_k{args.k}.json").read_text())["per_sample"]
    seeded_run = run / f"indexer_k{args.k}"

    cnrs = Path(args.cnrs_dir)
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    truth_by_sid = {}
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        if sid not in pool:
            continue
        cif = cnrs / f"{sid}_sg.cif"
        if not cif.exists():
            continue
        truth_by_sid[sid] = truth_cells(cif)

    rows = []
    for sid, rec in sorted(pool.items()):
        t = truth_by_sid.get(sid)
        if t is None:
            continue
        truth = t["prim"]
        raw = [c[:6] for c in rec["candidates"][: args.k]]
        valid = [c for c in raw if _seed_physically_valid(c)]
        sym = [list(c) for c, _ifi in build_seed_rows(valid, symmetrize=True, keep_original=True)]
        allc = seeded_run / sid / f"{sid.replace('-', '_')}.allcells"
        final = [c["params"] for c in parse_allcells(allc)] if allc.exists() else []

        near = min((cell_dist(c, truth) for c in raw), key=lambda d: d["len_relerr"])
        rows.append(
            {
                "sample_id": sid,
                "system": t["system"],
                "truth_prim": [round(x, 4) for x in truth],
                "prim_vol": round(t["prim_vol"], 2),
                "z_ratio": round(t["z_ratio"], 3) if t["z_ratio"] else None,
                "L0_raw": pool_hit(raw, truth),
                "L1_valid": pool_hit(valid, truth),
                "L2_sym": pool_hit(sym, truth),
                "L3_final": pool_hit(final, truth),
                "nearest_seed": near,
                "n_dropped_by_validity": len(raw) - len(valid),
            }
        )

    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)

    print(f"{'system':14s} {'n':>3s}  {'L0raw':>6s} {'L1val':>6s} {'L2sym':>6s} {'L3fin':>6s}   "
          f"{'medNearRelErr':>13s}")
    summary = {}
    for s, rs in sorted(by_sys.items()):
        m = len(rs)
        vals = {
            lay: sum(1 for r in rs if r[lay]["strict"]) / m
            for lay in ("L0_raw", "L1_valid", "L2_sym", "L3_final")
        }
        med = float(np.median([r["nearest_seed"]["len_relerr"] for r in rs]))
        summary[s] = {"n": m, **vals, "median_nearest_len_relerr": med}
        print(f"{s:14s} {m:3d}  {vals['L0_raw']:6.1%} {vals['L1_valid']:6.1%} "
              f"{vals['L2_sym']:6.1%} {vals['L3_final']:6.1%}   {med:13.3f}")

    m = len(rows)
    overall = {
        lay: sum(1 for r in rows if r[lay]["strict"]) / m
        for lay in ("L0_raw", "L1_valid", "L2_sym", "L3_final")
    }
    print(f"{'ALL':14s} {m:3d}  {overall['L0_raw']:6.1%} {overall['L1_valid']:6.1%} "
          f"{overall['L2_sym']:6.1%} {overall['L3_final']:6.1%}")

    out = Path(args.out) if args.out else run / "ortho_attribution.json"
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps({"by_system": summary, "overall": overall, "per_sample": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
