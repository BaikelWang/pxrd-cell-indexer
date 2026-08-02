#!/usr/bin/env python3
"""Score reseeded McMaille runs: library recall + Top-K mapping (prim/conv).

Same criterion as the stage-wise analysis:
  loose  = find_mapping(ltol=0.05, atol=3deg)
  strict = loose AND |det(scale)-1| < 0.25
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from remeasure_l4_prim_vs_conv import (  # noqa: E402
    CIF_DIR,
    l4,
    parse_allcells,
    truth_cells,
)

KS = [1, 5, 10, 20, 50, 100]


def eval_sid(job):
    sid, run_dir, rerank = job if len(job) == 3 else (*job, "none")
    allc = Path(run_dir) / sid / f"{sid.replace('-', '_')}.allcells"
    pool = []
    if allc.exists():
        if rerank == "linear":
            from pxrd_cell_indexing.rerank import order_allcells

            pool = order_allcells(allc)
        else:
            cands = parse_allcells(allc)
            cands.sort(key=lambda c: -c["McM20"])
            pool = [c["params"] for c in cands]

    t = truth_cells(CIF_DIR / f"{sid}.cif")
    row = {"sample_id": sid, "n_pool": len(pool)}
    for tag in ("conv", "prim"):
        flags = [l4(p, t[tag]) for p in pool]
        dets = [f[2] for f in flags if f[2] is not None]
        row[tag] = {
            "lib_loose": any(f[0] for f in flags),
            "lib_strict": any(f[1] for f in flags),
            "topk_loose": {k: any(f[0] for f in flags[:k]) for k in KS},
            "topk_strict": {k: any(f[1] for f in flags[:k]) for k in KS},
            "best_det": min(dets) if dets else None,
        }
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n_samples": n, "mean_pool": sum(r["n_pool"] for r in rows) / n}
    for tag in ("conv", "prim"):
        dets = [r[tag]["best_det"] for r in rows if r[tag]["best_det"] is not None]
        out[tag] = {
            "lib_loose": sum(r[tag]["lib_loose"] for r in rows) / n,
            "lib_strict": sum(r[tag]["lib_strict"] for r in rows) / n,
            "topk_loose": {
                str(k): sum(r[tag]["topk_loose"][k] for r in rows) / n for k in KS
            },
            "topk_strict": {
                str(k): sum(r[tag]["topk_strict"][k] for r in rows) / n for k in KS
            },
            "best_det_median": st.median(dets) if dets else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True,
                    help="label=path, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--rerank",
        choices=["none", "linear"],
        default="none",
        help="none=McM20; linear=V0 equal-weight reranker",
    )
    args = ap.parse_args()

    report = {}
    for spec in args.run_dir:
        label, path = spec.rsplit("=", 1)
        run_dir = Path(path)
        sids = sorted(p.name for p in run_dir.iterdir() if p.name.startswith("mp-"))
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(eval_sid, (s, str(run_dir), args.rerank)) for s in sids]
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: r["sample_id"])
        report[label] = {"run_dir": str(run_dir), **summarize(rows), "per_sample": rows}

        s = report[label]
        print(f"\n===== {label}  (pool avg {s['mean_pool']:.0f}) =====")
        print(f"{'K':>5} {'conv loose':>11} {'conv strict':>12} "
              f"{'prim loose':>11} {'prim strict':>12}")
        for k in KS:
            print(f"{k:>5} "
                  f"{s['conv']['topk_loose'][str(k)]:11.0%} "
                  f"{s['conv']['topk_strict'][str(k)]:12.0%} "
                  f"{s['prim']['topk_loose'][str(k)]:11.0%} "
                  f"{s['prim']['topk_strict'][str(k)]:12.0%}")
        for tag in ("conv", "prim"):
            print(f"  {tag} library: loose {s[tag]['lib_loose']:.0%} "
                  f"strict {s[tag]['lib_strict']:.0%} "
                  f"| best|det| median {s[tag]['best_det_median']}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
