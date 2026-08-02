#!/usr/bin/env python3
"""Compare ranking policies over the same reseeded McMaille library.

The library content is fixed; only the ordering changes. Each ``.allcells`` row
carries ``seed_src`` (1-based index into the seed file, i.e. the NN candidate
rank) and ``stage`` (1=RAW, 2=SUPCEL, 3=LOCAL_MC, 4=CELREF), so rows can be
re-ordered by NN confidence instead of McM20.

Policies:
  nn_only    NN candidates verbatim, McMaille never runs (reference ceiling)
  mcm20      current pipeline: sort by McM20 desc
  nn_stage   NN seed order; within a seed prefer CELREF > LOCAL_MC > SUPCEL > RAW
  nn_mcm20   NN seed order; within a seed sort by McM20 desc
  nn_refined NN seed order, exactly one row per seed (CELREF if present, else RAW)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

KS = [1, 5, 10, 20, 50]
POLICIES = ["nn_only", "mcm20", "nn_stage", "nn_mcm20", "nn_refined"]

ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)

STAGE_PRIORITY = {4: 0, 3: 1, 2: 2, 1: 3}  # CELREF first, RAW last


def parse_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        g = m.groups()
        out.append(
            {
                "seed_src": int(g[1]),
                "stage": int(g[2]),
                "n_indexed": int(g[3]),
                "McM20": float(g[4]),
                "params": [float(g[i]) for i in range(7, 13)],
            }
        )
    return out


def order(rows: list[dict], policy: str) -> list[list[float]]:
    if policy == "mcm20":
        rs = sorted(rows, key=lambda r: -r["McM20"])
    elif policy == "nn_stage":
        rs = sorted(rows, key=lambda r: (r["seed_src"], STAGE_PRIORITY[r["stage"]]))
    elif policy == "nn_mcm20":
        rs = sorted(rows, key=lambda r: (r["seed_src"], -r["McM20"]))
    elif policy == "nn_refined":
        best: dict[int, dict] = {}
        for r in rows:
            cur = best.get(r["seed_src"])
            if cur is None or STAGE_PRIORITY[r["stage"]] < STAGE_PRIORITY[cur["stage"]]:
                best[r["seed_src"]] = r
        rs = [best[k] for k in sorted(best)]
    else:
        raise ValueError(policy)
    return [r["params"] for r in rs]


def eval_sid(job):
    sid, run_dir, nn_cands, top_k = job
    rows = []
    allc = Path(run_dir) / sid / f"{sid.replace('-', '_')}.allcells"
    if allc.exists():
        rows = parse_rows(allc)

    t = truth_cells(CIF_DIR / f"{sid}.cif")
    res = {"sample_id": sid}
    for policy in POLICIES:
        pool = nn_cands[:top_k] if policy == "nn_only" else order(rows, policy)
        entry = {"n_pool": len(pool)}
        for tag in ("conv", "prim"):
            flags = [l4(p, t[tag]) for p in pool]
            entry[tag] = {
                "lib_strict": any(f[1] for f in flags),
                "topk_strict": {k: any(f[1] for f in flags[:k]) for k in KS},
                "topk_loose": {k: any(f[0] for f in flags[:k]) for k in KS},
            }
        res[policy] = entry
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pool-json", required=True)
    ap.add_argument("--top-k", type=int, default=20, help="NN seeds used in the run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pool = json.loads(Path(args.pool_json).read_text())["per_sample"]
    run_dir = Path(args.run_dir)
    sids = sorted(p.name for p in run_dir.iterdir() if p.name.startswith("mp-"))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(eval_sid, (s, str(run_dir), pool[s]["candidates"], args.top_k))
            for s in sids
        ]
        for f in as_completed(futs):
            rows.append(f.result())
    rows.sort(key=lambda r: r["sample_id"])
    n = len(rows)

    report = {"run_dir": str(run_dir), "n_samples": n, "policies": {}}
    for policy in POLICIES:
        agg = {
            "mean_pool": sum(r[policy]["n_pool"] for r in rows) / n,
        }
        for tag in ("conv", "prim"):
            agg[tag] = {
                "lib_strict": sum(r[policy][tag]["lib_strict"] for r in rows) / n,
                "topk_strict": {
                    str(k): sum(r[policy][tag]["topk_strict"][k] for r in rows) / n
                    for k in KS
                },
                "topk_loose": {
                    str(k): sum(r[policy][tag]["topk_loose"][k] for r in rows) / n
                    for k in KS
                },
            }
        report["policies"][policy] = agg

    print(f"\n=== prim strict, {run_dir.name} (n={n}) ===")
    hdr = "  ".join(f"K={k}".rjust(6) for k in KS)
    print(f"{'policy':<12} {'pool':>5}  {hdr}  {'lib':>6}")
    for policy in POLICIES:
        a = report["policies"][policy]
        cells = "  ".join(
            f"{a['prim']['topk_strict'][str(k)]:6.0%}" for k in KS
        )
        print(
            f"{policy:<12} {a['mean_pool']:5.0f}  {cells}  "
            f"{a['prim']['lib_strict']:6.0%}"
        )

    Path(args.out).write_text(json.dumps({**report, "per_sample": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
