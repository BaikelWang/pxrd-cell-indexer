#!/usr/bin/env python3
"""Does the flow ODE step count limit seed *precision*?

The seed pool is scored at L4-strict (``ltol=0.05``), but McMaille's Rp gate
needs the cell to be right to ~0.2%. So a step count that is fine for "lands in
the 5% window" can still be the thing capping the sub-1% tail.

Integration is Euler with a fixed z0 (same generator seed for every setting), so
differences between step counts are the discretisation error alone, not
sampling noise.

Reports, per step count, the fraction of MP100 with a candidate whose aligned
length error beats each threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

THRESHOLDS = [0.002, 0.005, 0.01, 0.02, 0.05]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/flow_seedgen/full6m_equiv_off/best.pt")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json")
    ap.add_argument("--steps", default="25,50,100,200,400")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="results/flow_seedgen/probe_sample_steps.json")
    return ap.parse_args()


def score_sample(payload):
    """Best aligned length error over K candidates. Runs in a worker process."""
    from diagnose_mcmaille_value import cell_err
    from remeasure_l4_prim_vs_conv import l4

    sid, cells, truth = payload
    best = None
    first_hit = {t: None for t in THRESHOLDS}
    for rank, row in enumerate(cells, 1):
        if not np.isfinite(row).all():
            continue
        if not l4(row.tolist(), truth)[1]:
            continue
        err, _ = cell_err(row.tolist(), truth)
        if not np.isfinite(err):
            continue
        if best is None or err < best:
            best = err
        for t in THRESHOLDS:
            if first_hit[t] is None and err < t:
                first_hit[t] = rank
    return sid, best, first_hit


def main() -> None:
    args = parse_args()
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from pxrd_cell_indexing.geometry import gstar6_to_lattice
    from remeasure_l4_prim_vs_conv import CIF_DIR, truth_cells
    from train_flow_seedgen import SeedGenerator, load_mp100_eval

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(args.stats)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = argparse.Namespace(**ck["args"])
    model = SeedGenerator(ckpt_args).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    print(
        f"loaded {args.ckpt} (epoch {ck.get('epoch')}, "
        f"reported library_strict={ck.get('mp100', {}).get('library_strict')})",
        flush=True,
    )

    items = load_mp100_eval(device)
    truths = {it["sample_id"]: it["prim"] for it in items}
    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)

    step_list = [int(s) for s in args.steps.split(",")]
    results = {}

    for steps in step_list:
        pools = {}
        with torch.no_grad():
            for it in items:
                emb = model.encode(it["pxrd_x"], it["pxrd_y"], it["peak_num"])
                # Same generator seed per sample per setting => identical z0.
                gen = torch.Generator(device=device).manual_seed(args.seed)
                z = model.sample(emb, num_samples=args.k, steps=steps, generator=gen)[0]
                cells = gstar6_to_lattice(std * z + mean).cpu().numpy()
                pools[it["sample_id"]] = cells

        payloads = [(sid, cells, truths[sid]) for sid, cells in pools.items()]
        best_err, first_hits = {}, {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(score_sample, p) for p in payloads]
            for fut in as_completed(futs):
                sid, best, fh = fut.result()
                best_err[sid] = best
                first_hits[sid] = fh

        n = len(best_err)
        hits = [v for v in best_err.values() if v is not None]
        row = {
            "n": n,
            "library_l4_strict": len(hits) / n,
            "median_best_err": float(np.median(hits)) if hits else None,
            "library_at_threshold": {
                str(t): sum(1 for e in hits if e < t) / n for t in THRESHOLDS
            },
            "coverage_1pct_at_k": {
                str(k): sum(
                    1 for fh in first_hits.values() if fh[0.01] is not None and fh[0.01] <= k
                )
                / n
                for k in (1, 5, 20, 50, args.k)
            },
        }
        results[str(steps)] = row
        at = row["library_at_threshold"]
        print(
            f"steps={steps:4d}  L4-strict={row['library_l4_strict']:.0%}  "
            f"<1%={at['0.01']:.0%}  <0.5%={at['0.005']:.0%}  <0.2%={at['0.002']:.0%}  "
            f"median={row['median_best_err']:.4%}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"ckpt": args.ckpt, "k": args.k, "seed": args.seed, "by_steps": results}, indent=2)
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
