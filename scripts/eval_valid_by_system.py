#!/usr/bin/env python3
"""Score a seed-generator checkpoint on the stratified valid subset, per system.

The old selection metric read a plain prefix of ``valid1400``, which is stored in
contiguous crystal-system blocks, so it only ever saw cubic and tetragonal and
sat at ~99%. This reports the same hit rates over a balanced subset with a
per-system breakdown, which is what ``--select-metric valid_macro`` now uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pxrd_cell_indexing.data.normalization import GStar6Normalizer  # noqa: E402
from train_flow_seedgen import (  # noqa: E402
    SELECT_TOLS,
    SeedGenerator,
    evaluate_valid_precision,
    load_valid_eval_batches,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/flow_seedgen/pxrd_indexer_full6m_v2/best.pt")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json")
    ap.add_argument("--valid-jsonl", default="data/processed/valid1400_niggli_seed42.jsonl")
    ap.add_argument("--valid-lmdb", default="/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_valid.lmdb")
    ap.add_argument("--valid-eval-n", type=int, default=700)
    ap.add_argument("--eval-k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--eval-workers", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    normalizer = GStar6Normalizer.from_json(str(ROOT / args.stats))
    model = SeedGenerator(argparse.Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    loader_args = argparse.Namespace(
        valid_jsonl=str(ROOT / args.valid_jsonl),
        valid_lmdb=args.valid_lmdb,
        train_jsonl=str(ROOT / args.valid_jsonl),
        train_lmdb=args.valid_lmdb,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    batches, labels = load_valid_eval_batches(loader_args, args.valid_eval_n)
    print(f"ckpt={args.ckpt} epoch={ck.get('epoch')} n={len(labels)} K={args.eval_k}", flush=True)

    prec = evaluate_valid_precision(
        model,
        normalizer,
        batches,
        device,
        k=args.eval_k,
        steps=args.sample_steps,
        seed=args.seed,
        workers=args.eval_workers,
        labels=labels,
    )

    tols = [str(t) for t in SELECT_TOLS]
    head = "  ".join(f"<{float(t):.1%}".rjust(7) for t in tols)
    print(f"\n{'system':14s} {'n':>4s}  {head}")
    for s, v in sorted(prec.get("by_system", {}).items()):
        cells = "  ".join(f"{v['hit_rate'][t]:7.1%}" for t in tols)
        print(f"{s:14s} {v['n']:4d}  {cells}")
    pooled = "  ".join(f"{prec['hit_rate'][t]:7.1%}" for t in tols)
    print(f"{'POOLED':14s} {prec['n']:4d}  {pooled}")
    if "macro_hit" in prec:
        macro = "  ".join(f"{prec['macro_hit'][t]:7.1%}" for t in tols)
        worst = "  ".join(f"{prec['worst_system_hit'][t]:7.1%}" for t in tols)
        print(f"{'MACRO':14s} {'':4s}  {macro}")
        print(f"{'WORST':14s} {'':4s}  {worst}")

    out = Path(args.out) if args.out else ROOT / "results/flow_seedgen/valid_by_system.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "epoch": ck.get("epoch"), **prec}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
