#!/usr/bin/env python3
"""Dump K lattice candidates from a trained flow seed-generator checkpoint.

Output schema matches ``run_mp100_reseed_nn.py``:
  {per_sample: {sid: {raw_pred, candidates: [[a,b,c,α,β,γ], ...]}}}
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pxrd_cell_indexing.data.normalization import (  # noqa: E402
    GStar6Normalizer,
    GStar6Stats,
)
from pxrd_cell_indexing.geometry import gstar6_to_lattice  # noqa: E402
from remeasure_l4_prim_vs_conv import l4  # noqa: E402
from train_flow_seedgen import SeedGenerator, load_mp100_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = Namespace(**ck["args"])
    # encoder_cfg is baked into SeedGenerator via ENCODER_CFG module constant;
    # flow hyperparams come from train_args.
    model = SeedGenerator(train_args).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    normalizer = GStar6Normalizer.from_stats(
        GStar6Stats(
            component_mean=tuple(ck["normalizer"]["component_mean"]),
            component_std=tuple(ck["normalizer"]["component_std"]),
        )
    )

    items = load_mp100_eval(device)
    mean = torch.tensor(normalizer.component_mean, device=device, dtype=torch.float32)
    std = torch.tensor(normalizer.component_std, device=device, dtype=torch.float32)

    per = {}
    hits_loose = hits_strict = 0
    topk_strict = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}

    with torch.no_grad():
        for i, item in enumerate(items, 1):
            emb = model.encode(item["pxrd_x"], item["pxrd_y"], item["peak_num"])
            z = model.flow.sample(
                emb, num_samples=args.top_k, steps=args.sample_steps
            )[0]
            cells = gstar6_to_lattice(z * std + mean).cpu().numpy()
            cands = [[float(x) for x in row.tolist()] for row in cells]
            per[item["sample_id"]] = {
                "raw_pred": cands[0],
                "candidates": cands,
            }
            flags = [l4(c, item["prim"]) for c in cands]
            if any(f[0] for f in flags):
                hits_loose += 1
            if any(f[1] for f in flags):
                hits_strict += 1
            for kk in topk_strict:
                if any(f[1] for f in flags[:kk]):
                    topk_strict[kk] += 1
            if i % 20 == 0:
                print(f"{i}/{len(items)}", flush=True)

    n = len(items)
    summary = {
        "ckpt": str(args.ckpt),
        "epoch": ck.get("epoch"),
        "top_k": args.top_k,
        "sample_steps": args.sample_steps,
        "n_samples": n,
        "L4_loose_library": hits_loose / n,
        "L4_strict_library": hits_strict / n,
        "L4_strict_topk": {str(k): topk_strict[k] / n for k in sorted(topk_strict)},
        "ckpt_mp100": ck.get("mp100"),
    }
    out = {
        "experiment": "flow_seedgen_full6m_equiv_off",
        "checkpoint": str(args.ckpt),
        "top_k": args.top_k,
        "summary": summary,
        "per_sample": per,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
