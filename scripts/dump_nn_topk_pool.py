#!/usr/bin/env python3
"""Dump the Top-K candidate pool of an indexing checkpoint on MP100.

Writes per-sample candidate lists so the pool can be scored with the same
strict criterion used for the seeded-McMaille stage-wise analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_top_k_candidates,
    parse_length_scale_factors,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--mp100-dir", type=Path, default=DEFAULT_MP100_DIR)
    ap.add_argument("--convention", default="niggli")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--scale-set", default="default")
    ap.add_argument("--bravais-set", default="default")
    ap.add_argument("--pool-max-log-volume-ratio", type=float, default=-1.0)
    ap.add_argument("--no-axis-scale-variants", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device)

    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    model.eval()
    normalizer = build_lattice_normalizer(config.data)

    samples = load_mp100_dataset(args.mp100_dir, convention=args.convention)

    vol_vs_base = (
        None
        if args.pool_max_log_volume_ratio < 0
        else args.pool_max_log_volume_ratio
    )
    topk_config = TopKConfig(
        k=args.top_k,
        length_scale_factors=parse_length_scale_factors(args.scale_set),
        max_log_volume_ratio_vs_base=vol_vs_base,
        include_axis_scale_variants=not args.no_axis_scale_variants,
        bravais_set=args.bravais_set,
    )

    per_sample: dict[str, dict] = {}
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        xs, ys, ns, ids, truths = [], [], [], [], []
        for s in batch:
            px, py, pn = peaks_to_model_tensors(s.two_theta, s.intensity)
            xs.append(torch.from_numpy(px))
            ys.append(torch.from_numpy(py))
            ns.append(pn)
            ids.append(s.sample_id)
            truths.append(s.truth_lattice.tolist())

        pxrd_x = torch.cat(xs, dim=0).to(device)
        pxrd_y = torch.cat(ys, dim=0).to(device)
        peak_num = torch.tensor(ns, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(pxrd_x, pxrd_y, peak_num)
            lattice = normalizer.denormalize(outputs["lattice_norm"])
            pools = build_top_k_candidates(lattice, k=args.top_k, config=topk_config)

        for i, sid in enumerate(ids):
            per_sample[sid] = {
                "raw_pred": lattice[i].cpu().tolist(),
                "eval_truth": truths[i],
                "candidates": [
                    [c.a, c.b, c.c, c.alpha, c.beta, c.gamma] for c in pools[i]
                ],
            }
        print(f"{start + len(batch)}/{len(samples)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "experiment": experiment_name,
                "checkpoint": str(args.checkpoint),
                "top_k": args.top_k,
                "convention": args.convention,
                "scale_set": args.scale_set,
                "bravais_set": args.bravais_set,
                "per_sample": per_sample,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
