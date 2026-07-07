#!/usr/bin/env python3
"""Evaluate checkpoint on valid1400 with pymatgen lattice match + Top-K recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.dataset import (
    PXRDDatasetConfig,
    PeakFilterConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import LatticeNormalizer
from pxrd_cell_indexing.eval import (
    crystal_system_accuracy,
    evaluate_by_crystal_system,
    lattice_mae,
    length_mape,
    top1_joint_match_rate,
    top1_lattice_match_proxy,
    top1_lattice_match_rate,
    topk_lattice_match_rate,
)
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "experiments" / "smoke_unfrozen_seed42" / "checkpoints" / "best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "valid1400_real_match_smoke_unfrozen_seed42.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = LatticeNormalizer.from_json(config.data.lattice_stats)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )

    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    top1_preds: list[list[float]] = []
    truths: list[list[float]] = []
    pred_cs_idx: list[int] = []
    target_cs_idx: list[int] = []
    candidate_lists = []

    metric_sums = {
        "crystal_system_accuracy": 0.0,
        "lattice_mae": 0.0,
        "length_mape": 0.0,
        "top1_lattice_match_proxy": 0.0,
    }
    metric_count = 0
    per_cs_accum: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
            pred = normalizer.denormalize(outputs["lattice_norm"])
            target = batch["lattice"]
            pred_cs = outputs["crystal_system_logits"].argmax(dim=-1)

            batch_candidates = build_top_k_candidates(
                outputs["crystal_system_logits"],
                pred,
                k=args.top_k,
                config=TopKConfig(k=args.top_k),
            )

            for idx in range(pred.shape[0]):
                top1 = batch_candidates[idx][0]
                top1_preds.append(
                    [top1.a, top1.b, top1.c, top1.alpha, top1.beta, top1.gamma]
                )
                truths.append(target[idx].cpu().tolist())
                pred_cs_idx.append(int(pred_cs[idx].item()))
                target_cs_idx.append(int(batch["crystal_system_idx"][idx].item()))
                candidate_lists.append(batch_candidates[idx])

            metric_sums["crystal_system_accuracy"] += crystal_system_accuracy(
                outputs["crystal_system_logits"], batch["crystal_system_idx"]
            )
            metric_sums["lattice_mae"] += lattice_mae(pred, target)
            metric_sums["length_mape"] += length_mape(pred, target)
            metric_sums["top1_lattice_match_proxy"] += top1_lattice_match_proxy(pred, target)
            metric_count += 1

            per_cs = evaluate_by_crystal_system(
                pred, target, batch["crystal_system_idx"]
            )
            for cs_name, cs_metrics in per_cs.items():
                bucket = per_cs_accum.setdefault(
                    cs_name,
                    {"lattice_mae": 0.0, "top1_lattice_match_proxy": 0.0, "count": 0.0},
                )
                count = cs_metrics["count"]
                bucket["lattice_mae"] += cs_metrics["lattice_mae"] * count
                bucket["top1_lattice_match_proxy"] += (
                    cs_metrics["top1_lattice_match_proxy"] * count
                )
                bucket["count"] += count

    averaged = {key: value / max(metric_count, 1) for key, value in metric_sums.items()}
    per_cs_summary: dict[str, dict[str, float]] = {}
    for cs_name, bucket in per_cs_accum.items():
        count = max(bucket["count"], 1.0)
        per_cs_summary[cs_name] = {
            "lattice_mae": bucket["lattice_mae"] / count,
            "top1_lattice_match_proxy": bucket["top1_lattice_match_proxy"] / count,
            "count": bucket["count"],
        }

    top1_match = top1_lattice_match_rate(top1_preds, truths)
    joint_match = top1_joint_match_rate(
        top1_preds, truths, pred_cs_idx, target_cs_idx
    )
    result = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "valid_jsonl": config.data.valid_jsonl,
        "n_samples": len(truths),
        "top_k": args.top_k,
        "metrics": {
            **averaged,
            "top1_lattice_match_rate": top1_match,
            "top1_joint_match_rate": joint_match,
            "topk_lattice_match_rate": topk_lattice_match_rate(candidate_lists, truths),
        },
        "per_crystal_system": per_cs_summary,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on valid1400")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
