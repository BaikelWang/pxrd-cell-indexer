#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the MP100 benchmark (lattice match only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import LatticeNormalizer
from pxrd_cell_indexing.eval import (
    top1_joint_match_rate,
    top1_lattice_match_rate,
    topk_lattice_match_rate,
)
from pxrd_cell_indexing.model.heads import IndexingModel
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "experiments" / "smoke_unfrozen_seed42" / "checkpoints" / "best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mp100_eval_smoke_unfrozen_seed42.json"

# Reference baselines from docs/开发日志/起点.md (ideal peaks, ltol=0.3, atol=10°).
REFERENCE_BASELINES = {
    "mcmaille_top1_lattice_match": 0.764,
    "jade9_top1_lattice_match": 0.725,
    "realpxrd_without_l_top1_lattice_match": 0.05,
}


def run_inference_on_mp100(
    model: IndexingModel,
    normalizer: LatticeNormalizer,
    samples: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    top1_preds: list[list[float]] = []
    truths: list[list[float]] = []
    pred_cs_idx: list[int] = []
    target_cs_idx: list[int] = []
    candidate_lists = []
    per_sample: list[dict[str, Any]] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        pxrd_x_parts = []
        pxrd_y_parts = []
        peak_nums = []
        batch_truths = []
        batch_ids = []
        batch_target_cs = []

        for sample in batch:
            pxrd_x, pxrd_y, peak_num = peaks_to_model_tensors(sample.two_theta, sample.intensity)
            pxrd_x_parts.append(torch.from_numpy(pxrd_x))
            pxrd_y_parts.append(torch.from_numpy(pxrd_y))
            peak_nums.append(peak_num)
            batch_truths.append(sample.truth_lattice.tolist())
            batch_ids.append(sample.sample_id)
            batch_target_cs.append(CRYSTAL_SYSTEM_TO_IDX[sample.crystal_system])

        pxrd_x = torch.cat(pxrd_x_parts, dim=0).to(device)
        pxrd_y = torch.cat(pxrd_y_parts, dim=0).to(device)
        peak_num = torch.tensor(peak_nums, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(pxrd_x, pxrd_y, peak_num)
            lattice = normalizer.denormalize(outputs["lattice_norm"])
            pred_cs = outputs["crystal_system_logits"].argmax(dim=-1)
            batch_candidates = build_top_k_candidates(
                outputs["crystal_system_logits"],
                lattice,
                k=top_k,
                config=TopKConfig(k=top_k),
            )

        for idx, sample_id in enumerate(batch_ids):
            truth = batch_truths[idx]
            candidates = batch_candidates[idx]
            top1 = candidates[0]
            top1_pred = [top1.a, top1.b, top1.c, top1.alpha, top1.beta, top1.gamma]
            top1_preds.append(top1_pred)
            truths.append(truth)
            pred_cs_idx.append(int(pred_cs[idx].item()))
            target_cs_idx.append(batch_target_cs[idx])
            candidate_lists.append(candidates)
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "truth_lattice": truth,
                    "top1_pred": top1_pred,
                    "top1_crystal_system": top1.crystal_system,
                    "top1_confidence": top1.confidence,
                    "topk_size": len(candidates),
                }
            )

    return {
        "n_samples": len(samples),
        "top1_lattice_match_rate": top1_lattice_match_rate(top1_preds, truths),
        "top1_joint_match_rate": top1_joint_match_rate(
            top1_preds, truths, pred_cs_idx, target_cs_idx
        ),
        "topk_lattice_match_rate": topk_lattice_match_rate(candidate_lists, truths),
        "per_sample": per_sample,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    samples = load_mp100_dataset(args.mp100_dir)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    normalizer = LatticeNormalizer.from_json(config.data.lattice_stats)

    metrics = run_inference_on_mp100(
        model,
        normalizer,
        samples,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    result = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "mp100_dir": str(args.mp100_dir),
        "top_k": args.top_k,
        "metrics": {
            "top1_lattice_match_rate": metrics["top1_lattice_match_rate"],
            "top1_joint_match_rate": metrics["top1_joint_match_rate"],
            "topk_lattice_match_rate": metrics["topk_lattice_match_rate"],
            "n_samples": metrics["n_samples"],
        },
        "reference_baselines": REFERENCE_BASELINES,
        "per_sample": metrics["per_sample"],
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(result["reference_baselines"], indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on MP100 benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mp100-dir", type=Path, default=DEFAULT_MP100_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
