#!/usr/bin/env python3
"""Diagnose 10k model errors on valid1400: confusion matrix and per-CS breakdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pxrd_cell_indexing.data.dataset import (
    PXRDDatasetConfig,
    PeakFilterConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import (
    infer_crystal_system_idx_from_lattice,
    lattice_match_pymatgen,
    lattice_match_proxy,
    top1_lattice_match_rate,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "experiments" / "smoke_unfrozen_seed42" / "checkpoints" / "best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnose_10k_smoke_unfrozen_seed42.json"

PARAM_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


def _build_confusion_matrix(
    true_idx: list[int],
    pred_idx: list[int],
    num_classes: int = 7,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(true_idx, pred_idx, strict=True):
        matrix[truth][pred] += 1
    return matrix


def _per_param_mae(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.abs(pred[:, i] - target[:, i]).mean())
        for i, name in enumerate(PARAM_NAMES)
    }


def _recall_per_class(true_idx: list[int], pred_idx: list[int]) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for cs_idx, cs_name in enumerate(CRYSTAL_SYSTEMS):
        mask = [idx == cs_idx for idx in true_idx]
        total = sum(mask)
        if total == 0:
            continue
        correct = sum(1 for truth, pred, is_true in zip(true_idx, pred_idx, mask, strict=True) if is_true and truth == pred)
        recalls[cs_name] = correct / total
    return recalls


def _analyze_hex_trig(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Root-cause analysis for hexagonal/trigonal failures."""
    focus_systems = {"hexagonal", "trigonal"}
    analysis: dict[str, Any] = {}

    for cs_name in focus_systems:
        cs_records = [record for record in records if record["true_crystal_system"] == cs_name]
        if not cs_records:
            continue

        truths = np.array([record["truth_lattice"] for record in cs_records], dtype=np.float64)
        preds = np.array([record["pred_lattice"] for record in cs_records], dtype=np.float64)
        pred_cs = [record["pred_crystal_system"] for record in cs_records]
        cls_correct = sum(record["cls_correct"] for record in cs_records)
        match_count = sum(record["lattice_match"] for record in cs_records)
        match_when_cls_correct = sum(
            record["lattice_match"]
            for record in cs_records
            if record["cls_correct"]
        )
        cls_correct_count = cls_correct

        pred_cs_counts: dict[str, int] = defaultdict(int)
        for name in pred_cs:
            pred_cs_counts[name] += 1

        analysis[cs_name] = {
            "n_samples": len(cs_records),
            "cls_recall": cls_correct / len(cs_records),
            "top1_lattice_match_rate": match_count / len(cs_records),
            "match_rate_when_cls_correct": (
                match_when_cls_correct / cls_correct_count if cls_correct_count else 0.0
            ),
            "pred_crystal_system_distribution": dict(pred_cs_counts),
            "truth_param_stats": {
                name: {
                    "mean": float(truths[:, i].mean()),
                    "std": float(truths[:, i].std()),
                }
                for i, name in enumerate(PARAM_NAMES)
            },
            "pred_param_stats": {
                name: {
                    "mean": float(preds[:, i].mean()),
                    "std": float(preds[:, i].std()),
                }
                for i, name in enumerate(PARAM_NAMES)
            },
            "per_param_mae": _per_param_mae(preds, truths),
            "truth_ab_ratio_mean": float((truths[:, 0] / np.clip(truths[:, 1], 1e-6, None)).mean()),
            "pred_ab_ratio_mean": float((preds[:, 0] / np.clip(preds[:, 1], 1e-6, None)).mean()),
        }

        if cls_correct_count == 0:
            analysis[cs_name]["dominant_failure"] = "classification"
        elif match_when_cls_correct / cls_correct_count < 0.1:
            analysis[cs_name]["dominant_failure"] = "regression"
        else:
            analysis[cs_name]["dominant_failure"] = "mixed"

    return analysis


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, checkpoint, experiment_name = load_indexing_model_from_checkpoint(
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

    records: list[dict[str, Any]] = []
    true_idx: list[int] = []
    pred_idx: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
            pred = normalizer.denormalize(outputs["lattice_norm"])
            pred_cs_idx = infer_crystal_system_idx_from_lattice(pred.cpu())

            for idx in range(pred.shape[0]):
                truth_lattice = batch["lattice"][idx].cpu().tolist()
                pred_lattice = pred[idx].cpu().tolist()
                truth_cs_idx = int(batch["crystal_system_idx"][idx].item())
                predicted_cs_idx = int(pred_cs_idx[idx])
                cls_correct = predicted_cs_idx >= 0 and truth_cs_idx == predicted_cs_idx
                lattice_match = lattice_match_pymatgen(pred_lattice, truth_lattice)
                proxy_match = bool(
                    lattice_match_proxy(
                        pred[idx : idx + 1], batch["lattice"][idx : idx + 1]
                    )
                    .reshape(-1)[0]
                    .item()
                )

                record = {
                    "sample_id": batch["sample_id"][idx],
                    "true_crystal_system": CRYSTAL_SYSTEMS[truth_cs_idx],
                    "pred_crystal_system": (
                        CRYSTAL_SYSTEMS[predicted_cs_idx]
                        if predicted_cs_idx >= 0
                        else "unknown"
                    ),
                    "cls_correct": cls_correct,
                    "lattice_match": lattice_match,
                    "lattice_match_proxy": proxy_match,
                    "truth_lattice": truth_lattice,
                    "pred_lattice": pred_lattice,
                }
                records.append(record)
                true_idx.append(truth_cs_idx)
                pred_idx.append(predicted_cs_idx)

    all_preds = np.array([record["pred_lattice"] for record in records], dtype=np.float64)
    all_truths = np.array([record["truth_lattice"] for record in records], dtype=np.float64)

    per_cs_breakdown: dict[str, Any] = {}
    for cs_idx, cs_name in enumerate(CRYSTAL_SYSTEMS):
        cs_records = [record for record in records if record["true_crystal_system"] == cs_name]
        if not cs_records:
            continue
        truths = np.array([record["truth_lattice"] for record in cs_records], dtype=np.float64)
        preds = np.array([record["pred_lattice"] for record in cs_records], dtype=np.float64)
        per_cs_breakdown[cs_name] = {
            "n_samples": len(cs_records),
            "cls_recall": _recall_per_class(
                [CRYSTAL_SYSTEM_TO_IDX[record["true_crystal_system"]] for record in cs_records],
                [CRYSTAL_SYSTEM_TO_IDX[record["pred_crystal_system"]] for record in cs_records],
            ).get(cs_name, 0.0),
            "top1_lattice_match_rate": sum(record["lattice_match"] for record in cs_records)
            / len(cs_records),
            "top1_lattice_match_proxy": sum(record["lattice_match_proxy"] for record in cs_records)
            / len(cs_records),
            "per_param_mae": _per_param_mae(preds, truths),
        }

    result = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "n_samples": len(records),
        "overall": {
            "crystal_system_accuracy": sum(record["cls_correct"] for record in records) / len(records),
            "top1_lattice_match_rate": top1_lattice_match_rate(all_preds, all_truths),
            "top1_lattice_match_proxy": sum(record["lattice_match_proxy"] for record in records)
            / len(records),
        },
        "confusion_matrix": {
            "labels": list(CRYSTAL_SYSTEMS),
            "matrix": _build_confusion_matrix(true_idx, pred_idx),
        },
        "per_crystal_system_recall": _recall_per_class(true_idx, pred_idx),
        "per_crystal_system_breakdown": per_cs_breakdown,
        "hex_trig_root_cause": _analyze_hex_trig(records),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(result["hex_trig_root_cause"], indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose 10k model on valid1400")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
