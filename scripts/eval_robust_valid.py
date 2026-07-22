#!/usr/bin/env python3
"""A4 (v3 §8.5 / v4 §7): evaluate a checkpoint on V-clean + the frozen
robust-valid perturbation sets, reporting the Gate metric
(strict raw Top-1 elementwise, ltol=0.05/atol=3deg) per set and per CS.

Usage:
    python scripts/eval_robust_valid.py \
        --config configs/scale_100k_a3_g1_gstar6.yaml \
        --checkpoint results/experiments/scale_100k_a3_g1_gstar6_seed42/checkpoints/best.pt \
        --output-path results/beat_engine/a4_robust/c0_robust_valid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.dataset import PeakFilterConfig, PXRDDatasetConfig, build_dataloader
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import (
    angle_mae,
    evaluate_by_crystal_system,
    lattice_mae,
    length_mae,
    top1_elementwise_match_rate,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUST_VALID_DIR = PROJECT_ROOT / "data" / "processed" / "robust_valid"

NAMED_SETS = ["clean", "zero", "jitter", "drop", "impurity", "mixed"]


def _dataset_paths(name: str, seed: int, base_lmdb: Path, base_jsonl: Path) -> tuple[Path, Path]:
    if name == "clean":
        return base_lmdb, base_jsonl
    return (
        ROBUST_VALID_DIR / f"robust_valid_{name}_seed{seed}.lmdb",
        ROBUST_VALID_DIR / f"robust_valid_{name}_seed{seed}.jsonl",
    )


def evaluate_one_set(
    model,
    normalizer,
    lmdb_path: Path,
    jsonl_path: Path,
    config: TrainConfig,
    device: torch.device,
    ltol: float,
    atol_deg: float,
) -> dict[str, Any]:
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=lmdb_path,
        split="valid",
        sample_list_path=jsonl_path,
        peak_filter=PeakFilterConfig(
            intensity_min=config.model.intensity_min,
            max_peaks=config.model.max_peaks,
        ),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=config.data.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    preds: list[list[float]] = []
    truths: list[list[float]] = []
    cs_idx_all: list[torch.Tensor] = []
    sums = {"lattice_mae": 0.0, "length_mae": 0.0, "angle_mae": 0.0}
    per_cs_accum: dict[str, dict[str, float]] = {}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(
                batch["pxrd_x"],
                batch["pxrd_y"],
                batch["peak_num"],
                crystal_system_idx=batch["crystal_system_idx"],
                cs_route=config.model.cs_route,
                lattice_phys=batch["lattice"],
                setting_route=config.model.setting_route,
            )
            pred = normalizer.denormalize(outputs["lattice_norm"])
            target = batch["lattice"]
            preds.extend(pred.cpu().tolist())
            truths.extend(target.cpu().tolist())
            cs_idx_all.append(batch["crystal_system_idx"].cpu())

            sums["lattice_mae"] += lattice_mae(pred, target)
            sums["length_mae"] += length_mae(pred, target)
            sums["angle_mae"] += angle_mae(pred, target)
            n_batches += 1

            per_cs = evaluate_by_crystal_system(pred, target, batch["crystal_system_idx"])
            for cs_name, cs_metrics in per_cs.items():
                bucket = per_cs_accum.setdefault(cs_name, {"count": 0.0, "strict_hits": 0.0})
                bucket["count"] += cs_metrics["count"]

    tol_kw = {"ltol": ltol, "atol_deg": atol_deg}
    strict = top1_elementwise_match_rate(preds, truths, **tol_kw)

    # Per-CS strict elementwise (recompute directly; evaluate_by_crystal_system
    # doesn't expose the strict-tolerance elementwise rate at arbitrary tol).
    cs_idx_cat = torch.cat(cs_idx_all) if cs_idx_all else torch.zeros(0, dtype=torch.long)
    per_cs_strict: dict[str, dict[str, float]] = {}
    from pxrd_cell_indexing.eval import lattice_match_elementwise

    for cs_value in range(len(CRYSTAL_SYSTEMS)):
        mask = (cs_idx_cat == cs_value).nonzero(as_tuple=True)[0].tolist()
        if not mask:
            continue
        hits = [
            lattice_match_elementwise(preds[i], truths[i], **tol_kw) for i in mask
        ]
        per_cs_strict[CRYSTAL_SYSTEMS[cs_value]] = {
            "count": len(mask),
            "strict_elementwise_rate": float(sum(hits) / len(hits)),
        }

    return {
        "n_samples": len(truths),
        "strict_raw_top1_elementwise_rate": strict,
        "lattice_mae": sums["lattice_mae"] / max(n_batches, 1),
        "length_mae": sums["length_mae"] / max(n_batches, 1),
        "angle_mae": sums["angle_mae"] / max(n_batches, 1),
        "per_crystal_system_strict_elementwise": per_cs_strict,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from pxrd_cell_indexing.training.checkpoint import apply_checkpoint_protocol_to_config

    config = apply_checkpoint_protocol_to_config(config, raw_ckpt)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    model.set_normalizer(normalizer)
    model.eval()

    base_lmdb = Path(config.data.valid_lmdb)
    base_jsonl = Path(config.data.valid_jsonl)

    results: dict[str, Any] = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "sets": {},
    }
    for name in NAMED_SETS:
        lmdb_path, jsonl_path = _dataset_paths(name, args.seed, base_lmdb, base_jsonl)
        if not Path(jsonl_path).exists():
            print(f"[skip] {name}: {jsonl_path} not found")
            continue
        metrics = evaluate_one_set(
            model, normalizer, lmdb_path, jsonl_path, config, device, args.ltol, args.atol_deg
        )
        results["sets"][name] = metrics
        print(
            f"[{name}] n={metrics['n_samples']} "
            f"strict={metrics['strict_raw_top1_elementwise_rate']:.4f} "
            f"angMAE={metrics['angle_mae']:.3f} lenMAE={metrics['length_mae']:.3f}"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Robust-valid frozen-set seed suffix")
    parser.add_argument("--ltol", type=float, default=0.05)
    parser.add_argument("--atol-deg", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
