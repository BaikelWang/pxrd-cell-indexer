#!/usr/bin/env python3
"""Decompose valid1400 predictions into joint-solution funnel metrics."""

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
from pxrd_cell_indexing.eval import lattice_match_pymatgen
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_joint_candidate(cand: Any, truth_lat: list[float], truth_cs_idx: int) -> bool:
    cs_ok = CRYSTAL_SYSTEM_TO_IDX[cand.crystal_system] == truth_cs_idx
    lat = [cand.a, cand.b, cand.c, cand.alpha, cand.beta, cand.gamma]
    return cs_ok and lattice_match_pymatgen(lat, truth_lat)


def decompose(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    top_k: int = 20,
) -> dict[str, Any]:
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        checkpoint_path, config, device
    )
    normalizer = LatticeNormalizer.from_json(config.data.lattice_stats)
    loader = build_dataloader(
        PXRDDatasetConfig(
            lmdb_path=Path(config.data.valid_lmdb),
            split="valid",
            sample_list_path=Path(config.data.valid_jsonl),
            peak_filter=PeakFilterConfig(),
            xrd_augment=False,
            strict=False,
            seed_base=config.seed,
        ),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    n = 0
    joint_top1 = 0
    joint_in_pool = 0
    joint_in_pool_not_top1 = 0
    lattice_top1 = 0
    lattice_in_pool = 0
    cs_top1 = 0
    fail_no_joint_in_pool = 0
    fail_cs_ok_lattice_bad_top1 = 0
    fail_cs_wrong_lattice_ok_top1 = 0
    fail_both_bad_top1 = 0
    rank_when_in_pool: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
            pred = normalizer.denormalize(outputs["lattice_norm"])
            cands_batch = build_top_k_candidates(
                outputs["crystal_system_logits"],
                pred,
                k=top_k,
                config=TopKConfig(k=top_k),
            )

            for idx in range(pred.shape[0]):
                truth_lat = batch["lattice"][idx].cpu().tolist()
                truth_cs = int(batch["crystal_system_idx"][idx].item())
                cands = cands_batch[idx]
                n += 1

                top1 = cands[0]
                top1_lat = [top1.a, top1.b, top1.c, top1.alpha, top1.beta, top1.gamma]
                t1_lat = lattice_match_pymatgen(top1_lat, truth_lat)
                t1_cs = CRYSTAL_SYSTEM_TO_IDX[top1.crystal_system] == truth_cs

                if t1_cs:
                    cs_top1 += 1
                if t1_lat:
                    lattice_top1 += 1
                if t1_cs and t1_lat:
                    joint_top1 += 1
                else:
                    if t1_cs and not t1_lat:
                        fail_cs_ok_lattice_bad_top1 += 1
                    elif (not t1_cs) and t1_lat:
                        fail_cs_wrong_lattice_ok_top1 += 1
                    else:
                        fail_both_bad_top1 += 1

                pool_lat = False
                pool_joint = False
                joint_rank = None
                for rank, cand in enumerate(cands, start=1):
                    lat = [cand.a, cand.b, cand.c, cand.alpha, cand.beta, cand.gamma]
                    if lattice_match_pymatgen(lat, truth_lat):
                        pool_lat = True
                    if _is_joint_candidate(cand, truth_lat, truth_cs):
                        pool_joint = True
                        if joint_rank is None:
                            joint_rank = rank

                if pool_lat:
                    lattice_in_pool += 1
                if pool_joint:
                    joint_in_pool += 1
                    rank_when_in_pool.append(joint_rank)
                    if joint_rank != 1:
                        joint_in_pool_not_top1 += 1
                else:
                    fail_no_joint_in_pool += 1

    def rate(count: int) -> float:
        return count / max(n, 1)

    rank_dist: dict[str, float] = {}
    if rank_when_in_pool:
        from collections import Counter

        counter = Counter(rank_when_in_pool)
        for rank in sorted(counter):
            rank_dist[f"rank_{rank}"] = counter[rank] / len(rank_when_in_pool)

    return {
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path),
        "n_samples": n,
        "top_k": top_k,
        "rates": {
            "crystal_system_top1": rate(cs_top1),
            "lattice_top1": rate(lattice_top1),
            "joint_top1": rate(joint_top1),
            "lattice_in_pool": rate(lattice_in_pool),
            "joint_in_pool": rate(joint_in_pool),
            "joint_in_pool_not_top1": rate(joint_in_pool_not_top1),
            "joint_not_in_pool": rate(fail_no_joint_in_pool),
            "fail_cs_ok_lattice_bad_top1": rate(fail_cs_ok_lattice_bad_top1),
            "fail_cs_wrong_lattice_ok_top1": rate(fail_cs_wrong_lattice_ok_top1),
            "fail_both_bad_top1": rate(fail_both_bad_top1),
        },
        "counts": {
            "crystal_system_top1": cs_top1,
            "lattice_top1": lattice_top1,
            "joint_top1": joint_top1,
            "lattice_in_pool": lattice_in_pool,
            "joint_in_pool": joint_in_pool,
            "joint_in_pool_not_top1": joint_in_pool_not_top1,
            "joint_not_in_pool": fail_no_joint_in_pool,
            "fail_cs_ok_lattice_bad_top1": fail_cs_ok_lattice_bad_top1,
            "fail_cs_wrong_lattice_ok_top1": fail_cs_wrong_lattice_ok_top1,
            "fail_both_bad_top1": fail_both_bad_top1,
        },
        "joint_rank_when_in_pool": rank_dist,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    result = decompose(args.config, args.checkpoint, device=device, top_k=args.top_k)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result["rates"], indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint-solution funnel decomposition")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
