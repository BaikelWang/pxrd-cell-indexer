#!/usr/bin/env python3
"""Grid-search FOM rerank configs on valid1400 (no retraining)."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.dataset import (
    PXRDDatasetConfig,
    PeakFilterConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import lattice_match_pymatgen
from pxrd_cell_indexing.model.fom import FomRerankConfig
from pxrd_cell_indexing.model.fom_rerank import maybe_rerank_candidates
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "scale_100k_no_cs_matrix6_seed42"
    / "checkpoints"
    / "best.pt"
)


def _params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def evaluate_config(
    *,
    config: TrainConfig,
    checkpoint_path: Path,
    device: torch.device,
    fom_config: FomRerankConfig,
    top_k: int = 20,
) -> dict[str, Any]:
    model, _, _ = load_indexing_model_from_checkpoint(checkpoint_path, config, device)
    normalizer = build_lattice_normalizer(config.data)
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
    in_pool = 0
    top1 = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
            pred = normalizer.denormalize(outputs["lattice_norm"])
            cands_batch = build_top_k_candidates(
                pred,
                k=top_k,
                config=TopKConfig(k=top_k),
            )
            for idx in range(pred.shape[0]):
                n += 1
                truth = batch["lattice"][idx].cpu().tolist()
                cands = cands_batch[idx]
                hits = [lattice_match_pymatgen(_params(c), truth) for c in cands]
                if not any(hits):
                    continue
                in_pool += 1
                reranked = maybe_rerank_candidates(
                    cands,
                    rerank="fom",
                    pxrd_x=batch["pxrd_x"],
                    pxrd_y=batch["pxrd_y"],
                    peak_num=batch["peak_num"],
                    sample_index=idx,
                    fom_config=fom_config,
                )
                if lattice_match_pymatgen(_params(reranked[0]), truth):
                    top1 += 1

    return {
        "n_samples": n,
        "lattice_in_pool": in_pool,
        "lattice_in_pool_rate": in_pool / max(n, 1),
        "top1_lattice_match_rate": top1 / max(n, 1),
        "top1_among_in_pool": top1 / max(in_pool, 1),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    train_config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)

    grid: list[FomRerankConfig] = []
    for mode, collapse, abs_tol in itertools.product(
        args.modes,
        args.collapse_variants,
        args.q_abs_tols,
    ):
        grid.append(
            FomRerankConfig(
                mode=mode,  # type: ignore[arg-type]
                collapse_variants=collapse,
                q_match_abs_tol=abs_tol,
            )
        )

    results: list[dict[str, Any]] = []
    for fom_config in grid:
        metrics = evaluate_config(
            config=train_config,
            checkpoint_path=args.checkpoint,
            device=device,
            fom_config=fom_config,
            top_k=args.top_k,
        )
        entry = {
            "mode": fom_config.mode,
            "collapse_variants": fom_config.collapse_variants,
            "q_match_abs_tol": fom_config.q_match_abs_tol,
            **metrics,
        }
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    results.sort(key=lambda item: item["top1_lattice_match_rate"], reverse=True)
    best = results[0]
    output = {"best": best, "grid": results}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print("BEST:", json.dumps(best, ensure_ascii=False))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FOM rerank configs on valid1400")
    parser.add_argument("--config", type=Path, default="configs/scale_100k_no_cs_matrix6.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "fom_rerank_sweep_valid1400.json",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["heuristic", "strict_dewolff", "intensity_weighted"],
        choices=["heuristic", "strict_dewolff", "intensity_weighted"],
    )
    parser.add_argument(
        "--collapse-variants",
        nargs="+",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=[False, True],
    )
    parser.add_argument(
        "--q-abs-tols",
        nargs="+",
        type=float,
        default=[1e-4, 1e-5, 1e-6],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
