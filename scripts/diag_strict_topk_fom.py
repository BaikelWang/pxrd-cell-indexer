#!/usr/bin/env python3
"""R5-Diag: strict-tolerance Top-K pool recall + FOM Top-1, by crystal system.

Compares three rates on the same valid split:
  - raw Top-1 elementwise (ltol/atol)
  - Top-K pool recall (any candidate in the Bravais/scale pool matches)
  - FOM-reranked Top-1 elementwise

Does not retrain; consumes an existing indexing checkpoint.
"""

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
    lattice_match_elementwise,
    top1_elementwise_match_rate,
    topk_elementwise_match_rate,
)
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.model.fom_rerank import (
    add_fom_cli_args,
    fom_config_from_args,
    maybe_rerank_candidates,
)
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_top_k_candidates,
    parse_length_scale_factors,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def _rate(oks: list[bool]) -> float | None:
    if not oks:
        return None
    return float(np.mean(oks))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    model.set_normalizer(normalizer)

    scale_factors = parse_length_scale_factors(args.scale_set)
    vol_vs_base = None if args.pool_max_log_volume_ratio < 0 else args.pool_max_log_volume_ratio
    topk_config = TopKConfig(
        k=args.top_k,
        length_scale_factors=scale_factors,
        max_log_volume_ratio_vs_base=vol_vs_base,
        include_axis_scale_variants=not args.no_axis_scale_variants,
        bravais_set=args.bravais_set,
    )
    fom_config = fom_config_from_args(args)

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
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )

    by_cs: dict[str, dict[str, list[bool]]] = {
        cs: {"raw": [], "pool": [], "fom": []} for cs in CRYSTAL_SYSTEMS
    }
    overall = {"raw": [], "pool": [], "fom": []}
    angle_maes: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch_t = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            outputs = model(
                batch_t["pxrd_x"],
                batch_t["pxrd_y"],
                batch_t["peak_num"],
                crystal_system_idx=batch_t["crystal_system_idx"],
                cs_route=config.model.cs_route,
                lattice_phys=batch_t["lattice"],
                setting_route=config.model.setting_route,
            )
            pred = normalizer.denormalize(outputs["lattice_norm"])
            truth = batch_t["lattice"]
            pools = build_top_k_candidates(pred, k=args.top_k, config=topk_config)

            for i in range(pred.shape[0]):
                cs = CRYSTAL_SYSTEMS[int(batch_t["crystal_system_idx"][i].item())]
                t = truth[i].cpu().numpy()
                p = pred[i].cpu().numpy()
                angle_maes.append(float(np.abs(p[3:] - t[3:]).mean()))

                raw_ok = bool(
                    lattice_match_elementwise(p, t, ltol=args.ltol, atol_deg=args.atol_deg)
                )
                pool_params = [np.asarray(_candidate_params(c), dtype=np.float64) for c in pools[i]]
                pool_ok = any(
                    lattice_match_elementwise(c, t, ltol=args.ltol, atol_deg=args.atol_deg)
                    for c in pool_params
                )
                fom_pool = maybe_rerank_candidates(
                    pools[i],
                    rerank="fom",
                    pxrd_x=batch_t["pxrd_x"],
                    pxrd_y=batch_t["pxrd_y"],
                    peak_num=batch_t["peak_num"],
                    sample_index=i,
                    fom_config=fom_config,
                )
                fom_top = np.asarray(_candidate_params(fom_pool[0]), dtype=np.float64)
                fom_ok = bool(
                    lattice_match_elementwise(
                        fom_top, t, ltol=args.ltol, atol_deg=args.atol_deg
                    )
                )

                for bucket, ok in (("raw", raw_ok), ("pool", pool_ok), ("fom", fom_ok)):
                    by_cs[cs][bucket].append(ok)
                    overall[bucket].append(ok)

    non_cubic = [cs for cs in CRYSTAL_SYSTEMS if cs != "cubic"]
    result = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k": args.top_k,
        "bravais_set": args.bravais_set,
        "n": len(overall["raw"]),
        "angle_mae_raw": float(np.mean(angle_maes)) if angle_maes else None,
        "overall": {
            "raw_elem": _rate(overall["raw"]),
            "pool_recall": _rate(overall["pool"]),
            "fom_top1_elem": _rate(overall["fom"]),
        },
        "non_cubic": {
            "raw_elem": _rate(sum((by_cs[cs]["raw"] for cs in non_cubic), [])),
            "pool_recall": _rate(sum((by_cs[cs]["pool"] for cs in non_cubic), [])),
            "fom_top1_elem": _rate(sum((by_cs[cs]["fom"] for cs in non_cubic), [])),
        },
        "by_crystal_system": {
            cs: {
                "n": len(by_cs[cs]["raw"]),
                "raw_elem": _rate(by_cs[cs]["raw"]),
                "pool_recall": _rate(by_cs[cs]["pool"]),
                "fom_top1_elem": _rate(by_cs[cs]["fom"]),
            }
            for cs in CRYSTAL_SYSTEMS
        },
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Strict Top-K/FOM diagnosis by crystal system")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--scale-set", type=str, default="default")
    p.add_argument("--pool-max-log-volume-ratio", type=float, default=-1.0)
    p.add_argument("--no-axis-scale-variants", action="store_true")
    p.add_argument("--bravais-set", choices=["default", "extended"], default="default")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    add_fom_cli_args(p)
    # Force FOM defaults for diagnosis.
    args = p.parse_args()
    if not hasattr(args, "fom_mode") or args.fom_mode is None:
        args.fom_mode = "heuristic"
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    o = result["overall"]
    n = result["non_cubic"]
    print(
        f"overall raw={o['raw_elem']*100:.2f}% pool={o['pool_recall']*100:.2f}% "
        f"fom={o['fom_top1_elem']*100:.2f}%"
    )
    print(
        f"noncub raw={n['raw_elem']*100:.2f}% pool={n['pool_recall']*100:.2f}% "
        f"fom={n['fom_top1_elem']*100:.2f}%"
    )
    for cs, row in result["by_crystal_system"].items():
        print(
            f"  {cs:12s} n={row['n']:4d} raw={row['raw_elem']*100:5.1f}% "
            f"pool={row['pool_recall']*100:5.1f}% fom={row['fom_top1_elem']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
