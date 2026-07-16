#!/usr/bin/env python3
"""R6-C: volume-guard + FOM ref-volume scan (strict elementwise).

Zero-training diagnosis. Scans combinations of:
  - pool scale variants on/off
  - pool max_log_volume_ratio
  - FOM collapse_variants
  - FOM ref_volume preference (vs legacy smaller-volume bias)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.model.fom import (
    FomRerankConfig,
    rerank_candidates_by_fom,
    slice_observed_two_theta,
)
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_top_k_candidates,
    lattice_params_volume,
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


@dataclass(frozen=True)
class GuardVariant:
    name: str
    scale_set: str
    no_axis: bool
    pool_max_log_volume_ratio: float | None
    fom_ref_volume: bool
    fom_collapse: bool
    fom_max_log_volume_ratio: float | None
    volume_log_penalty: float = 1.0


DEFAULT_VARIANTS: tuple[GuardVariant, ...] = (
    GuardVariant("baseline_legacy", "default", False, None, False, False, None),
    GuardVariant("noscale", "none", True, None, False, False, None),
    GuardVariant("vol_guard_log2", "default", False, math.log(2.0), False, False, None),
    GuardVariant("noscale_vol_guard", "none", True, math.log(2.0), False, False, None),
    GuardVariant("fom_refvol", "default", False, None, True, False, math.log(2.0)),
    GuardVariant("noscale_fom_refvol", "none", True, math.log(2.0), True, True, math.log(2.0)),
    GuardVariant(
        "noscale_fom_refvol_pen2",
        "none",
        True,
        math.log(2.0),
        True,
        True,
        math.log(2.0),
        volume_log_penalty=2.0,
    ),
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    model.set_normalizer(normalizer)

    variants = list(DEFAULT_VARIANTS)
    if args.only:
        wanted = set(args.only.split(","))
        variants = [v for v in variants if v.name in wanted]

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

    # Precompute predictions once; apply pool/FOM variants offline.
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    cs_list: list[str] = []
    observed: list[np.ndarray] = []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if args.max_samples > 0 and n_done >= args.max_samples:
                break
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
            bsz = pred.shape[0]
            if args.max_samples > 0:
                remain = args.max_samples - n_done
                if remain < bsz:
                    for key in list(batch_t.keys()):
                        if torch.is_tensor(batch_t[key]):
                            batch_t[key] = batch_t[key][:remain]
                    pred = pred[:remain]
                    bsz = remain
            for i in range(bsz):
                preds.append(pred[i].cpu().numpy())
                truths.append(batch_t["lattice"][i].cpu().numpy())
                cs_list.append(CRYSTAL_SYSTEMS[int(batch_t["crystal_system_idx"][i].item())])
                observed.append(
                    np.asarray(
                        slice_observed_two_theta(batch_t["pxrd_x"], batch_t["peak_num"], i),
                        dtype=np.float64,
                    )
                )
                n_done += 1

    pred_t = torch.tensor(np.stack(preds), dtype=torch.float64)
    results: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "n": n_done,
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "variants": {},
    }

    for variant in variants:
        topk_cfg = TopKConfig(
            k=args.top_k,
            length_scale_factors=parse_length_scale_factors(variant.scale_set),
            max_log_volume_ratio_vs_base=variant.pool_max_log_volume_ratio,
            include_axis_scale_variants=not variant.no_axis,
            bravais_set=args.bravais_set,
        )
        pools = build_top_k_candidates(pred_t, k=args.top_k, config=topk_cfg)
        overall = {"raw": [], "pool": [], "fom": []}
        by_cs: dict[str, dict[str, list[bool]]] = {
            cs: {"raw": [], "pool": [], "fom": []} for cs in CRYSTAL_SYSTEMS
        }
        for i in range(n_done):
            t = truths[i]
            p = preds[i]
            cs = cs_list[i]
            raw_ok = bool(lattice_match_elementwise(p, t, ltol=args.ltol, atol_deg=args.atol_deg))
            pool = pools[i]
            pool_ok = any(
                lattice_match_elementwise(
                    np.asarray(_candidate_params(c), dtype=np.float64),
                    t,
                    ltol=args.ltol,
                    atol_deg=args.atol_deg,
                )
                for c in pool
            )
            ref_vol = lattice_params_volume(p)
            fom_cfg = FomRerankConfig(
                mode="heuristic",
                collapse_variants=variant.fom_collapse,
                ref_volume=ref_vol if variant.fom_ref_volume else None,
                max_log_volume_ratio=(
                    variant.fom_max_log_volume_ratio if variant.fom_ref_volume else None
                ),
                volume_log_penalty=variant.volume_log_penalty,
            )
            ranked = rerank_candidates_by_fom(pool, observed[i], config=fom_cfg) if pool else []
            fom_ok = False
            if ranked:
                fom_ok = bool(
                    lattice_match_elementwise(
                        np.asarray(_candidate_params(ranked[0]), dtype=np.float64),
                        t,
                        ltol=args.ltol,
                        atol_deg=args.atol_deg,
                    )
                )
            for bucket, ok in (("raw", raw_ok), ("pool", pool_ok), ("fom", fom_ok)):
                overall[bucket].append(ok)
                by_cs[cs][bucket].append(ok)

        non_cubic = [cs for cs in CRYSTAL_SYSTEMS if cs != "cubic"]
        block = {
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
            "settings": {
                "scale_set": variant.scale_set,
                "no_axis": variant.no_axis,
                "pool_max_log_volume_ratio": variant.pool_max_log_volume_ratio,
                "fom_ref_volume": variant.fom_ref_volume,
                "fom_collapse": variant.fom_collapse,
                "fom_max_log_volume_ratio": variant.fom_max_log_volume_ratio,
                "volume_log_penalty": variant.volume_log_penalty,
            },
        }
        results["variants"][variant.name] = block
        o = block["overall"]
        print(
            f"[{variant.name}] raw={o['raw_elem']*100:.2f}% "
            f"pool={o['pool_recall']*100:.2f}% fom={o['fom_top1_elem']*100:.2f}%",
            flush=True,
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="R6-C FOM / volume-guard scan")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--bravais-set", choices=["default", "extended"], default="default")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--only", type=str, default="", help="Comma-separated variant names")
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
