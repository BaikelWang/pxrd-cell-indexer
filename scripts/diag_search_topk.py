#!/usr/bin/env python3
"""R6-B: single Bravais snap vs multi-seed manifold local search (strict elementwise).

Zero-training diagnosis on an existing indexing checkpoint.
Compares pool recall (and optional FOM Top-1) for:
  - single: current build_top_k_candidates (Bravais snap + scale variants)
  - search: build_manifold_search_candidates (NN seeds → manifold Q-match refine)
"""

from __future__ import annotations

import argparse
import json
import time
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
from pxrd_cell_indexing.model.refine import (
    RefineConfig,
    SearchConfig,
    build_manifold_search_candidates,
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


def _pool_hit(pool, truth, *, ltol: float, atol_deg: float) -> bool:
    t = np.asarray(truth, dtype=np.float64)
    return any(
        lattice_match_elementwise(
            np.asarray(_candidate_params(c), dtype=np.float64),
            t,
            ltol=ltol,
            atol_deg=atol_deg,
        )
        for c in pool
    )


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
    single_cfg = TopKConfig(
        k=args.top_k,
        length_scale_factors=scale_factors,
        max_log_volume_ratio_vs_base=vol_vs_base,
        include_axis_scale_variants=not args.no_axis_scale_variants,
        bravais_set=args.bravais_set,
    )
    search_cfg = SearchConfig(
        max_seeds=args.max_seeds,
        keep_unrefined=not args.no_keep_unrefined,
        bravais_set=args.bravais_set,
        k=args.top_k,
        max_log_volume_ratio_vs_base=vol_vs_base if vol_vs_base is not None else float(np.log(2.0)),
        refine=RefineConfig(
            max_steps=args.refine_steps,
            max_hkl_cap=args.refine_hkl_cap,
            n_lines=args.refine_n_lines,
            length_rel_bound=args.length_rel_bound,
            angle_abs_bound_deg=args.angle_abs_bound_deg,
        ),
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
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )

    modes = ["single", "search"] if args.mode == "both" else [args.mode]
    by_cs: dict[str, dict[str, dict[str, list[bool]]]] = {
        cs: {m: {"raw": [], "pool": [], "fom": []} for m in modes} for cs in CRYSTAL_SYSTEMS
    }
    overall: dict[str, dict[str, list[bool]]] = {m: {"raw": [], "pool": [], "fom": []} for m in modes}
    n_done = 0
    t0 = time.time()

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
            truth = batch_t["lattice"]
            bsz = pred.shape[0]
            if args.max_samples > 0:
                remain = args.max_samples - n_done
                if remain < bsz:
                    # Truncate last batch.
                    for key in list(batch_t.keys()):
                        if torch.is_tensor(batch_t[key]):
                            batch_t[key] = batch_t[key][:remain]
                    pred = pred[:remain]
                    truth = truth[:remain]
                    bsz = remain

            observed_batch = [
                slice_observed_two_theta(batch_t["pxrd_x"], batch_t["peak_num"], i)
                for i in range(bsz)
            ]

            pools_by_mode: dict[str, list] = {}
            if "single" in modes:
                pools_by_mode["single"] = build_top_k_candidates(
                    pred, k=args.top_k, config=single_cfg
                )
            if "search" in modes:
                pools_by_mode["search"] = build_manifold_search_candidates(
                    pred, observed_batch, config=search_cfg
                )

            for i in range(bsz):
                cs = CRYSTAL_SYSTEMS[int(batch_t["crystal_system_idx"][i].item())]
                t = truth[i].cpu().numpy()
                p = pred[i].cpu().numpy()
                raw_ok = bool(
                    lattice_match_elementwise(p, t, ltol=args.ltol, atol_deg=args.atol_deg)
                )
                ref_vol = lattice_params_volume(p)
                for mode, pools in pools_by_mode.items():
                    pool = pools[i]
                    pool_ok = _pool_hit(pool, t, ltol=args.ltol, atol_deg=args.atol_deg)
                    fom_ok = False
                    if args.with_fom and pool:
                        fom_cfg = FomRerankConfig(
                            mode="heuristic",
                            collapse_variants=args.fom_collapse_variants,
                            ref_volume=ref_vol if args.fom_use_ref_volume else None,
                            max_log_volume_ratio=(
                                args.pool_max_log_volume_ratio
                                if args.fom_use_ref_volume and args.pool_max_log_volume_ratio >= 0
                                else None
                            ),
                            volume_log_penalty=args.fom_volume_log_penalty,
                        )
                        ranked = rerank_candidates_by_fom(
                            pool, observed_batch[i], config=fom_cfg
                        )
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
                        by_cs[cs][mode][bucket].append(ok)
                        overall[mode][bucket].append(ok)
                n_done += 1

            elapsed = time.time() - t0
            print(f"... {n_done} samples in {elapsed:.1f}s ({elapsed/max(n_done,1):.2f}s/sample)", flush=True)
            if args.max_samples > 0 and n_done >= args.max_samples:
                break

    non_cubic = [cs for cs in CRYSTAL_SYSTEMS if cs != "cubic"]
    result: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k": args.top_k,
        "bravais_set": args.bravais_set,
        "modes": modes,
        "n": n_done,
        "elapsed_sec": time.time() - t0,
        "search_config": {
            "max_seeds": args.max_seeds,
            "refine_steps": args.refine_steps,
            "pool_max_log_volume_ratio": args.pool_max_log_volume_ratio,
        },
        "by_mode": {},
    }
    for mode in modes:
        o = overall[mode]
        result["by_mode"][mode] = {
            "overall": {
                "raw_elem": _rate(o["raw"]),
                "pool_recall": _rate(o["pool"]),
                "fom_top1_elem": _rate(o["fom"]) if args.with_fom else None,
            },
            "non_cubic": {
                "raw_elem": _rate(sum((by_cs[cs][mode]["raw"] for cs in non_cubic), [])),
                "pool_recall": _rate(sum((by_cs[cs][mode]["pool"] for cs in non_cubic), [])),
                "fom_top1_elem": (
                    _rate(sum((by_cs[cs][mode]["fom"] for cs in non_cubic), []))
                    if args.with_fom
                    else None
                ),
            },
            "by_crystal_system": {
                cs: {
                    "n": len(by_cs[cs][mode]["raw"]),
                    "raw_elem": _rate(by_cs[cs][mode]["raw"]),
                    "pool_recall": _rate(by_cs[cs][mode]["pool"]),
                    "fom_top1_elem": _rate(by_cs[cs][mode]["fom"]) if args.with_fom else None,
                }
                for cs in CRYSTAL_SYSTEMS
            },
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="R6-B single vs manifold-search Top-K diagnosis")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", choices=["both", "single", "search"], default="both")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-seeds", type=int, default=8)
    p.add_argument("--refine-steps", type=int, default=25)
    p.add_argument("--refine-hkl-cap", type=int, default=15)
    p.add_argument("--refine-n-lines", type=int, default=12)
    p.add_argument("--length-rel-bound", type=float, default=0.25)
    p.add_argument("--angle-abs-bound-deg", type=float, default=15.0)
    p.add_argument("--scale-set", type=str, default="default")
    p.add_argument("--pool-max-log-volume-ratio", type=float, default=0.693)
    p.add_argument("--no-axis-scale-variants", action="store_true")
    p.add_argument("--no-keep-unrefined", action="store_true")
    p.add_argument("--bravais-set", choices=["default", "extended"], default="extended")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full valid")
    p.add_argument("--with-fom", action="store_true")
    p.add_argument("--fom-use-ref-volume", action="store_true", default=True)
    p.add_argument("--no-fom-use-ref-volume", action="store_false", dest="fom_use_ref_volume")
    p.add_argument("--fom-collapse-variants", action="store_true")
    p.add_argument("--fom-volume-log-penalty", type=float, default=1.0)
    args = p.parse_args()

    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    for mode, block in result["by_mode"].items():
        o = block["overall"]
        n = block["non_cubic"]
        def _pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x*100:.2f}%"

        fom_s = f" fom={_pct(o['fom_top1_elem'])}" if args.with_fom else ""
        print(
            f"[{mode}] n={result['n']} raw={_pct(o['raw_elem'])} "
            f"pool={_pct(o['pool_recall'])}{fom_s} | "
            f"noncub pool={_pct(n['pool_recall'])}"
        )
        for cs, row in block["by_crystal_system"].items():
            print(
                f"  {cs:12s} n={row['n']:4d} raw={_pct(row['raw_elem']):>7s} "
                f"pool={_pct(row['pool_recall']):>7s}"
            )


if __name__ == "__main__":
    main()
