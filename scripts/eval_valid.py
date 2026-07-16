#!/usr/bin/env python3
"""Evaluate checkpoint with dual metrics: raw regression + Top-K rerank."""

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
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import (
    DEFAULT_VOLUME_LOG_RATIO_MAX,
    crystal_system_accuracy_from_lattice,
    evaluate_by_crystal_system,
    infer_crystal_system_idx_from_lattice,
    lattice_mae,
    length_mae,
    angle_mae,
    length_mape,
    mapping_vs_elementwise_gap_rate,
    top1_elementwise_match_rate,
    top1_joint_match_rate,
    top1_lattice_match_proxy,
    top1_lattice_match_rate,
    top1_volume_guarded_match_rate,
    topk_elementwise_match_rate,
    topk_lattice_match_rate,
    topk_mapping_vs_elementwise_gap_rate,
    topk_volume_guarded_match_rate,
)
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.model.fom_rerank import (
    add_fom_cli_args,
    fom_config_from_args,
    maybe_rerank_candidates,
)
from pxrd_cell_indexing.model.refine import RefineConfig, refine_candidates
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_top_k_candidates,
    parse_length_scale_factors,
)
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "experiments" / "smoke_unfrozen_seed42" / "checkpoints" / "best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "valid1400_real_match_smoke_unfrozen_seed42.json"


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def run(args: argparse.Namespace) -> dict[str, Any]:
    active_rerank_config = fom_config_from_args(args) if args.rerank == "fom" else None
    fom_eval_config = fom_config_from_args(args)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    # Load ckpt first so A0 protocol can override representation / stats / jsonl.
    import torch as _torch
    from pxrd_cell_indexing.training.checkpoint import (
        apply_checkpoint_protocol_to_config,
        infer_canonical_convention_from_checkpoint,
    )

    raw_ckpt = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = apply_checkpoint_protocol_to_config(config, raw_ckpt)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    model.set_normalizer(normalizer)
    _ = infer_canonical_convention_from_checkpoint(raw_ckpt)
    scale_factors = parse_length_scale_factors(args.scale_set)
    vol_vs_base = None if args.pool_max_log_volume_ratio < 0 else args.pool_max_log_volume_ratio
    topk_config = TopKConfig(
        k=args.top_k,
        length_scale_factors=scale_factors,
        max_log_volume_ratio_vs_base=vol_vs_base,
        include_axis_scale_variants=not args.no_axis_scale_variants,
        bravais_set=args.bravais_set,
    )
    refine_config = None
    if args.refine_steps > 0:
        refine_config = RefineConfig(
            max_steps=args.refine_steps,
            top_n=args.refine_top_n,
            length_rel_bound=args.refine_length_rel_bound,
            angle_abs_bound_deg=args.refine_angle_bound,
            max_log_volume_ratio=args.refine_max_log_volume_ratio,
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

    raw_preds: list[list[float]] = []
    rerank_preds: list[list[float]] = []
    fom_preds: list[list[float]] = []
    truths: list[list[float]] = []
    pred_cs_idx: list[int] = []
    target_cs_idx: list[int] = []
    candidate_lists = []

    metric_sums = {
        "crystal_system_accuracy": 0.0,
        "lattice_mae": 0.0,
        "length_mae": 0.0,
        "angle_mae": 0.0,
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
            pred_cs = infer_crystal_system_idx_from_lattice(pred.cpu())

            batch_candidates = build_top_k_candidates(
                pred,
                k=args.top_k,
                config=topk_config,
            )

            for idx in range(pred.shape[0]):
                truth = target[idx].cpu().tolist()
                truths.append(truth)
                raw_preds.append(pred[idx].cpu().tolist())
                pred_cs_idx.append(int(pred_cs[idx]))
                target_cs_idx.append(int(batch["crystal_system_idx"][idx].item()))

                pool = batch_candidates[idx]
                if refine_config is not None and refine_config.max_steps > 0:
                    observed = slice_observed_two_theta(
                        batch["pxrd_x"], batch["peak_num"], idx
                    )
                    pool = refine_candidates(pool, observed, config=refine_config)

                fom_reranked = maybe_rerank_candidates(
                    pool,
                    rerank="fom",
                    pxrd_x=batch["pxrd_x"],
                    pxrd_y=batch["pxrd_y"],
                    peak_num=batch["peak_num"],
                    sample_index=idx,
                    fom_config=fom_eval_config,
                    ref_lattice_params=pred[idx].cpu().tolist(),
                )
                fom_preds.append(_candidate_params(fom_reranked[0]))

                sample_candidates = maybe_rerank_candidates(
                    pool,
                    rerank=args.rerank,
                    pxrd_x=batch["pxrd_x"],
                    pxrd_y=batch["pxrd_y"],
                    peak_num=batch["peak_num"],
                    sample_index=idx,
                    fom_config=active_rerank_config,
                    ref_lattice_params=pred[idx].cpu().tolist(),
                )
                candidate_lists.append(sample_candidates)
                rerank_preds.append(_candidate_params(sample_candidates[0]))

            metric_sums["crystal_system_accuracy"] += crystal_system_accuracy_from_lattice(
                pred, batch["crystal_system_idx"]
            )
            metric_sums["lattice_mae"] += lattice_mae(pred, target)
            metric_sums["length_mae"] += length_mae(pred, target)
            metric_sums["angle_mae"] += angle_mae(pred, target)
            metric_sums["length_mape"] += length_mape(pred, target)
            metric_sums["top1_lattice_match_proxy"] += top1_lattice_match_proxy(
                pred, target, ltol=args.ltol, atol_deg=args.atol_deg
            )
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

    tol_kw = {"ltol": args.ltol, "atol_deg": args.atol_deg}
    vol_kw = {**tol_kw, "max_log_volume_ratio": args.max_log_volume_ratio}
    raw_top1 = top1_lattice_match_rate(raw_preds, truths, **tol_kw)
    rerank_top1 = top1_lattice_match_rate(rerank_preds, truths, **tol_kw)
    fom_top1 = top1_lattice_match_rate(fom_preds, truths, **tol_kw)

    joint_match = top1_joint_match_rate(
        rerank_preds,
        truths,
        pred_cs_idx,
        target_cs_idx,
        **tol_kw,
    )

    result = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "valid_jsonl": config.data.valid_jsonl,
        "n_samples": len(truths),
        "top_k": args.top_k,
        "scale_set": args.scale_set,
        "length_scale_factors": list(scale_factors),
        "pool_max_log_volume_ratio": vol_vs_base,
        "include_axis_scale_variants": not args.no_axis_scale_variants,
        "bravais_set": args.bravais_set,
        "refine_steps": args.refine_steps,
        "refine_top_n": args.refine_top_n if args.refine_steps > 0 else None,
        "rerank": args.rerank,
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "max_log_volume_ratio": args.max_log_volume_ratio,
        "fom_mode": args.fom_mode if args.rerank == "fom" else None,
        "fom_collapse_variants": args.fom_collapse_variants if args.rerank == "fom" else None,
        "fom_q_abs_tol": args.fom_q_abs_tol if args.rerank == "fom" else None,
        "metrics": {
            **averaged,
            "raw_top1_lattice_match_rate": raw_top1,
            "raw_top1_elementwise_rate": top1_elementwise_match_rate(
                raw_preds, truths, **tol_kw
            ),
            "raw_top1_volume_guarded_rate": top1_volume_guarded_match_rate(
                raw_preds, truths, **vol_kw
            ),
            "top1_lattice_match_rate": rerank_top1,
            "fom_top1_lattice_match_rate": fom_top1,
            "fom_top1_elementwise_rate": top1_elementwise_match_rate(
                fom_preds, truths, **tol_kw
            ),
            "fom_top1_volume_guarded_rate": top1_volume_guarded_match_rate(
                fom_preds, truths, **vol_kw
            ),
            "top1_joint_match_rate": joint_match,
            "topk_lattice_match_rate": topk_lattice_match_rate(
                candidate_lists, truths, **tol_kw
            ),
            "topk_elementwise_rate": topk_elementwise_match_rate(
                candidate_lists, truths, **tol_kw
            ),
            "topk_volume_guarded_rate": topk_volume_guarded_match_rate(
                candidate_lists, truths, **vol_kw
            ),
            "raw_mapping_vs_elementwise_gap_rate": mapping_vs_elementwise_gap_rate(
                raw_preds, truths, **tol_kw
            ),
            "topk_mapping_vs_elementwise_gap_rate": topk_mapping_vs_elementwise_gap_rate(
                candidate_lists, truths, **tol_kw
            ),
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
    parser.add_argument(
        "--scale-set",
        type=str,
        default="default",
        help="Top-K length scale set: none|default|extended|comma floats (A2)",
    )
    parser.add_argument(
        "--pool-max-log-volume-ratio",
        type=float,
        default=-1.0,
        help="Filter pool vs raw pred volume; <0 disables (default). e.g. 0.693=log(2)",
    )
    parser.add_argument(
        "--no-axis-scale-variants",
        action="store_true",
        help="Disable single-axis ×2/×0.5 variants in Top-K pool",
    )
    parser.add_argument(
        "--bravais-set",
        type=str,
        choices=("default", "extended"),
        default="default",
        help="A4: default=Decision A 8 hyps; extended=+mono+hex_strict",
    )
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=0,
        help="A5: L-BFGS-B maxiter per candidate (0=off)",
    )
    parser.add_argument(
        "--refine-top-n",
        type=int,
        default=10,
        help="A5: refine only first N pool candidates",
    )
    parser.add_argument(
        "--refine-length-rel-bound",
        type=float,
        default=0.25,
        help="A5: relative ± bound on a,b,c during refine",
    )
    parser.add_argument(
        "--refine-angle-bound",
        type=float,
        default=15.0,
        help="A5: absolute ± angle bound (deg) during refine",
    )
    parser.add_argument(
        "--refine-max-log-volume-ratio",
        type=float,
        default=float(__import__("math").log(2.0)),
        help="A5: reject refine if |log(V/V_seed)| exceeds this",
    )
    parser.add_argument(
        "--rerank",
        type=str,
        choices=("none", "fom"),
        default="fom",
        help="Candidate reranking after Top-K pool generation (default fom; use none for ablation)",
    )
    parser.add_argument(
        "--ltol",
        type=float,
        default=0.3,
        help="Relative length tolerance for lattice match (default 0.3)",
    )
    parser.add_argument(
        "--atol-deg",
        type=float,
        default=10.0,
        help="Absolute angle tolerance in degrees (default 10)",
    )
    parser.add_argument(
        "--max-log-volume-ratio",
        type=float,
        default=DEFAULT_VOLUME_LOG_RATIO_MAX,
        help="Volume guard: |log(V_pred/V_truth)| max (default log(2)≈0.693)",
    )
    add_fom_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
