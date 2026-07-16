#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the MP100 benchmark (dual metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import (
    LatticeNormalizer,
    MatrixLatticeNormalizer,
    build_lattice_normalizer,
)
from pxrd_cell_indexing.eval import (
    DEFAULT_VOLUME_LOG_RATIO_MAX,
    infer_crystal_system_idx_from_lattice,
    mapping_vs_elementwise_gap_rate,
    top1_elementwise_match_rate,
    top1_joint_match_rate,
    top1_lattice_match_rate,
    top1_volume_guarded_match_rate,
    topk_elementwise_match_rate,
    topk_lattice_match_rate,
    topk_mapping_vs_elementwise_gap_rate,
    topk_volume_guarded_match_rate,
)
from pxrd_cell_indexing.model.fom_rerank import (
    add_fom_cli_args,
    fom_config_from_args,
    maybe_rerank_candidates,
)
from pxrd_cell_indexing.model.heads import IndexingModel
from pxrd_cell_indexing.model.refine import RefineConfig, refine_candidates
from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_top_k_candidates,
    parse_length_scale_factors,
)
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "scale_100k_no_cs_matrix6.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "scale_100k_no_cs_matrix6_seed42"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "mp100_eval_scale_100k_no_cs_matrix6_seed42.json"

# Reference baselines (ideal peaks). Loose = historical Mc/JADE compete口径;
# strict = user-provided numbers under ltol=0.05 / atol=3°.
REFERENCE_BASELINES = {
    "mcmaille_top1_lattice_match_ltol0.3_atol10": 0.764,
    "jade9_top1_lattice_match_ltol0.3_atol10": 0.725,
    "mcmaille_top1_lattice_match_ltol0.05_atol3": 0.659,
    "jade9_top1_lattice_match_ltol0.05_atol3": 0.681,
    "realpxrd_without_l_top1_lattice_match": 0.05,
    # Backward-compatible aliases (loose口径).
    "mcmaille_top1_lattice_match": 0.764,
    "jade9_top1_lattice_match": 0.725,
}


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def run_inference_on_mp100(
    model: IndexingModel,
    normalizer: LatticeNormalizer | MatrixLatticeNormalizer,
    samples: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    top_k: int,
    rerank: str,
    fom_config,
    ltol: float,
    atol_deg: float,
    max_log_volume_ratio: float,
    topk_config: TopKConfig,
    refine_config: RefineConfig | None,
) -> dict[str, Any]:
    raw_preds: list[list[float]] = []
    rerank_preds: list[list[float]] = []
    fom_preds: list[list[float]] = []
    truths: list[list[float]] = []
    pred_cs_idx: list[int] = []
    target_cs_idx: list[int] = []
    candidate_lists = []
    per_sample: list[dict[str, Any]] = []
    active_rerank_config = fom_config if rerank == "fom" else None
    vol_kw = {"max_log_volume_ratio": max_log_volume_ratio}

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
            pred_cs = infer_crystal_system_idx_from_lattice(lattice.cpu())
            batch_candidates = build_top_k_candidates(
                lattice,
                k=top_k,
                config=topk_config,
            )

        for idx, sample_id in enumerate(batch_ids):
            truth = batch_truths[idx]
            raw_pred = lattice[idx].cpu().tolist()
            raw_preds.append(raw_pred)
            truths.append(truth)
            pred_cs_idx.append(int(pred_cs[idx]))
            target_cs_idx.append(batch_target_cs[idx])

            pool = batch_candidates[idx]
            if refine_config is not None and refine_config.max_steps > 0:
                observed = slice_observed_two_theta(pxrd_x, peak_num, idx)
                pool = refine_candidates(pool, observed, config=refine_config)

            fom_reranked = maybe_rerank_candidates(
                pool,
                rerank="fom",
                pxrd_x=pxrd_x,
                pxrd_y=pxrd_y,
                peak_num=peak_num,
                sample_index=idx,
                fom_config=fom_config,
            )
            fom_top1 = _candidate_params(fom_reranked[0])
            fom_preds.append(fom_top1)

            sample_candidates = maybe_rerank_candidates(
                pool,
                rerank=rerank,
                pxrd_x=pxrd_x,
                pxrd_y=pxrd_y,
                peak_num=peak_num,
                sample_index=idx,
                fom_config=active_rerank_config,
            )
            candidate_lists.append(sample_candidates)
            rerank_top1 = _candidate_params(sample_candidates[0])
            rerank_preds.append(rerank_top1)

            per_sample.append(
                {
                    "sample_id": sample_id,
                    "truth_lattice": truth,
                    "raw_pred": raw_pred,
                    "top1_pred": rerank_top1,
                    "fom_top1_pred": fom_top1,
                    "top1_crystal_system": sample_candidates[0].crystal_system,
                    "top1_confidence": sample_candidates[0].confidence,
                    "topk_size": len(sample_candidates),
                }
            )

    return {
        "n_samples": len(samples),
        "raw_top1_lattice_match_rate": top1_lattice_match_rate(
            raw_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "raw_top1_elementwise_rate": top1_elementwise_match_rate(
            raw_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "raw_top1_volume_guarded_rate": top1_volume_guarded_match_rate(
            raw_preds, truths, ltol=ltol, atol_deg=atol_deg, **vol_kw
        ),
        "top1_lattice_match_rate": top1_lattice_match_rate(
            rerank_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "fom_top1_lattice_match_rate": top1_lattice_match_rate(
            fom_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "fom_top1_elementwise_rate": top1_elementwise_match_rate(
            fom_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "fom_top1_volume_guarded_rate": top1_volume_guarded_match_rate(
            fom_preds, truths, ltol=ltol, atol_deg=atol_deg, **vol_kw
        ),
        "top1_joint_match_rate": top1_joint_match_rate(
            rerank_preds, truths, pred_cs_idx, target_cs_idx, ltol=ltol, atol_deg=atol_deg
        ),
        "topk_lattice_match_rate": topk_lattice_match_rate(
            candidate_lists, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "topk_elementwise_rate": topk_elementwise_match_rate(
            candidate_lists, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "topk_volume_guarded_rate": topk_volume_guarded_match_rate(
            candidate_lists, truths, ltol=ltol, atol_deg=atol_deg, **vol_kw
        ),
        "raw_mapping_vs_elementwise_gap_rate": mapping_vs_elementwise_gap_rate(
            raw_preds, truths, ltol=ltol, atol_deg=atol_deg
        ),
        "topk_mapping_vs_elementwise_gap_rate": topk_mapping_vs_elementwise_gap_rate(
            candidate_lists, truths, ltol=ltol, atol_deg=atol_deg
        ),
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
    from pxrd_cell_indexing.training.checkpoint import (
        apply_checkpoint_protocol_to_config,
        infer_canonical_convention_from_checkpoint,
    )

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = apply_checkpoint_protocol_to_config(config, raw_ckpt)
    convention = args.convention or infer_canonical_convention_from_checkpoint(raw_ckpt)
    if convention == "unknown":
        convention = "primitive"
    samples = load_mp100_dataset(args.mp100_dir, convention=convention)
    model, _, experiment_name = load_indexing_model_from_checkpoint(
        args.checkpoint, config, device
    )
    normalizer = build_lattice_normalizer(config.data)
    fom_config = fom_config_from_args(args)

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
    metrics = run_inference_on_mp100(
        model,
        normalizer,
        samples,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        rerank=args.rerank,
        fom_config=fom_config,
        ltol=args.ltol,
        atol_deg=args.atol_deg,
        max_log_volume_ratio=args.max_log_volume_ratio,
        topk_config=topk_config,
        refine_config=refine_config,
    )
    metric_keys = [
        "raw_top1_lattice_match_rate",
        "raw_top1_elementwise_rate",
        "raw_top1_volume_guarded_rate",
        "top1_lattice_match_rate",
        "fom_top1_lattice_match_rate",
        "fom_top1_elementwise_rate",
        "fom_top1_volume_guarded_rate",
        "top1_joint_match_rate",
        "topk_lattice_match_rate",
        "topk_elementwise_rate",
        "topk_volume_guarded_rate",
        "raw_mapping_vs_elementwise_gap_rate",
        "topk_mapping_vs_elementwise_gap_rate",
        "n_samples",
    ]
    result = {
        "experiment": experiment_name,
        "checkpoint": str(args.checkpoint),
        "mp100_dir": str(args.mp100_dir),
        "convention": args.convention,
        "top_k": args.top_k,
        "scale_set": args.scale_set,
        "length_scale_factors": list(scale_factors),
        "pool_max_log_volume_ratio": vol_vs_base,
        "include_axis_scale_variants": not args.no_axis_scale_variants,
        "bravais_set": args.bravais_set,
        "refine_steps": args.refine_steps,
        "refine_top_n": args.refine_top_n if args.refine_steps > 0 else None,
        "rerank": args.rerank,
        "fom_mode": args.fom_mode if args.rerank == "fom" else None,
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "max_log_volume_ratio": args.max_log_volume_ratio,
        "metrics": {key: metrics[key] for key in metric_keys},
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
    parser.add_argument(
        "--convention",
        type=str,
        choices=("primitive", "reduced", "niggli"),
        default=None,
        help="Truth lattice convention; default inherits from checkpoint (A0)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
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
        help="Candidate reranking after Top-K pool generation (default fom)",
    )
    parser.add_argument(
        "--ltol",
        type=float,
        default=0.3,
        help="Relative length tolerance for lattice match (default 0.3, Mc/JADE口径)",
    )
    parser.add_argument(
        "--atol-deg",
        type=float,
        default=10.0,
        help="Absolute angle tolerance in degrees for lattice match (default 10)",
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
