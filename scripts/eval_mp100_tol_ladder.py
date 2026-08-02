#!/usr/bin/env python3
"""MP100 L1–L4 tolerance ladder for our production stack (vs RealPXRD A2).

Mirrors the table in ``docs/实验记录/20260723-RealPXRD-A2纯XRD-四级容差梯子.md``:

  L1 0.30/10° · L2 0.20/8° · L3 0.10/5° · L4 0.05/3°
  metrics: Top-1 / Top-20 × mapping / elementwise

Arms
----
- ``nn_raw``: A3-G1 single-point Top-1 (Top-20 = same singleton)
- ``b1b2_pred_cs``: q-search(NN CS) + legacy FOM + NN volume prior (deployable)
- ``b1b2_oracle_cs``: q-search(GT CS) + FOM (upper bound)
- ``merged_pred``: NN Bravais pool ∪ q-search(pred CS), FOM Top-1 / pool Top-20

Protocol notes (vs RealPXRD A2)
-------------------------------
- Truth / peaks: our ``load_mp100_dataset(convention='niggli')`` (A2 used primitive).
- Peaks still conventional→reduced→XRDCalculator, y>5 (same generator).
- Our Top-20 is **pool recall** (Bravais / q-search candidates), not A2's
  independent K=20 noise samples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core.lattice import Lattice

from pxrd_cell_indexing.data.canonical import canonicalize_lattice
from pxrd_cell_indexing.data.mp100 import load_mp100_dataset, peaks_to_model_tensors
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.fom import FomRerankConfig, rerank_candidates_by_fom
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.search.qsearch import DEFAULT_SEARCH_KWARGS, search_crystal_system
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, LatticeCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "beat_engine" / "tol_ladder" / "a3g1_b1b2_mp100_tol_ladder.json"
)
DEFAULT_WAVELENGTH = 1.54184

LADDER = (
    ("L1_loose", 0.30, 10.0),
    ("L2_mid_loose", 0.20, 8.0),
    ("L3_mid_strict", 0.10, 5.0),
    ("L4_engine", 0.05, 3.0),
)

ARMS = ("nn_raw", "b1b2_pred_cs", "b1b2_oracle_cs", "merged_pred")


def _params(c) -> list[float]:
    return [c.a, c.b, c.c, c.alpha, c.beta, c.gamma]


def _niggli6(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _volume(params6: list[float]) -> float:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return float(abs(np.linalg.det(matrix)))


def _to_lc(c, *, bravais_key: str) -> LatticeCandidate:
    return LatticeCandidate(
        crystal_system=getattr(c, "crystal_system", "unknown"),
        a=c.a,
        b=c.b,
        c=c.c,
        alpha=c.alpha,
        beta=c.beta,
        gamma=c.gamma,
        confidence=float(getattr(c, "n_matched", 1) / max(getattr(c, "n_peaks", 1), 1)),
        bravais_key=bravais_key,
    )


def _hit_elementwise(pred: list[float], truth: list[float], *, ltol: float, atol_deg: float) -> bool:
    cand_forms = (pred, _niggli6(pred))
    truth_forms = (truth, _niggli6(truth))
    return any(
        lattice_match_elementwise(c, t, ltol=ltol, atol_deg=atol_deg)
        for c in cand_forms
        for t in truth_forms
    )


def _hit_mapping(pred: list[float], truth: list[float], *, ltol: float, atol_deg: float) -> bool:
    for c in (pred, _niggli6(pred)):
        for t in (truth, _niggli6(truth)):
            try:
                pred_lat = Lattice.from_parameters(*c)
                truth_lat = Lattice.from_parameters(*t)
                if pred_lat.find_mapping(truth_lat, ltol=ltol, atol=atol_deg) is not None:
                    return True
            except Exception:
                continue
    return False


def _fom_ranked(pool: list, peaks: np.ndarray, nn_volume: float) -> list[list[float]]:
    if not pool:
        return []
    cfg = FomRerankConfig(mode="heuristic", ref_volume=nn_volume, volume_log_penalty=1.0)
    ranked = rerank_candidates_by_fom(pool, peaks, config=cfg)
    return [_params(c) for c in ranked]


def _qsearch(peaks: np.ndarray, system: str, *, pool_budget: int) -> list:
    if system not in DEFAULT_SEARCH_KWARGS:
        return []
    kwargs = dict(DEFAULT_SEARCH_KWARGS[system])
    kwargs["pool_budget"] = max(kwargs.get("pool_budget", 30), pool_budget)
    return search_crystal_system(peaks, system, wavelength_angstrom=DEFAULT_WAVELENGTH, **kwargs)


def _score_arm(
    top1: list[float] | None,
    pool: list[list[float]],
    truth: list[float],
) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for name, ltol, atol in LADDER:
        t1 = top1 if top1 is not None else (pool[0] if pool else None)
        pool_use = pool if pool else ([t1] if t1 is not None else [])
        out[name] = {
            "top1_map": bool(t1 is not None and _hit_mapping(t1, truth, ltol=ltol, atol_deg=atol)),
            "top20_map": any(_hit_mapping(p, truth, ltol=ltol, atol_deg=atol) for p in pool_use),
            "top1_ew": bool(
                t1 is not None and _hit_elementwise(t1, truth, ltol=ltol, atol_deg=atol)
            ),
            "top20_ew": any(_hit_elementwise(p, truth, ltol=ltol, atol_deg=atol) for p in pool_use),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    model.set_normalizer(normalizer)
    model.eval()

    samples = load_mp100_dataset(args.mp100_dir, convention="niggli")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    topk_cfg = TopKConfig(k=args.top_k_nn, bravais_set="extended")
    # arm -> level -> metric -> list[bool]
    hits: dict[str, dict[str, dict[str, list[bool]]]] = {
        arm: {lvl: {m: [] for m in ("top1_map", "top20_map", "top1_ew", "top20_ew")} for lvl, _, _ in LADDER}
        for arm in ARMS
    }
    per_sample: list[dict[str, Any]] = []
    t0 = time.time()

    with torch.no_grad():
        for i, sample in enumerate(samples):
            pxrd_x, pxrd_y, peak_num = peaks_to_model_tensors(sample.two_theta, sample.intensity)
            pxrd_x_t = torch.from_numpy(pxrd_x).to(device)
            pxrd_y_t = torch.from_numpy(pxrd_y).to(device)
            peak_num_t = torch.tensor([peak_num], dtype=torch.long, device=device)

            outputs = model(pxrd_x_t, pxrd_y_t, peak_num_t)
            pred = normalizer.denormalize(outputs["lattice_norm"])[0].cpu().numpy().tolist()
            pred_cs = CRYSTAL_SYSTEMS[int(outputs["crystal_system_logits"].argmax(-1).item())]
            gt_cs = sample.crystal_system
            truth = _niggli6(sample.truth_lattice.tolist())
            nn_volume = _volume(pred)
            peaks = np.asarray(sample.two_theta, dtype=np.float64)

            nn_pool = build_top_k_candidates(
                torch.tensor([pred], dtype=torch.float32), k=args.top_k_nn, config=topk_cfg
            )[0]
            q_oracle = _qsearch(peaks, gt_cs, pool_budget=args.top_k_qsearch)
            q_pred = (
                list(q_oracle)
                if pred_cs == gt_cs
                else _qsearch(peaks, pred_cs, pool_budget=args.top_k_qsearch)
            )

            nn_lc = [_to_lc(c, bravais_key=c.bravais_key or "nn") for c in nn_pool]
            q_oracle_lc = [_to_lc(c, bravais_key=f"qsearch:{gt_cs}") for c in q_oracle]
            q_pred_lc = [_to_lc(c, bravais_key=f"qsearch:{pred_cs}") for c in q_pred]

            ranked_pred = _fom_ranked(q_pred_lc, peaks, nn_volume)
            ranked_oracle = _fom_ranked(q_oracle_lc, peaks, nn_volume)
            ranked_merged = _fom_ranked(nn_lc + q_pred_lc, peaks, nn_volume)

            arm_payload = {
                "nn_raw": (pred, [pred]),
                "b1b2_pred_cs": (
                    ranked_pred[0] if ranked_pred else pred,
                    ranked_pred if ranked_pred else [pred],
                ),
                "b1b2_oracle_cs": (
                    ranked_oracle[0] if ranked_oracle else pred,
                    ranked_oracle if ranked_oracle else [pred],
                ),
                "merged_pred": (
                    ranked_merged[0] if ranked_merged else pred,
                    ranked_merged if ranked_merged else [pred],
                ),
            }

            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "gt_cs": gt_cs,
                "pred_cs": pred_cs,
                "cs_correct": pred_cs == gt_cs,
                "q_pred_n": len(q_pred),
                "q_oracle_n": len(q_oracle),
                "nn_pool_n": len(nn_pool),
            }
            for arm, (top1, pool) in arm_payload.items():
                scored = _score_arm(top1, pool[: args.top_k_report], truth)
                row[arm] = scored
                for lvl, metrics in scored.items():
                    for m, ok in metrics.items():
                        hits[arm][lvl][m].append(ok)

            per_sample.append(row)
            elapsed = time.time() - t0
            l4 = row["merged_pred"]["L4_engine"]
            print(
                f"... {i+1}/{len(samples)} {sample.sample_id} gt={gt_cs} pred={pred_cs} "
                f"merged L4 t1_ew={l4['top1_ew']} t20_ew={l4['top20_ew']} "
                f"total={elapsed:.0f}s",
                flush=True,
            )

    def _rate(xs: list[bool]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary: dict[str, Any] = {}
    for arm in ARMS:
        summary[arm] = {}
        for lvl, _, _ in LADDER:
            summary[arm][lvl] = {m: _rate(hits[arm][lvl][m]) for m in hits[arm][lvl]}

    report = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "config": str(config_path),
        "convention": "niggli",
        "n_samples": len(samples),
        "top_k_nn": args.top_k_nn,
        "top_k_qsearch": args.top_k_qsearch,
        "top_k_report": args.top_k_report,
        "ladder": [{"name": n, "ltol": lt, "atol_deg": at} for n, lt, at in LADDER],
        "elapsed_sec": time.time() - t0,
        "cs_accuracy": _rate([s["cs_correct"] for s in per_sample]),
        "summary": summary,
        "reference_realpxrd_a2": {
            "L1_loose": {"top1_map": 0.80, "top20_map": 0.98, "top1_ew": 0.02, "top20_ew": 0.18},
            "L2_mid_loose": {"top1_map": 0.58, "top20_map": 0.96, "top1_ew": 0.01, "top20_ew": 0.08},
            "L3_mid_strict": {"top1_map": 0.22, "top20_map": 0.80, "top1_ew": 0.01, "top20_ew": 0.04},
            "L4_engine": {"top1_map": 0.03, "top20_map": 0.22, "top1_ew": 0.01, "top20_ew": 0.02},
            "note": "RealPXRD A2: primitive truth, K=20 noise samples (not our pool)",
        },
        "per_sample": per_sample,
    }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scale_100k_a3_g1_gstar6.yaml")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "results/experiments/scale_100k_a3_g1_gstar6_seed42/checkpoints/best.pt",
    )
    p.add_argument("--mp100-dir", type=Path, default=PROJECT_ROOT / "data/MP-100samples-benchmark")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--top-k-nn", type=int, default=20)
    p.add_argument("--top-k-qsearch", type=int, default=20)
    p.add_argument("--top-k-report", type=int, default=20, help="cap for Top-20 metrics")
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    def pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    print("\n=== MP100 tolerance ladder (our stack) ===")
    for arm in ARMS:
        print(f"\n## {arm}")
        print(f"{'level':16} {'T1 map':>8} {'T20 map':>8} {'T1 ew':>8} {'T20 ew':>8}")
        for lvl, _, _ in LADDER:
            m = report["summary"][arm][lvl]
            print(
                f"{lvl:16} {pct(m['top1_map']):>8} {pct(m['top20_map']):>8} "
                f"{pct(m['top1_ew']):>8} {pct(m['top20_ew']):>8}"
            )
    print(f"\nWrote {args.output}  elapsed={report['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
