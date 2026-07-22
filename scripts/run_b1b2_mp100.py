#!/usr/bin/env python3
"""MP100 end-to-end: A3-G1 raw vs B1+B2 (q-search + FOM) for baseline update.

Compares on the same niggli / ltol=0.05 / atol=3° strict elementwise KPI:

  - ``nn_raw``: A3-G1 single-point Top-1 (locked production baseline)
  - ``nn_pool_fom``: NN Bravais neighborhood pool + legacy FOM (ref_volume)
  - ``b1b2_oracle_cs``: q-search with GT crystal system + FOM (upper bound)
  - ``b1b2_pred_cs``: q-search with NN-predicted CS + FOM (realistic)
  - ``merged_oracle``: nn_pool ∪ q-search(oracle CS), FOM-ranked Top-1
  - ``merged_pred``: nn_pool ∪ q-search(pred CS), FOM-ranked Top-1

Also reports high-symmetry-only (cubic/tet/hex/trig) vs full MP100.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "b1b2_mp100.json"
DEFAULT_WAVELENGTH = 1.54184
HIGH_SYM = {"cubic", "tetragonal", "hexagonal", "trigonal"}
ARMS = (
    "nn_raw",
    "nn_pool_fom",
    "b1b2_oracle_cs",
    "b1b2_pred_cs",
    "merged_oracle",
    "merged_pred",
)


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


def _hit(params6: list[float], truth: list[float], *, ltol: float, atol_deg: float) -> bool:
    """Strict elementwise hit, invariant to Niggli 60°/120° angle flips.

    S0 ``eval_mp100`` compares raw pred vs truth without re-Niggli; forcing
    Niggli on hexagonal/trigonal preds alone can flip γ=120→60 and false-negate
    a correct cell (see mp-1391). Accept any of the four (raw/niggli)×(raw/niggli)
    pairings so both the locked S0 protocol and Niggli-reduced q-search cells
    are counted fairly.
    """
    cand_forms = (params6, _niggli6(params6))
    truth_forms = (truth, _niggli6(truth))
    return any(
        lattice_match_elementwise(c, t, ltol=ltol, atol_deg=atol_deg)
        for c in cand_forms
        for t in truth_forms
    )


def _pool_hit(pool: list, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> bool:
    return any(_hit(_params(c), truth_niggli, ltol=ltol, atol_deg=atol_deg) for c in pool)


def _fom_top1(pool: list, peaks: np.ndarray, nn_volume: float) -> list[float] | None:
    if not pool:
        return None
    cfg = FomRerankConfig(mode="heuristic", ref_volume=nn_volume, volume_log_penalty=1.0)
    ranked = rerank_candidates_by_fom(pool, peaks, config=cfg)
    return _params(ranked[0]) if ranked else None


def _qsearch(peaks: np.ndarray, system: str, *, pool_budget: int) -> list:
    if system not in DEFAULT_SEARCH_KWARGS:
        return []
    kwargs = dict(DEFAULT_SEARCH_KWARGS[system])
    kwargs["pool_budget"] = max(kwargs.get("pool_budget", 30), pool_budget)
    return search_crystal_system(peaks, system, wavelength_angstrom=DEFAULT_WAVELENGTH, **kwargs)


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
    hits: dict[str, list[bool]] = {arm: [] for arm in ARMS}
    pool_hits = {"q_oracle": [], "q_pred": [], "nn_pool": [], "merged_oracle": [], "merged_pred": []}
    by_cs: dict[str, dict[str, list[bool]]] = {
        cs: {arm: [] for arm in ARMS} for cs in CRYSTAL_SYSTEMS
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
            pred_cs_idx = int(outputs["crystal_system_logits"].argmax(-1).item())
            pred_cs = CRYSTAL_SYSTEMS[pred_cs_idx]
            gt_cs = sample.crystal_system
            truth_niggli = _niggli6(sample.truth_lattice.tolist())
            nn_volume = _volume(pred)
            peaks = np.asarray(sample.two_theta, dtype=np.float64)

            nn_pools = build_top_k_candidates(
                torch.tensor([pred], dtype=torch.float32), k=args.top_k_nn, config=topk_cfg
            )
            nn_pool = nn_pools[0]

            sample_t0 = time.time()
            q_oracle = _qsearch(peaks, gt_cs, pool_budget=args.top_k_qsearch)
            q_pred = _qsearch(peaks, pred_cs, pool_budget=args.top_k_qsearch) if pred_cs != gt_cs else list(q_oracle)
            search_s = time.time() - sample_t0

            nn_lc = [_to_lc(c, bravais_key=c.bravais_key or "nn") for c in nn_pool]
            q_oracle_lc = [_to_lc(c, bravais_key=f"qsearch:{gt_cs}") for c in q_oracle]
            q_pred_lc = [_to_lc(c, bravais_key=f"qsearch:{pred_cs}") for c in q_pred]

            arm_top1 = {
                "nn_raw": pred,
                "nn_pool_fom": _fom_top1(nn_lc, peaks, nn_volume) or pred,
                "b1b2_oracle_cs": _fom_top1(q_oracle_lc, peaks, nn_volume) or pred,
                "b1b2_pred_cs": _fom_top1(q_pred_lc, peaks, nn_volume) or pred,
                "merged_oracle": _fom_top1(nn_lc + q_oracle_lc, peaks, nn_volume) or pred,
                "merged_pred": _fom_top1(nn_lc + q_pred_lc, peaks, nn_volume) or pred,
            }

            row = {
                "sample_id": sample.sample_id,
                "gt_cs": gt_cs,
                "pred_cs": pred_cs,
                "cs_correct": pred_cs == gt_cs,
                "n_peaks": int(sample.peak_num),
                "search_time_s": search_s,
                "q_oracle_n": len(q_oracle),
                "q_pred_n": len(q_pred),
            }
            for arm, params in arm_top1.items():
                ok = _hit(params, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                hits[arm].append(ok)
                by_cs[gt_cs][arm].append(ok)
                row[arm] = ok

            ph = {
                "q_oracle": _pool_hit(q_oracle, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg),
                "q_pred": _pool_hit(q_pred, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg),
                "nn_pool": _pool_hit(nn_pool, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg),
                "merged_oracle": _pool_hit(nn_pool + q_oracle, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg),
                "merged_pred": _pool_hit(nn_pool + q_pred, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg),
            }
            for k, v in ph.items():
                pool_hits[k].append(v)
                row[f"pool_{k}"] = v

            per_sample.append(row)
            elapsed = time.time() - t0
            print(
                f"... {i+1}/{len(samples)} {sample.sample_id} gt={gt_cs} pred={pred_cs} "
                f"nn_raw={row['nn_raw']} merged_oracle={row['merged_oracle']} "
                f"search={search_s:.1f}s total={elapsed:.0f}s",
                flush=True,
            )

    def _rate(xs: list[bool]) -> float | None:
        return float(np.mean(xs)) if xs else None

    def _subset(predicate) -> dict[str, Any]:
        idx = [i for i, s in enumerate(per_sample) if predicate(s)]
        out: dict[str, Any] = {"n": len(idx)}
        for arm in ARMS:
            out[arm] = _rate([hits[arm][i] for i in idx])
        for k in pool_hits:
            out[f"pool_{k}"] = _rate([pool_hits[k][i] for i in idx])
        return out

    report: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "mp100_dir": str(args.mp100_dir),
        "convention": "niggli",
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k_nn": args.top_k_nn,
        "top_k_qsearch": args.top_k_qsearch,
        "n_samples": len(samples),
        "elapsed_sec": time.time() - t0,
        "overall": _subset(lambda _: True),
        "high_symmetry": _subset(lambda s: s["gt_cs"] in HIGH_SYM),
        "low_symmetry": _subset(lambda s: s["gt_cs"] not in HIGH_SYM),
        "by_crystal_system": {},
        "cs_accuracy": _rate([s["cs_correct"] for s in per_sample]),
        "per_sample": per_sample,
        "baseline_reference": {
            "a3_g1_s0_raw_elementwise": 0.23,
            "jade9_strict": 0.681,
            "mcmaille_strict": 0.659,
        },
    }
    for cs in CRYSTAL_SYSTEMS:
        report["by_crystal_system"][cs] = {
            "n": len(by_cs[cs]["nn_raw"]),
            **{arm: _rate(by_cs[cs][arm]) for arm in ARMS},
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
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full MP100")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Slim JSON for humans (drop per_sample lattice dumps already in row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    def _pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    for scope in ("overall", "high_symmetry", "low_symmetry"):
        block = report[scope]
        print(f"\n=== {scope} (n={block['n']}) ===")
        for arm in ARMS:
            print(f"  {arm:18s} Top-1={_pct(block[arm])}")
        print(
            f"  pool_q_oracle={_pct(block['pool_q_oracle'])} "
            f"pool_merged_oracle={_pct(block['pool_merged_oracle'])} "
            f"pool_merged_pred={_pct(block['pool_merged_pred'])}"
        )
    print(f"\nCS accuracy={_pct(report['cs_accuracy'])}")
    print("by crystal system (Top-1):")
    for cs, row in report["by_crystal_system"].items():
        print(
            f"  {cs:12s} n={row['n']:2d} "
            f"nn_raw={_pct(row['nn_raw']):>6s} "
            f"merged_oracle={_pct(row['merged_oracle']):>6s} "
            f"merged_pred={_pct(row['merged_pred']):>6s}"
        )


if __name__ == "__main__":
    main()
