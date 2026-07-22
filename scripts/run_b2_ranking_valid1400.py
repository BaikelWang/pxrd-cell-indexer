#!/usr/bin/env python3
"""B2 (v3 §12.3): candidate ranking comparison on a fixed B1 q-search pool.

For each valid1400 sample (cubic/tetragonal/hexagonal/trigonal, the systems
that cleared B1-S1), generate the independent q-search Top-K pool once (oracle
CS routing, same as B1-S1), then rank that *same fixed pool* three ways:

  1. ``nn_proximity``: ignore peak fit, pick the pool candidate whose
     reciprocal metric is closest to the NN's raw single-point proposal
     ("NN confidence" baseline per v3 §12.3 item 1).
  2. ``legacy_fom``: existing R6-C de Wolff-style heuristic FOM
     (``model/fom.rerank_candidates_by_fom``, item 2).
  3. ``deterministic``: new peaks-only score (``search/rank.py``, item 3;
     matched count + supercell flag + theoretical-density + volume/metric
     proximity to NN + fit residual, v3 §12.1).

Reports, per v3 §12.4 Gate:
  - ranking efficiency = ranked Top-1 strict / TopK(20) strict recall
  - ranked Top-1 strict rate
Gate: ranking efficiency >=75%, valid ranked Top-1 >=25%.
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
from pxrd_cell_indexing.data.dataset import PXRDDatasetConfig, PeakFilterConfig, build_dataloader
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.fom import FomRerankConfig, rerank_candidates_by_fom, slice_observed_two_theta
from pxrd_cell_indexing.search.qsearch import DEFAULT_SEARCH_KWARGS, search_crystal_system
from pxrd_cell_indexing.search.rank import RankConfig, rank_by_deterministic_score, rank_by_nn_proximity
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, LatticeCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "b2_ranking_valid1400.json"
DEFAULT_WAVELENGTH = 1.54184
METHODS = ("nn_proximity", "legacy_fom", "deterministic")


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def _niggli_params6(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _gstar_from_params6(params6: list[float]) -> np.ndarray:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return np.linalg.inv(matrix @ matrix.T)


def _to_lattice_candidate(candidate) -> LatticeCandidate:
    """``rerank_candidates_by_fom`` (legacy R6-C FOM) expects pydantic
    ``LatticeCandidate`` objects (uses ``model_copy``); convert the plain
    ``QSearchCandidate`` dataclass to compare on the same fixed pool."""
    return LatticeCandidate(
        crystal_system=candidate.crystal_system,
        a=candidate.a,
        b=candidate.b,
        c=candidate.c,
        alpha=candidate.alpha,
        beta=candidate.beta,
        gamma=candidate.gamma,
        confidence=candidate.n_matched / max(candidate.n_peaks, 1),
        bravais_key=f"qsearch:{candidate.crystal_system}",
    )


def _top1_hit(pool: list, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> bool:
    if not pool:
        return False
    return bool(
        lattice_match_elementwise(_niggli_params6(_candidate_params(pool[0])), truth_niggli, ltol=ltol, atol_deg=atol_deg)
    )


def _any_hit(pool: list, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> bool:
    return any(
        lattice_match_elementwise(_niggli_params6(_candidate_params(c)), truth_niggli, ltol=ltol, atol_deg=atol_deg)
        for c in pool
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    model.set_normalizer(normalizer)
    model.eval()

    rank_cfg = RankConfig()

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
        dataset_cfg, batch_size=config.data.batch_size, num_workers=0, shuffle=False, pin_memory=False
    )

    systems = args.systems.split(",")
    target_n = {cs: args.n_per_system for cs in systems}
    collected_n = {cs: 0 for cs in systems}
    per_system: dict[str, dict[str, list]] = {
        cs: {"topk_hit": [], **{m: [] for m in METHODS}, "elapsed_s": []} for cs in systems
    }
    t0 = time.time()
    n_done = 0

    with torch.no_grad():
        for batch in loader:
            if all(collected_n[cs] >= target_n[cs] for cs in systems):
                break
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
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

            for i in range(bsz):
                cs = CRYSTAL_SYSTEMS[int(batch_t["crystal_system_idx"][i].item())]
                if cs not in systems or collected_n[cs] >= target_n[cs]:
                    continue

                t_np = truth[i].cpu().numpy().tolist()
                truth_niggli = _niggli_params6(t_np)
                observed = slice_observed_two_theta(batch_t["pxrd_x"], batch_t["peak_num"], i)
                observed_np = observed.cpu().numpy() if torch.is_tensor(observed) else np.asarray(observed)

                nn_params = pred[i].cpu().numpy().tolist()
                nn_volume = float(abs(np.linalg.det(lattice_params_to_matrix(torch.tensor(nn_params)).numpy())))
                nn_gstar = _gstar_from_params6(nn_params)

                sample_t0 = time.time()
                kwargs = dict(DEFAULT_SEARCH_KWARGS.get(cs, {}))
                kwargs["pool_budget"] = args.top_k  # fixed B1 pool = same K as the B1-S1 Gate
                pool = search_crystal_system(observed_np, cs, wavelength_angstrom=DEFAULT_WAVELENGTH, **kwargs)
                elapsed = time.time() - sample_t0

                topk_hit = _any_hit(pool, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)

                # Give legacy FOM the same NN-volume prior as nn_proximity/deterministic
                # get (fair comparison -- all 3 methods see the same NN proposal).
                fom_cfg = FomRerankConfig(mode="heuristic", ref_volume=nn_volume, volume_log_penalty=1.0)
                ranked = {
                    "nn_proximity": rank_by_nn_proximity(list(pool), nn_gstar),
                    "legacy_fom": rerank_candidates_by_fom(
                        [_to_lattice_candidate(c) for c in pool], observed_np, config=fom_cfg
                    )
                    if pool
                    else [],
                    "deterministic": rank_by_deterministic_score(
                        list(pool), observed_np, config=rank_cfg, nn_volume=nn_volume, nn_gstar=nn_gstar
                    ),
                }

                per_system[cs]["topk_hit"].append(topk_hit)
                for method in METHODS:
                    hit = _top1_hit(ranked[method], truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                    per_system[cs][method].append(hit)
                per_system[cs]["elapsed_s"].append(elapsed)
                collected_n[cs] += 1
                n_done += 1

                total_elapsed = time.time() - t0
                print(
                    f"... {n_done} samples in {total_elapsed:.1f}s ({dict(collected_n)}) "
                    f"last={cs} topk_hit={topk_hit} search_time={elapsed:.1f}s",
                    flush=True,
                )
                if all(collected_n[cs] >= target_n[cs] for cs in systems):
                    break

    def _rate(xs: list[bool]) -> float | None:
        return float(np.mean(xs)) if xs else None

    def _efficiency(top1_rate: float | None, topk_rate: float | None) -> float | None:
        if top1_rate is None or topk_rate is None or topk_rate <= 1e-12:
            return None
        return top1_rate / topk_rate

    report: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k": args.top_k,
        "n_per_system": args.n_per_system,
        "elapsed_sec": time.time() - t0,
        "by_crystal_system": {},
    }
    all_topk: list[bool] = []
    all_methods: dict[str, list[bool]] = {m: [] for m in METHODS}
    noncubic_topk: list[bool] = []
    noncubic_methods: dict[str, list[bool]] = {m: [] for m in METHODS}
    for cs in systems:
        rec = per_system[cs]
        topk_rate = _rate(rec["topk_hit"])
        row = {
            "n": len(rec["topk_hit"]),
            "topk_recall": topk_rate,
            "mean_search_time_s": float(np.mean(rec["elapsed_s"])) if rec["elapsed_s"] else None,
        }
        for m in METHODS:
            top1_rate = _rate(rec[m])
            row[m] = {"ranked_top1": top1_rate, "ranking_efficiency": _efficiency(top1_rate, topk_rate)}
        report["by_crystal_system"][cs] = row
        all_topk.extend(rec["topk_hit"])
        if cs != "cubic":
            noncubic_topk.extend(rec["topk_hit"])
        for m in METHODS:
            all_methods[m].extend(rec[m])
            if cs != "cubic":
                noncubic_methods[m].extend(rec[m])

    overall_topk = _rate(all_topk)
    noncubic_topk_rate = _rate(noncubic_topk)
    report["overall"] = {"topk_recall": overall_topk}
    report["non_cubic"] = {"topk_recall": noncubic_topk_rate}
    for m in METHODS:
        top1 = _rate(all_methods[m])
        report["overall"][m] = {"ranked_top1": top1, "ranking_efficiency": _efficiency(top1, overall_topk)}
        nc_top1 = _rate(noncubic_methods[m])
        report["non_cubic"][m] = {"ranked_top1": nc_top1, "ranking_efficiency": _efficiency(nc_top1, noncubic_topk_rate)}
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--systems", type=str, default="cubic,tetragonal,hexagonal,trigonal")
    p.add_argument("--n-per-system", type=int, default=40)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    def _pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    for scope in ("overall", "non_cubic"):
        block = report[scope]
        print(f"\n=== B2 {scope} (TopK recall={_pct(block['topk_recall'])}) ===")
        for m in METHODS:
            r = block[m]
            print(f"  {m:14s} ranked_top1={_pct(r['ranked_top1'])} efficiency={_pct(r['ranking_efficiency'])}")
    print("\n=== by crystal system ===")
    for cs, row in report["by_crystal_system"].items():
        print(f"  [{cs}] n={row['n']} topk_recall={_pct(row['topk_recall'])} mean_time={row['mean_search_time_s']:.2f}s")
        for m in METHODS:
            r = row[m]
            print(f"    {m:14s} ranked_top1={_pct(r['ranked_top1'])} efficiency={_pct(r['ranking_efficiency'])}")


if __name__ == "__main__":
    main()
