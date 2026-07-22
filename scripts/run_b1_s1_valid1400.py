#!/usr/bin/env python3
"""B1-S1 (v3 §11.3): high-symmetry valid1400 pool-recall comparison.

Compares three candidate sources on real valid1400 samples (cubic/tetragonal/
hexagonal/trigonal -- the 4 systems that cleared the B1-S0 95% synthetic Gate
after the vectorization rewrite, see
``docs/实验记录/20260720-B1-S0独立q-search原型.md``):

  - ``q_search``: independent q-search only (no NN), GT crystal-system routing
    (oracle CS -- q-search itself has no CS classifier yet).
  - ``nn_pool``: existing NN neighborhood search (Bravais snap + scale variants,
    ``build_top_k_candidates``), same as the S0.2 diagnostic.
  - ``merged``: ``nn_pool`` ∪ ``q_search`` candidates.

Gate (v3 §11.4): valid Top-20 strict elementwise >=30%; non-cubic Top-20
>=15%; q-search (or merged) >= NN-local pool recall + 8pp; no learned rerank.

Usage:
    python scripts/run_b1_s1_valid1400.py \
        --config configs/scale_100k_a3_g1_gstar6.yaml \
        --checkpoint results/experiments/scale_100k_a3_g1_gstar6_seed42/checkpoints/best.pt \
        --systems cubic,tetragonal,hexagonal,trigonal \
        --n-per-system 40 --top-k 20
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
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.model.topk import TopKConfig, build_top_k_candidates
from pxrd_cell_indexing.search.qsearch import DEFAULT_SEARCH_KWARGS, search_crystal_system
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "b1_s1_valid1400.json"
DEFAULT_WAVELENGTH = 1.54184
TOP_KS = (5, 20, 100)


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def _niggli_params6(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _hit_rank(pool_params: list[list[float]], truth_niggli: list[float], *, ltol: float, atol_deg: float) -> int | None:
    for rank, params in enumerate(pool_params):
        if lattice_match_elementwise(params, truth_niggli, ltol=ltol, atol_deg=atol_deg):
            return rank
    return None


def _recall_at_ks(hit_ranks: list[int | None]) -> dict[str, float | None]:
    n = len(hit_ranks)
    if n == 0:
        return {str(k): None for k in TOP_KS}
    return {
        str(k): float(np.mean([r is not None and r < k for r in hit_ranks]))
        for k in TOP_KS
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    model.set_normalizer(normalizer)
    model.eval()

    single_cfg = TopKConfig(k=args.top_k_nn, bravais_set="extended")

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
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    systems = args.systems.split(",")
    target_n = {cs: args.n_per_system for cs in systems}
    collected_n = {cs: 0 for cs in systems}

    per_system: dict[str, dict[str, list]] = {
        cs: {"q_search": [], "nn_pool": [], "merged": [], "elapsed_s": []} for cs in systems
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

            nn_pools = build_top_k_candidates(pred, k=args.top_k_nn, config=single_cfg)

            for i in range(bsz):
                cs = CRYSTAL_SYSTEMS[int(batch_t["crystal_system_idx"][i].item())]
                if cs not in systems or collected_n[cs] >= target_n[cs]:
                    continue

                t_np = truth[i].cpu().numpy().tolist()
                truth_niggli = _niggli_params6(t_np)
                observed = slice_observed_two_theta(batch_t["pxrd_x"], batch_t["peak_num"], i)
                observed_np = observed.cpu().numpy() if torch.is_tensor(observed) else np.asarray(observed)

                sample_t0 = time.time()
                kwargs = dict(DEFAULT_SEARCH_KWARGS.get(cs, {}))
                kwargs["pool_budget"] = max(kwargs.get("pool_budget", 30), args.top_k_qsearch)
                q_candidates = search_crystal_system(
                    observed_np, cs, wavelength_angstrom=DEFAULT_WAVELENGTH, **kwargs
                )
                elapsed = time.time() - sample_t0

                q_params = [_niggli_params6(_candidate_params(c)) for c in q_candidates]
                nn_params = [_niggli_params6(_candidate_params(c)) for c in nn_pools[i]]
                merged_params = nn_params + q_params

                per_system[cs]["q_search"].append(
                    _hit_rank(q_params, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                )
                per_system[cs]["nn_pool"].append(
                    _hit_rank(nn_params, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                )
                per_system[cs]["merged"].append(
                    _hit_rank(merged_params, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                )
                per_system[cs]["elapsed_s"].append(elapsed)
                collected_n[cs] += 1
                n_done += 1

                total_elapsed = time.time() - t0
                print(
                    f"... {n_done} samples in {total_elapsed:.1f}s "
                    f"({dict(collected_n)}) last={cs} q_search_time={elapsed:.1f}s",
                    flush=True,
                )
                if all(collected_n[cs] >= target_n[cs] for cs in systems):
                    break

    report: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k_nn": args.top_k_nn,
        "top_k_qsearch": args.top_k_qsearch,
        "n_per_system": args.n_per_system,
        "elapsed_sec": time.time() - t0,
        "by_crystal_system": {},
    }
    all_ranks = {"q_search": [], "nn_pool": [], "merged": []}
    noncubic_ranks = {"q_search": [], "nn_pool": [], "merged": []}
    for cs in systems:
        rec = per_system[cs]
        row = {
            "n": len(rec["q_search"]),
            "mean_qsearch_time_s": float(np.mean(rec["elapsed_s"])) if rec["elapsed_s"] else None,
            "max_qsearch_time_s": float(np.max(rec["elapsed_s"])) if rec["elapsed_s"] else None,
        }
        for arm in ("q_search", "nn_pool", "merged"):
            row[arm] = _recall_at_ks(rec[arm])
            all_ranks[arm].extend(rec[arm])
            if cs != "cubic":
                noncubic_ranks[arm].extend(rec[arm])
        report["by_crystal_system"][cs] = row

    report["overall"] = {arm: _recall_at_ks(all_ranks[arm]) for arm in all_ranks}
    report["non_cubic"] = {arm: _recall_at_ks(noncubic_ranks[arm]) for arm in noncubic_ranks}
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--systems", type=str, default="cubic,tetragonal,hexagonal,trigonal")
    p.add_argument("--n-per-system", type=int, default=40)
    p.add_argument("--top-k-nn", type=int, default=20)
    p.add_argument("--top-k-qsearch", type=int, default=100)
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

    print("\n=== B1-S1 overall (all target systems) ===")
    for arm, rec in report["overall"].items():
        print(f"  {arm:10s} " + " ".join(f"recall@{k}={_pct(rec[k])}" for k in rec))
    print("=== B1-S1 non-cubic ===")
    for arm, rec in report["non_cubic"].items():
        print(f"  {arm:10s} " + " ".join(f"recall@{k}={_pct(rec[k])}" for k in rec))
    print("=== by crystal system ===")
    for cs, row in report["by_crystal_system"].items():
        print(f"  [{cs}] n={row['n']} mean_time={row['mean_qsearch_time_s']:.2f}s")
        for arm in ("q_search", "nn_pool", "merged"):
            rec = row[arm]
            print(f"    {arm:10s} " + " ".join(f"recall@{k}={_pct(rec[k])}" for k in rec))


if __name__ == "__main__":
    main()
