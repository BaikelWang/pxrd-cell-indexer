#!/usr/bin/env python3
"""B3 (v3 §13): optional iterative local refiner on the B1+B2 pipeline.

Pipeline (only entered because B1 recall and B2 ranking already passed
their Gates, per v3 §13's precondition):

  1. generate the fixed B1 q-search pool (pool_budget=20, oracle CS routing,
     same as B1-S1/B2);
  2. rank it with the B2 winner (legacy R6-C FOM + NN volume prior);
  3. take the ranked Top-M (default 5);
  4. run 0/1/3 iterative refine steps (least-squares metric refit over all
     currently-matched peaks, ``search/refine.py``) on each of the M
     candidates;
  5. re-rank the refined Top-M at each step count with the same FOM, and
     check whether ranked Top-1 strict-match rate changes.

Gate (v3 §12.4/§13): ranked Top-1 >=+2pp vs step 0, OR Top-20 recall
unchanged with a significant angle-MAE drop on the step-0 ranked Top-1
candidate. If peak fit improves but strict match rate drops, stop (would
indicate the residual objective prefers a degenerate/pseudo cell).
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
from pxrd_cell_indexing.search.refine import iterative_refine
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, LatticeCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "b3_refine_valid1400.json"
DEFAULT_WAVELENGTH = 1.54184
STEP_COUNTS = (0, 1, 3)


def _candidate_params(candidate) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def _niggli_params6(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _gstar_from_params6(params6: list[float]) -> np.ndarray:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return np.linalg.inv(matrix @ matrix.T)


def _to_lattice_candidate(candidate) -> LatticeCandidate:
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


def _angle_free_lattice_error(params6: list[float], truth_niggli: list[float]) -> float:
    """Mean angular deviation (deg) between candidate and truth Niggli params
    (alpha/beta/gamma only -- a lightweight 'angle MAE' diagnostic per v3 §13
    Gate's alternate clause)."""
    cand_niggli = _niggli_params6(params6)
    return float(np.mean(np.abs(np.array(cand_niggli[3:]) - np.array(truth_niggli[3:]))))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    normalizer = build_lattice_normalizer(config.data)
    model, _, experiment_name = load_indexing_model_from_checkpoint(args.checkpoint, config, device)
    model.set_normalizer(normalizer)
    model.eval()

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
        cs: {"topk_hit": [], **{f"top1_step{s}": [] for s in STEP_COUNTS}, "angle_mae_step0": [], "angle_mae_step3": [], "elapsed_s": []}
        for cs in systems
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

                sample_t0 = time.time()
                kwargs = dict(DEFAULT_SEARCH_KWARGS.get(cs, {}))
                kwargs["pool_budget"] = args.top_k
                pool = search_crystal_system(observed_np, cs, wavelength_angstrom=DEFAULT_WAVELENGTH, **kwargs)
                elapsed = time.time() - sample_t0

                topk_hit = any(
                    lattice_match_elementwise(_niggli_params6(_candidate_params(c)), truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg)
                    for c in pool
                )
                per_system[cs]["topk_hit"].append(topk_hit)

                if not pool:
                    for s in STEP_COUNTS:
                        per_system[cs][f"top1_step{s}"].append(False)
                    per_system[cs]["angle_mae_step0"].append(float("nan"))
                    per_system[cs]["angle_mae_step3"].append(float("nan"))
                    per_system[cs]["elapsed_s"].append(elapsed)
                    collected_n[cs] += 1
                    n_done += 1
                    continue

                fom_cfg = FomRerankConfig(mode="heuristic", ref_volume=nn_volume, volume_log_penalty=1.0)
                ranked0 = rerank_candidates_by_fom([_to_lattice_candidate(c) for c in pool], observed_np, config=fom_cfg)
                top_m = ranked0[: args.top_m]
                # Map ranked LatticeCandidate back to the underlying QSearchCandidate
                # (by identity of the 6 params) so we can iteratively refine it.
                by_params = {tuple(round(v, 9) for v in _candidate_params(c)): c for c in pool}
                top_m_qsearch = [by_params[tuple(round(v, 9) for v in (c.a, c.b, c.c, c.alpha, c.beta, c.gamma))] for c in top_m]

                traces = [iterative_refine(c, observed_np, max_steps=max(STEP_COUNTS)) for c in top_m_qsearch]

                for s in STEP_COUNTS:
                    refined_at_s = [trace[min(s, len(trace) - 1)] for trace in traces]
                    ranked_s = rerank_candidates_by_fom(
                        [_to_lattice_candidate(c) for c in refined_at_s], observed_np, config=fom_cfg
                    )
                    hit = bool(
                        ranked_s
                        and lattice_match_elementwise(
                            _niggli_params6(_candidate_params(ranked_s[0])), truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg
                        )
                    )
                    per_system[cs][f"top1_step{s}"].append(hit)

                mae0 = _angle_free_lattice_error(_candidate_params(traces[0][0]), truth_niggli)
                mae3 = _angle_free_lattice_error(_candidate_params(traces[0][min(3, len(traces[0]) - 1)]), truth_niggli)
                per_system[cs]["angle_mae_step0"].append(mae0)
                per_system[cs]["angle_mae_step3"].append(mae3)
                per_system[cs]["elapsed_s"].append(elapsed)
                collected_n[cs] += 1
                n_done += 1

                total_elapsed = time.time() - t0
                print(
                    f"... {n_done} samples in {total_elapsed:.1f}s ({dict(collected_n)}) last={cs} "
                    f"top1_step0={per_system[cs]['top1_step0'][-1]} top1_step3={per_system[cs]['top1_step3'][-1]} "
                    f"search_time={elapsed:.1f}s",
                    flush=True,
                )
                if all(collected_n[cs] >= target_n[cs] for cs in systems):
                    break

    def _rate(xs: list[bool]) -> float | None:
        return float(np.mean(xs)) if xs else None

    report: dict[str, Any] = {
        "experiment_name": experiment_name,
        "checkpoint": str(args.checkpoint),
        "ltol": args.ltol,
        "atol_deg": args.atol_deg,
        "top_k": args.top_k,
        "top_m": args.top_m,
        "n_per_system": args.n_per_system,
        "elapsed_sec": time.time() - t0,
        "by_crystal_system": {},
    }
    all_topk: list[bool] = []
    all_steps: dict[int, list[bool]] = {s: [] for s in STEP_COUNTS}
    all_mae0: list[float] = []
    all_mae3: list[float] = []
    for cs in systems:
        rec = per_system[cs]
        row = {"n": len(rec["topk_hit"]), "topk_recall": _rate(rec["topk_hit"])}
        for s in STEP_COUNTS:
            row[f"top1_step{s}"] = _rate(rec[f"top1_step{s}"])
            all_steps[s].extend(rec[f"top1_step{s}"])
        row["angle_mae_step0_deg"] = float(np.nanmean(rec["angle_mae_step0"])) if rec["angle_mae_step0"] else None
        row["angle_mae_step3_deg"] = float(np.nanmean(rec["angle_mae_step3"])) if rec["angle_mae_step3"] else None
        report["by_crystal_system"][cs] = row
        all_topk.extend(rec["topk_hit"])
        all_mae0.extend(rec["angle_mae_step0"])
        all_mae3.extend(rec["angle_mae_step3"])

    report["overall"] = {"topk_recall": _rate(all_topk)}
    for s in STEP_COUNTS:
        report["overall"][f"top1_step{s}"] = _rate(all_steps[s])
    report["overall"]["angle_mae_step0_deg"] = float(np.nanmean(all_mae0)) if all_mae0 else None
    report["overall"]["angle_mae_step3_deg"] = float(np.nanmean(all_mae3)) if all_mae3 else None
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
    p.add_argument("--top-m", type=int, default=5)
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

    o = report["overall"]
    print(f"\n=== B3 overall (TopK recall={_pct(o['topk_recall'])}) ===")
    for s in STEP_COUNTS:
        print(f"  step={s} ranked_top1={_pct(o[f'top1_step{s}'])}")
    print(f"  angle_mae step0={o['angle_mae_step0_deg']:.3f} deg -> step3={o['angle_mae_step3_deg']:.3f} deg")
    print("\n=== by crystal system ===")
    for cs, row in report["by_crystal_system"].items():
        print(f"  [{cs}] n={row['n']} topk_recall={_pct(row['topk_recall'])}")
        for s in STEP_COUNTS:
            print(f"    step={s} ranked_top1={_pct(row[f'top1_step{s}'])}")
        print(f"    angle_mae step0={row['angle_mae_step0_deg']:.3f} deg -> step3={row['angle_mae_step3_deg']:.3f} deg")


if __name__ == "__main__":
    main()
