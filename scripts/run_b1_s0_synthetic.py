#!/usr/bin/env python3
"""B1-S0 (v3 §11.3): synthetic unit test for the independent q-search.

For each crystal system: generate a known lattice + ideal (noise-free) peaks
(no NN dependence anywhere), run the independent search, and check whether
the true cell is recovered in the candidate pool at various Top-K widths.

Gate (v3 §11.4, applied per-system here since this is the pre-valid1400
sanity check): "Synthetic Top-K recall 未过 95%，不得进入 valid1400."

Usage:
    python scripts/run_b1_s0_synthetic.py --n-samples 20 --systems cubic,tetragonal,hexagonal,orthorhombic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pxrd_cell_indexing.data.canonical import canonicalize_lattice
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.fom import theoretical_two_theta
from pxrd_cell_indexing.search.qsearch import DEFAULT_SEARCH_KWARGS, search_crystal_system

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "b1_s0_synthetic.json"

TOP_KS = (5, 10, 20, 30)


def _random_lattice(system: str, rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    a = float(rng.uniform(4.0, 11.0))
    if system == "cubic":
        return a, a, a, 90.0, 90.0, 90.0
    if system == "tetragonal":
        c = float(rng.uniform(4.0, 14.0))
        return a, a, c, 90.0, 90.0, 90.0
    if system == "hexagonal":
        c = float(rng.uniform(4.0, 14.0))
        return a, a, c, 90.0, 90.0, 120.0
    if system == "trigonal":
        # Mix hexagonal-axis and rhombohedral-primitive synthetics (matches
        # the two internal search variants trigonal_hex / trigonal_rhomb).
        if rng.random() < 0.5:
            c = float(rng.uniform(4.0, 14.0))
            return a, a, c, 90.0, 90.0, 120.0
        alpha = float(rng.uniform(55.0, 110.0))
        return a, a, a, alpha, alpha, alpha
    if system == "orthorhombic":
        b = float(rng.uniform(4.0, 11.0))
        c = float(rng.uniform(4.0, 14.0))
        return a, b, c, 90.0, 90.0, 90.0
    if system == "monoclinic":
        b = float(rng.uniform(4.0, 11.0))
        c = float(rng.uniform(4.0, 14.0))
        beta = float(rng.uniform(95.0, 122.0))
        return a, b, c, 90.0, beta, 90.0
    if system == "triclinic":
        b = float(rng.uniform(4.0, 11.0))
        c = float(rng.uniform(4.0, 14.0))
        alpha = float(rng.uniform(75.0, 105.0))
        beta = float(rng.uniform(75.0, 105.0))
        gamma = float(rng.uniform(75.0, 105.0))
        return a, b, c, alpha, beta, gamma
    raise ValueError(system)


def _niggli_params(params6: tuple[float, ...]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def run(args: argparse.Namespace) -> dict:
    systems = args.systems.split(",")
    wavelength = 1.54184

    report: dict = {"n_samples": args.n_samples, "systems": {}}

    for system in systems:
        # Per-system RNG so multi-system runs match single-system runs at the
        # same --seed (otherwise earlier systems consume the stream).
        rng = np.random.default_rng(args.seed)
        kwargs = {"match_fraction_min": args.match_fraction_min}
        kwargs.update(DEFAULT_SEARCH_KWARGS.get(system, {}))
        recall_counts = {k: 0 for k in TOP_KS}
        n_ok = 0
        times = []
        per_sample_records = []

        for i in range(args.n_samples):
            truth_raw = _random_lattice(system, rng)
            truth_niggli = _niggli_params(truth_raw)

            two_theta_all = theoretical_two_theta(
                truth_raw, wavelength_angstrom=wavelength, two_theta_max=args.two_theta_max
            )
            two_theta_all = two_theta_all[two_theta_all >= args.two_theta_min]
            if two_theta_all.shape[0] < args.n_peaks_min:
                continue
            observed = two_theta_all[: args.n_peaks]

            t0 = time.monotonic()
            candidates = search_crystal_system(
                observed,
                system,
                wavelength_angstrom=wavelength,
                **kwargs,
            )
            elapsed = time.monotonic() - t0
            times.append(elapsed)
            n_ok += 1

            cand_niggli = [c.niggli_params6() for c in candidates]
            hit_rank = None
            for rank, cparams in enumerate(cand_niggli):
                if lattice_match_elementwise(cparams, truth_niggli, ltol=args.ltol, atol_deg=args.atol_deg):
                    hit_rank = rank
                    break
            for k in TOP_KS:
                if hit_rank is not None and hit_rank < k:
                    recall_counts[k] += 1

            per_sample_records.append(
                {
                    "truth": truth_niggli,
                    "n_observed_peaks": int(observed.shape[0]),
                    "n_candidates": len(candidates),
                    "hit_rank": hit_rank,
                    "elapsed_s": elapsed,
                }
            )

        n_total = max(n_ok, 1)
        report["systems"][system] = {
            "n_tested": n_ok,
            "recall_at_k": {str(k): recall_counts[k] / n_total for k in TOP_KS},
            "mean_search_time_s": float(np.mean(times)) if times else None,
            "max_search_time_s": float(np.max(times)) if times else None,
            "per_sample": per_sample_records,
        }
        print(
            f"[{system}] n={n_ok} "
            + " ".join(f"recall@{k}={recall_counts[k] / n_total:.2f}" for k in TOP_KS)
            + f" mean_time={np.mean(times):.2f}s"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument(
        "--systems",
        type=str,
        default="cubic,tetragonal,hexagonal,orthorhombic",
    )
    parser.add_argument("--n-peaks", type=int, default=15, help="Observed low-angle peak count")
    parser.add_argument("--n-peaks-min", type=int, default=8)
    parser.add_argument("--two-theta-min", type=float, default=5.0)
    parser.add_argument("--two-theta-max", type=float, default=90.0)
    parser.add_argument("--match-fraction-min", type=float, default=0.95)
    parser.add_argument("--ltol", type=float, default=0.05)
    parser.add_argument("--atol-deg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
