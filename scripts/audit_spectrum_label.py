#!/usr/bin/env python3
"""Stage-0 audit: stored PXRD vs primitive / conventional-reduced simulations.

Quantifies whether LMDB peaks match conventional→reduced simulation and how
far primitive / conventional / reduced lattice labels diverge.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAVELENGTH = 1.54184
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 80.0
INTENSITY_MIN = 5.0
SYMPREC = 0.01


def _filter_peaks(two_theta: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = intensity > INTENSITY_MIN
    return two_theta[mask].astype(np.float64), intensity[mask].astype(np.float64)


def _simulate(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    pattern = XRDCalculator(wavelength=DEFAULT_WAVELENGTH).get_pattern(
        structure,
        scaled=True,
        two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX),
    )
    return _filter_peaks(np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64))


def _inverse_d2(two_theta: np.ndarray, wavelength: float = DEFAULT_WAVELENGTH) -> np.ndarray:
    theta = np.deg2rad(two_theta) / 2.0
    d = wavelength / (2.0 * np.sin(np.clip(theta, 1e-8, None)))
    return 1.0 / np.clip(d, 1e-8, None) ** 2


def _match_rate(obs: np.ndarray, theory: np.ndarray, atol: float = 0.05) -> float:
    if obs.size == 0 or theory.size == 0:
        return float("nan")
    diffs = np.abs(obs[:, None] - theory[None, :])
    return float(np.mean(np.min(diffs, axis=1) <= atol))


def _hungarian_mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Greedy nearest-neighbor mean abs distance on sorted 1D arrays."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    aa = np.sort(a)
    bb = np.sort(b)
    # Match shorter to longer by nearest neighbor.
    if aa.size > bb.size:
        aa, bb = bb, aa
    used = np.zeros(bb.size, dtype=bool)
    errs = []
    for x in aa:
        dists = np.abs(bb - x)
        dists[used] = np.inf
        j = int(np.argmin(dists))
        if not np.isfinite(dists[j]):
            break
        used[j] = True
        errs.append(float(dists[j]))
    return float(np.mean(errs)) if errs else float("nan")


def _lattice_params(structure: Structure) -> np.ndarray:
    lat = structure.lattice
    return np.array([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma], dtype=np.float64)


def _volume(params: np.ndarray) -> float:
    a, b, c, alpha, beta, gamma = params
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    return float(a * b * c * np.sqrt(max(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0.0)))


def _load_records(jsonl: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=PROJECT_ROOT / "data/processed/overfit700_seed42.jsonl")
    parser.add_argument(
        "--lmdb",
        type=Path,
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/beat_engine/raw_diag/spectrum_label_audit.json",
    )
    args = parser.parse_args()

    records = _load_records(args.jsonl, args.n, args.seed)
    env = lmdb.open(str(args.lmdb), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

    by_cs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overall: list[dict[str, Any]] = []

    for rec in records:
        raw = env.begin().get(str(rec["lmdb_key"]).encode("ascii"))
        if raw is None:
            continue
        data = pickle.loads(gzip.decompress(raw))
        stored_tt, stored_i = _filter_peaks(
            np.asarray(data["pxrd_x"], dtype=np.float64),
            np.asarray(data["pxrd_y"], dtype=np.float64),
        )
        lattice = Lattice(np.asarray(data["p_lattice_matrix"], dtype=np.float64))
        structure = Structure(lattice, list(data["p_atom_type"]), np.asarray(data["p_atom_pos"], dtype=np.float64))
        analyzer = SpacegroupAnalyzer(structure, symprec=SYMPREC)
        try:
            primitive = analyzer.find_primitive()
            conventional = analyzer.get_conventional_standard_structure()
            reduced = conventional.get_reduced_structure()
            crystal_system = analyzer.get_crystal_system()
        except Exception as exc:  # noqa: BLE001
            print(f"skip {rec['lmdb_key']}: {exc}", flush=True)
            continue

        prim_tt, _ = _simulate(primitive)
        conv_tt, _ = _simulate(conventional)
        red_tt, _ = _simulate(reduced)

        prim_params = _lattice_params(primitive)
        conv_params = _lattice_params(conventional)
        red_params = _lattice_params(reduced)
        label_params = np.array(
            [
                rec["lattice_a"],
                rec["lattice_b"],
                rec["lattice_c"],
                rec["lattice_alpha"],
                rec["lattice_beta"],
                rec["lattice_gamma"],
            ],
            dtype=np.float64,
        )

        stored_g = _inverse_d2(stored_tt)
        red_g = _inverse_d2(red_tt)
        prim_g = _inverse_d2(prim_tt)

        row = {
            "sample_id": rec["lmdb_key"],
            "crystal_system": crystal_system,
            "n_stored": int(stored_tt.size),
            "n_reduced": int(red_tt.size),
            "n_primitive": int(prim_tt.size),
            "n_conventional": int(conv_tt.size),
            "match_stored_vs_reduced_2theta": _match_rate(stored_tt, red_tt, atol=0.05),
            "match_stored_vs_primitive_2theta": _match_rate(stored_tt, prim_tt, atol=0.05),
            "match_stored_vs_conventional_2theta": _match_rate(stored_tt, conv_tt, atol=0.05),
            "mean_abs_2theta_stored_vs_reduced": _hungarian_mean_abs(stored_tt, red_tt),
            "mean_abs_inverse_d2_stored_vs_reduced": _hungarian_mean_abs(stored_g, red_g),
            "mean_abs_inverse_d2_stored_vs_primitive": _hungarian_mean_abs(stored_g, prim_g),
            "label_vs_primitive_abs_max": float(np.max(np.abs(label_params - prim_params))),
            "label_vs_conventional_abs_max": float(np.max(np.abs(label_params - conv_params))),
            "primitive_vs_conventional_abs_max": float(np.max(np.abs(prim_params - conv_params))),
            "primitive_vs_reduced_abs_max": float(np.max(np.abs(prim_params - red_params))),
            "log_volume_ratio_prim_over_conv": float(
                np.log(max(_volume(prim_params), 1e-12) / max(_volume(conv_params), 1e-12))
            ),
            "primitive_params": prim_params.tolist(),
            "conventional_params": conv_params.tolist(),
            "reduced_params": red_params.tolist(),
            "label_params": label_params.tolist(),
            "truth_orth_dev_primitive": float(np.mean(np.abs(prim_params[3:] - 90.0))),
            "truth_orth_dev_conventional": float(np.mean(np.abs(conv_params[3:] - 90.0))),
        }
        overall.append(row)
        by_cs[crystal_system].append(row)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}

        def mean(key: str) -> float:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if vals.size else float("nan")

        return {
            "n": len(rows),
            "match_stored_vs_reduced_2theta": mean("match_stored_vs_reduced_2theta"),
            "match_stored_vs_primitive_2theta": mean("match_stored_vs_primitive_2theta"),
            "match_stored_vs_conventional_2theta": mean("match_stored_vs_conventional_2theta"),
            "mean_abs_2theta_stored_vs_reduced": mean("mean_abs_2theta_stored_vs_reduced"),
            "mean_abs_inverse_d2_stored_vs_reduced": mean("mean_abs_inverse_d2_stored_vs_reduced"),
            "mean_abs_inverse_d2_stored_vs_primitive": mean("mean_abs_inverse_d2_stored_vs_primitive"),
            "label_vs_primitive_abs_max": mean("label_vs_primitive_abs_max"),
            "primitive_vs_conventional_abs_max": mean("primitive_vs_conventional_abs_max"),
            "abs_log_volume_ratio_prim_over_conv": float(
                np.mean(np.abs([r["log_volume_ratio_prim_over_conv"] for r in rows]))
            ),
            "frac_primitive_equals_label": float(
                np.mean([r["label_vs_primitive_abs_max"] < 1e-3 for r in rows])
            ),
            "frac_peak_count_equal_reduced": float(
                np.mean([r["n_stored"] == r["n_reduced"] for r in rows])
            ),
        }

    result = {
        "n_requested": args.n,
        "n_ok": len(overall),
        "jsonl": str(args.jsonl),
        "lmdb": str(args.lmdb),
        "overall": _agg(overall),
        "by_crystal_system": {cs: _agg(rows) for cs, rows in sorted(by_cs.items())},
        "samples": overall[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "by_crystal_system": result["by_crystal_system"]}, indent=2))


if __name__ == "__main__":
    main()
