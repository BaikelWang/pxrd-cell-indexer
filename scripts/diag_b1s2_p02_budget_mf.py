#!/usr/bin/env python3
"""P0.2: 2×2 probe on best stratum — time_budget × match_fraction_min.

Stratum: label CS == Niggli geometry AND axial_ok==3.
Production search code unchanged; kwargs overridden per arm only.

Arms
----
A  mf=0.95, time_scale=1.0   (production-like)
B  mf=0.80, time_scale=1.0
C  mf=0.95, time_scale=2.0
D  mf=0.80, time_scale=2.0

Also reports the subset that is empty under arm A *and* has all three
diagonals inside the production axial window (≤12) — the residual puzzle
from P0.

Usage:
    python scripts/diag_b1s2_p02_budget_mf.py \
        --config configs/scale_100k_a3_g1_gstar6.yaml \
        --n-per-label 20
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
from pxrd_cell_indexing.data.dataset import PeakFilterConfig, PXRDDatasetConfig, build_dataloader
from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.fom import slice_observed_two_theta
from pxrd_cell_indexing.search.qsearch import (
    DEFAULT_SEARCH_KWARGS,
    DEFAULT_WAVELENGTH_ANGSTROM,
    _axial_index_pool,
    inverse_d2_from_two_theta_f64,
    search_crystal_system,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "diag_b1s2_p02_budget_mf.json"

ARMS = (
    ("A_prod", 0.95, 1.0),
    ("B_mf080", 0.80, 1.0),
    ("C_tb2x", 0.95, 2.0),
    ("D_mf080_tb2x", 0.80, 2.0),
)


def _niggli(params6: list[float]) -> list[float]:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    return canonicalize_lattice(matrix, convention="niggli").as_params6()


def _geom_system(tn: list[float], tol_deg: float = 1.0) -> str:
    n90 = sum(abs(tn[k] - 90.0) <= tol_deg for k in (3, 4, 5))
    if n90 == 3:
        return "orthorhombic"
    if n90 == 2:
        return "monoclinic"
    return "triclinic"


def _gstar(truth: list[float]) -> np.ndarray:
    matrix = lattice_params_to_matrix(torch.tensor(truth, dtype=torch.float64)).numpy()
    return np.linalg.inv(matrix @ matrix.T)


def _axial_ok(gstar: np.ndarray, q_obs: np.ndarray, tol: float = 1e-5) -> int:
    ok = 0
    for gii in (gstar[0, 0], gstar[1, 1], gstar[2, 2]):
        if any(float(np.min(np.abs(q_obs - gii * h * h))) <= tol for h in (1, 2, 3, 4)):
            ok += 1
    return ok


def _diags_in_prod_window(q_obs: np.ndarray, gstar: np.ndarray, label: str, tol: float = 1e-6) -> bool:
    kwargs = DEFAULT_SEARCH_KWARGS.get(label, {})
    n_low = int(kwargs.get("n_low_peaks", 8))
    n_peaks = int(q_obs.shape[0])
    n_axial = max(n_low, min(n_peaks, 12))
    sparse = int(kwargs.get("sparse_hkl_index", 6))
    axial_max = max(sparse, 8)
    axial_idx = _axial_index_pool(axial_max)
    targets = (gstar[0, 0], gstar[1, 1], gstar[2, 2])
    for gii in targets:
        found = False
        for i in range(n_axial):
            for h in axial_idx:
                if abs(float(q_obs[i] / (h * h)) - gii) <= tol:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _hit20(cands, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> bool:
    for rank, cand in enumerate(cands):
        if rank >= 20:
            break
        if lattice_match_elementwise(cand.niggli_params6(), truth_niggli, ltol=ltol, atol_deg=atol_deg):
            return True
    return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    ds = PXRDDatasetConfig(
        lmdb_path=Path(config.data.valid_lmdb),
        split="valid",
        sample_list_path=Path(config.data.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(ds, batch_size=config.data.batch_size, num_workers=0, shuffle=False, pin_memory=False)

    labels = ("orthorhombic", "monoclinic")
    stratum: dict[str, list[dict[str, Any]]] = {cs: [] for cs in labels}

    print("=== Collect best stratum ===", flush=True)
    with torch.no_grad():
        for batch in loader:
            if all(len(stratum[cs]) >= args.n_per_label for cs in labels):
                break
            for i in range(batch["lattice"].shape[0]):
                label = CRYSTAL_SYSTEMS[int(batch["crystal_system_idx"][i].item())]
                if label not in labels or len(stratum[label]) >= args.n_per_label:
                    continue
                truth = batch["lattice"][i].cpu().numpy().tolist()
                tn = _niggli(truth)
                if _geom_system(tn, tol_deg=args.angle_tol_deg) != label:
                    continue
                obs = np.asarray(slice_observed_two_theta(batch["pxrd_x"], batch["peak_num"], i), dtype=np.float64)
                obs = obs[np.isfinite(obs)]
                q = inverse_d2_from_two_theta_f64(obs, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM)
                gstar = _gstar(truth)
                if _axial_ok(gstar, q, tol=args.axial_q_tol) < 3:
                    continue
                stratum[label].append(
                    {
                        "label": label,
                        "truth_niggli": tn,
                        "obs": obs,
                        "q": q,
                        "gstar": gstar,
                        "diags_in_prod_window": _diags_in_prod_window(q, gstar, label),
                    }
                )
            if all(len(stratum[cs]) >= args.n_per_label for cs in labels):
                break

    rows = stratum["orthorhombic"] + stratum["monoclinic"]
    print(
        f"n={len(rows)} "
        f"diags_in_window={sum(1 for r in rows if r['diags_in_prod_window'])}/"
        f"{len(rows)}",
        flush=True,
    )

    # results[arm][sample_idx] = dict
    per_arm: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in ARMS}
    t0 = time.time()

    for arm_i, (arm_name, mf, tscale) in enumerate(ARMS):
        print(f"\n=== Arm {arm_name} (mf={mf}, time_scale={tscale}) ===", flush=True)
        for j, row in enumerate(rows):
            label = row["label"]
            kwargs = dict(DEFAULT_SEARCH_KWARGS.get(label, {}))
            kwargs["match_fraction_min"] = mf
            kwargs["pool_budget"] = max(int(kwargs.get("pool_budget", 30)), 100)
            base_tb = float(kwargs.get("time_budget_s", 20.0))
            kwargs["time_budget_s"] = base_tb * tscale
            st = time.time()
            cands = search_crystal_system(
                row["obs"], label, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM, **kwargs
            )
            elapsed = time.time() - st
            hit = _hit20(cands, row["truth_niggli"], ltol=args.ltol, atol_deg=args.atol_deg)
            empty = len(cands) == 0
            per_arm[arm_name].append(
                {
                    "label": label,
                    "hit20": hit,
                    "empty": empty,
                    "n_cand": len(cands),
                    "elapsed_s": elapsed,
                    "budget_s": kwargs["time_budget_s"],
                    "budget_exhausted": elapsed >= kwargs["time_budget_s"] * 0.95,
                    "diags_in_prod_window": row["diags_in_prod_window"],
                }
            )
            if (j + 1) % 5 == 0 or j + 1 == len(rows):
                print(
                    f"  [{arm_name}] {j + 1}/{len(rows)} "
                    f"hit_so_far={sum(1 for x in per_arm[arm_name] if x['hit20'])} "
                    f"empty_so_far={sum(1 for x in per_arm[arm_name] if x['empty'])} "
                    f"wall={time.time() - t0:.0f}s",
                    flush=True,
                )

    def _summarize(recs: list[dict[str, Any]], pred=None) -> dict[str, Any]:
        use = [r for r in recs if pred is None or pred(r)]
        if not use:
            return {"n": 0}
        return {
            "n": len(use),
            "recall20": float(np.mean([r["hit20"] for r in use])),
            "empty_rate": float(np.mean([r["empty"] for r in use])),
            "budget_exhausted_rate": float(np.mean([r["budget_exhausted"] for r in use])),
            "mean_time_s": float(np.mean([r["elapsed_s"] for r in use])),
        }

    # Index of samples empty under A with diags in window
    a_recs = per_arm["A_prod"]
    residual_idx = [
        i
        for i, r in enumerate(a_recs)
        if r["empty"] and r["diags_in_prod_window"]
    ]

    arm_tables: dict[str, Any] = {}
    for arm_name, mf, tscale in ARMS:
        recs = per_arm[arm_name]
        arm_tables[arm_name] = {
            "mf": mf,
            "time_scale": tscale,
            "overall": _summarize(recs),
            "orthorhombic": _summarize(recs, lambda r: r["label"] == "orthorhombic"),
            "monoclinic": _summarize(recs, lambda r: r["label"] == "monoclinic"),
            "residual_emptyA_diags_in_window": _summarize(
                [recs[i] for i in residual_idx]
            ),
        }

    # Pairwise deltas vs A
    base = arm_tables["A_prod"]["overall"]
    deltas = {}
    for arm_name, _, _ in ARMS:
        if arm_name == "A_prod":
            continue
        cur = arm_tables[arm_name]["overall"]
        deltas[arm_name] = {
            "delta_recall20": cur["recall20"] - base["recall20"],
            "delta_empty_rate": cur["empty_rate"] - base["empty_rate"],
        }
        rb = arm_tables[arm_name]["residual_emptyA_diags_in_window"]
        ra = arm_tables["A_prod"]["residual_emptyA_diags_in_window"]
        if ra.get("n", 0) and rb.get("n", 0):
            deltas[arm_name]["residual_recall20"] = rb.get("recall20")
            deltas[arm_name]["residual_empty_rate"] = rb.get("empty_rate")
            deltas[arm_name]["residual_delta_recall20"] = (
                (rb["recall20"] - ra["recall20"]) if ra.get("recall20") is not None else None
            )

    report = {
        "protocol": {
            "ltol": args.ltol,
            "atol_deg": args.atol_deg,
            "n_per_label": args.n_per_label,
            "stratum": "label==geom AND axial_ok==3",
            "arms": [{"name": n, "mf": mf, "time_scale": ts} for n, mf, ts in ARMS],
            "note": "Production search defaults overridden per-arm only.",
        },
        "n_samples": len(rows),
        "n_residual_emptyA_diags_in_window": len(residual_idx),
        "arms": arm_tables,
        "deltas_vs_A": deltas,
        "wall_time_s": time.time() - t0,
    }

    print("\n=== 2×2 SUMMARY (overall) ===", flush=True)
    for arm_name, mf, tscale in ARMS:
        o = arm_tables[arm_name]["overall"]
        print(
            f"  {arm_name}: recall@20={o['recall20']*100:.1f}% "
            f"empty={o['empty_rate']*100:.1f}% "
            f"budget_exh={o['budget_exhausted_rate']*100:.1f}% "
            f"n={o['n']}",
            flush=True,
        )
    print("\n=== Residual (empty under A ∧ diags in window) ===", flush=True)
    print(f"  n_residual={len(residual_idx)}", flush=True)
    for arm_name, _, _ in ARMS:
        r = arm_tables[arm_name]["residual_emptyA_diags_in_window"]
        if r.get("n"):
            print(
                f"  {arm_name}: recall@20={r['recall20']*100:.1f}% "
                f"empty={r['empty_rate']*100:.1f}%",
                flush=True,
            )
    print("\n=== Deltas vs A ===", flush=True)
    for k, v in deltas.items():
        print(f"  {k}: {v}", flush=True)

    # Decision hint
    dr_b = deltas.get("B_mf080", {}).get("delta_recall20", 0) or 0
    dr_c = deltas.get("C_tb2x", {}).get("delta_recall20", 0) or 0
    dr_d = deltas.get("D_mf080_tb2x", {}).get("delta_recall20", 0) or 0
    rr_b = deltas.get("B_mf080", {}).get("residual_recall20")
    rr_c = deltas.get("C_tb2x", {}).get("residual_recall20")
    rr_d = deltas.get("D_mf080_tb2x", {}).get("residual_recall20")
    hint = []
    if dr_c >= 0.08 or (rr_c is not None and rr_c >= 0.25):
        hint.append("budget-sensitive")
    if dr_b >= 0.08 or (rr_b is not None and rr_b >= 0.25):
        hint.append("match_fraction-sensitive")
    if dr_d >= max(dr_b, dr_c) + 0.05:
        hint.append("needs both")
    if max(dr_b, dr_c, dr_d) < 0.05 and (rr_d is None or rr_d < 0.15):
        hint.append("logic/seed-limited (kwargs won't save)")
    report["auto_hint"] = hint or ["inconclusive"]
    print(f"\nauto_hint: {report['auto_hint']}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/scale_100k_a3_g1_gstar6.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-label", type=int, default=20)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--angle-tol-deg", type=float, default=1.0)
    p.add_argument("--axial-q-tol", type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
