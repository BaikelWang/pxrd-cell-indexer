#!/usr/bin/env python3
"""P0 probe: why does consistent ∧ axial_ok==3 still fail?

Only samples where label CS == Niggli geometry AND all three axial Gii
appear in the observed q-list (h=1..4). Production search unchanged.

Failure taxonomy (mutually exclusive, checked in order):
  hit20          – true cell in Top-20 (success)
  empty          – zero candidates kept
  pool_miss      – candidates >0 but none match truth (any rank)
  rank_gt20      – truth in pool but rank >= 20

Extra diagnostics on empty / pool_miss:
  - truth_match_frac: fraction of obs peaks explained by true G* @ 1e-6
  - min_matched required by DEFAULT match_fraction_min
  - whether true (G11,G22,G33) appear in axial option lists under
    production n_axial_peaks=min(12, n_peaks) vs all-peaks window
  - n_peaks, cell lengths, time_budget hit (elapsed ≈ budget)

Usage:
    python scripts/diag_b1s2_best_stratum.py \
        --config configs/scale_100k_a3_g1_gstar6.yaml \
        --n-per-label 25
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
    _fast_match_count,
    inverse_d2_from_two_theta_f64,
    search_crystal_system,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "diag_b1s2_best_stratum.json"


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
        found = any(float(np.min(np.abs(q_obs - gii * h * h))) <= tol for h in (1, 2, 3, 4))
        ok += int(found)
    return ok


def _diag_in_axial_opts(
    q_obs: np.ndarray,
    gstar: np.ndarray,
    *,
    n_axial_peaks: int,
    axial_max_index: int,
    tol: float = 1e-6,
) -> dict[str, bool]:
    axial_idx = _axial_index_pool(axial_max_index)
    n = min(n_axial_peaks, int(q_obs.shape[0]))
    opts = {
        "G11": [(i, h, float(q_obs[i] / (h * h))) for i in range(n) for h in axial_idx],
        "G22": [(i, k, float(q_obs[i] / (k * k))) for i in range(n) for k in axial_idx],
        "G33": [(i, l, float(q_obs[i] / (l * l))) for i in range(n) for l in axial_idx],
    }
    targets = {"G11": gstar[0, 0], "G22": gstar[1, 1], "G33": gstar[2, 2]}
    return {name: any(abs(g - targets[name]) <= tol for _, _, g in opts[name]) for name in targets}


def _hit_rank(cands, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> int | None:
    for rank, cand in enumerate(cands):
        if lattice_match_elementwise(cand.niggli_params6(), truth_niggli, ltol=ltol, atol_deg=atol_deg):
            return rank
    return None


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

    print("=== Collect consistent ∧ axial_ok==3 ===", flush=True)
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
                stratum[label].append({"label": label, "truth": truth, "truth_niggli": tn, "obs": obs, "q": q, "gstar": gstar})
            if all(len(stratum[cs]) >= args.n_per_label for cs in labels):
                break

    for cs in labels:
        print(f"[{cs}] stratum n={len(stratum[cs])}", flush=True)

    records: list[dict[str, Any]] = []
    t0 = time.time()
    all_rows = stratum["orthorhombic"] + stratum["monoclinic"]
    for idx, row in enumerate(all_rows):
        label = row["label"]
        kwargs = dict(DEFAULT_SEARCH_KWARGS.get(label, {}))
        kwargs["pool_budget"] = max(int(kwargs.get("pool_budget", 30)), 100)
        budget = float(kwargs.get("time_budget_s", 20.0))
        match_frac = float(kwargs.get("match_fraction_min", 0.95))
        n_peaks = int(row["obs"].shape[0])
        min_matched = int(np.ceil(match_frac * n_peaks))
        truth_m = _fast_match_count(row["q"], row["gstar"], q_match_abs_tol=1e-6)

        n_low = int(kwargs.get("n_low_peaks", 8))
        n_axial_prod = max(n_low, min(n_peaks, 12))
        sparse = int(kwargs.get("sparse_hkl_index", 6))
        axial_max = max(sparse, 8)  # search_crystal_system uses max(sparse, 8)
        in_prod = _diag_in_axial_opts(row["q"], row["gstar"], n_axial_peaks=n_axial_prod, axial_max_index=axial_max)
        in_all = _diag_in_axial_opts(row["q"], row["gstar"], n_axial_peaks=n_peaks, axial_max_index=axial_max)

        st = time.time()
        cands = search_crystal_system(
            row["obs"], label, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM, **kwargs
        )
        elapsed = time.time() - st
        hit = _hit_rank(cands, row["truth_niggli"], ltol=args.ltol, atol_deg=args.atol_deg)

        if hit is not None and hit < 20:
            status = "hit20"
        elif len(cands) == 0:
            status = "empty"
        elif hit is None:
            status = "pool_miss"
        else:
            status = "rank_gt20"

        # Best candidate proximity (lengths only) when pool non-empty
        best_len_rel = None
        if cands:
            cn = cands[0].niggli_params6()
            tn = row["truth_niggli"]
            best_len_rel = float(np.mean([abs(cn[k] - tn[k]) / max(tn[k], 1e-6) for k in range(3)]))

        rec = {
            "label": label,
            "status": status,
            "n_peaks": n_peaks,
            "truth_matched": int(truth_m),
            "truth_match_frac": float(truth_m / max(n_peaks, 1)),
            "min_matched_required": min_matched,
            "n_cand": len(cands),
            "hit_rank": hit,
            "top_matched": int(cands[0].n_matched) if cands else 0,
            "elapsed_s": elapsed,
            "time_budget_s": budget,
            "budget_exhausted": elapsed >= budget * 0.95,
            "diag_in_prod_window": in_prod,
            "diag_in_all_peaks": in_all,
            "missing_diag_in_prod": [k for k, v in in_prod.items() if not v],
            "missing_diag_in_all": [k for k, v in in_all.items() if not v],
            "best_len_rel_mae": best_len_rel,
            "cell_abc": [round(row["truth_niggli"][k], 3) for k in range(3)],
            "anisotropy": float(max(row["truth_niggli"][:3]) / max(min(row["truth_niggli"][:3]), 1e-6)),
        }
        records.append(rec)
        print(
            f"{idx + 1:02d}/{len(all_rows)} {label[:4]} {status:10s} "
            f"peaks={n_peaks} truth_m={truth_m}/{n_peaks} n_cand={len(cands)} "
            f"miss_prod={rec['missing_diag_in_prod']} t={elapsed:.1f}/{budget:.0f} "
            f"abc={rec['cell_abc']}",
            flush=True,
        )

    # Aggregate
    def _agg(rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        statuses = [r["status"] for r in rows]
        counts = {s: statuses.count(s) for s in ("hit20", "empty", "pool_miss", "rank_gt20")}
        empties = [r for r in rows if r["status"] == "empty"]
        misses = [r for r in rows if r["status"] == "pool_miss"]
        return {
            "n": len(rows),
            "status_counts": counts,
            "recall20": counts.get("hit20", 0) / len(rows),
            "empty_rate": counts.get("empty", 0) / len(rows),
            "among_empty": {
                "n": len(empties),
                "frac_missing_any_diag_prod": (
                    float(np.mean([len(r["missing_diag_in_prod"]) > 0 for r in empties])) if empties else None
                ),
                "frac_missing_any_diag_all": (
                    float(np.mean([len(r["missing_diag_in_all"]) > 0 for r in empties])) if empties else None
                ),
                "frac_budget_exhausted": (
                    float(np.mean([r["budget_exhausted"] for r in empties])) if empties else None
                ),
                "mean_n_peaks": float(np.mean([r["n_peaks"] for r in empties])) if empties else None,
                "mean_anisotropy": float(np.mean([r["anisotropy"] for r in empties])) if empties else None,
                "mean_truth_match_frac": float(np.mean([r["truth_match_frac"] for r in empties])) if empties else None,
            },
            "among_pool_miss": {
                "n": len(misses),
                "mean_n_cand": float(np.mean([r["n_cand"] for r in misses])) if misses else None,
                "mean_top_matched": float(np.mean([r["top_matched"] for r in misses])) if misses else None,
                "mean_best_len_rel_mae": (
                    float(np.mean([r["best_len_rel_mae"] for r in misses if r["best_len_rel_mae"] is not None]))
                    if misses
                    else None
                ),
                "frac_budget_exhausted": (
                    float(np.mean([r["budget_exhausted"] for r in misses])) if misses else None
                ),
            },
        }

    report = {
        "protocol": {
            "ltol": args.ltol,
            "atol_deg": args.atol_deg,
            "n_per_label": args.n_per_label,
            "stratum": "label==geom AND axial_ok==3",
            "note": "Production search unchanged.",
        },
        "overall": _agg(records),
        "by_label": {cs: _agg([r for r in records if r["label"] == cs]) for cs in labels},
        "records": [{k: v for k, v in r.items() if k not in ("obs", "q", "gstar")} for r in records],
        "wall_time_s": time.time() - t0,
    }

    print("\n=== SUMMARY ===", flush=True)
    for name, block in (("overall", report["overall"]), ("orthorhombic", report["by_label"]["orthorhombic"]), ("monoclinic", report["by_label"]["monoclinic"])):
        print(
            f"[{name}] n={block.get('n')} recall@20={block.get('recall20')} "
            f"counts={block.get('status_counts')}",
            flush=True,
        )
        if block.get("among_empty", {}).get("n"):
            print(f"  empty detail: {block['among_empty']}", flush=True)
        if block.get("among_pool_miss", {}).get("n"):
            print(f"  pool_miss detail: {block['among_pool_miss']}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/scale_100k_a3_g1_gstar6.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-label", type=int, default=25)
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
