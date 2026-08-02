#!/usr/bin/env python3
"""B1-S2 hypothesis probes (test-first; does NOT change production search).

Hypotheses
----------
A. CS-label ≠ Niggli geometry is the main failure driver.
   → Compare q-search recall on geometry-consistent vs mismatch subsets
     (both routed with the *label* CS, i.e. current B1-S2 protocol).

B. Axial extinctions break sequential solve even when geometry matches.
   → On geometry-consistent samples, compare axial_ok==3 vs axial_ok<3.

C. Routing by Niggli geometry (not label) is enough to recover mismatch cases.
   → On mismatch samples, compare label-CS search vs geom-CS search.

Usage
-----
    python scripts/diag_b1s2_hypotheses.py \
        --config configs/scale_100k_a3_g1_gstar6.yaml \
        --n-scan 120 --n-probe 20 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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
    inverse_d2_from_two_theta_f64,
    search_crystal_system,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "diag_b1s2_hypotheses.json"


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


def _axial_ok(truth_params: list[float], q_obs: np.ndarray, tol: float = 1e-5) -> int:
    """Count how many of G11/G22/G33 appear as (100)/(010)/(001) or (200)/…/(00l) l<=4."""
    matrix = lattice_params_to_matrix(torch.tensor(truth_params, dtype=torch.float64)).numpy()
    gstar = np.linalg.inv(matrix @ matrix.T)
    ok = 0
    for gii in (gstar[0, 0], gstar[1, 1], gstar[2, 2]):
        found = False
        for h in (1, 2, 3, 4):
            if float(np.min(np.abs(q_obs - gii * h * h))) <= tol:
                found = True
                break
        ok += int(found)
    return ok


def _hit_rank(cands, truth_niggli: list[float], *, ltol: float, atol_deg: float) -> int | None:
    for rank, cand in enumerate(cands):
        if lattice_match_elementwise(cand.niggli_params6(), truth_niggli, ltol=ltol, atol_deg=atol_deg):
            return rank
    return None


def _rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return float(np.mean(flags))


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)

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

    target_labels = ("orthorhombic", "monoclinic")
    # Collect up to n_scan per label for stratification census.
    census: dict[str, list[dict[str, Any]]] = {cs: [] for cs in target_labels}

    print("=== Pass 1: stratify valid1400 ortho/mono (no search) ===", flush=True)
    with torch.no_grad():
        for batch in loader:
            if all(len(census[cs]) >= args.n_scan for cs in target_labels):
                break
            for i in range(batch["lattice"].shape[0]):
                label = CRYSTAL_SYSTEMS[int(batch["crystal_system_idx"][i].item())]
                if label not in target_labels or len(census[label]) >= args.n_scan:
                    continue
                truth = batch["lattice"][i].cpu().numpy().tolist()
                tn = _niggli(truth)
                geom = _geom_system(tn, tol_deg=args.angle_tol_deg)
                obs = slice_observed_two_theta(batch["pxrd_x"], batch["peak_num"], i)
                obs = np.asarray(obs, dtype=np.float64)
                obs = obs[np.isfinite(obs)]
                q = inverse_d2_from_two_theta_f64(obs, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM)
                axial = _axial_ok(truth, q, tol=args.axial_q_tol)
                census[label].append(
                    {
                        "label": label,
                        "geom": geom,
                        "consistent": label == geom,
                        "axial_ok": axial,
                        "n_peaks": int(obs.shape[0]),
                        "truth": truth,
                        "truth_niggli": tn,
                        "obs": obs,
                    }
                )
            if all(len(census[cs]) >= args.n_scan for cs in target_labels):
                break

    # Census tables
    census_summary: dict[str, Any] = {}
    for label, rows in census.items():
        n = len(rows)
        n_cons = sum(1 for r in rows if r["consistent"])
        by_geom = defaultdict(int)
        by_axial_cons = defaultdict(int)
        by_axial_all = defaultdict(int)
        for r in rows:
            by_geom[r["geom"]] += 1
            by_axial_all[r["axial_ok"]] += 1
            if r["consistent"]:
                by_axial_cons[r["axial_ok"]] += 1
        census_summary[label] = {
            "n": n,
            "consistent_frac": n_cons / n if n else None,
            "by_geom": dict(by_geom),
            "axial_ok_all": {str(k): by_axial_all[k] for k in sorted(by_axial_all)},
            "axial_ok_consistent_only": {str(k): by_axial_cons[k] for k in sorted(by_axial_cons)},
        }
        print(
            f"[{label}] n={n} consistent={n_cons}/{n} ({100 * n_cons / n:.1f}%) "
            f"by_geom={dict(by_geom)} axial_cons={dict(by_axial_cons)}",
            flush=True,
        )

    def _pick(rows: list[dict], predicate, limit: int) -> list[dict]:
        out = [r for r in rows if predicate(r)]
        return out[:limit]

    # Probe sets (shared pool; each sample searched at most once per CS).
    probe_rows: list[dict] = []
    for label in target_labels:
        rows = census[label]
        probe_rows.extend(_pick(rows, lambda r: r["consistent"], args.n_probe))
        probe_rows.extend(_pick(rows, lambda r: not r["consistent"], args.n_probe))

    # Dedup by id(obs) — each row is unique object from census.
    seen_ids: set[int] = set()
    unique_probes: list[dict] = []
    for r in probe_rows:
        rid = id(r)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        unique_probes.append(r)

    print(
        f"\n=== Pass 2: q-search on {len(unique_probes)} probe samples "
        f"(label CS + geom CS if different) ===",
        flush=True,
    )

    search_cache: dict[tuple[int, str], dict[str, Any]] = {}
    t0 = time.time()
    for idx, row in enumerate(unique_probes):
        systems = [row["label"]]
        if row["geom"] != row["label"]:
            systems.append(row["geom"])
        for sys in systems:
            key = (id(row), sys)
            if key in search_cache:
                continue
            kwargs = dict(DEFAULT_SEARCH_KWARGS.get(sys, {}))
            if args.time_scale != 1.0 and "time_budget_s" in kwargs:
                kwargs["time_budget_s"] = float(kwargs["time_budget_s"]) * args.time_scale
            kwargs["pool_budget"] = max(int(kwargs.get("pool_budget", 30)), 100)
            st = time.time()
            cands = search_crystal_system(
                row["obs"], sys, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM, **kwargs
            )
            elapsed = time.time() - st
            hit = _hit_rank(cands, row["truth_niggli"], ltol=args.ltol, atol_deg=args.atol_deg)
            search_cache[key] = {
                "n_cand": len(cands),
                "empty": len(cands) == 0,
                "hit_rank": hit,
                "hit20": hit is not None and hit < 20,
                "top_matched": int(cands[0].n_matched) if cands else 0,
                "elapsed_s": elapsed,
            }
        done = idx + 1
        if done % 5 == 0 or done == len(unique_probes):
            print(
                f"... {done}/{len(unique_probes)} samples, "
                f"wall={time.time() - t0:.0f}s cache={len(search_cache)}",
                flush=True,
            )

    def _arm_stats(rows: list[dict], route: str) -> dict[str, Any]:
        """route: 'label' | 'geom'."""
        hits20: list[bool] = []
        empty: list[bool] = []
        times: list[float] = []
        for r in rows:
            sys = r["label"] if route == "label" else r["geom"]
            rec = search_cache.get((id(r), sys))
            if rec is None:
                continue
            hits20.append(bool(rec["hit20"]))
            empty.append(bool(rec["empty"]))
            times.append(float(rec["elapsed_s"]))
        return {
            "n": len(hits20),
            "recall20": _rate(hits20),
            "empty_rate": _rate(empty),
            "mean_time_s": float(np.mean(times)) if times else None,
        }

    # Build strata from unique_probes that actually have label-route cache.
    all_probed = [r for r in unique_probes if (id(r), r["label"]) in search_cache]
    cons = [r for r in all_probed if r["consistent"]]
    mism = [r for r in all_probed if not r["consistent"]]
    cons_ax3 = [r for r in cons if r["axial_ok"] == 3]
    cons_axlt = [r for r in cons if r["axial_ok"] < 3]

    hyp_a = {
        "consistent_label_route": _arm_stats(cons, "label"),
        "mismatch_label_route": _arm_stats(mism, "label"),
    }
    hyp_b = {
        "consistent_axial3_label_route": _arm_stats(cons_ax3, "label"),
        "consistent_axial_lt3_label_route": _arm_stats(cons_axlt, "label"),
    }
    hyp_c = {
        "mismatch_label_route": _arm_stats(mism, "label"),
        "mismatch_geom_route": _arm_stats(mism, "geom"),
    }

    # Per-label breakdown for A
    hyp_a_by_label: dict[str, Any] = {}
    for label in target_labels:
        hyp_a_by_label[label] = {
            "consistent": _arm_stats([r for r in cons if r["label"] == label], "label"),
            "mismatch": _arm_stats([r for r in mism if r["label"] == label], "label"),
        }

    report = {
        "protocol": {
            "ltol": args.ltol,
            "atol_deg": args.atol_deg,
            "angle_tol_deg": args.angle_tol_deg,
            "axial_q_tol": args.axial_q_tol,
            "n_scan": args.n_scan,
            "n_probe": args.n_probe,
            "time_scale": args.time_scale,
            "note": "Production search code unchanged; DEFAULT_SEARCH_KWARGS used as-is.",
        },
        "census": census_summary,
        "hypothesis_A": hyp_a,
        "hypothesis_A_by_label": hyp_a_by_label,
        "hypothesis_B": hyp_b,
        "hypothesis_C": hyp_c,
        "n_unique_probes": len(unique_probes),
        "wall_time_s": time.time() - t0,
    }

    print("\n=== Hypothesis A (label route): consistent vs mismatch ===", flush=True)
    for k, v in hyp_a.items():
        print(
            f"  {k}: n={v['n']} recall@20={_pct(v['recall20'])} empty={_pct(v['empty_rate'])}",
            flush=True,
        )
    print("\n=== Hypothesis B (consistent only): axial==3 vs <3 ===", flush=True)
    for k, v in hyp_b.items():
        print(
            f"  {k}: n={v['n']} recall@20={_pct(v['recall20'])} empty={_pct(v['empty_rate'])}",
            flush=True,
        )
    print("\n=== Hypothesis C (mismatch only): label vs geom route ===", flush=True)
    for k, v in hyp_c.items():
        print(
            f"  {k}: n={v['n']} recall@20={_pct(v['recall20'])} empty={_pct(v['empty_rate'])}",
            flush=True,
        )
    return report


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/scale_100k_a3_g1_gstar6.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-scan", type=int, default=120, help="Max samples/label for census")
    p.add_argument("--n-probe", type=int, default=20, help="Max probes/label/stratum for search")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    p.add_argument("--angle-tol-deg", type=float, default=1.0)
    p.add_argument("--axial-q-tol", type=float, default=1e-5)
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiply per-system time_budget_s (1.0 = production defaults)",
    )
    p.add_argument("--device", type=str, default="cpu", help="Unused; kept for CLI symmetry")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Drop non-JSON-serializable obs arrays from any accidental leak — report has none.
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
