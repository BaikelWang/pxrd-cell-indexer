#!/usr/bin/env python3
"""P0.3: seed-level trace for best-stratum empties (diags already in window).

For each residual sample, check without relying on full search success:
  1) true G11/G22/G33 present in axial option lists (prod window)
  2) exists a distinct-peak triple (i1,i2,i3) realizing (G11,G22,G33)
  3) true gvec (ortho zeros / mono true offdiag) passes SPD→params
  4) exact match count vs min_matched (mf=0.95)
  5) approx match count on confirm grid
  6) for monoclinic: whether true off-diagonal is recoverable from a zone peak

Does NOT modify production search. Optionally runs one production search to
confirm empty status.

Usage:
    python scripts/diag_b1s2_p03_seed_trace.py --n-per-label 20 --max-trace 12
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
    _approx_match_counts,
    _axial_index_pool,
    _fast_match_count,
    _hkl_pool,
    _offdiag_from_peak,
    _zone_hkl_pool,
    gstar_to_lattice_params,
    inverse_d2_from_two_theta_f64,
    search_crystal_system,
)
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "beat_engine" / "b1_search" / "diag_b1s2_p03_seed_trace.json"


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


def _axial_ok(gstar: np.ndarray, q: np.ndarray, tol: float = 1e-5) -> int:
    ok = 0
    for gii in (gstar[0, 0], gstar[1, 1], gstar[2, 2]):
        if any(float(np.min(np.abs(q - gii * h * h))) <= tol for h in (1, 2, 3, 4)):
            ok += 1
    return ok


def _build_opts(q: np.ndarray, n_axial: int, axial_max: int):
    idx = _axial_index_pool(axial_max)
    n = min(n_axial, len(q))
    return {
        "G11": [(i, h, float(q[i] / (h * h))) for i in range(n) for h in idx],
        "G22": [(i, k, float(q[i] / (k * k))) for i in range(n) for k in idx],
        "G33": [(i, l, float(q[i] / (l * l))) for i in range(n) for l in idx],
    }


def _find_opts(opts, gii: float, tol: float = 1e-6):
    return [t for t in opts if abs(t[2] - gii) <= tol]


def trace_one(row: dict[str, Any], *, ltol: float, atol_deg: float) -> dict[str, Any]:
    label = row["label"]
    q = row["q"]
    gstar = row["gstar"]
    obs = row["obs"]
    tn = row["truth_niggli"]
    n_peaks = len(q)
    kwargs = dict(DEFAULT_SEARCH_KWARGS.get(label, {}))
    mf = float(kwargs.get("match_fraction_min", 0.95))
    min_matched = int(np.ceil(mf * n_peaks))
    n_low = int(kwargs.get("n_low_peaks", 8))
    n_axial = max(n_low, min(n_peaks, 12))
    axial_max = max(int(kwargs.get("sparse_hkl_index", 6)), 8)
    zone_max = max(int(kwargs.get("max_hkl_index", 3)), 4)

    opts = _build_opts(q, n_axial, axial_max)
    g11, g22, g33 = float(gstar[0, 0]), float(gstar[1, 1]), float(gstar[2, 2])
    o11, o22, o33 = _find_opts(opts["G11"], g11), _find_opts(opts["G22"], g22), _find_opts(opts["G33"], g33)

    # Distinct-peak triple?
    triples = []
    for i1, h, _ in o11:
        for i2, k, _ in o22:
            for i3, l, _ in o33:
                if len({i1, i2, i3}) == 3:
                    triples.append(((i1, h), (i2, k), (i3, l)))
    # Also check if same peak forces failure when only non-distinct available
    any_triple_ignoring_distinct = bool(o11 and o22 and o33)

    # True gvec for system
    g12 = float(gstar[0, 1])
    g13 = float(gstar[0, 2])
    g23 = float(gstar[1, 2])
    if label == "orthorhombic":
        gvec = np.array([g11, g22, g33, 0.0, 0.0, 0.0], dtype=np.float64)
        # ortho search forces zeros; report true offdiag magnitudes
        true_offdiag_abs = {"g12": abs(g12), "g13": abs(g13), "g23": abs(g23)}
    else:
        # monoclinic: try all three unique-axis embeddings of the true metric
        gvec = np.array([g11, g22, g33, g12, g13, g23], dtype=np.float64)
        true_offdiag_abs = {"g12": abs(g12), "g13": abs(g13), "g23": abs(g23)}

    gmat = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]], dtype=np.float64)
    params_true = gstar_to_lattice_params(gmat)
    exact_true = _fast_match_count(q, gmat, q_match_abs_tol=1e-6)

    # Ortho forced-zero match (what sequential ortho actually builds)
    gmat_ortho = np.array([[g11, 0, 0], [0, g22, 0], [0, 0, g33]], dtype=np.float64)
    params_ortho = gstar_to_lattice_params(gmat_ortho)
    exact_ortho = _fast_match_count(q, gmat_ortho, q_match_abs_tol=1e-6)

    confirm = np.array(_hkl_pool(min(12, max(axial_max, zone_max) + 3), max_nonzero=3), dtype=np.float64)
    coeff = np.stack(
        [
            confirm[:, 0] ** 2,
            confirm[:, 1] ** 2,
            confirm[:, 2] ** 2,
            2 * confirm[:, 0] * confirm[:, 1],
            2 * confirm[:, 0] * confirm[:, 2],
            2 * confirm[:, 1] * confirm[:, 2],
        ],
        axis=1,
    )
    approx_true = int(
        _approx_match_counts(
            np.array([[g11, g22, g33, g12, g13, g23]]),
            coeff,
            q,
            q_match_abs_tol=1e-6,
            deadline=time.monotonic() + 30,
        )[0]
    )
    approx_ortho = int(
        _approx_match_counts(
            np.array([[g11, g22, g33, 0.0, 0.0, 0.0]]),
            coeff,
            q,
            q_match_abs_tol=1e-6,
            deadline=time.monotonic() + 30,
        )[0]
    )

    # Mono: can we recover the needed offdiag from a zone peak given true diags?
    offdiag_recover = {}
    if label == "monoclinic":
        for which, true_val in (("g12", g12), ("g13", g13), ("g23", g23)):
            best = None
            for pi in range(n_peaks):
                for zhkl in _zone_hkl_pool(zone_max, which):
                    val = _offdiag_from_peak(float(q[pi]), zhkl, g11, g22, g33, which)
                    if val is None or not np.isfinite(val):
                        continue
                    err = abs(val - true_val)
                    if best is None or err < best[0]:
                        best = (err, val, zhkl, pi)
            offdiag_recover[which] = {
                "true": true_val,
                "best_err": None if best is None else best[0],
                "best_val": None if best is None else best[1],
                "hkl": None if best is None else list(best[2]),
                "peak": None if best is None else best[3],
                "ok_1e6": bool(best is not None and best[0] <= 1e-6),
            }

    # Production search confirm empty / hit
    kwargs["pool_budget"] = 100
    st = time.time()
    cands = search_crystal_system(obs, label, wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM, **kwargs)
    elapsed = time.time() - st
    hit = None
    for r, c in enumerate(cands):
        if lattice_match_elementwise(c.niggli_params6(), tn, ltol=ltol, atol_deg=atol_deg):
            hit = r
            break

    # Failure tag
    if hit is not None and hit < 20:
        tag = "hit20"
    elif not o11 or not o22 or not o33:
        tag = "diag_missing_in_opts"
    elif not triples:
        tag = "no_distinct_peak_triple"
    elif label == "orthorhombic" and exact_ortho < min_matched:
        tag = "ortho_zero_offdiag_under_matches"
    elif exact_true < min_matched:
        tag = "true_gvec_under_matches"
    elif params_true is None:
        tag = "true_gvec_not_spd"
    elif label == "monoclinic" and not any(v["ok_1e6"] for v in offdiag_recover.values() if abs(v["true"]) > 1e-8):
        # needed nonzero offdiag not recoverable
        needed = [k for k, v in offdiag_recover.items() if abs(v["true"]) > 1e-6]
        if needed and not any(offdiag_recover[k]["ok_1e6"] for k in needed):
            tag = "mono_offdiag_not_recoverable"
        else:
            tag = "search_miss_despite_ok_seed"
    elif len(cands) == 0:
        tag = "search_empty_despite_ok_seed"
    else:
        tag = "pool_miss_despite_ok_seed"

    return {
        "label": label,
        "n_peaks": n_peaks,
        "min_matched": min_matched,
        "cell_abc": [round(tn[k], 3) for k in range(3)],
        "cell_angles": [round(tn[k], 2) for k in range(3, 6)],
        "n_opts": {"G11": len(o11), "G22": len(o22), "G33": len(o33)},
        "n_distinct_triples": len(triples),
        "any_opts_ignoring_distinct": any_triple_ignoring_distinct,
        "true_offdiag_abs": true_offdiag_abs,
        "params_true_ok": params_true is not None,
        "params_ortho_ok": params_ortho is not None,
        "exact_true": exact_true,
        "exact_ortho_zeros": exact_ortho,
        "approx_true": approx_true,
        "approx_ortho_zeros": approx_ortho,
        "exact_true_ge_min": exact_true >= min_matched,
        "exact_ortho_ge_min": exact_ortho >= min_matched,
        "offdiag_recover": offdiag_recover,
        "search_n_cand": len(cands),
        "search_hit_rank": hit,
        "search_elapsed_s": elapsed,
        "tag": tag,
    }


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
    pool: dict[str, list] = {cs: [] for cs in labels}
    print("=== Collect best stratum, keep empties with diags in opts ===", flush=True)

    # First gather candidates then filter by quick prod search emptiness — expensive.
    # Instead: gather stratum, run search, keep empties up to max_trace.
    with torch.no_grad():
        for batch in loader:
            if all(len(pool[cs]) >= args.n_per_label for cs in labels):
                break
            for i in range(batch["lattice"].shape[0]):
                label = CRYSTAL_SYSTEMS[int(batch["crystal_system_idx"][i].item())]
                if label not in labels or len(pool[label]) >= args.n_per_label:
                    continue
                truth = batch["lattice"][i].cpu().numpy().tolist()
                tn = _niggli(truth)
                if _geom_system(tn) != label:
                    continue
                obs = np.asarray(slice_observed_two_theta(batch["pxrd_x"], batch["peak_num"], i), dtype=np.float64)
                obs = obs[np.isfinite(obs)]
                q = inverse_d2_from_two_theta_f64(obs)
                gstar = _gstar(truth)
                if _axial_ok(gstar, q) < 3:
                    continue
                pool[label].append({"label": label, "truth_niggli": tn, "obs": obs, "q": q, "gstar": gstar})
            if all(len(pool[cs]) >= args.n_per_label for cs in labels):
                break

    rows = pool["orthorhombic"] + pool["monoclinic"]
    traces = []
    tag_counts: dict[str, int] = {}
    for j, row in enumerate(rows):
        tr = trace_one(row, ltol=args.ltol, atol_deg=args.atol_deg)
        # Only keep failures for the residual analysis focus, but count all tags
        tag_counts[tr["tag"]] = tag_counts.get(tr["tag"], 0) + 1
        print(
            f"{j+1:02d}/{len(rows)} {tr['label'][:4]} tag={tr['tag']:32s} "
            f"opts={tr['n_opts']} triples={tr['n_distinct_triples']} "
            f"exact_true={tr['exact_true']}/{tr['min_matched']} "
            f"exact_ortho0={tr['exact_ortho_zeros']} n_cand={tr['search_n_cand']}",
            flush=True,
        )
        if tr["tag"] != "hit20":
            traces.append(tr)
        if sum(1 for t in traces if t["tag"] != "hit20") >= args.max_trace and j + 1 >= args.max_trace:
            # continue until we scanned full collected stratum for tag_counts
            pass

    # Summaries by tag
    fail_traces = [t for t in traces if t["tag"] != "hit20"]
    report = {
        "protocol": {
            "n_per_label": args.n_per_label,
            "max_trace": args.max_trace,
            "ltol": args.ltol,
            "atol_deg": args.atol_deg,
        },
        "n_scanned": len(rows),
        "tag_counts": tag_counts,
        "n_fail_traced": len(fail_traces),
        "fails": fail_traces,
        "ortho_zero_issue_rate_among_ortho_fails": None,
    }
    ortho_fails = [t for t in fail_traces if t["label"] == "orthorhombic"]
    if ortho_fails:
        report["ortho_zero_issue_rate_among_ortho_fails"] = float(
            np.mean([t["tag"] == "ortho_zero_offdiag_under_matches" for t in ortho_fails])
        )

    print("\n=== TAG COUNTS ===", flush=True)
    for k, v in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/scale_100k_a3_g1_gstar6.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-label", type=int, default=20)
    p.add_argument("--max-trace", type=int, default=20)
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
