#!/usr/bin/env python3
"""P0: deterministic rerank probe over frozen seeded-McMaille libraries.

No training, no McMaille re-run. Reads an existing run directory and reorders the
``.allcells`` rows with a handful of hand-built scoring keys, to answer whether the
signal needed for a learned reranker is present in the features at all.

Every policy is a *permutation* of the same candidate list, so ``lib_strict`` is
invariant by construction (asserted). Reordering is restricted to the top-``S``
rows by McM20, because measured first-hit ranks show nearly all recoverable mass
sits in rank <= 20.

Inputs per sample, all local to the run dir:
  ``<stem>.dat``       observed peaks + wavelength actually fed to McMaille
  ``<stem>.seed``      NN seed pool (``seed_src`` is a 1-based index into it)
  ``<stem>.allcells``  candidate rows (14 columns)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

from pxrd_cell_indexing.model.fom import (  # noqa: E402
    _compute_match_stats,
    lattice_params_to_matrix,
    reciprocal_metric_tensor,
)

KS = [1, 5, 10, 20]
POLICIES = [
    "mcm20",
    "fom",
    "strict_M",
    "matched_frac",
    "vol_anchor",
    "nn_dist",
    "combo",
]

CNRS_DIR = Path("/nanolab/users/wyx/CNRS")

# 14 columns: idx seed_src stage n_indexed McM20 volume Rp a b c al be ga bravais
ALLCELLS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)


def parse_allcells_full(path: Path) -> list[dict]:
    """Parse all 14 columns (existing parsers each drop half of them)."""
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = ALLCELLS_ROW.match(line)
        if not m:
            continue
        g = m.groups()
        out.append(
            {
                "seed_src": int(g[1]),
                "stage": int(g[2]),
                "n_indexed": int(g[3]),
                "McM20": float(g[4]),
                "volume": float(g[5]),
                "Rp": float(g[6]),
                "params": [float(g[i]) for i in range(7, 13)],
                "bravais": g[13],
            }
        )
    return out


def read_dat(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (wavelength, two_theta, intensity) from a McMaille .dat."""
    lines = path.read_text(errors="replace").splitlines()
    wavelength = 1.5406
    tt: list[float] = []
    ii: list[float] = []
    for idx, line in enumerate(lines):
        if line.startswith("!"):
            continue
        parts = line.split()
        if idx <= 2 and len(parts) >= 2:
            try:
                wavelength = float(parts[0])
            except ValueError:
                pass
            continue
        if len(parts) == 2:
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            tt.append(a)
            ii.append(b)
    return wavelength, np.asarray(tt), np.asarray(ii)


def read_seeds(path: Path) -> np.ndarray:
    """Return (n_seeds, 6) lattice params of the NN seed pool."""
    rows: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:6]])
        except ValueError:
            continue
    return np.asarray(rows) if rows else np.zeros((0, 6))


def gstar6(params) -> np.ndarray:
    import torch

    matrix = lattice_params_to_matrix(torch.tensor(np.asarray(params, dtype=np.float64))).numpy()
    g = reciprocal_metric_tensor(matrix)
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 2], g[0, 2], g[0, 1]])


def cell_volume(params) -> float:
    import torch

    matrix = lattice_params_to_matrix(torch.tensor(np.asarray(params, dtype=np.float64))).numpy()
    return float(abs(np.linalg.det(matrix)))


def pct_rank(values: np.ndarray, higher_better: bool = True) -> np.ndarray:
    """Within-sample percentile rank in [0, 1]; absolute scales are not comparable."""
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isfinite(v), v, np.nanmin(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0)
    if v.size <= 1:
        return np.ones_like(v)
    order = np.argsort(-v if higher_better else v)
    out = np.empty_like(v)
    out[order] = np.linspace(1.0, 0.0, v.size)
    return out


def build_orderings(rows: list[dict], dat: tuple, seeds: np.ndarray, shortlist: int) -> dict:
    """Return {policy: [row_index, ...]}; each is a permutation of ``range(len(rows))``.

    ``rows`` must already be sorted by McM20 desc (the baseline order).
    """
    wavelength, tt, _ii = dat
    head, tail_idx = rows[:shortlist], list(range(shortlist, len(rows)))
    if not head:
        return {p: list(range(len(rows))) for p in POLICIES}, {}

    seed_g = np.asarray([gstar6(s) for s in seeds]) if seeds.size else np.zeros((0, 6))
    v_nn = float(np.median([cell_volume(s) for s in seeds])) if seeds.size else 0.0

    feats = []
    for r in head:
        st = _compute_match_stats(tt, r["params"], wavelength_angstrom=wavelength)
        if seed_g.size:
            d = float(np.min(np.linalg.norm(seed_g - gstar6(r["params"]), axis=1)))
        else:
            d = 0.0
        vol = st.volume if st.volume > 0 else max(r["volume"], 1e-6)
        vol_dev = abs(np.log(vol / v_nn)) if v_nn > 0 else 0.0
        feats.append(
            {
                "de_wolff": st.de_wolff_m,
                "strict_M": st.strict_dewolff_m,
                "matched_frac": st.intensity_score,
                "nn_dist": d,
                "vol_dev": min(vol_dev, 3.0) / 3.0,
            }
        )

    mcm = np.asarray([r["McM20"] for r in head])
    p_mcm = pct_rank(mcm)
    p_strict = pct_rank(np.asarray([f["strict_M"] for f in feats]))
    p_frac = pct_rank(np.asarray([f["matched_frac"] for f in feats]))
    vol_dev = np.asarray([f["vol_dev"] for f in feats])

    keys = {
        "mcm20": -mcm,
        "fom": -np.asarray([f["de_wolff"] for f in feats]),
        "strict_M": -np.asarray([f["strict_M"] for f in feats]),
        "matched_frac": -(p_frac + 0.01 * p_mcm),
        "vol_anchor": -(p_mcm - 0.5 * vol_dev),
        "nn_dist": np.asarray([f["nn_dist"] for f in feats]),
        "combo": -((p_mcm + p_strict + p_frac) / 3.0 - 0.3 * vol_dev),
    }

    out = {}
    for policy, key in keys.items():
        idx = np.argsort(np.where(np.isfinite(key), key, np.inf), kind="stable")
        out[policy] = [int(i) for i in idx] + tail_idx

    # Cached so downstream sweeps / model training never re-run peak matching.
    cache = {
        "p_mcm": p_mcm.tolist(),
        "p_strict": p_strict.tolist(),
        "p_frac": p_frac.tolist(),
        "vol_dev": vol_dev.tolist(),
        "p_nn_dist": pct_rank(np.asarray([f["nn_dist"] for f in feats]), False).tolist(),
        "p_dewolff": pct_rank(np.asarray([f["de_wolff"] for f in feats])).tolist(),
        "Rp": [r["Rp"] for r in head],
        "n_indexed": [r["n_indexed"] for r in head],
        "stage": [r["stage"] for r in head],
        "seed_src": [r["seed_src"] for r in head],
        "n_tail": len(tail_idx),
    }
    return out, cache


def truth_prim(dataset: str, sid: str):
    cif = CIF_DIR / f"{sid}.cif" if dataset == "mp100" else CNRS_DIR / f"{sid}_sg.cif"
    return truth_cells(cif)["prim"]


def eval_sid(job) -> dict:
    sid, run_dir, dataset, shortlist = job
    d = Path(run_dir) / sid
    stem = sid.replace("-", "_")
    allc = d / f"{stem}.allcells"
    res = {"sample_id": sid, "n_pool": 0, "policies": {}, "features": {}, "is_hit": []}
    if not allc.exists():
        for p in POLICIES:
            res["policies"][p] = {"lib": False, "first": None}
        return res

    rows = sorted(parse_allcells_full(allc), key=lambda r: -r["McM20"])
    dat = read_dat(d / f"{stem}.dat")
    seeds = read_seeds(d / f"{stem}.seed") if (d / f"{stem}.seed").exists() else np.zeros((0, 6))
    orderings, feats = build_orderings(rows, dat, seeds, shortlist)

    truth = truth_prim(dataset, sid)
    is_hit = [bool(l4(r["params"], truth)[1]) for r in rows]

    res["n_pool"] = len(rows)
    res["features"] = feats
    res["is_hit"] = is_hit
    for policy, perm in orderings.items():
        assert sorted(perm) == list(range(len(rows))), f"{policy} is not a permutation"
        first = next((i for i, j in enumerate(perm, 1) if is_hit[j]), None)
        res["policies"][policy] = {"lib": any(is_hit), "first": first}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", choices=["mp100", "cnrs"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shortlist", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    prefix = "mp-" if args.dataset == "mp100" else ""
    sids = sorted(
        p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if args.dataset == "cnrs":
        sids = [s for s in sids if s.isdigit()]

    rows: list[dict] = []
    jobs = [(s, str(run_dir), args.dataset, args.shortlist) for s in sids]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_sid, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    rows.sort(key=lambda r: r["sample_id"])
    n = max(len(rows), 1)

    report = {"run_dir": str(run_dir), "dataset": args.dataset, "n": len(rows), "policies": {}}
    libs = set()
    for policy in POLICIES:
        firsts = [r["policies"][policy]["first"] for r in rows]
        lib = sum(1 for r in rows if r["policies"][policy]["lib"]) / n
        libs.add(round(lib, 9))
        report["policies"][policy] = {
            "lib_strict": lib,
            "topk_strict": {str(k): sum(1 for f in firsts if f and f <= k) / n for k in KS},
        }
    assert len(libs) == 1, f"lib_strict must be invariant across policies, got {libs}"

    base = report["policies"]["mcm20"]["topk_strict"]["1"]
    print(f"\n=== {args.dataset} prim L4-strict | {run_dir.name} (n={len(rows)}) ===")
    print(f"{'policy':<14} " + "  ".join(f"K={k}".rjust(6) for k in KS) + f"  {'lib':>6}  {'ΔTop1':>7}")
    for policy in POLICIES:
        a = report["policies"][policy]
        cells = "  ".join(f"{a['topk_strict'][str(k)]:6.1%}" for k in KS)
        delta = a["topk_strict"]["1"] - base
        print(f"{policy:<14} {cells}  {a['lib_strict']:6.1%}  {delta:+7.1%}")

    Path(args.out).write_text(json.dumps({**report, "per_sample": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
