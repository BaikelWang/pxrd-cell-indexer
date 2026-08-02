#!/usr/bin/env python3
"""Test the four steps of the proposed NN-seed -> McMaille pipeline on real data.

Step 1  score seeds by Rp / n_indexed          -> is it a usable gate?
Step 2  small-step MC refine                   -> does a trust region save CELREF?
Step 3  dedup + Bravais + constrained LSQ      -> how much does dedup alone buy?
Step 4  rank by fit / symmetry / volume        -> does the prior beat plain McM20?

Everything is measured against L4-strict vs the primitive-standard truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_mcmaille_value import STAGE_NAME, parse_allcells  # noqa: E402
from pymatgen.core import Lattice  # noqa: E402
from remeasure_l4_prim_vs_conv import ATOL, CIF_DIR, LTOL, l4, truth_cells  # noqa: E402


def symmetry_score(cell: list[float], ltol: float = 0.02, atol: float = 1.5) -> int:
    """Count satisfied metric constraints. Higher = more symmetric cell."""
    a, b, c, al, be, ga = cell
    s = 0
    s += abs(a - b) / max(a, 1e-9) < ltol
    s += abs(b - c) / max(b, 1e-9) < ltol
    s += abs(a - c) / max(a, 1e-9) < ltol
    for ang in (al, be, ga):
        s += abs(ang - 90.0) < atol or abs(ang - 120.0) < atol
    return int(s)


def drift(a: list[float], b: list[float]) -> float:
    """Max relative parameter change between a seed and its refined version."""
    d = max(abs(a[i] - b[i]) / max(b[i], 1e-9) for i in range(3))
    return max(d, max(abs(a[i] - b[i]) for i in range(3, 6)) / 90.0)


def cluster(cells: np.ndarray, tol: float = 0.03) -> list[list[int]]:
    """Greedy dedup in log-length + angle space."""
    feat = np.concatenate(
        [np.log(np.clip(cells[:, :3], 1e-6, None)), cells[:, 3:] / 90.0], axis=1
    )
    clusters: list[list[int]] = []
    centers: list[np.ndarray] = []
    for i in range(len(feat)):
        for ci, cen in enumerate(centers):
            if float(np.max(np.abs(feat[i] - cen))) < tol:
                clusters[ci].append(i)
                centers[ci] = feat[clusters[ci]].mean(axis=0)
                break
        else:
            centers.append(feat[i].copy())
            clusters.append([i])
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="third_party/McMaille/run_lab/mp100_reseed_flow6m_k100_noproj",
    )
    ap.add_argument("--out", default="results/flow_seedgen/proposed_pipeline.json")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    samples = []
    for d in sorted(run_dir.iterdir()):
        cif = CIF_DIR / f"{d.name}.cif"
        if not (d.is_dir() and cif.exists()):
            continue
        files = list(d.glob("*.allcells"))
        if not files:
            continue
        rows = parse_allcells(files[0])
        if not rows:
            continue
        tcell = truth_cells(cif)["prim"]
        for r in rows:
            r["hit"] = l4(r["cell"], tcell)[1]
            r["sym"] = symmetry_score(r["cell"])
        samples.append({"sid": d.name, "truth": tcell, "rows": rows})

    n = len(samples)
    top1 = defaultdict(int)
    gate_stats = defaultdict(lambda: {"kept": 0, "hits": 0, "lost_samples": 0})
    tr_curve = {}
    dedup_sizes = []

    TAUS = [0.005, 0.01, 0.02, 0.05, 0.10, 1.0]
    tr_res = {t: {"hit_to_miss": 0, "miss_to_hit": 0} for t in TAUS}

    for s in samples:
        rows, tcell = s["rows"], s["truth"]
        raw = [r for r in rows if STAGE_NAME.get(r["stage"]) == "raw"]
        if not raw:
            continue

        # ---------- step 4: ranking policies on the raw seed pool ----------
        cells = np.array([r["cell"] for r in raw])
        flags = np.array([r["hit"] for r in raw])
        mcm = np.array([r["mcm20"] for r in raw])
        rp = np.array([r["rp"] for r in raw])
        nidx = np.array([r["n_indexed"] for r in raw], dtype=float)
        sym = np.array([r["sym"] for r in raw], dtype=float)
        vol = np.array([r["volume"] for r in raw])

        def pick(score: np.ndarray, name: str) -> None:
            if flags[int(np.argmax(score))]:
                top1[name] += 1

        zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
        pick(mcm, "mcm20")
        pick(-rp, "rp")
        pick(nidx, "n_indexed")
        pick(zs(mcm) + 0.5 * zs(sym), "mcm20+sym")
        pick(zs(mcm) + 0.5 * zs(sym) - 0.5 * zs(np.log(vol)), "mcm20+sym-vol")
        pick(zs(mcm) - 0.5 * zs(np.log(vol)), "mcm20-vol")

        clusters = cluster(cells)
        clusters.sort(key=len, reverse=True)
        dedup_sizes.append(len(clusters))
        size = np.zeros(len(raw))
        for c in clusters:
            for i in c:
                size[i] = len(c)
        pick(size, "cluster_size")
        pick(zs(size) + 0.5 * zs(mcm), "cluster+mcm20")
        pick(zs(size) + 0.5 * zs(sym), "cluster+sym")
        pick(zs(size) + 0.5 * zs(mcm) + 0.5 * zs(sym), "cluster+mcm20+sym")

        # ---------- step 1: is Rp / n_indexed a usable gate? ----------
        for name, score, frac in (
            ("rp_top25", -rp, 0.25),
            ("rp_top50", -rp, 0.50),
            ("nidx_top25", nidx, 0.25),
            ("nidx_top50", nidx, 0.50),
        ):
            k = max(1, int(len(raw) * frac))
            keep = np.argsort(-score)[:k]
            g = gate_stats[name]
            g["kept"] += k
            g["hits"] += int(flags[keep].sum())
            if flags.any() and not flags[keep].any():
                g["lost_samples"] += 1

        # ---------- step 2: does a trust region rescue CELREF? ----------
        celref = {}
        for r in rows:
            if STAGE_NAME.get(r["stage"]) == "celref":
                celref.setdefault(r["seed_src"], r)
        for r in raw:
            c = celref.get(r["seed_src"])
            if c is None:
                continue
            d = drift(c["cell"], r["cell"])
            for t in TAUS:
                acc = c if d <= t else r  # reject refinement outside trust region
                if r["hit"] and not acc["hit"]:
                    tr_res[t]["hit_to_miss"] += 1
                elif not r["hit"] and acc["hit"]:
                    tr_res[t]["miss_to_hit"] += 1

    for t in TAUS:
        tr_curve[str(t)] = tr_res[t]

    out = {
        "run_dir": args.run_dir,
        "n_samples": n,
        "oracle_raw_pool": sum(
            1 for s in samples if any(r["hit"] for r in s["rows"] if STAGE_NAME.get(r["stage"]) == "raw")
        )
        / n,
        "step4_top1_by_policy": {k: v / n for k, v in sorted(top1.items(), key=lambda kv: -kv[1])},
        "step3_median_unique_cells": float(np.median(dedup_sizes)),
        "step1_gate": {
            k: {
                "precision": v["hits"] / max(v["kept"], 1),
                "samples_losing_their_only_hit": v["lost_samples"] / n,
            }
            for k, v in gate_stats.items()
        },
        "step2_trust_region": tr_curve,
    }
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
