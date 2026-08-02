#!/usr/bin/env python3
"""Diagnose what McMaille would need to add on top of the flow seed pool.

Three questions, matching the three stated goals for keeping McMaille:
  G1 ranking  -- can the true primitive cell be pushed to Top-1 without McM20?
  G3 recall   -- does drawing more flow samples raise library recall on its own?
  (G2 refinement precision is measured separately, it needs CELREF output.)

All metrics are L4 vs primitive-standard truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pxrd_cell_indexing.data.normalization import (  # noqa: E402
    GStar6Normalizer,
    GStar6Stats,
)
from pxrd_cell_indexing.geometry import gstar6_to_lattice  # noqa: E402
from pxrd_cell_indexing.model.fom import de_wolff_fom  # noqa: E402
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402
from train_flow_seedgen import SeedGenerator, load_mp100_eval  # noqa: E402


def cluster_by_gstar(z: np.ndarray, tol: float) -> tuple[np.ndarray, list[list[int]]]:
    """Greedy clustering in normalized gstar6 space. Returns (labels, clusters)."""
    n = len(z)
    labels = -np.ones(n, dtype=int)
    centers: list[np.ndarray] = []
    clusters: list[list[int]] = []
    for i in range(n):
        placed = False
        for ci, center in enumerate(centers):
            if float(np.linalg.norm(z[i] - center)) < tol:
                clusters[ci].append(i)
                centers[ci] = z[clusters[ci]].mean(axis=0)
                labels[i] = ci
                placed = True
                break
        if not placed:
            centers.append(z[i].copy())
            clusters.append([i])
            labels[i] = len(centers) - 1
    return labels, clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/flow_seedgen/full6m_equiv_off/best.pt")
    ap.add_argument("--max-k", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--cluster-tol", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/flow_seedgen/pool_diagnostics.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SeedGenerator(Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    normalizer = GStar6Normalizer.from_stats(
        GStar6Stats(
            component_mean=tuple(ck["normalizer"]["component_mean"]),
            component_std=tuple(ck["normalizer"]["component_std"]),
        )
    )
    mean = torch.tensor(normalizer.component_mean, device=device, dtype=torch.float32)
    std = torch.tensor(normalizer.component_std, device=device, dtype=torch.float32)

    items = load_mp100_eval(device)
    k_grid = [1, 5, 10, 20, 50, 100, 200, 500, args.max_k]
    k_grid = sorted({k for k in k_grid if k <= args.max_k})

    cov_hits = {k: 0 for k in k_grid}
    top1 = {"draw_order": 0, "cluster_size": 0, "dewolff": 0, "cluster_then_fom": 0}
    oracle = 0
    n_clusters_all, hit_cluster_rank = [], []
    per_sample = []

    with torch.no_grad():
        for idx, item in enumerate(items, 1):
            emb = model.encode(item["pxrd_x"], item["pxrd_y"], item["peak_num"])
            z = model.flow.sample(
                emb, num_samples=args.max_k, steps=args.sample_steps
            )[0]
            cells = gstar6_to_lattice(z * std + mean).cpu().numpy()
            zn = z.cpu().numpy()
            truth = item["prim"]

            flags = np.array([l4(c.tolist(), truth)[1] for c in cells], dtype=bool)
            for k in k_grid:
                if flags[:k].any():
                    cov_hits[k] += 1
            if flags.any():
                oracle += 1

            # --- ranking experiments on the first 100 draws ---
            sub_cells, sub_flags, sub_z = cells[:100], flags[:100], zn[:100]
            if sub_flags[0]:
                top1["draw_order"] += 1

            _, clusters = cluster_by_gstar(sub_z, args.cluster_tol)
            clusters.sort(key=len, reverse=True)
            n_clusters_all.append(len(clusters))
            rep = [c[0] for c in clusters]
            if sub_flags[rep[0]]:
                top1["cluster_size"] += 1
            for rank, c in enumerate(clusters, 1):
                if sub_flags[c[0]]:
                    hit_cluster_rank.append(rank)
                    break

            obs = item["pxrd_x"].cpu().numpy().reshape(-1)
            obs_i = item["pxrd_y"].cpu().numpy().reshape(-1)
            foms = np.array(
                [
                    de_wolff_fom(obs, c.tolist())
                    for c in sub_cells
                ]
            )
            foms = np.nan_to_num(foms, nan=0.0, posinf=0.0)
            if sub_flags[int(np.argmax(foms))]:
                top1["dewolff"] += 1

            # rank clusters by best FoM inside the cluster
            cl_score = [max(foms[i] for i in c) for c in clusters]
            best_cluster = clusters[int(np.argmax(cl_score))]
            best_in_cluster = max(best_cluster, key=lambda i: foms[i])
            if sub_flags[best_in_cluster]:
                top1["cluster_then_fom"] += 1

            per_sample.append(
                {
                    "sample_id": item["sample_id"],
                    "n_clusters": len(clusters),
                    "largest_cluster": len(clusters[0]),
                    "any_hit_100": bool(sub_flags.any()),
                    "any_hit_max": bool(flags.any()),
                    "n_hits_100": int(sub_flags.sum()),
                }
            )
            if idx % 20 == 0:
                print(f"{idx}/{len(items)}", flush=True)

    n = len(items)
    out = {
        "ckpt": args.ckpt,
        "epoch": ck.get("epoch"),
        "max_k": args.max_k,
        "cluster_tol": args.cluster_tol,
        "coverage_vs_k": {str(k): cov_hits[k] / n for k in k_grid},
        "top1_by_ranking_K100": {k: v / n for k, v in top1.items()},
        "oracle_top1_K100": sum(1 for r in per_sample if r["any_hit_100"]) / n,
        "median_n_clusters": float(np.median(n_clusters_all)),
        "median_hit_cluster_rank": (
            float(np.median(hit_cluster_rank)) if hit_cluster_rank else None
        ),
        "median_n_hits_in_100": float(np.median([r["n_hits_100"] for r in per_sample])),
        "per_sample": per_sample,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
