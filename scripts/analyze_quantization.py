#!/usr/bin/env python3
"""D-C: quantify 2θ-quantization damage and peaks->lattice ambiguity.

The discrete encoder casts 2θ to integer degrees (nn.Embedding index), so peaks
at 20.1° and 20.9° collapse to index 20. This script measures:
  1. d-spacing relative error induced by rounding 2θ to the nearest integer.
  2. Input ambiguity: for each sample, build the integer-2θ occupancy signature
     (exactly the position information the discrete encoder receives) and find
     its nearest neighbour by Jaccard similarity; report the lattice geometry
     gap. If near-identical position signatures map to very different lattices,
     the discrete position channel is fundamentally ambiguous at 1° resolution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    build_dataloader,
)
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAVELENGTH = 1.54184


def _d_from_2theta(two_theta_deg: np.ndarray) -> np.ndarray:
    theta = np.radians(two_theta_deg) / 2.0
    return WAVELENGTH / (2.0 * np.sin(np.clip(theta, 1e-6, None)))


def _lengths_angles(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = torch.tensor(lat, dtype=torch.float64).reshape(-1, 6)
    l, a = lattice_lengths_angles(lattice_params_to_matrix(t))
    return l.numpy(), a.numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scale_100k_b3a_no_shift.yaml")
    p.add_argument("--jsonl", type=str, default="data/processed/valid1400_seed42.jsonl")
    p.add_argument("--lmdb", type=str, default=None, help="override lmdb (default valid_lmdb)")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/beat_engine/raw_diag/quantization.json")
    args = p.parse_args()

    from pxrd_cell_indexing.training.config import TrainConfig

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    jsonl = args.jsonl if Path(args.jsonl).is_absolute() else str(PROJECT_ROOT / args.jsonl)
    lmdb = args.lmdb or config.data.valid_lmdb

    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(lmdb),
        split="valid",
        sample_list_path=Path(jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg, batch_size=128, num_workers=4, shuffle=False,
        pin_memory=False, prefetch_factor=2, persistent_workers=False,
    )

    peaks_list: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    all_d_rel_err: list[float] = []
    for batch in loader:
        px = batch["pxrd_x"].view(-1).numpy()
        pn = batch["peak_num"].numpy()
        lat = batch["lattice"].numpy()
        idx = 0
        for i in range(len(pn)):
            n = int(pn[i])
            tt = px[idx: idx + n].astype(np.float64)
            idx += n
            peaks_list.append(tt)
            lats.append(lat[i])
            d = _d_from_2theta(tt)
            d_round = _d_from_2theta(np.round(tt))
            rel = np.abs(d_round - d) / np.clip(d, 1e-9, None)
            all_d_rel_err.extend(rel.tolist())

    lats_arr = np.stack(lats, 0)
    len_all, ang_all = _lengths_angles(lats_arr)
    n = len(peaks_list)

    # --- 1. d-spacing rounding error ---
    d_rel = np.array(all_d_rel_err)
    d_summary = {
        "n_peaks": int(d_rel.size),
        "d_rel_err_mean": float(d_rel.mean()),
        "d_rel_err_p50": float(np.percentile(d_rel, 50)),
        "d_rel_err_p90": float(np.percentile(d_rel, 90)),
        "d_rel_err_p99": float(np.percentile(d_rel, 99)),
        "frac_d_rel_gt_5pct": float(np.mean(d_rel > 0.05)),
        "frac_d_rel_gt_2pct": float(np.mean(d_rel > 0.02)),
    }

    # --- 2. integer-2θ signature ambiguity ---
    # occupancy over integer bins in [5, 90]; exactly the discrete encoder's position indices.
    lo, hi = 5, 90
    sigs = []
    for tt in peaks_list:
        occ = np.zeros(hi - lo + 1, dtype=np.float32)
        idxs = np.clip(np.round(tt).astype(int), lo, hi) - lo
        occ[idxs] = 1.0
        sigs.append(occ)
    sigs = np.stack(sigs, 0)  # [n, bins]

    # Jaccard similarity via matrix ops: inter / union
    inter = sigs @ sigs.T
    card = sigs.sum(1, keepdims=True)
    union = card + card.T - inter
    jac = inter / np.clip(union, 1e-9, None)
    np.fill_diagonal(jac, -1.0)
    nn_idx = np.argmax(jac, axis=1)
    nn_jac = jac[np.arange(n), nn_idx]

    ang_gap = np.abs(ang_all - ang_all[nn_idx]).max(1)
    len_gap = (np.abs(len_all - len_all[nn_idx]) / np.clip(len_all, 1e-6, None)).max(1)

    # exact collisions (identical integer occupancy)
    sig_keys = defaultdict(list)
    for i, tt in enumerate(peaks_list):
        key = tuple(sorted(set(np.clip(np.round(tt).astype(int), lo, hi).tolist())))
        sig_keys[key].append(i)
    collision_groups = {k: v for k, v in sig_keys.items() if len(v) > 1}
    n_in_collision = sum(len(v) for v in collision_groups.values())
    # within-collision lattice spread
    coll_ang_spreads, coll_len_spreads = [], []
    for members in collision_groups.values():
        m = np.array(members)
        aa = ang_all[m]
        ll = len_all[m]
        coll_ang_spreads.append(float(np.abs(aa - aa.mean(0)).max()))
        coll_len_spreads.append(float((np.abs(ll - ll.mean(0)) / np.clip(ll.mean(0), 1e-6, None)).max()))

    amb_summary = {
        "n_samples": n,
        "nn_jaccard_p50": float(np.percentile(nn_jac, 50)),
        "nn_jaccard_p90": float(np.percentile(nn_jac, 90)),
        "frac_nn_jaccard_ge_0.8": float(np.mean(nn_jac >= 0.8)),
        "frac_nn_jaccard_eq_1.0": float(np.mean(nn_jac >= 0.999)),
        # lattice gap to nearest-signature neighbour
        "nn_angle_gap_max_p50": float(np.percentile(ang_gap, 50)),
        "nn_length_gap_max_p50": float(np.percentile(len_gap, 50)),
        # among highly-similar (jac>=0.8) pairs, how far apart are lattices?
        "highsim_angle_gap_p50": float(np.percentile(ang_gap[nn_jac >= 0.8], 50)) if np.any(nn_jac >= 0.8) else None,
        "highsim_length_gap_p50": float(np.percentile(len_gap[nn_jac >= 0.8], 50)) if np.any(nn_jac >= 0.8) else None,
        "highsim_within_3deg_5pct": float(np.mean((ang_gap[nn_jac >= 0.8] <= 3.0) & (len_gap[nn_jac >= 0.8] <= 0.05))) if np.any(nn_jac >= 0.8) else None,
        # exact integer-signature collisions
        "n_exact_collision_samples": n_in_collision,
        "n_collision_groups": len(collision_groups),
        "collision_angle_spread_p50": float(np.percentile(coll_ang_spreads, 50)) if coll_ang_spreads else None,
        "collision_length_spread_p50": float(np.percentile(coll_len_spreads, 50)) if coll_len_spreads else None,
    }

    result = {
        "config": str(config_path),
        "jsonl": jsonl,
        "wavelength": WAVELENGTH,
        "d_spacing_rounding": d_summary,
        "integer_2theta_ambiguity": amb_summary,
    }
    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
