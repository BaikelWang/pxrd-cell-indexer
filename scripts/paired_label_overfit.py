#!/usr/bin/env python3
"""Stage-0 paired-label experiment: same peaks, primitive vs conventional labels.

Uses the inverse_d2 histogram MLP on the balanced 700 set. Both label conventions
should be memorizable; a large generalization gap on a held-out slice would imply
spectrum→cell-convention mapping complexity is an independent bottleneck.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch import nn

from pxrd_cell_indexing.data.peak_features import (
    PeakFeatureConfig,
    build_inverse_d2_histogram_features,
    histogram_feature_dim,
)
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYMPREC = 0.01


def _geom(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    def la(p):
        t = torch.tensor(p, dtype=torch.float64).reshape(-1, 6)
        l, a = lattice_lengths_angles(lattice_params_to_matrix(t))
        return l.numpy(), a.numpy()

    lp, ap = la(pred)
    lt, at = la(truth)
    ang = np.abs(ap - at)
    lr = np.abs(lp - lt) / np.clip(lt, 1e-6, None)
    elem = (ang.max(1) <= 3.0) & (lr.max(1) <= 0.05)
    return {
        "angle_mae": float(ang.mean()),
        "length_rel_mean": float(lr.mean()),
        "elem_ok_rate": float(elem.mean()),
    }


def _normalize(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y.mean(0)
    std = y.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((y - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _train_eval(
    x: torch.Tensor,
    y_norm: torch.Tensor,
    y_phys: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
) -> dict[str, float]:
    model = nn.Sequential(
        nn.Linear(x.shape[1], 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 6),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x)
        loss = lossf(pred, y_norm)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_n = model(x).cpu().numpy()
    pred_p = pred_n * std + mean
    g = _geom(pred_p, y_phys)
    g["loss"] = float(loss.item())
    return g


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, default=PROJECT_ROOT / "data/processed/overfit700_seed42.jsonl")
    p.add_argument(
        "--lmdb",
        type=Path,
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
    )
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/beat_engine/raw_diag/paired_label_overfit.json",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    env = lmdb.open(str(args.lmdb), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    cfg = PeakFeatureConfig()
    feats, prim_y, conv_y, label_y = [], [], [], []
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = env.begin().get(str(rec["lmdb_key"]).encode("ascii"))
            if raw is None:
                continue
            data = pickle.loads(gzip.decompress(raw))
            tt = np.asarray(data["pxrd_x"], dtype=np.float32)
            inten = np.asarray(data["pxrd_y"], dtype=np.float32)
            mask = inten > 5.0
            tt, inten = tt[mask], inten[mask]
            feats.append(build_inverse_d2_histogram_features(tt, inten, config=cfg))
            lattice = Lattice(np.asarray(data["p_lattice_matrix"], dtype=np.float64))
            structure = Structure(
                lattice, list(data["p_atom_type"]), np.asarray(data["p_atom_pos"], dtype=np.float64)
            )
            analyzer = SpacegroupAnalyzer(structure, symprec=SYMPREC)
            primitive = analyzer.find_primitive()
            conventional = analyzer.get_conventional_standard_structure()
            pl = primitive.lattice
            cl = conventional.lattice
            prim_y.append([pl.a, pl.b, pl.c, pl.alpha, pl.beta, pl.gamma])
            conv_y.append([cl.a, cl.b, cl.c, cl.alpha, cl.beta, cl.gamma])
            label_y.append(
                [
                    rec["lattice_a"],
                    rec["lattice_b"],
                    rec["lattice_c"],
                    rec["lattice_alpha"],
                    rec["lattice_beta"],
                    rec["lattice_gamma"],
                ]
            )

    x = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
    results = {}
    for name, arr in [("primitive", prim_y), ("conventional", conv_y), ("jsonl_label", label_y)]:
        y_phys = np.asarray(arr, dtype=np.float32)
        y_norm, mean, std = _normalize(y_phys)
        y_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
        torch.manual_seed(42)
        results[name] = _train_eval(x, y_t, y_phys, mean, std, epochs=args.epochs, device=device)
        print(name, results[name], flush=True)

    out = {
        "n_samples": int(x.shape[0]),
        "feature_dim": histogram_feature_dim(cfg),
        "epochs": args.epochs,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
