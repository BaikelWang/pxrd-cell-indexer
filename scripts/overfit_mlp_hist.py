#!/usr/bin/env python3
"""D-A3: can a plain MLP on a physical featurization overfit the 700-set?

Bypasses the RealPXRD peak-token transformer entirely. Each sample is turned
into a fixed-length feature vector from its peak list (Q=1/d^2 histogram + top-K
sorted Q + peak count), then a small MLP regresses the (normalized) lattice.

If this trivially memorizes 700 while the transformer encoder cannot, the
bottleneck is the encoder featurization/architecture, not the task or capacity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer, head_output_dim
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAVELENGTH = 1.54184
N_BINS = 256
TOPK = 24
Q_MAX = 2.2  # 1/d^2 range for 2theta up to ~90 deg, CuKa


def _featurize(two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    theta = np.radians(two_theta) / 2.0
    d = WAVELENGTH / (2.0 * np.sin(np.clip(theta, 1e-6, None)))
    q = 1.0 / np.clip(d, 1e-6, None) ** 2
    hist = np.zeros(N_BINS, dtype=np.float32)
    bins = np.clip((q / Q_MAX * N_BINS).astype(int), 0, N_BINS - 1)
    for b, inten in zip(bins, intensity):
        hist[b] = max(hist[b], float(inten))
    if hist.max() > 0:
        hist = hist / hist.max()
    order = np.argsort(q)
    topq = np.zeros(TOPK, dtype=np.float32)
    qs = np.sort(q)[:TOPK]
    topq[: len(qs)] = qs
    npk = np.array([len(q) / 50.0], dtype=np.float32)
    return np.concatenate([hist, topq, npk])


def _geom(pred_phys: np.ndarray, truth_phys: np.ndarray) -> dict[str, float]:
    def la(p):
        t = torch.tensor(p, dtype=torch.float64).reshape(-1, 6)
        l, a = lattice_lengths_angles(lattice_params_to_matrix(t))
        return l.numpy(), a.numpy()
    lp, ap = la(pred_phys); lt, at = la(truth_phys)
    ang = np.abs(ap - at); lr = np.abs(lp - lt) / np.clip(lt, 1e-6, None)
    elem = (ang.max(1) <= 3.0) & (lr.max(1) <= 0.05)
    return {"angle_mae": float(ang.mean()), "length_rel_mean": float(lr.mean()), "elem_ok_rate": float(elem.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/diag_overfit_discrete.yaml")
    p.add_argument("--jsonl", type=str, default="data/processed/overfit700_seed42.jsonl")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/beat_engine/raw_diag/overfit_mlp_hist.json")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    normalizer = build_lattice_normalizer(config.data)
    jsonl = args.jsonl if Path(args.jsonl).is_absolute() else str(PROJECT_ROOT / args.jsonl)

    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb), split="valid", sample_list_path=Path(jsonl),
        peak_filter=PeakFilterConfig(), xrd_augment=False, strict=False, seed_base=config.seed,
    )
    loader = build_dataloader(dataset_cfg, batch_size=128, num_workers=4, shuffle=False,
                              pin_memory=False, prefetch_factor=2, persistent_workers=False)

    feats, targets_norm, targets_phys = [], [], []
    for batch in loader:
        px = batch["pxrd_x"].view(-1).numpy(); py = batch["pxrd_y"].view(-1).numpy()
        pn = batch["peak_num"].numpy(); lat = batch["lattice"]
        tn = normalizer.normalize(lat).numpy(); tp = lat.numpy()
        idx = 0
        for i in range(len(pn)):
            n = int(pn[i])
            feats.append(_featurize(px[idx:idx+n].astype(np.float64), py[idx:idx+n].astype(np.float64)))
            idx += n
            targets_norm.append(tn[i]); targets_phys.append(tp[i])
    X = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
    Y = torch.tensor(np.stack(targets_norm), dtype=torch.float32, device=device)
    Yp = np.stack(targets_phys)

    out_dim = head_output_dim(config.data.representation)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 512), nn.GELU(),
        nn.Linear(512, 512), nn.GELU(),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, out_dim),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()
    n_params = sum(pp.numel() for pp in model.parameters())

    curve = []
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = lossf(pred, Y)
        loss.backward(); opt.step()
        if ep % 25 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_phys = normalizer.denormalize(model(X)).cpu().numpy()
            g = _geom(pred_phys, Yp); g["epoch"] = ep; g["loss"] = float(loss.item())
            curve.append(g)
            print(f"ep{ep:4} loss={loss.item():.4f} ang={g['angle_mae']:.2f} len_rel={g['length_rel_mean']:.3f} elem={g['elem_ok_rate']*100:.1f}%", flush=True)

    result = {"n_params": n_params, "feature_dim": X.shape[1], "n_samples": int(X.shape[0]), "curve": curve, "final": curve[-1]}
    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(curve[-1]))


if __name__ == "__main__":
    main()
