#!/usr/bin/env python3
"""D-B: Is the bottleneck the encoder representation or the regression head?

Extract CLS embeddings from (a) the raw pretrained encoder and (b) a trained
checkpoint's encoder, then:
  1. Linear probe: least-squares map embedding -> normalized lattice, fit on a
     train subset, evaluate angle/length error on valid. If a *linear* head on
     trained embeddings already matches the full MLP model, the head is not the
     bottleneck (the representation is).
  2. kNN geometry preservation: for each valid sample find nearest train sample
     by embedding cosine distance; report the lattice geometry gap. If nearest
     neighbours in embedding space have very different lattices, the embedding
     does not encode lattice geometry (encoder/info bottleneck).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix
from pxrd_cell_indexing.model.heads import build_indexing_model
from pxrd_cell_indexing.training.checkpoint import load_indexing_model_from_checkpoint
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_lengths_angles(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = torch.tensor(params, dtype=torch.float64).reshape(-1, 6)
    m = lattice_params_to_matrix(t)
    lengths, angles = lattice_lengths_angles(m)
    return lengths.numpy(), angles.numpy()


def _extract(
    model,
    loader,
    device: torch.device,
    normalizer,
) -> dict[str, np.ndarray]:
    embs, lat_norm, lat_phys, cs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            emb = model.encode(bt["pxrd_x"], bt["pxrd_y"], bt["peak_num"])
            embs.append(emb.cpu().numpy())
            phys = bt["lattice"].cpu().numpy()
            lat_phys.append(phys)
            lat_norm.append(normalizer.normalize(bt["lattice"]).cpu().numpy())
            cs.append(bt["crystal_system_idx"].cpu().numpy())
    return {
        "emb": np.concatenate(embs, 0),
        "lat_norm": np.concatenate(lat_norm, 0),
        "lat_phys": np.concatenate(lat_phys, 0),
        "cs": np.concatenate(cs, 0),
    }


def _build_loader(config, jsonl: str, device, *, lmdb_path: str):
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(lmdb_path),
        split="valid",
        sample_list_path=Path(jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    return build_dataloader(
        dataset_cfg,
        batch_size=128,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=False,
    )


def _linear_probe(
    fit: dict[str, np.ndarray],
    ev: dict[str, np.ndarray],
    normalizer,
    *,
    ltol: float,
    atol_deg: float,
) -> dict[str, Any]:
    # Solve W: emb_aug @ W ~= lat_norm (ridge for stability)
    xf = np.concatenate([fit["emb"], np.ones((fit["emb"].shape[0], 1))], axis=1)
    yf = fit["lat_norm"]
    lam = 1e-2
    a = xf.T @ xf + lam * np.eye(xf.shape[1])
    w = np.linalg.solve(a, xf.T @ yf)
    xe = np.concatenate([ev["emb"], np.ones((ev["emb"].shape[0], 1))], axis=1)
    pred_norm = xe @ w
    pred_phys = normalizer.denormalize(torch.tensor(pred_norm, dtype=torch.float32)).numpy()
    truth_phys = ev["lat_phys"]
    return _geometry_metrics(pred_phys, truth_phys, ev["cs"], ltol=ltol, atol_deg=atol_deg)


def _knn_geometry(fit: dict[str, np.ndarray], ev: dict[str, np.ndarray], *, k: int = 1) -> dict[str, Any]:
    # cosine NN from ev -> fit embeddings
    ef = fit["emb"] / (np.linalg.norm(fit["emb"], axis=1, keepdims=True) + 1e-9)
    ee = ev["emb"] / (np.linalg.norm(ev["emb"], axis=1, keepdims=True) + 1e-9)
    sims = ee @ ef.T  # [n_ev, n_fit]
    nn_idx = np.argmax(sims, axis=1)
    nn_phys = fit["lat_phys"][nn_idx]
    truth_phys = ev["lat_phys"]
    len_t, ang_t = _to_lengths_angles(truth_phys)
    len_n, ang_n = _to_lengths_angles(nn_phys)
    ang_err = np.abs(ang_t - ang_n).max(axis=1)
    len_rel = (np.abs(len_t - len_n) / np.clip(len_t, 1e-6, None)).max(axis=1)
    return {
        "nn_angle_max_median": float(np.median(ang_err)),
        "nn_angle_max_mean": float(np.mean(ang_err)),
        "nn_length_rel_max_median": float(np.median(len_rel)),
        "nn_within_3deg_5pct": float(np.mean((ang_err <= 3.0) & (len_rel <= 0.05))),
        "nn_within_10deg_20pct": float(np.mean((ang_err <= 10.0) & (len_rel <= 0.20))),
    }


def _geometry_metrics(
    pred_phys: np.ndarray,
    truth_phys: np.ndarray,
    cs: np.ndarray,
    *,
    ltol: float,
    atol_deg: float,
) -> dict[str, Any]:
    len_p, ang_p = _to_lengths_angles(pred_phys)
    len_t, ang_t = _to_lengths_angles(truth_phys)
    ang_err = np.abs(ang_p - ang_t)
    len_rel = np.abs(len_p - len_t) / np.clip(len_t, 1e-6, None)
    ang_max = ang_err.max(axis=1)
    len_max = len_rel.max(axis=1)
    elem_ok = (ang_max <= atol_deg) & (len_max <= ltol)
    pred_dev = np.abs(ang_p - 90.0).mean(axis=1)
    truth_dev = np.abs(ang_t - 90.0).mean(axis=1)
    out = {
        "n": int(pred_phys.shape[0]),
        "elem_ok_rate": float(np.mean(elem_ok)),
        "angle_mae": float(ang_err.mean()),
        "angle_max_median": float(np.median(ang_max)),
        "length_rel_mean": float(len_rel.mean()),
        "length_rel_max_median": float(np.median(len_max)),
        "pulled_to_90_rate": float(np.mean(pred_dev < truth_dev)),
    }
    by_cs = {}
    for i, name in enumerate(CRYSTAL_SYSTEMS):
        m = cs == i
        if not m.any():
            continue
        by_cs[name] = {
            "n": int(m.sum()),
            "elem_ok_rate": float(np.mean(elem_ok[m])),
            "angle_mae": float(ang_err[m].mean()),
        }
    out["by_crystal_system"] = by_cs
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="trained ckpt (optional)")
    p.add_argument("--fit-jsonl", type=str, default="data/processed/overfit700_seed42.jsonl")
    p.add_argument("--eval-jsonl", type=str, default="data/processed/valid1400_seed42.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ltol", type=float, default=0.05)
    p.add_argument("--atol-deg", type=float, default=3.0)
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    normalizer = build_lattice_normalizer(config.data)

    fit_jsonl = args.fit_jsonl if Path(args.fit_jsonl).is_absolute() else str(PROJECT_ROOT / args.fit_jsonl)
    eval_jsonl = args.eval_jsonl if Path(args.eval_jsonl).is_absolute() else str(PROJECT_ROOT / args.eval_jsonl)

    result: dict[str, Any] = {"config": str(config_path), "ltol": args.ltol, "atol_deg": args.atol_deg}

    # --- Raw pretrained encoder ---
    enc_ckpt = config.model.encoder_checkpoint
    pre_model = build_indexing_model(
        checkpoint_path=enc_ckpt,
        head_config=None,
        freeze_encoder=True,
        normalize_embedding=config.model.normalize_embedding,
    ).to(device)
    fit_loader = _build_loader(config, fit_jsonl, device, lmdb_path=config.data.train_lmdb)
    ev_loader = _build_loader(config, eval_jsonl, device, lmdb_path=config.data.valid_lmdb)
    pre_fit = _extract(pre_model, fit_loader, device, normalizer)
    pre_ev = _extract(pre_model, ev_loader, device, normalizer)
    result["pretrained_encoder"] = {
        "linear_probe": _linear_probe(pre_fit, pre_ev, normalizer, ltol=args.ltol, atol_deg=args.atol_deg),
        "knn": _knn_geometry(pre_fit, pre_ev),
    }

    # --- Trained checkpoint encoder ---
    if args.checkpoint is not None:
        ckpt = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
        tr_model, _, _ = load_indexing_model_from_checkpoint(ckpt, config, device)
        tr_fit = _extract(tr_model, fit_loader, device, normalizer)
        tr_ev = _extract(tr_model, ev_loader, device, normalizer)
        result["trained_encoder"] = {
            "checkpoint": str(ckpt),
            "linear_probe": _linear_probe(tr_fit, tr_ev, normalizer, ltol=args.ltol, atol_deg=args.atol_deg),
            "knn": _knn_geometry(tr_fit, tr_ev),
        }

    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
