#!/usr/bin/env python3
"""G0 full-batch overfit sweep across R1 input representations.

Uses the real encoder modules (legacy Bert / physical peak-token / histogram)
with full-batch Adam on the balanced 700 set — same protocol as D-A3 — so
memorization capacity is comparable across variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from pxrd_cell_indexing.model.heads import HeadConfig, build_indexing_model
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANTS: dict[str, dict[str, Any]] = {
    "legacy_discrete": {
        "encoder_type": "bert",
        "position_encoding": "discrete",
        "peak_feature_mode": "legacy",
        "normalize_embedding": True,
        "encoder_checkpoint": None,
    },
    "continuous_2theta": {
        "encoder_type": "bert",
        "position_encoding": "continuous",
        "peak_feature_mode": "legacy",
        "normalize_embedding": True,
        "encoder_checkpoint": None,
    },
    "phys_continuous_2theta_i": {
        "encoder_type": "bert",
        "position_encoding": "physical",
        "peak_feature_mode": "continuous_2theta_i",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
    },
    "phys_reciprocal_d_i": {
        "encoder_type": "bert",
        "position_encoding": "physical",
        "peak_feature_mode": "reciprocal_d_i",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
    },
    "phys_inverse_d2_i": {
        "encoder_type": "bert",
        "position_encoding": "physical",
        "peak_feature_mode": "inverse_d2_i",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
    },
    "phys_inverse_d2_logi": {
        "encoder_type": "bert",
        "position_encoding": "physical",
        "peak_feature_mode": "inverse_d2_logi",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
    },
    "phys_inverse_d2_only": {
        "encoder_type": "bert",
        "position_encoding": "physical",
        "peak_feature_mode": "inverse_d2_only",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
    },
    "hist_mlp": {
        "encoder_type": "histogram",
        "position_encoding": "discrete",
        "peak_feature_mode": "legacy",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
        "histogram_dropout": 0.0,
    },
    "hist_mlp_sum": {
        "encoder_type": "histogram",
        "position_encoding": "discrete",
        "peak_feature_mode": "legacy",
        "normalize_embedding": False,
        "encoder_checkpoint": None,
        "hist_pool": "sum",
        "histogram_dropout": 0.0,
    },
}


def _geom(pred_phys: np.ndarray, truth_phys: np.ndarray, cs: np.ndarray) -> dict[str, Any]:
    def la(p):
        t = torch.tensor(p, dtype=torch.float64).reshape(-1, 6)
        l, a = lattice_lengths_angles(lattice_params_to_matrix(t))
        return l.numpy(), a.numpy()

    lp, ap = la(pred_phys)
    lt, at = la(truth_phys)
    ang = np.abs(ap - at)
    lr = np.abs(lp - lt) / np.clip(lt, 1e-6, None)
    elem = (ang.max(1) <= 3.0) & (lr.max(1) <= 0.05)
    pred_dev = np.abs(ap - 90.0).mean(1)
    truth_dev = np.abs(at - 90.0).mean(1)
    by_cs = {}
    for i, name in enumerate(CRYSTAL_SYSTEMS):
        m = cs == i
        if not m.any():
            continue
        by_cs[name] = {
            "n": int(m.sum()),
            "elem_ok_rate": float(elem[m].mean()),
            "angle_mae": float(ang[m].mean()),
        }
    return {
        "angle_mae": float(ang.mean()),
        "length_rel_mean": float(lr.mean()),
        "elem_ok_rate": float(elem.mean()),
        "pulled_to_90_rate": float(np.mean(pred_dev < truth_dev)),
        "min_cs_elem_ok_rate": float(min(v["elem_ok_rate"] for v in by_cs.values())) if by_cs else float("nan"),
        "by_crystal_system": by_cs,
    }


def _load_tensors(config: TrainConfig, device: torch.device):
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb),
        split="valid",
        sample_list_path=Path(config.data.train_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=256,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
    )
    xs, ys, ns, lats, css = [], [], [], [], []
    for batch in loader:
        xs.append(batch["pxrd_x"])
        ys.append(batch["pxrd_y"])
        ns.append(batch["peak_num"])
        lats.append(batch["lattice"])
        css.append(batch["crystal_system_idx"])
    # Keep variable-length flat layout by concatenating samples sequentially.
    # Rebuild as one mega-batch: concatenate peaks and peak_num.
    pxrd_x = torch.cat(xs, dim=0).to(device)
    pxrd_y = torch.cat(ys, dim=0).to(device)
    peak_num = torch.cat(ns, dim=0).to(device)
    lattice = torch.cat(lats, dim=0).to(device)
    cs = torch.cat(css, dim=0).cpu().numpy()
    return pxrd_x, pxrd_y, peak_num, lattice, cs


def run_variant(
    name: str,
    variant: dict[str, Any],
    config: TrainConfig,
    tensors,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    pxrd_x, pxrd_y, peak_num, lattice, cs = tensors
    normalizer = build_lattice_normalizer(config.data)
    target = normalizer.normalize(lattice)
    enc_cfg = {
        "encoder_type": variant["encoder_type"],
        "position_encoding": variant["position_encoding"],
        "peak_feature_mode": variant["peak_feature_mode"],
        "wavelength_angstrom": 1.54184,
        "intensity_transform": "linear",
        "hist_bins": 256,
        "sorted_peak_count": 24,
        "hist_pool": variant.get("hist_pool", "max"),
        "histogram_hidden_dim": 512,
        "histogram_dropout": float(variant.get("histogram_dropout", 0.0)),
    }
    torch.manual_seed(config.seed)
    model = build_indexing_model(
        checkpoint_path=variant.get("encoder_checkpoint"),
        encoder_config=enc_cfg,
        head_config=HeadConfig(
            hidden_dim=256,
            dropout=0.0,
            output_dim=head_output_dim(config.data.representation),
        ),
        freeze_encoder=False,
        normalize_embedding=bool(variant.get("normalize_embedding", False)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(pxrd_x, pxrd_y, peak_num)
        loss = lossf(out["lattice_norm"], target)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs or ep in (1, 10, 25):
            model.eval()
            with torch.no_grad():
                pred = normalizer.denormalize(model(pxrd_x, pxrd_y, peak_num)["lattice_norm"])
            g = _geom(pred.cpu().numpy(), lattice.cpu().numpy(), cs)
            g["epoch"] = ep
            g["loss"] = float(loss.item())
            curve.append(g)
            print(
                f"[{name}] ep{ep:4} loss={loss.item():.4f} ang={g['angle_mae']:.2f} "
                f"len={g['length_rel_mean']:.3f} elem={g['elem_ok_rate']*100:.1f}% "
                f"minCS={g['min_cs_elem_ok_rate']*100:.1f}%",
                flush=True,
            )
    return {"variant": name, "config": variant, "curve": curve, "final": curve[-1]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/diag_overfit_discrete.yaml")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--variants",
        type=str,
        default="legacy_discrete,phys_inverse_d2_i,hist_mlp,phys_reciprocal_d_i,phys_continuous_2theta_i,phys_inverse_d2_only,hist_mlp_sum",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/beat_engine/raw_diag/g0/overfit_input_sweep.json",
    )
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tensors = _load_tensors(config, device)
    print(f"loaded n_samples={int(tensors[2].shape[0])} peaks={int(tensors[0].shape[0])}", flush=True)

    results = {}
    for name in args.variants.split(","):
        name = name.strip()
        if name not in VARIANTS:
            raise KeyError(name)
        results[name] = run_variant(
            name, VARIANTS[name], config, tensors, epochs=args.epochs, lr=args.lr, device=device
        )

    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== G0 SUMMARY ===")
    for name, r in results.items():
        f = r["final"]
        gate = (
            f["elem_ok_rate"] >= 0.80
            and f["angle_mae"] <= 2.0
            and f["length_rel_mean"] <= 0.02
            and f["min_cs_elem_ok_rate"] >= 0.50
        )
        print(
            f"{name:28} loss={f['loss']:.4f} ang={f['angle_mae']:.2f} "
            f"len={f['length_rel_mean']:.3f} elem={f['elem_ok_rate']*100:5.1f}% "
            f"minCS={f['min_cs_elem_ok_rate']*100:5.1f}% pull90={f['pulled_to_90_rate']*100:4.0f}% "
            f"{'PASS' if gate else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
