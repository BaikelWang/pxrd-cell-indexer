#!/usr/bin/env python3
"""R3 P0-700: cubic split / joint_phys overfit on histogram + cs_conditional."""

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
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer, head_output_dim
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix
from pxrd_cell_indexing.losses import IndexingLoss, LossWeights
from pxrd_cell_indexing.model.heads import HeadConfig, build_indexing_model
from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NON_CUBIC = [i for i, n in enumerate(CRYSTAL_SYSTEMS) if n != "cubic"]
HIST_ENC = {
    "encoder_type": "histogram",
    "position_encoding": "discrete",
    "peak_feature_mode": "legacy",
    "wavelength_angstrom": 1.54184,
    "intensity_transform": "linear",
    "hist_bins": 256,
    "sorted_peak_count": 24,
    "hist_pool": "max",
    "histogram_hidden_dim": 512,
    "histogram_dropout": 0.0,
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
    non_cubic = np.isin(cs, NON_CUBIC)
    cubic = cs == 0
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
        "elem_ok_rate": float(elem.mean()),
        "non_cubic_elem_ok_rate": float(elem[non_cubic].mean()) if non_cubic.any() else float("nan"),
        "cubic_elem_ok_rate": float(elem[cubic].mean()) if cubic.any() else float("nan"),
        "pulled_to_90_rate": float(np.mean(pred_dev < truth_dev)),
        "min_cs_elem_ok_rate": float(min(v["elem_ok_rate"] for v in by_cs.values())),
        "by_crystal_system": by_cs,
    }


def _load(config: TrainConfig, device: torch.device):
    ds = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb),
        split="valid",
        sample_list_path=Path(config.data.train_jsonl),
        peak_filter=PeakFilterConfig(intensity_min=5.0),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        ds, batch_size=256, num_workers=4, shuffle=False, pin_memory=False,
        prefetch_factor=2, persistent_workers=False,
    )
    xs, ys, ns, lats, css = [], [], [], [], []
    for batch in loader:
        xs.append(batch["pxrd_x"]); ys.append(batch["pxrd_y"]); ns.append(batch["peak_num"])
        lats.append(batch["lattice"]); css.append(batch["crystal_system_idx"])
    return (
        torch.cat(xs).to(device),
        torch.cat(ys).to(device),
        torch.cat(ns).to(device),
        torch.cat(lats).to(device),
        torch.cat(css).to(device),
    )


def run_one(name, *, cubic_split, loss_mode, phys_w, config, tensors, epochs, lr, device):
    pxrd_x, pxrd_y, peak_num, lattice, cs_t = tensors
    normalizer = build_lattice_normalizer(config.data)
    target = normalizer.normalize(lattice)
    torch.manual_seed(config.seed)
    model = build_indexing_model(
        encoder_config=dict(HIST_ENC),
        head_config=HeadConfig(
            hidden_dim=256, dropout=0.0,
            output_dim=head_output_dim(config.data.representation),
            head_type="cs_conditional",
            use_cs_classifier=False,
            default_cs_route="oracle",
            cubic_bravais_split=cubic_split,
            default_setting_route="oracle",
        ),
        freeze_encoder=False,
        normalize_embedding=False,
    ).to(device)
    model.set_normalizer(normalizer)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = IndexingLoss(
        LossWeights(mode=loss_mode, regression=1.0, physical_weight=phys_w),
        normalizer=normalizer,
    )
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(
            pxrd_x, pxrd_y, peak_num,
            crystal_system_idx=cs_t, cs_route="oracle",
            lattice_phys=lattice, setting_route="oracle",
        )
        losses = loss_fn(
            out["lattice_norm"], target,
            lattice_phys_target=lattice, crystal_system_idx=cs_t,
        )
        losses["loss_total"].backward()
        opt.step()
        if ep % 50 == 0 or ep in (1, 10, 25) or ep == epochs:
            model.eval()
            with torch.no_grad():
                pred = normalizer.denormalize(
                    model(
                        pxrd_x, pxrd_y, peak_num,
                        crystal_system_idx=cs_t, cs_route="oracle",
                        lattice_phys=lattice, setting_route="oracle",
                    )["lattice_norm"]
                )
            g = _geom(pred.cpu().numpy(), lattice.cpu().numpy(), cs_t.cpu().numpy())
            g["epoch"] = ep
            g["loss"] = float(losses["loss_total"].item())
            curve.append(g)
            print(
                f"[{name}] ep{ep:4} loss={g['loss']:.4f} elem={g['elem_ok_rate']*100:.1f}% "
                f"cub={g['cubic_elem_ok_rate']*100:.1f}% noncub={g['non_cubic_elem_ok_rate']*100:.1f}% "
                f"pull90={g['pulled_to_90_rate']*100:.0f}% ang={g['angle_mae']:.2f}",
                flush=True,
            )
    return {"variant": name, "curve": curve, "final": curve[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/diag_overfit_hist_mlp.yaml")
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/beat_engine/raw_diag/r3/p0_700.json")
    args = p.parse_args()
    config = TrainConfig.from_yaml(
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    ).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tensors = _load(config, device)
    print(f"n={int(tensors[2].shape[0])}", flush=True)

    variants = [
        ("cs_oracle_baseline", False, "baseline", 0.0),
        ("cs_oracle_cubic_split", True, "baseline", 0.0),
        ("cs_oracle_joint_phys_0.1", False, "joint_phys", 0.1),
        ("cs_oracle_split_joint_0.1", True, "joint_phys", 0.1),
        ("cs_oracle_joint_phys_0.05", False, "joint_phys", 0.05),
        ("cs_oracle_split_joint_0.05", True, "joint_phys", 0.05),
    ]
    results = {}
    for name, split, mode, pw in variants:
        results[name] = run_one(
            name, cubic_split=split, loss_mode=mode, phys_w=pw,
            config=config, tensors=tensors, epochs=args.epochs, lr=args.lr, device=device,
        )
    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== R3 P0-700 ===")
    for name, r in results.items():
        f = r["final"]
        print(
            f"{name:28} elem={f['elem_ok_rate']*100:5.1f}% cub={f['cubic_elem_ok_rate']*100:5.1f}% "
            f"noncub={f['non_cubic_elem_ok_rate']*100:5.1f}% pull90={f['pulled_to_90_rate']*100:4.0f}% "
            f"ang={f['angle_mae']:.2f}"
        )


if __name__ == "__main__":
    main()
