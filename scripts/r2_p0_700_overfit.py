#!/usr/bin/env python3
"""R2 P0-A/B: full-batch 700 overfit — shared vs oracle-CS heads vs angle prior.

Frozen histogram encoder settings match G2 champion (intensity_min=5).
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
from pxrd_cell_indexing.losses import LossWeights, IndexingLoss, bravais_angle_prior_loss
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
    by_cs = {}
    for i, name in enumerate(CRYSTAL_SYSTEMS):
        m = cs == i
        if not m.any():
            continue
        by_cs[name] = {
            "n": int(m.sum()),
            "elem_ok_rate": float(elem[m].mean()),
            "angle_mae": float(ang[m].mean()),
            "pulled_to_90_rate": float(np.mean(pred_dev[m] < truth_dev[m])),
        }
    return {
        "angle_mae": float(ang.mean()),
        "length_rel_mean": float(lr.mean()),
        "elem_ok_rate": float(elem.mean()),
        "non_cubic_elem_ok_rate": float(elem[non_cubic].mean()) if non_cubic.any() else float("nan"),
        "pulled_to_90_rate": float(np.mean(pred_dev < truth_dev)),
        "min_cs_elem_ok_rate": float(min(v["elem_ok_rate"] for v in by_cs.values())) if by_cs else float("nan"),
        "by_crystal_system": by_cs,
    }


def _load_tensors(config: TrainConfig, device: torch.device, intensity_min: float):
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb),
        split="valid",
        sample_list_path=Path(config.data.train_jsonl),
        peak_filter=PeakFilterConfig(intensity_min=intensity_min),
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
    pxrd_x = torch.cat(xs, dim=0).to(device)
    pxrd_y = torch.cat(ys, dim=0).to(device)
    peak_num = torch.cat(ns, dim=0).to(device)
    lattice = torch.cat(lats, dim=0).to(device)
    cs = torch.cat(css, dim=0)
    return pxrd_x, pxrd_y, peak_num, lattice, cs


def run_variant(
    name: str,
    *,
    head_type: str,
    loss_mode: str,
    config: TrainConfig,
    tensors,
    epochs: int,
    lr: float,
    device: torch.device,
    angle_prior_weight: float = 0.25,
) -> dict[str, Any]:
    pxrd_x, pxrd_y, peak_num, lattice, cs_t = tensors
    cs_np = cs_t.cpu().numpy()
    normalizer = build_lattice_normalizer(config.data)
    target = normalizer.normalize(lattice)
    torch.manual_seed(config.seed)
    model = build_indexing_model(
        checkpoint_path=None,
        encoder_config=dict(HIST_ENC),
        head_config=HeadConfig(
            hidden_dim=256,
            dropout=0.0,
            output_dim=head_output_dim(config.data.representation),
            head_type=head_type,  # type: ignore[arg-type]
            use_cs_classifier=False,
            default_cs_route="oracle",
        ),
        freeze_encoder=False,
        normalize_embedding=False,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = IndexingLoss(
        LossWeights(
            mode=loss_mode,  # type: ignore[arg-type]
            regression=1.0,
            angle_prior_weight=angle_prior_weight,
        ),
        normalizer=normalizer,
    )
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(
            pxrd_x,
            pxrd_y,
            peak_num,
            crystal_system_idx=cs_t,
            cs_route="oracle",
        )
        losses = loss_fn(
            out["lattice_norm"],
            target,
            lattice_phys_target=lattice,
            crystal_system_idx=cs_t,
        )
        losses["loss_total"].backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs or ep in (1, 10, 25):
            model.eval()
            with torch.no_grad():
                pred = normalizer.denormalize(
                    model(
                        pxrd_x,
                        pxrd_y,
                        peak_num,
                        crystal_system_idx=cs_t,
                        cs_route="oracle",
                    )["lattice_norm"]
                )
            g = _geom(pred.cpu().numpy(), lattice.cpu().numpy(), cs_np)
            g["epoch"] = ep
            g["loss"] = float(losses["loss_total"].item())
            curve.append(g)
            print(
                f"[{name}] ep{ep:4} loss={g['loss']:.4f} ang={g['angle_mae']:.2f} "
                f"elem={g['elem_ok_rate']*100:.1f}% noncub={g['non_cubic_elem_ok_rate']*100:.1f}% "
                f"pull90={g['pulled_to_90_rate']*100:.0f}% minCS={g['min_cs_elem_ok_rate']*100:.1f}%",
                flush=True,
            )
    return {"variant": name, "head_type": head_type, "loss_mode": loss_mode, "curve": curve, "final": curve[-1]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/diag_overfit_hist_mlp.yaml")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--intensity-min", type=float, default=5.0)
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/beat_engine/raw_diag/r2/p0_700_overfit.json",
    )
    args = p.parse_args()

    config = TrainConfig.from_yaml(
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    ).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tensors = _load_tensors(config, device, intensity_min=args.intensity_min)
    print(f"loaded n={int(tensors[2].shape[0])}", flush=True)

    variants = [
        ("shared_baseline", "shared", "baseline"),
        ("oracle_cs_baseline", "cs_conditional", "baseline"),
        ("shared_angle_prior", "shared", "angle_prior"),
        ("oracle_cs_angle_prior", "cs_conditional", "angle_prior"),
    ]
    results = {}
    for name, head_type, loss_mode in variants:
        results[name] = run_variant(
            name,
            head_type=head_type,
            loss_mode=loss_mode,
            config=config,
            tensors=tensors,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== R2 P0-700 SUMMARY ===")
    shared = results["shared_baseline"]["final"]
    for name, r in results.items():
        f = r["final"]
        # P0-A gate on oracle_cs: non-cubic ≥80%, pull90 ≤40%
        gate_a = (
            f["non_cubic_elem_ok_rate"] >= 0.80
            and f["pulled_to_90_rate"] <= 0.40
            and f["elem_ok_rate"] >= 0.80
        )
        d_noncub = f["non_cubic_elem_ok_rate"] - shared["non_cubic_elem_ok_rate"]
        d_pull = f["pulled_to_90_rate"] - shared["pulled_to_90_rate"]
        print(
            f"{name:24} elem={f['elem_ok_rate']*100:5.1f}% noncub={f['non_cubic_elem_ok_rate']*100:5.1f}% "
            f"(Δ{d_noncub*100:+.1f}) pull90={f['pulled_to_90_rate']*100:4.0f}% (Δ{d_pull*100:+.0f}) "
            f"ang={f['angle_mae']:.2f} {'PASS_A' if gate_a and 'oracle_cs' in name else ''}"
        )


if __name__ == "__main__":
    main()
