#!/usr/bin/env python3
"""D-A capacity sweep: can a LARGER encoder memorize the 700-sample set?

Builds IndexingModel with configurable (random-init) encoder sizes and trains
each on the tiny overfit set. If a bigger encoder drives train angle/length
error toward zero while the 32-dim/2-layer default plateaus, capacity is the
bottleneck. If even a large encoder plateaus, the limit is input ambiguity.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAPACITY_PRESETS: dict[str, dict[str, Any]] = {
    "tiny_32x2": {"encoder_embed_dim": 32, "encoder_layers": 2, "encoder_ffn_embed_dim": 32, "encoder_attention_heads": 4},
    "mid_128x4": {"encoder_embed_dim": 128, "encoder_layers": 4, "encoder_ffn_embed_dim": 256, "encoder_attention_heads": 8},
    "large_256x6": {"encoder_embed_dim": 256, "encoder_layers": 6, "encoder_ffn_embed_dim": 512, "encoder_attention_heads": 8},
}


def _geom(pred_phys: np.ndarray, truth_phys: np.ndarray) -> dict[str, float]:
    def la(p):
        t = torch.tensor(p, dtype=torch.float64).reshape(-1, 6)
        l, a = lattice_lengths_angles(lattice_params_to_matrix(t))
        return l.numpy(), a.numpy()
    lp, ap = la(pred_phys)
    lt, at = la(truth_phys)
    ang_err = np.abs(ap - at)
    len_rel = np.abs(lp - lt) / np.clip(lt, 1e-6, None)
    elem = (ang_err.max(1) <= 3.0) & (len_rel.max(1) <= 0.05)
    return {
        "angle_mae": float(ang_err.mean()),
        "length_rel_mean": float(len_rel.mean()),
        "elem_ok_rate": float(elem.mean()),
        "pulled_to_90_rate": float(np.mean(np.abs(ap - 90).mean(1) < np.abs(at - 90).mean(1))),
    }


def _load_all(loader, device):
    xs, ys, ns, lats = [], [], [], []
    for batch in loader:
        xs.append(batch["pxrd_x"]); ys.append(batch["pxrd_y"])
        ns.append(batch["peak_num"]); lats.append(batch["lattice"])
    return xs, ys, ns, lats


def run_preset(name: str, enc_cfg: dict, config, loader, device, *, epochs: int, lr: float) -> dict:
    model = build_indexing_model(
        checkpoint_path=None,  # random init to isolate capacity
        encoder_config={**enc_cfg, "position_encoding": "continuous"},
        head_config=HeadConfig(hidden_dim=256, dropout=0.0, output_dim=head_output_dim(config.data.representation)),
        freeze_encoder=False,
        normalize_embedding=True,
    ).to(device)
    normalizer = build_lattice_normalizer(config.data)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader:
            bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(bt["pxrd_x"], bt["pxrd_y"], bt["peak_num"])
            target = normalizer.normalize(bt["lattice"])
            loss = lossf(out["lattice_norm"], target)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for batch in loader:
                    bt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    out = model(bt["pxrd_x"], bt["pxrd_y"], bt["peak_num"])
                    preds.append(normalizer.denormalize(out["lattice_norm"]).cpu().numpy())
                    truths.append(bt["lattice"].cpu().numpy())
            g = _geom(np.concatenate(preds), np.concatenate(truths))
            g["epoch"] = ep
            g["train_loss"] = float(loss.item())
            curve.append(g)
            print(f"[{name}] ep{ep:3} loss={loss.item():.4f} ang={g['angle_mae']:.2f} len_rel={g['length_rel_mean']:.3f} elem={g['elem_ok_rate']*100:.1f}%", flush=True)
    return {"preset": name, "encoder": enc_cfg, "n_params": n_params, "curve": curve, "final": curve[-1]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/diag_overfit_discrete.yaml")
    p.add_argument("--jsonl", type=str, default="data/processed/overfit700_seed42.jsonl")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/beat_engine/raw_diag/overfit_capacity.json")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--presets", type=str, default="tiny_32x2,mid_128x4,large_256x6")
    args = p.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    jsonl = args.jsonl if Path(args.jsonl).is_absolute() else str(PROJECT_ROOT / args.jsonl)
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb),
        split="valid",
        sample_list_path=Path(jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=False,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg, batch_size=128, num_workers=4, shuffle=True,
        pin_memory=device.type == "cuda", prefetch_factor=2, persistent_workers=False,
    )

    results = {}
    for name in args.presets.split(","):
        name = name.strip()
        torch.manual_seed(config.seed)
        results[name] = run_preset(name, CAPACITY_PRESETS[name], config, loader, device, epochs=args.epochs, lr=args.lr)

    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (train-set fit) ===")
    for name, r in results.items():
        f = r["final"]
        print(f"{name:14} params={r['n_params']:>9,} ang={f['angle_mae']:.2f} len_rel={f['length_rel_mean']:.3f} elem={f['elem_ok_rate']*100:.1f}% pull90={f['pulled_to_90_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
