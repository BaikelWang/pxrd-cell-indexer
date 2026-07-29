#!/usr/bin/env python3
"""Train RealPXRD FT arms A / B / C on the shared 10k subset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import (
    PROJECT,
    PeakLatticeSubset,
    build_or_load_subset,
    collate_peaks,
    fit_normalizer,
    load_bert_from_ckpt,
    load_cspflow_from_ckpt,
    volume_from_six,
)
from .models import ArmAModel, ArmBModel, ArmCModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-pretrained", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: results/ft_realpxrd/arm{A,B,C}",
    )
    return ap.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, step, total_steps, base_lrs, warmup_frac=0.05):
    warm = max(1, int(total_steps * warmup_frac))
    if step < warm:
        scale = (step + 1) / warm
    else:
        progress = (step - warm) / max(1, total_steps - warm)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg, base in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base * scale


@torch.no_grad()
def eval_regression(model, loader, normalizer, device, arm: str) -> dict:
    model.eval()
    losses = []
    rel_lens = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if arm == "A":
            loss = model.forward_flow_loss(batch)
            losses.append(float(loss.item()))
            continue
        pred_n = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
        tgt_n = batch["lattice_norm"]
        loss = F.smooth_l1_loss(pred_n, tgt_n)
        # volume aux (already in train; report only)
        losses.append(float(loss.item()))
        for i in range(pred_n.size(0)):
            pred_six = normalizer.decode(pred_n[i].detach().cpu().numpy())
            true_six = batch["lattice_six"][i].detach().cpu().tolist()
            try:
                pl = np.sort(pred_six[:3])
                tl = np.sort(true_six[:3])
                rel_lens.append(float(np.mean(np.abs(pl - tl) / np.maximum(tl, 1e-6))))
            except Exception:
                pass
    out = {"loss": float(np.mean(losses)) if losses else 1e9}
    if rel_lens:
        out["mean_rel_len"] = float(np.mean(rel_lens))
    return out


def train_one(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or (PROJECT / "results/ft_realpxrd" / f"arm{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = build_or_load_subset(seed=42)
    print(f"subset train={len(subset['train_indices'])} valid={len(subset['valid_indices'])}")

    normalizer = fit_normalizer(Path(subset["train_lmdb"]), subset["train_indices"])
    (out_dir / "normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2))

    need_matrix = args.arm == "A"
    train_ds = PeakLatticeSubset(
        subset["train_lmdb"],
        subset["train_indices"],
        xrd_augment=True,
        normalizer=normalizer,
        need_matrix=need_matrix,
    )
    valid_ds = PeakLatticeSubset(
        subset["valid_lmdb"],
        subset["valid_indices"],
        xrd_augment=False,
        normalizer=normalizer,
        need_matrix=need_matrix,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_peaks,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_peaks,
        pin_memory=True,
    )

    if args.arm == "A":
        bundle, hp, missing, unexpected = load_cspflow_from_ckpt(device)
        print("ArmA load missing", len(missing), "unexpected", len(unexpected))
        model = ArmAModel(bundle, timesteps=hp["timesteps"]).to(device)
        # smaller batch recommended
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        base_lrs = [args.lr]
    elif args.arm == "B":
        enc, missing, unexpected = load_bert_from_ckpt(device, continuous_pos=False)
        print("ArmB bert missing", missing, "unexpected", unexpected)
        model = ArmBModel(enc, freeze_encoder=True).to(device)
        opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
        base_lrs = [args.lr]
    else:
        enc, missing, unexpected = load_bert_from_ckpt(device, continuous_pos=True)
        print("ArmC bert missing", [m for m in missing if "pos" not in m][:5], "n_missing", len(missing))
        model = ArmCModel(enc).to(device)
        pretrained_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
        head_params = list(model.head.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": args.lr},
                {"params": pretrained_params, "lr": args.lr_pretrained},
            ],
            weight_decay=1e-4,
        )
        base_lrs = [args.lr, args.lr_pretrained]

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params {n_train:,} / {n_all:,}")

    total_steps = args.epochs * max(1, len(train_loader))
    step = 0
    best = {"loss": 1e9, "epoch": -1}
    history = []
    bad = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            if args.arm == "A":
                loss = model.forward_flow_loss(batch)
            else:
                pred = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
                tgt = batch["lattice_norm"]
                loss = F.smooth_l1_loss(pred, tgt)
                # weak volume prior via sum of log edge lengths (differentiable)
                mean = torch.tensor(normalizer.mean, device=device, dtype=pred.dtype)
                std = torch.tensor(normalizer.std, device=device, dtype=pred.dtype)
                r = pred * std + mean
                log_abc_pred = r[:, 0] + r[:, 1] + r[:, 2]
                true_six = batch["lattice_six"]
                log_abc_true = (
                    torch.log(true_six[:, 0].clamp(min=1e-3))
                    + torch.log(true_six[:, 1].clamp(min=1e-3))
                    + torch.log(true_six[:, 2].clamp(min=1e-3))
                )
                loss = loss + 0.1 * F.smooth_l1_loss(log_abc_pred, log_abc_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            cosine_lr(opt, step, total_steps, base_lrs)
            step += 1
            tr_losses.append(float(loss.item()))

        val = eval_regression(model, valid_loader, normalizer, device, args.arm)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)),
            "valid": val,
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        print(
            f"arm{args.arm} ep{epoch:03d} train={row['train_loss']:.4f} "
            f"valid_loss={val['loss']:.4f} "
            f"rel_len={val.get('mean_rel_len', float('nan')):.3f} lr={row['lr']:.2e}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        score = val["loss"]
        if score < best["loss"] - 1e-5:
            best = {"loss": score, "epoch": epoch, **val}
            torch.save(
                {
                    "arm": args.arm,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "normalizer": normalizer.to_dict(),
                    "args": vars(args),
                    "valid": val,
                },
                out_dir / "best.pt",
            )
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

        torch.save(
            {
                "arm": args.arm,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "normalizer": normalizer.to_dict(),
                "args": vars(args),
            },
            out_dir / "last.pt",
        )

    (out_dir / "best_meta.json").write_text(json.dumps(best, indent=2))
    print(f"done arm{args.arm} best={best}", flush=True)


def main():
    args = parse_args()
    # Fix Path serialization
    if args.out_dir is not None:
        args.out_dir = Path(args.out_dir)
    train_one(args)


if __name__ == "__main__":
    main()
