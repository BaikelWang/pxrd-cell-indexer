"""Training loop for indexing smoke experiments."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import LatticeNormalizer
from pxrd_cell_indexing.eval import (
    evaluate_batch,
    evaluate_by_crystal_system,
    top1_joint_match_rate,
    top1_lattice_match_rate,
)
from pxrd_cell_indexing.losses import IndexingLoss, compute_best_metric_score
from pxrd_cell_indexing.model.heads import HeadConfig, build_indexing_model
from pxrd_cell_indexing.training.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loader(config: TrainConfig, split: str) -> Any:
    is_train = split == "train"
    data_cfg = config.data
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(data_cfg.train_lmdb if is_train else data_cfg.valid_lmdb),
        split=split,
        sample_list_path=Path(data_cfg.train_jsonl if is_train else data_cfg.valid_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=data_cfg.train_augment if is_train else data_cfg.valid_augment,
        strict=False,
        seed_base=config.seed,
    )
    return build_dataloader(
        dataset_cfg,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=is_train,
        pin_memory=config.device.startswith("cuda"),
    )


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        set_seed(config.seed)
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)

        self.normalizer = LatticeNormalizer.from_json(config.data.lattice_stats)
        self.model = build_indexing_model(
            checkpoint_path=config.model.encoder_checkpoint,
            head_config=HeadConfig(
                hidden_dim=config.model.hidden_dim,
                dropout=config.model.dropout,
            ),
            freeze_encoder=config.model.freeze_encoder,
            normalize_embedding=config.model.normalize_embedding,
        ).to(self.device)
        self.loss_fn = IndexingLoss(config.loss)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.best_valid_metric = -math.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.global_step = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        head_params = list(self.model.crystal_system_head.parameters()) + list(
            self.model.lattice_head.parameters()
        )
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        param_groups: list[dict[str, Any]] = [
            {"params": head_params, "lr": self.config.optim.head_lr}
        ]
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.config.optim.encoder_lr}
            )
        uncertainty_params = self.loss_fn.uncertainty_parameters()
        if uncertainty_params:
            param_groups.append({"params": uncertainty_params, "lr": self.config.optim.head_lr})
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.optim.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        total_steps = self._estimate_total_steps()
        train_loader = _build_loader(self.config, "train")
        warmup_steps = int(self.config.optim.warmup_epochs * len(train_loader))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 1e-8)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _estimate_total_steps(self) -> int:
        train_loader = _build_loader(self.config, "train")
        return len(train_loader) * self.config.optim.max_epochs

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prepared["lattice_norm"] = self.normalizer.normalize(prepared["lattice"])
        return prepared

    def train(self) -> dict[str, Any]:
        train_loader = _build_loader(self.config, "train")
        valid_loader = _build_loader(self.config, "valid")
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.optim.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            valid_metrics = self._run_epoch(valid_loader, epoch, train=False)
            epoch_summary = {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
            history.append(epoch_summary)
            should_stop = self._maybe_save_best(valid_metrics, epoch)
            with (self.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            if should_stop:
                break

        self.writer.close()
        return {
            "history": history,
            "best_valid_metric": self.best_valid_metric,
            "best_epoch": self.best_epoch,
            "best_metric_name": self.config.best_metric,
        }

    def _run_epoch(self, loader: Any, epoch: int, *, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_totals: list[float] = []
        metric_sums: dict[str, float] = {}
        metric_count = 0
        per_cs_accum: dict[str, dict[str, float]] = {}
        valid_top1_preds: list[list[float]] = []
        valid_truths: list[list[float]] = []
        valid_pred_cs: list[int] = []
        valid_target_cs: list[int] = []

        for step, batch in enumerate(loader, start=1):
            batch = self._prepare_batch(batch)
            if train:
                outputs = self.model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
                losses = self.loss_fn(
                    outputs["crystal_system_logits"],
                    outputs["lattice_norm"],
                    batch["crystal_system_idx"],
                    batch["lattice_norm"],
                )
                if not torch.isfinite(losses["loss_total"]):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                if step % self.config.log_every == 0:
                    step_id = self.global_step
                    self.writer.add_scalar("train/loss_total", losses["loss_total"].item(), step_id)
                    self.writer.add_scalar("train/loss_cls", losses["loss_cls"].item(), step_id)
                    self.writer.add_scalar("train/loss_reg", losses["loss_reg"].item(), step_id)
            else:
                with torch.no_grad():
                    outputs = self.model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
                    losses = self.loss_fn(
                        outputs["crystal_system_logits"],
                        outputs["lattice_norm"],
                        batch["crystal_system_idx"],
                        batch["lattice_norm"],
                    )

            metrics = evaluate_batch(outputs, batch, self.normalizer)
            loss_totals.append(float(losses["loss_total"].item()))
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_count += 1

            if not train:
                pred = self.normalizer.denormalize(outputs["lattice_norm"])
                pred_cs_idx = outputs["crystal_system_logits"].argmax(dim=-1)
                for idx in range(pred.shape[0]):
                    valid_top1_preds.append(pred[idx].cpu().tolist())
                    valid_truths.append(batch["lattice"][idx].cpu().tolist())
                    valid_pred_cs.append(int(pred_cs_idx[idx].item()))
                    valid_target_cs.append(int(batch["crystal_system_idx"][idx].item()))
                per_cs = evaluate_by_crystal_system(
                    pred, batch["lattice"], batch["crystal_system_idx"]
                )
                for cs_name, cs_metrics in per_cs.items():
                    bucket = per_cs_accum.setdefault(
                        cs_name, {"lattice_mae": 0.0, "top1_lattice_match_proxy": 0.0, "count": 0.0}
                    )
                    count = cs_metrics["count"]
                    bucket["lattice_mae"] += cs_metrics["lattice_mae"] * count
                    bucket["top1_lattice_match_proxy"] += (
                        cs_metrics["top1_lattice_match_proxy"] * count
                    )
                    bucket["count"] += count

        averaged = {key: value / max(metric_count, 1) for key, value in metric_sums.items()}
        averaged["loss"] = sum(loss_totals) / max(len(loss_totals), 1)
        if not train and valid_top1_preds:
            averaged["top1_lattice_match_rate"] = top1_lattice_match_rate(
                valid_top1_preds, valid_truths
            )
            averaged["top1_joint_match_rate"] = top1_joint_match_rate(
                valid_top1_preds,
                valid_truths,
                valid_pred_cs,
                valid_target_cs,
            )
        prefix = "train" if train else "valid"
        for key, value in averaged.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, epoch)

        if not train and per_cs_accum:
            per_cs_summary = {}
            for cs_name, bucket in per_cs_accum.items():
                count = max(bucket["count"], 1.0)
                per_cs_summary[cs_name] = {
                    "lattice_mae": bucket["lattice_mae"] / count,
                    "top1_lattice_match_proxy": bucket["top1_lattice_match_proxy"] / count,
                    "count": bucket["count"],
                }
            averaged["per_crystal_system"] = per_cs_summary
            per_cs_path = self.run_dir / f"valid_per_cs_epoch{epoch}.json"
            with per_cs_path.open("w", encoding="utf-8") as handle:
                json.dump(per_cs_summary, handle, indent=2)

        return averaged

    def _maybe_save_best(self, valid_metrics: dict[str, float], epoch: int) -> bool:
        score = compute_best_metric_score(
            valid_metrics,
            best_metric=self.config.best_metric,
        )
        if score > self.best_valid_metric:
            self.best_valid_metric = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            ckpt_path = self.run_dir / "checkpoints" / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "valid_metrics": valid_metrics,
                    "best_metric_name": self.config.best_metric,
                    "best_metric_score": score,
                    "config": self.config.to_dict(),
                },
                ckpt_path,
            )
            return False

        self.epochs_without_improvement += 1
        patience = self.config.optim.early_stop_patience
        if patience is not None and self.epochs_without_improvement >= patience:
            return True
        return False
