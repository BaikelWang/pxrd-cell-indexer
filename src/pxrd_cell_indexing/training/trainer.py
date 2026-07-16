"""Training loop for indexing smoke experiments."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pxrd_cell_indexing.data.dataset import (
    PeakFilterConfig,
    PXRDDatasetConfig,
    SpectrumAugmentConfig,
    build_dataloader,
)
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer, head_output_dim
from pxrd_cell_indexing.eval import (
    build_a0_metrics_block,
    evaluate_batch,
    evaluate_by_crystal_system,
    infer_crystal_system_idx_from_lattice,
    oracle_hyp_elementwise_rate,
    top1_elementwise_match_rate,
    top1_joint_match_rate,
    top1_lattice_match_rate,
)
from pxrd_cell_indexing.losses import IndexingLoss, compute_best_metric_score
from pxrd_cell_indexing.model.heads import (
    HeadConfig,
    build_indexing_model,
    load_warm_start_state_dict,
)
from pxrd_cell_indexing.training.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_canonical_convention(config_dict: dict[str, Any]) -> str:
    """Infer label convention from paths / explicit fields (A0 protocol)."""
    data = config_dict.get("data") or {}
    explicit = data.get("canonical_convention") or data.get("label_convention")
    if explicit:
        return str(explicit)
    for key in ("train_jsonl", "valid_jsonl", "lattice_stats"):
        path = str(data.get(key) or "").lower()
        if "niggli" in path:
            return "niggli"
        if "reduced" in path:
            return "reduced"
        if "primitive" in path:
            return "primitive"
    return "unknown"


def _encoder_runtime_config(config: TrainConfig) -> dict[str, Any]:
    model = config.model
    return {
        "position_encoding": model.position_encoding,
        "encoder_type": model.encoder_type,
        "peak_feature_mode": model.peak_feature_mode,
        "wavelength_angstrom": model.wavelength_angstrom,
        "intensity_transform": model.intensity_transform,
        "hist_bins": model.hist_bins,
        "sorted_peak_count": model.sorted_peak_count,
        "hist_pool": model.hist_pool,
        "histogram_hidden_dim": model.histogram_hidden_dim,
        "histogram_dropout": model.histogram_dropout,
        "histogram_num_blocks": model.histogram_num_blocks,
        "spectrum_bins": model.spectrum_bins,
        "spectrum_sigma_deg": model.spectrum_sigma_deg,
        "spectrum_cnn_channels": list(model.spectrum_cnn_channels),
        "spectrum_cnn_kernel": model.spectrum_cnn_kernel,
        "fusion_mode": model.fusion_mode,
        "fusion_branch_dim": model.fusion_branch_dim,
        "output_dim": model.embedding_dim,
        "peak_transformer_max_peaks": model.peak_transformer_max_peaks,
        "peak_transformer_token_mode": model.peak_transformer_token_mode,
        "peak_transformer_d_model": model.peak_transformer_d_model,
        "peak_transformer_num_layers": model.peak_transformer_num_layers,
        "peak_transformer_num_heads": model.peak_transformer_num_heads,
        "peak_transformer_ffn_dim": model.peak_transformer_ffn_dim,
        "peak_transformer_dropout": model.peak_transformer_dropout,
        "peak_transformer_fourier_freqs": model.peak_transformer_fourier_freqs,
        "peak_transformer_fourier_mode": model.peak_transformer_fourier_mode,
        "peak_transformer_g_floor": model.peak_transformer_g_floor,
        "peak_transformer_pool": model.peak_transformer_pool,
    }


def _build_loader(config: TrainConfig, split: str) -> Any:
    is_train = split == "train"
    data_cfg = config.data
    sample_list = Path(data_cfg.train_jsonl if is_train else data_cfg.valid_jsonl)
    if is_train and data_cfg.hard_cs_upsample > 1.0 + 1e-9:
        sample_list = _maybe_write_upsampled_jsonl(
            sample_list,
            run_dir=config.run_dir,
            upsample=data_cfg.hard_cs_upsample,
            hard_cs_names=data_cfg.hard_cs_names,
            seed=config.seed,
        )
    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(data_cfg.train_lmdb if is_train else data_cfg.valid_lmdb),
        split=split,
        sample_list_path=sample_list,
        peak_filter=PeakFilterConfig(
            intensity_min=config.model.intensity_min,
            max_peaks=config.model.max_peaks,
        ),
        xrd_augment=data_cfg.train_augment if is_train else data_cfg.valid_augment,
        augment=SpectrumAugmentConfig(shift_range=data_cfg.augment_shift_range),
        strict=False,
        seed_base=config.seed,
    )
    return build_dataloader(
        dataset_cfg,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=is_train,
        pin_memory=config.device.startswith("cuda"),
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
    )


def _maybe_write_upsampled_jsonl(
    source: Path,
    *,
    run_dir: Path,
    upsample: float,
    hard_cs_names: tuple[str, ...],
    seed: int,
) -> Path:
    """Duplicate hard-CS rows ``upsample`` times (integer factor) into run_dir."""
    factor = max(int(round(upsample)), 1)
    if factor <= 1:
        return source
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"train_upsampled_x{factor}_seed{seed}.jsonl"
    if out_path.exists():
        return out_path
    hard = set(hard_cs_names)
    lines = source.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        cs = rec.get("crystal_system")
        reps = factor if cs in hard else 1
        out_lines.extend([line] * reps)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(
        f"[B2 upsample] wrote {out_path} ({len(lines)} -> {len(out_lines)} rows, x{factor})",
        flush=True,
    )
    return out_path


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

        self.normalizer = build_lattice_normalizer(config.data)
        encoder_config = _encoder_runtime_config(config)
        self.model = build_indexing_model(
            checkpoint_path=config.model.encoder_checkpoint,
            encoder_config=encoder_config,
            head_config=HeadConfig(
                embedding_dim=config.model.embedding_dim,
                hidden_dim=config.model.hidden_dim,
                dropout=config.model.dropout,
                output_dim=head_output_dim(config.data.representation),
                head_type=config.model.head_type,
                use_cs_classifier=config.model.use_cs_classifier,
                default_cs_route=config.model.cs_route,
                cubic_bravais_split=config.model.cubic_bravais_split,
                default_setting_route=config.model.setting_route,
                use_cubic_setting_classifier=config.model.use_cubic_setting_classifier,
                multi_hypothesis=config.model.multi_hypothesis,
                num_hypotheses=config.model.num_hypotheses,
                head_num_layers=config.model.head_num_layers,
                cubic_exact=config.model.cubic_exact,
            ),
            freeze_encoder=config.model.freeze_encoder,
            normalize_embedding=config.model.normalize_embedding,
        ).to(self.device)
        self.model.set_normalizer(self.normalizer)
        if config.model.warm_start_checkpoint:
            # Skip only modules that are known to be incompatible across variants.
            # Histogram→histogram must load the encoder (R4 hard-CS finetune needs it).
            # Continuous/physical position encodings rename token/position embeds.
            skip: tuple[str, ...]
            if config.model.position_encoding in ("continuous", "physical"):
                skip = ("embed_positions", "embed_tokens")
            else:
                skip = ()
            report = load_warm_start_state_dict(
                self.model,
                config.model.warm_start_checkpoint,
                skip_key_substrings=skip,
                map_location=self.device,
            )
            print(f"[warm_start] skip={skip} {report}", flush=True)
        self.loss_fn = IndexingLoss(config.loss, normalizer=self.normalizer)
        self.optimizer = self._build_optimizer()
        self._train_loader_len: int | None = None
        self.scheduler = self._build_scheduler()
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.best_valid_metric = -math.inf
        self.best_valid_loss = math.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.best_loss_epoch = 0
        self.global_step = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        head_params = self.model.head_parameters()
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        param_groups: list[dict[str, Any]] = [
            {"params": head_params, "lr": self.config.optim.head_lr}
        ]
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.config.optim.encoder_lr}
            )
        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.optim.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        total_steps = self._estimate_total_steps()
        train_loader_len = self._train_loader_len
        assert train_loader_len is not None
        warmup_steps = int(self.config.optim.warmup_epochs * train_loader_len)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 1e-8)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _estimate_total_steps(self) -> int:
        if self._train_loader_len is None:
            train_loader = _build_loader(self.config, "train")
            self._train_loader_len = len(train_loader)
        accum = max(self.config.optim.accumulate_grad_batches, 1)
        return (self._train_loader_len * self.config.optim.max_epochs) // accum

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prepared["lattice_norm"] = self.normalizer.normalize(prepared["lattice"])
        return prepared

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return self.loss_fn(
            outputs["lattice_norm"],
            batch["lattice_norm"],
            lattice_phys_target=batch["lattice"],
            crystal_system_idx=batch["crystal_system_idx"],
            crystal_system_logits=outputs.get("crystal_system_logits"),
            cubic_setting_logits=outputs.get("cubic_setting_logits"),
            pxrd_x=batch["pxrd_x"],
            peak_num=batch["peak_num"],
            lattice_hyp_pred=outputs.get("lattice_hyp_selected"),
        )

    def _forward_model(
        self,
        batch: dict[str, Any],
        *,
        train: bool,
        cs_route_override: str | None = None,
    ) -> dict[str, torch.Tensor]:
        route = cs_route_override or (
            self.config.model.train_cs_route if train else self.config.model.cs_route
        )
        setting_route = (
            self.config.model.train_setting_route
            if train
            else self.config.model.setting_route
        )
        return self.model(
            batch["pxrd_x"],
            batch["pxrd_y"],
            batch["peak_num"],
            crystal_system_idx=batch["crystal_system_idx"],
            cs_route=route,
            lattice_phys=batch["lattice"],
            setting_route=setting_route,
        )

    def train(self) -> dict[str, Any]:
        train_loader = _build_loader(self.config, "train")
        valid_loader = _build_loader(self.config, "valid")
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.optim.max_epochs + 1):
            ft_ep = self.config.model.hard_cs_finetune_epoch
            if ft_ep is not None and epoch == ft_ep:
                self.model.set_encoder_trainable(False)
                self.model.set_cubic_heads_trainable(False)
                self.model.set_non_cubic_heads_trainable(True)
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()
                print(
                    f"[hard_cs_finetune] epoch>={ft_ep}: freeze encoder+cubic; "
                    f"rebuilt optimizer",
                    flush=True,
                )
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
            "best_valid_loss": self.best_valid_loss,
            "best_loss_epoch": self.best_loss_epoch,
            "best_metric_name": self.config.best_metric,
        }

    def _run_epoch(self, loader: Any, epoch: int, *, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_totals: dict[str, float] = {}
        metric_sums: dict[str, float] = {}
        metric_count = 0
        per_cs_accum: dict[str, dict[str, float]] = {}
        valid_top1_preds: list[list[float]] = []
        valid_oracle_preds: list[list[float]] = []
        valid_truths: list[list[float]] = []
        valid_pred_cs: list[int] = []
        valid_target_cs: list[int] = []
        valid_classifier_cs: list[int] = []
        valid_peak_counts: list[int] = []
        valid_hyp_oracle_preds: list[list[list[float]]] = []
        collect_oracle_route = (
            not train
            and bool(self.config.model.use_cs_classifier)
            and self.config.model.cs_route != "oracle"
        )

        data_wait_s = 0.0
        compute_s = 0.0
        step_count = 0
        accum_steps = max(self.config.optim.accumulate_grad_batches, 1)
        profile_timing = train and self.config.optim.profile_timing and epoch == 1

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        iter_start = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            if profile_timing:
                data_wait_s += time.perf_counter() - iter_start

            batch = self._prepare_batch(batch)
            compute_start = time.perf_counter() if profile_timing else 0.0

            if train:
                outputs = self._forward_model(batch, train=True)
                losses = self._compute_losses(outputs, batch)
                if not torch.isfinite(losses["loss_total"]):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
                scaled_loss = losses["loss_total"] / accum_steps
                scaled_loss.backward()
                if step % accum_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.optim.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.global_step % self.config.log_every == 0:
                        step_id = self.global_step
                        self.writer.add_scalar(
                            "train/loss_total", losses["loss_total"].item(), step_id
                        )
                        self.writer.add_scalar(
                            "train/loss_reg", losses["loss_reg"].item(), step_id
                        )
                        if "loss_phys" in losses:
                            self.writer.add_scalar(
                                "train/loss_phys", losses["loss_phys"].item(), step_id
                            )
                        if "loss_cls" in losses:
                            self.writer.add_scalar(
                                "train/loss_cls", losses["loss_cls"].item(), step_id
                            )
            else:
                with torch.no_grad():
                    outputs = self._forward_model(batch, train=False)
                    losses = self._compute_losses(outputs, batch)

            if profile_timing:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                compute_s += time.perf_counter() - compute_start
                step_count += 1

            metrics = evaluate_batch(outputs, batch, self.normalizer)
            for key, value in losses.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.item())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_count += 1

            if not train:
                pred = self.normalizer.denormalize(outputs["lattice_norm"])
                pred_cs_idx = infer_crystal_system_idx_from_lattice(pred.cpu())
                clf_logits = outputs.get("crystal_system_logits")
                if clf_logits is not None:
                    clf_idx = clf_logits.argmax(dim=-1).detach().cpu().tolist()
                else:
                    clf_idx = [-1] * int(pred.shape[0])
                oracle_pred = None
                if collect_oracle_route:
                    oracle_out = self._forward_model(
                        batch, train=False, cs_route_override="oracle"
                    )
                    oracle_pred = self.normalizer.denormalize(oracle_out["lattice_norm"])
                for idx in range(pred.shape[0]):
                    valid_top1_preds.append(pred[idx].cpu().tolist())
                    valid_truths.append(batch["lattice"][idx].cpu().tolist())
                    valid_pred_cs.append(int(pred_cs_idx[idx]))
                    valid_target_cs.append(int(batch["crystal_system_idx"][idx].item()))
                    valid_classifier_cs.append(int(clf_idx[idx]))
                    valid_peak_counts.append(int(batch["peak_num"][idx].item()))
                    if oracle_pred is not None:
                        valid_oracle_preds.append(oracle_pred[idx].cpu().tolist())
                if "lattice_hyp_all" in outputs:
                    # R10: gather each sample's *own ground-truth* CS hypothesis set
                    # (not the routed one) for an oracle candidate-recall diagnostic.
                    hyp_all = outputs["lattice_hyp_all"]  # [B, 7, K, D] normalized
                    gt_idx = batch["crystal_system_idx"].long()
                    hyp_gt = hyp_all[torch.arange(hyp_all.shape[0], device=hyp_all.device), gt_idx]
                    hyp_gt_phys = self.normalizer.denormalize(
                        hyp_gt.reshape(-1, hyp_gt.shape[-1])
                    ).reshape(hyp_gt.shape)
                    valid_hyp_oracle_preds.extend(hyp_gt_phys.cpu().tolist())
                per_cs = evaluate_by_crystal_system(
                    pred, batch["lattice"], batch["crystal_system_idx"]
                )
                for cs_name, cs_metrics in per_cs.items():
                    bucket = per_cs_accum.setdefault(
                        cs_name,
                        {
                            "lattice_mae": 0.0,
                            "top1_lattice_match_proxy": 0.0,
                            "count": 0.0,
                        },
                    )
                    count = cs_metrics["count"]
                    bucket["lattice_mae"] += cs_metrics["lattice_mae"] * count
                    bucket["top1_lattice_match_proxy"] += (
                        cs_metrics["top1_lattice_match_proxy"] * count
                    )
                    bucket["count"] += count

            iter_start = time.perf_counter()

        averaged = {key: value / max(metric_count, 1) for key, value in metric_sums.items()}
        averaged["loss"] = loss_totals.get("loss_total", 0.0) / max(metric_count, 1)
        for loss_key in (
            "loss_reg",
            "loss_matrix6",
            "loss_phys",
            "loss_length_phys",
            "loss_angle_phys",
            "loss_cls",
            "loss_setting",
        ):
            if loss_key in loss_totals:
                averaged[loss_key] = loss_totals[loss_key] / max(metric_count, 1)

        if not train and valid_top1_preds:
            loose_kw = {
                "ltol": self.config.eval_ltol,
                "atol_deg": self.config.eval_atol_deg,
            }
            strict_kw = {
                "ltol": self.config.strict_ltol,
                "atol_deg": self.config.strict_atol_deg,
            }
            # Historical / funnel metrics (loose by default).
            averaged["top1_lattice_match_rate"] = top1_lattice_match_rate(
                valid_top1_preds, valid_truths, **loose_kw
            )
            averaged["raw_top1_lattice_match_rate"] = averaged["top1_lattice_match_rate"]
            averaged["top1_joint_match_rate"] = top1_joint_match_rate(
                valid_top1_preds,
                valid_truths,
                valid_pred_cs,
                valid_target_cs,
                **loose_kw,
            )
            # North-star strict metrics (B4 dual-track).
            averaged["strict_raw_top1_lattice_match_rate"] = top1_lattice_match_rate(
                valid_top1_preds, valid_truths, **strict_kw
            )
            averaged["strict_raw_top1_elementwise_rate"] = top1_elementwise_match_rate(
                valid_top1_preds, valid_truths, **strict_kw
            )
            a0 = build_a0_metrics_block(
                preds_predicted=valid_top1_preds,
                preds_oracle=valid_oracle_preds or None,
                targets=valid_truths,
                peak_counts=valid_peak_counts,
                target_cs_idx=valid_target_cs,
                classifier_cs_idx=valid_classifier_cs
                if any(c >= 0 for c in valid_classifier_cs)
                else None,
                lattice_inferred_cs_idx=valid_pred_cs,
                ltol=self.config.strict_ltol,
                atol_deg=self.config.strict_atol_deg,
            )
            # Flatten selected A0 scalars into epoch metrics; keep nested dicts too.
            for key in (
                "classifier_cs_accuracy",
                "lattice_inferred_cs_accuracy",
                "cs_correct_subset_lattice_elementwise",
                "oracle_cs_strict_elementwise_rate",
                "predicted_cs_strict_elementwise_rate",
                "oracle_predicted_route_gap_pp",
            ):
                val = a0.get(key)
                if val is not None:
                    averaged[key] = float(val)
            averaged["a0"] = a0
            if valid_hyp_oracle_preds:
                # R10 gate diagnostics: oracle best-of-K recall vs the K=1 raw
                # baseline above, overall and restricted to non-cubic (cubic K=1
                # duplicates so it never differs from strict_raw_top1_elementwise).
                averaged["oracle_topk_elementwise_rate"] = oracle_hyp_elementwise_rate(
                    valid_hyp_oracle_preds, valid_truths, **strict_kw
                )
                non_cubic_idx = [i for i, cs in enumerate(valid_target_cs) if cs != 0]
                if non_cubic_idx:
                    averaged["oracle_topk_elementwise_rate_noncubic"] = (
                        oracle_hyp_elementwise_rate(
                            [valid_hyp_oracle_preds[i] for i in non_cubic_idx],
                            [valid_truths[i] for i in non_cubic_idx],
                            **strict_kw,
                        )
                    )

        if profile_timing and step_count > 0:
            averaged["data_wait_ms_per_step"] = 1000.0 * data_wait_s / step_count
            averaged["compute_ms_per_step"] = 1000.0 * compute_s / step_count

        prefix = "train" if train else "valid"
        for key, value in averaged.items():
            if key in ("per_crystal_system", "a0") or isinstance(value, dict):
                continue
            if value is None:
                continue
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

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        valid_metrics: dict[str, float],
        best_metric_name: str,
        best_metric_score: float,
    ) -> dict[str, Any]:
        cfg = self.config.to_dict()
        convention = _infer_canonical_convention(cfg)
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "valid_metrics": {
                k: v for k, v in valid_metrics.items() if k != "a0" and not isinstance(v, dict)
            },
            "valid_metrics_a0": valid_metrics.get("a0"),
            "best_metric_name": best_metric_name,
            "best_metric_score": best_metric_score,
            "config": cfg,
            # A0 protocol fields (also nested under config for older loaders).
            "representation": self.config.data.representation,
            "canonical_convention": convention,
            "lattice_stats": self.config.data.lattice_stats,
        }

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
                self._checkpoint_payload(
                    epoch=epoch,
                    valid_metrics=valid_metrics,
                    best_metric_name=self.config.best_metric,
                    best_metric_score=score,
                ),
                ckpt_path,
            )
            # Also keep an explicit elementwise alias for R9/R10 dual-track.
            ew_path = self.run_dir / "checkpoints" / "best_valid_elementwise.pt"
            torch.save(
                self._checkpoint_payload(
                    epoch=epoch,
                    valid_metrics=valid_metrics,
                    best_metric_name="strict_raw_top1_elementwise_rate",
                    best_metric_score=float(
                        valid_metrics.get("strict_raw_top1_elementwise_rate", score)
                    ),
                ),
                ew_path,
            )
        else:
            self.epochs_without_improvement += 1

        if self.config.optim.save_best_loss:
            valid_loss = float(valid_metrics.get("loss", math.inf))
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_loss_epoch = epoch
                torch.save(
                    self._checkpoint_payload(
                        epoch=epoch,
                        valid_metrics=valid_metrics,
                        best_metric_name="loss",
                        best_metric_score=-valid_loss,
                    ),
                    self.run_dir / "checkpoints" / "best_valid_loss.pt",
                )

        # Always refresh last.pt for crash recovery / late-epoch inspection.
        torch.save(
            self._checkpoint_payload(
                epoch=epoch,
                valid_metrics=valid_metrics,
                best_metric_name="last",
                best_metric_score=score,
            ),
            self.run_dir / "checkpoints" / "last.pt",
        )

        patience = self.config.optim.early_stop_patience
        min_epochs = max(int(self.config.optim.min_epochs), 1)
        if (
            patience is not None
            and epoch >= min_epochs
            and self.epochs_without_improvement >= patience
        ):
            return True
        return False