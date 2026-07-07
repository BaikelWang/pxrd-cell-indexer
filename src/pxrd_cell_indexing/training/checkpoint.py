"""Shared checkpoint loading and config validation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.model.heads import HeadConfig, IndexingModel, build_indexing_model
from pxrd_cell_indexing.training.config import TrainConfig


def resolve_checkpoint_experiment_name(
    checkpoint: dict[str, Any],
    cli_config: TrainConfig,
) -> tuple[str, bool]:
    """Return experiment name to use and whether CLI vs checkpoint mismatched."""
    ckpt_config = checkpoint.get("config") or {}
    ckpt_exp = ckpt_config.get("experiment_name", cli_config.experiment_name)
    mismatch = ckpt_exp != cli_config.experiment_name
    if mismatch:
        warnings.warn(
            f"CLI config experiment_name={cli_config.experiment_name!r} "
            f"differs from checkpoint experiment_name={ckpt_exp!r}; "
            f"using checkpoint value for result metadata.",
            stacklevel=2,
        )
    return ckpt_exp, mismatch


def load_indexing_model_from_checkpoint(
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> tuple[IndexingModel, dict[str, Any], str]:
    """Build model, load weights, and resolve experiment name from checkpoint."""
    model = build_indexing_model(
        checkpoint_path=config.model.encoder_checkpoint,
        head_config=HeadConfig(
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        ),
        freeze_encoder=config.model.freeze_encoder,
        normalize_embedding=config.model.normalize_embedding,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    experiment_name, _ = resolve_checkpoint_experiment_name(checkpoint, cli_config=config)
    return model, checkpoint, experiment_name
