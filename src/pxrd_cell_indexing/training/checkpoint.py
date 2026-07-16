"""Shared checkpoint loading and config validation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from pxrd_cell_indexing.data.normalization import head_output_dim
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


def infer_canonical_convention_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Read A0 convention from checkpoint payload or nested config paths."""
    if checkpoint.get("canonical_convention"):
        return str(checkpoint["canonical_convention"])
    cfg = checkpoint.get("config") or {}
    data = cfg.get("data") or {}
    for key in ("canonical_convention", "label_convention"):
        if data.get(key):
            return str(data[key])
    for key in ("train_jsonl", "valid_jsonl", "lattice_stats"):
        path = str(data.get(key) or checkpoint.get("lattice_stats") or "").lower()
        if "niggli" in path:
            return "niggli"
        if "reduced" in path:
            return "reduced"
        if "primitive" in path:
            return "primitive"
    return "unknown"


def apply_checkpoint_protocol_to_config(
    config: TrainConfig,
    checkpoint: dict[str, Any],
) -> TrainConfig:
    """Override representation / lattice_stats from checkpoint when present (A0)."""
    cfg = checkpoint.get("config") or {}
    data = cfg.get("data") or {}
    representation = checkpoint.get("representation") or data.get("representation")
    lattice_stats = checkpoint.get("lattice_stats") or data.get("lattice_stats")
    if representation is not None:
        config.data.representation = representation  # type: ignore[assignment]
    if lattice_stats is not None:
        config.data.lattice_stats = str(lattice_stats)
    # Prefer checkpoint jsonl paths when present so Niggli labels match training.
    for key in ("train_jsonl", "valid_jsonl"):
        val = data.get(key)
        if val:
            setattr(config.data, key, str(val))
    return config


def load_indexing_model_from_checkpoint(
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
) -> tuple[IndexingModel, dict[str, Any], str]:
    """Build model, load weights, and resolve experiment name from checkpoint."""
    from pxrd_cell_indexing.training.trainer import _encoder_runtime_config

    model = build_indexing_model(
        checkpoint_path=config.model.encoder_checkpoint,
        encoder_config=_encoder_runtime_config(config),
        head_config=HeadConfig(
            embedding_dim=getattr(config.model, "embedding_dim", 512),
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            output_dim=head_output_dim(config.data.representation),
            head_type=config.model.head_type,
            use_cs_classifier=config.model.use_cs_classifier,
            default_cs_route=config.model.cs_route,
            cubic_bravais_split=config.model.cubic_bravais_split,
            default_setting_route=config.model.setting_route,
            use_cubic_setting_classifier=config.model.use_cubic_setting_classifier,
            multi_hypothesis=getattr(config.model, "multi_hypothesis", False),
            num_hypotheses=getattr(config.model, "num_hypotheses", 3),
            head_num_layers=getattr(config.model, "head_num_layers", 2),
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
