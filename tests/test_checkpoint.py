"""Tests for checkpoint/config consistency and path resolution."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from pxrd_cell_indexing.training.checkpoint import (
    load_indexing_model_from_checkpoint,
    resolve_checkpoint_experiment_name,
)
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _minimal_config(experiment_name: str = "cli_experiment") -> TrainConfig:
    config = TrainConfig.from_yaml(PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml")
    config.experiment_name = experiment_name
    return config


def test_resolve_checkpoint_experiment_name_mismatch_warns() -> None:
    cli_config = _minimal_config("cli_experiment")
    checkpoint = {"config": {"experiment_name": "ckpt_experiment"}}
    with pytest.warns(UserWarning, match="differs from checkpoint"):
        name, mismatch = resolve_checkpoint_experiment_name(checkpoint, cli_config)
    assert name == "ckpt_experiment"
    assert mismatch is True


def test_resolve_checkpoint_experiment_name_match() -> None:
    cli_config = _minimal_config("same_experiment")
    checkpoint = {"config": {"experiment_name": "same_experiment"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        name, mismatch = resolve_checkpoint_experiment_name(checkpoint, cli_config)
    assert name == "same_experiment"
    assert mismatch is False


def test_train_config_resolve_paths_makes_absolute() -> None:
    config = TrainConfig.from_yaml(PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml")
    config.resolve_paths(PROJECT_ROOT)
    assert Path(config.data.train_jsonl).is_absolute()
    assert Path(config.data.valid_jsonl).is_absolute()
    assert Path(config.data.lattice_stats).is_absolute()
    assert Path(config.model.encoder_checkpoint).is_absolute()
    assert Path(config.output_dir).is_absolute()


def test_load_indexing_model_from_checkpoint_uses_checkpoint_experiment_name(
    tmp_path: Path,
) -> None:
    config = TrainConfig.from_yaml(PROJECT_ROOT / "configs" / "smoke_unfrozen.yaml")
    config.resolve_paths(PROJECT_ROOT)
    config.experiment_name = "cli_wrong_name"
    device = torch.device("cpu")
    from pxrd_cell_indexing.model.heads import HeadConfig, build_indexing_model

    built = build_indexing_model(
        checkpoint_path=config.model.encoder_checkpoint,
        head_config=HeadConfig(
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        ),
        freeze_encoder=config.model.freeze_encoder,
        normalize_embedding=config.model.normalize_embedding,
    )
    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": built.state_dict(),
            "config": {"experiment_name": "ckpt_true_name"},
        },
        ckpt_path,
    )
    with pytest.warns(UserWarning, match="differs from checkpoint"):
        _, _, experiment_name = load_indexing_model_from_checkpoint(
            ckpt_path, config, device
        )
    assert experiment_name == "ckpt_true_name"
