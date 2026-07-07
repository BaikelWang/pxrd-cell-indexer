"""Training configuration (native yaml + dataclass, D18)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pxrd_cell_indexing.losses import LossWeights

BestMetricName = Literal[
    "top1_lattice_match_proxy",
    "top1_lattice_match_rate",
    "top1_joint_match_rate",
    "crystal_system_accuracy",
    "composite",
]


@dataclass
class DataConfig:
    train_lmdb: str
    valid_lmdb: str
    train_jsonl: str
    valid_jsonl: str
    lattice_stats: str
    batch_size: int = 64
    num_workers: int = 4
    train_augment: bool = True
    valid_augment: bool = False


@dataclass
class ModelConfig:
    encoder_checkpoint: str
    freeze_encoder: bool = True
    normalize_embedding: bool = True
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class OptimConfig:
    head_lr: float = 1e-3
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: float = 1.0
    max_epochs: int = 5
    grad_clip: float = 1.0
    early_stop_patience: int | None = None


@dataclass
class TrainConfig:
    experiment_name: str
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "results/experiments"
    data: DataConfig = field(default_factory=DataConfig)  # type: ignore[arg-type]
    model: ModelConfig = field(default_factory=ModelConfig)  # type: ignore[arg-type]
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    best_metric: BestMetricName = "top1_lattice_match_rate"
    log_every: int = 20
    eval_every: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        loss_raw = dict(raw.get("loss", {}))
        return cls(
            experiment_name=raw["experiment_name"],
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
            output_dir=raw.get("output_dir", "results/experiments"),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            optim=OptimConfig(**raw["optim"]),
            loss=LossWeights(**loss_raw),
            best_metric=raw.get("best_metric", "top1_lattice_match_rate"),
            log_every=raw.get("log_every", 20),
            eval_every=raw.get("eval_every", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    def resolve_paths(self, project_root: str | Path) -> TrainConfig:
        """Resolve relative data/model/output paths against project root."""
        root = Path(project_root)
        data = self.data
        for field_name in (
            "train_lmdb",
            "valid_lmdb",
            "train_jsonl",
            "valid_jsonl",
            "lattice_stats",
        ):
            value = getattr(data, field_name)
            path = Path(value)
            if not path.is_absolute():
                setattr(data, field_name, str((root / path).resolve()))
        if not Path(self.model.encoder_checkpoint).is_absolute():
            self.model.encoder_checkpoint = str(
                (root / self.model.encoder_checkpoint).resolve()
            )
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str((root / self.output_dir).resolve())
        return self
