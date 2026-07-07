"""Model heads and full indexing model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pxrd_cell_indexing.model.encoder.bert import BertModel
from pxrd_cell_indexing.model.encoder.loader import (
    DEFAULT_ENCODER_CONFIG,
    build_bert_model,
    load_xrd_encoder_from_checkpoint,
)


@dataclass(frozen=True)
class HeadConfig:
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_crystal_systems: int = 7
    dropout: float = 0.1


class CrystalSystemHead(nn.Module):
    """Seven-way crystal-system classifier (auxiliary task)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(config.embedding_dim, config.num_crystal_systems)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


class LatticeRegressionHead(nn.Module):
    """Predict all six primitive lattice parameters in normalized space."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 6),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class IndexingModel(nn.Module):
    """Encoder + crystal-system classifier + lattice regression head."""

    def __init__(
        self,
        encoder: BertModel,
        *,
        head_config: HeadConfig | None = None,
        freeze_encoder: bool = False,
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head_config = head_config or HeadConfig()
        self.normalize_embedding = normalize_embedding
        self.crystal_system_head = CrystalSystemHead(self.head_config)
        self.lattice_head = LatticeRegressionHead(self.head_config)
        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    @property
    def encoder_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.encoder.parameters())

    def encode(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encoder(pxrd_x, pxrd_y, peak_num)
        if self.normalize_embedding:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedding = self.encode(pxrd_x, pxrd_y, peak_num)
        return {
            "embedding": embedding,
            "crystal_system_logits": self.crystal_system_head(embedding),
            "lattice_norm": self.lattice_head(embedding),
        }


def build_indexing_model(
    *,
    checkpoint_path: str | None = None,
    encoder_config: dict[str, Any] | None = None,
    head_config: HeadConfig | None = None,
    freeze_encoder: bool = False,
    normalize_embedding: bool = True,
) -> IndexingModel:
    """Construct IndexingModel with optional pretrained encoder weights."""
    if checkpoint_path is None:
        encoder = build_bert_model(encoder_config or DEFAULT_ENCODER_CONFIG)
    else:
        encoder, _ = load_xrd_encoder_from_checkpoint(checkpoint_path, config=encoder_config)
    return IndexingModel(
        encoder,
        head_config=head_config,
        freeze_encoder=freeze_encoder,
        normalize_embedding=normalize_embedding,
    )
