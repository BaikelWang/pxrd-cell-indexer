"""Load RealPXRD ``xrd_encoder.*`` weights into the vendored BertModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import nn

from pxrd_cell_indexing.model.encoder.bert import BertModel

XRD_ENCODER_PREFIX = "xrd_encoder."

DEFAULT_ENCODER_CONFIG: dict[str, Any] = {
    "output_dim": 512,
    "max_seq_len": 180,
    "encoder_layers": 2,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 32,
    "encoder_attention_heads": 4,
    "dropout": 0.1,
    "emb_dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "activation_fn": "gelu",
    "post_ln": False,
}

REALPXRD_ENCODER_CHECKPOINT = Path(
    "/nanolab/users/wyx/archive/RealPXRD-Solver/pretrained/weight/2501/pxrd-all/last_one.ckpt"
)


class EncoderLoadReport(TypedDict):
    missing_keys: list[str]
    unexpected_keys: list[str]
    loaded_key_count: int
    checkpoint_path: str


def extract_xrd_encoder_state_dict(
    full_state_dict: dict[str, torch.Tensor],
    prefix: str = XRD_ENCODER_PREFIX,
) -> dict[str, torch.Tensor]:
    """Strip the Lightning ``xrd_encoder.`` prefix from a full checkpoint state dict."""
    return {
        key[len(prefix) :]: value
        for key, value in full_state_dict.items()
        if key.startswith(prefix)
    }


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a RealPXRD Lightning checkpoint and return its ``state_dict``."""
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def build_bert_model(config: dict[str, Any] | None = None) -> BertModel:
    """Instantiate BertModel with RealPXRD default hyperparameters."""
    model_cfg = dict(DEFAULT_ENCODER_CONFIG)
    if config:
        model_cfg.update(config)
    return BertModel(**model_cfg)


def load_xrd_encoder_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[BertModel, EncoderLoadReport]:
    """Build BertModel and load ``xrd_encoder.*`` weights from a RealPXRD checkpoint."""
    encoder = build_bert_model(config)
    full_state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=map_location)
    encoder_state_dict = extract_xrd_encoder_state_dict(full_state_dict)
    load_result = encoder.load_state_dict(encoder_state_dict, strict=strict)
    report: EncoderLoadReport = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "loaded_key_count": len(encoder_state_dict),
        "checkpoint_path": str(checkpoint_path),
    }
    return encoder, report


def count_encoder_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
