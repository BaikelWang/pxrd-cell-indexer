# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from archive/RealPXRD-Solver/app/model/bert.py (D17).

from __future__ import annotations

import torch
import torch.nn as nn

from pxrd_cell_indexing.model.encoder.transformer.transformer_encoder import (
    TransformerEncoder,
    init_bert_params,
)


class DotDict:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str) -> object:
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'DotDict' object has no attribute '{name}'")


class BertModel(nn.Module):
    """RealPXRD-style XRD peak-table encoder."""

    def __init__(self, pretrained: object = None, *arg: object, **kwargs: object) -> None:
        super().__init__()
        args = DotDict(**kwargs)
        self.padding_idx = 0

        self.embed_tokens = nn.Sequential(
            nn.Linear(1, args.encoder_embed_dim),
            nn.LayerNorm(args.encoder_embed_dim),
            nn.ReLU(True),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.vnode_encoder = nn.Embedding(1, args.encoder_embed_dim)
        self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)

        self.sentence_encoder = TransformerEncoder(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            rel_pos=False,
            rel_pos_bins=320,
            max_rel_pos=1280,
            post_ln=args.post_ln,
        )

        self.apply(init_bert_params)
        self.out = nn.Linear(args.encoder_embed_dim, args.output_dim)

    def half(self) -> BertModel:
        super().half()
        self.embed_tokens = self.embed_tokens.float()
        self.embed_positions = self.embed_positions.float()
        self.vnode_encoder = self.vnode_encoder.float()
        self.dtype = torch.half
        return self

    def bfloat16(self) -> BertModel:
        super().bfloat16()
        self.embed_tokens = self.embed_tokens.float()
        self.embed_positions = self.embed_positions.float()
        self.vnode_encoder = self.vnode_encoder.float()
        self.dtype = torch.bfloat16
        return self

    def float(self) -> BertModel:
        super().float()
        self.dtype = torch.float32
        return self

    def batch_input(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_peak_num = int(peak_num.max().item())
        batch_peak_x = torch.zeros((peak_num.shape[0], max_peak_num, 1), device=pxrd_x.device)
        batch_peak_y = torch.zeros((peak_num.shape[0], max_peak_num, 1), device=pxrd_y.device)
        idx = 0
        for i in range(len(peak_num)):
            n = int(peak_num[i].item())
            batch_peak_x[i, :n] = pxrd_x[idx : idx + n]
            batch_peak_y[i, :n] = pxrd_y[idx : idx + n]
            idx += n
        return batch_peak_x.long(), batch_peak_y

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        src_pos, src_tokens = self.batch_input(pxrd_x, pxrd_y, peak_num)
        x = self.embed_tokens(src_tokens).squeeze(-2)
        pos_embed = self.embed_positions(src_pos).squeeze(-2)
        x = x + pos_embed

        cls_token = self.vnode_encoder.weight.unsqueeze(0).repeat(src_tokens.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = x.type(self.sentence_encoder.emb_layer_norm.weight.dtype)

        padding_mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        padding_mask[:, 1:] = src_tokens.eq(self.padding_idx).squeeze()
        if not padding_mask.any():
            padding_mask = None

        x = self.sentence_encoder(x, padding_mask=padding_mask)
        out_ = x[:, 0, :]
        return self.out(out_)
