# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from archive/RealPXRD-Solver/app/model/bert.py (D17), extended for R1
# physical peak-token features (inverse_d2 / reciprocal_d).

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pxrd_cell_indexing.data.peak_features import (
    PeakFeatureConfig,
    PeakFeatureMode,
    padding_mask_from_peak_num,
    peak_feature_dim,
)
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
    """RealPXRD-style XRD peak-table encoder with optional physical peak tokens."""

    def __init__(self, pretrained: object = None, *arg: object, **kwargs: object) -> None:
        super().__init__()
        args = DotDict(**kwargs)
        self.padding_idx = 0
        # discrete: 2θ cast to long → nn.Embedding(max_seq_len) (RealPXRD default; loses sub-degree).
        # continuous: float 2θ → MLP (A3; preserves peak-position precision).
        # physical: multi-channel peak features (inverse_d2/I etc.); no separate pos embed.
        self.position_encoding = str(getattr(args, "position_encoding", "discrete"))
        if self.position_encoding not in ("discrete", "continuous", "physical"):
            raise ValueError(
                "position_encoding must be 'discrete', 'continuous', or 'physical', "
                f"got {self.position_encoding!r}"
            )
        self.peak_feature_mode: PeakFeatureMode = str(  # type: ignore[assignment]
            getattr(args, "peak_feature_mode", "legacy")
        )
        self.wavelength_angstrom = float(getattr(args, "wavelength_angstrom", 1.54184))
        self.intensity_transform = str(getattr(args, "intensity_transform", "linear"))
        self.two_theta_min_deg = float(getattr(args, "two_theta_min_deg", 5.0))
        self.two_theta_max_deg = float(getattr(args, "two_theta_max_deg", 80.0))
        if self.position_encoding == "physical" and self.peak_feature_mode == "legacy":
            self.peak_feature_mode = "inverse_d2_i"
        token_dim = (
            peak_feature_dim(self.peak_feature_mode)
            if self.position_encoding == "physical"
            else 1
        )
        self.token_dim = token_dim

        self.embed_tokens = nn.Sequential(
            nn.Linear(token_dim, args.encoder_embed_dim),
            nn.LayerNorm(args.encoder_embed_dim),
            nn.ReLU(True),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.vnode_encoder = nn.Embedding(1, args.encoder_embed_dim)
        if self.position_encoding == "physical":
            self.embed_positions = None
        elif self.position_encoding == "continuous":
            self.embed_positions = nn.Sequential(
                nn.Linear(1, args.encoder_embed_dim),
                nn.LayerNorm(args.encoder_embed_dim),
                nn.ReLU(True),
                nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
            )
        else:
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

    def _feature_config(self) -> PeakFeatureConfig:
        return PeakFeatureConfig(
            feature_mode=self.peak_feature_mode,
            wavelength_angstrom=self.wavelength_angstrom,
            intensity_transform=self.intensity_transform,  # type: ignore[arg-type]
            two_theta_min_deg=self.two_theta_min_deg,
            two_theta_max_deg=self.two_theta_max_deg,
        )

    def half(self) -> BertModel:
        super().half()
        self.embed_tokens = self.embed_tokens.float()
        if self.embed_positions is not None:
            self.embed_positions = self.embed_positions.float()
        self.vnode_encoder = self.vnode_encoder.float()
        self.dtype = torch.half
        return self

    def bfloat16(self) -> BertModel:
        super().bfloat16()
        self.embed_tokens = self.embed_tokens.float()
        if self.embed_positions is not None:
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
        batch_peak_x = torch.zeros(
            (peak_num.shape[0], max_peak_num, 1), device=pxrd_x.device, dtype=pxrd_x.dtype
        )
        batch_peak_y = torch.zeros(
            (peak_num.shape[0], max_peak_num, 1), device=pxrd_y.device, dtype=pxrd_y.dtype
        )
        idx = 0
        for i in range(len(peak_num)):
            n = int(peak_num[i].item())
            batch_peak_x[i, :n] = pxrd_x[idx : idx + n]
            batch_peak_y[i, :n] = pxrd_y[idx : idx + n]
            idx += n
        if self.position_encoding == "discrete":
            return batch_peak_x.long(), batch_peak_y
        return batch_peak_x, batch_peak_y

    def _batch_physical_tokens(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        """Pad per-peak physical features to ``[B, max_N, C]`` (vectorized)."""
        from pxrd_cell_indexing.data.peak_features import (
            inverse_d2_from_two_theta,
            inverse_d2_max,
            normalize_intensity,
            reciprocal_d_from_two_theta,
        )

        max_peak_num = int(peak_num.max().item())
        batch = peak_num.shape[0]
        device = pxrd_x.device
        tt = torch.zeros((batch, max_peak_num), device=device, dtype=torch.float32)
        inten = torch.zeros((batch, max_peak_num), device=device, dtype=torch.float32)
        idx = 0
        for i in range(batch):
            n = int(peak_num[i].item())
            if n <= 0:
                continue
            tt[i, :n] = pxrd_x[idx : idx + n].reshape(-1).to(dtype=torch.float32)
            inten[i, :n] = pxrd_y[idx : idx + n].reshape(-1).to(dtype=torch.float32)
            idx += n

        mode = self.peak_feature_mode
        if mode == "legacy":
            return inten.unsqueeze(-1)

        inv = inverse_d2_from_two_theta(tt, wavelength_angstrom=self.wavelength_angstrom)
        assert isinstance(inv, torch.Tensor)
        g_max = max(
            inverse_d2_max(
                wavelength_angstrom=self.wavelength_angstrom,
                two_theta_max_deg=self.two_theta_max_deg,
            ),
            1e-8,
        )
        inv_norm = inv / g_max
        recip = reciprocal_d_from_two_theta(tt, wavelength_angstrom=self.wavelength_angstrom)
        assert isinstance(recip, torch.Tensor)
        recip_max = max(
            float(
                reciprocal_d_from_two_theta(
                    np.array([self.two_theta_max_deg]),
                    wavelength_angstrom=self.wavelength_angstrom,
                )[0]
            ),
            1e-8,
        )
        recip_norm = recip / recip_max
        # Per-sample intensity normalization (ignore padded zeros via peak_num).
        inten_n = torch.zeros_like(inten)
        for i in range(batch):
            n = int(peak_num[i].item())
            if n <= 0:
                continue
            transform = "log" if mode == "inverse_d2_logi" else self.intensity_transform
            inten_n[i, :n] = normalize_intensity(  # type: ignore[assignment]
                inten[i, :n], transform=transform  # type: ignore[arg-type]
            )
        tt_norm = (tt - self.two_theta_min_deg) / max(
            self.two_theta_max_deg - self.two_theta_min_deg, 1e-8
        )
        if mode == "continuous_2theta_i":
            return torch.stack([tt_norm, inten_n], dim=-1)
        if mode == "reciprocal_d_i":
            return torch.stack([recip_norm, inten_n], dim=-1)
        if mode in ("inverse_d2_i", "inverse_d2_logi"):
            return torch.stack([inv_norm, inten_n], dim=-1)
        if mode == "inverse_d2_only":
            return inv_norm.unsqueeze(-1)
        raise ValueError(f"Unknown peak_feature_mode: {mode}")

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        if self.position_encoding == "physical":
            tokens = self._batch_physical_tokens(pxrd_x, pxrd_y, peak_num)
            x = self.embed_tokens(tokens)
            padding_mask = padding_mask_from_peak_num(peak_num, tokens.shape[1], include_cls=True)
        else:
            src_pos, src_tokens = self.batch_input(pxrd_x, pxrd_y, peak_num)
            x = self.embed_tokens(src_tokens).squeeze(-2)
            assert self.embed_positions is not None
            if self.position_encoding == "continuous":
                pos_embed = self.embed_positions(src_pos.float()).squeeze(-2)
            else:
                pos_embed = self.embed_positions(src_pos).squeeze(-2)
            x = x + pos_embed
            # Prefer peak_num mask; fall back to intensity==0 for legacy parity.
            padding_mask = padding_mask_from_peak_num(
                peak_num, src_tokens.shape[1], include_cls=True
            )

        cls_token = self.vnode_encoder.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x.type(self.sentence_encoder.emb_layer_norm.weight.dtype)
        if not padding_mask.any():
            padding_mask = None
        x = self.sentence_encoder(x, padding_mask=padding_mask)
        out_ = x[:, 0, :]
        return self.out(out_)
