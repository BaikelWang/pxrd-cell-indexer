"""AIdex-style peak-geometry Transformer encoder (Phase A2).

Sorts peaks by ascending 2θ / inverse_d², builds physical tokens, and encodes
with a small pre-LN Transformer. Does **not** reuse RealPXRD discrete-2θ Bert.

P0 diagnosis (2026-07-15): a plain Linear(g) projection stalls at the mean-floor
on Niggli-700 (~5% strict / ~45% loose). Fourier features on ``g`` plus
CLS+mean pooling restore learning (probe: ~93% loose @800 steps).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn

from pxrd_cell_indexing.data.peak_features import (
    inverse_d2_from_two_theta,
    inverse_d2_max,
)

TokenMode = Literal["pos", "pos_i", "geom"]
PoolMode = Literal["cls", "mean", "cls_mean", "attn", "cls_mean_max"]
FourierMode = Literal["linear", "log", "loglinear"]


def _token_feature_dim(mode: TokenMode) -> int:
    if mode == "pos":
        return 1
    if mode == "pos_i":
        return 2
    if mode == "geom":
        return 4
    raise ValueError(f"Unknown peak_transformer_token_mode: {mode}")


def _extra_feat_dim(mode: TokenMode) -> int:
    """Non-g channels concatenated after Fourier(g)."""
    return _token_feature_dim(mode) - 1


class PeakGeometryTransformerEncoder(nn.Module):
    """Low-angle peak sequence → embedding (A2)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.output_dim = int(cfg.get("output_dim", 512))
        self.max_peaks = int(cfg.get("peak_transformer_max_peaks", 20))
        mode = str(cfg.get("peak_transformer_token_mode", "geom"))
        if mode not in ("pos", "pos_i", "geom"):
            raise ValueError(f"peak_transformer_token_mode must be pos|pos_i|geom, got {mode}")
        self.token_mode: TokenMode = mode  # type: ignore[assignment]
        pool = str(cfg.get("peak_transformer_pool", "cls_mean"))
        if pool not in ("cls", "mean", "cls_mean", "attn", "cls_mean_max"):
            raise ValueError(
                f"peak_transformer_pool must be cls|mean|cls_mean|attn|cls_mean_max, got {pool}"
            )
        self.pool_mode: PoolMode = pool  # type: ignore[assignment]
        self.wavelength = float(cfg.get("wavelength_angstrom", 1.54184))
        self.tt_max = float(cfg.get("two_theta_max_deg", 80.0))
        self.intensity_transform = str(cfg.get("intensity_transform", "linear"))

        d_model = int(cfg.get("peak_transformer_d_model", 256))
        n_layers = int(cfg.get("peak_transformer_num_layers", 4))
        n_heads = int(cfg.get("peak_transformer_num_heads", 8))
        ffn_dim = int(cfg.get("peak_transformer_ffn_dim", 1024))
        dropout = float(cfg.get("peak_transformer_dropout", 0.1))
        self.d_model = d_model

        # Fourier features on normalized g=1/d². 0 disables (legacy Linear-on-raw).
        self.n_fourier = int(cfg.get("peak_transformer_fourier_freqs", 16))
        if self.n_fourier < 0:
            raise ValueError("peak_transformer_fourier_freqs must be >= 0")
        fmode = str(cfg.get("peak_transformer_fourier_mode", "linear"))
        if fmode not in ("linear", "log", "loglinear"):
            raise ValueError(
                f"peak_transformer_fourier_mode must be linear|log|loglinear, got {fmode}"
            )
        self.fourier_mode: FourierMode = fmode  # type: ignore[assignment]
        # Small g-floor so log(g) is finite; below observed min g_norm (~0.0046).
        self.g_floor = float(cfg.get("peak_transformer_g_floor", 1e-3))

        if self.n_fourier > 0:
            # Split frequencies between linear-g and log-g bands per mode.
            if self.fourier_mode == "linear":
                n_lin, n_log = self.n_fourier, 0
            elif self.fourier_mode == "log":
                n_lin, n_log = 0, self.n_fourier
            else:  # loglinear
                n_log = self.n_fourier // 2
                n_lin = self.n_fourier - n_log
            self._n_lin = n_lin
            self._n_log = n_log
            lin = (
                torch.exp(torch.linspace(math.log(1.0), math.log(64.0), n_lin))
                if n_lin > 0
                else torch.empty(0)
            )
            log = (
                torch.exp(torch.linspace(math.log(1.0), math.log(64.0), n_log))
                if n_log > 0
                else torch.empty(0)
            )
            self.register_buffer("_freqs_lin", lin, persistent=False)
            self.register_buffer("_freqs_log", log, persistent=False)
            in_dim = 2 * self.n_fourier + _extra_feat_dim(self.token_mode)
        else:
            self._n_lin = 0
            self._n_log = 0
            self.register_buffer("_freqs_lin", torch.empty(0), persistent=False)
            self.register_buffer("_freqs_log", torch.empty(0), persistent=False)
            in_dim = _token_feature_dim(self.token_mode)

        self.input_proj = nn.Linear(in_dim, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rank_embed = nn.Embedding(self.max_peaks + 1, d_model)  # 0=CLS, 1..N peaks
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.rank_embed.weight, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Learned query for attention pooling (only used when pool_mode == "attn").
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        pool_out_dim = 3 * d_model if self.pool_mode == "cls_mean_max" else d_model
        self.out_norm = nn.LayerNorm(pool_out_dim)
        self.out_proj = nn.Linear(pool_out_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

        g_max = inverse_d2_max(
            wavelength_angstrom=self.wavelength,
            two_theta_max_deg=self.tt_max,
        )
        self.register_buffer(
            "_g_max",
            torch.tensor(g_max, dtype=torch.float32),
            persistent=False,
        )

    def _featurize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map raw token channels → input_proj features (Fourier on g when enabled).

        ``linear`` gives uniform *absolute* resolution in g; ``log`` gives uniform
        *relative* resolution (scale-equivariant to lattice scaling, matching the
        log-length target normalization); ``loglinear`` splits frequencies across
        both bands.
        """
        if self.n_fourier <= 0:
            return tokens
        g = tokens[..., :1]
        parts: list[torch.Tensor] = []
        if self._n_lin > 0:
            ang = g * self._freqs_lin.view(1, 1, -1) * (2.0 * math.pi)
            parts.extend([torch.sin(ang), torch.cos(ang)])
        if self._n_log > 0:
            log_floor = math.log(self.g_floor)
            u = (torch.log(g.clamp_min(self.g_floor)) - log_floor) / (-log_floor)
            u = u.clamp(0.0, 1.0)
            ang = u * self._freqs_log.view(1, 1, -1) * (2.0 * math.pi)
            parts.extend([torch.sin(ang), torch.cos(ang)])
        fourier = torch.cat(parts, dim=-1)
        if tokens.shape[-1] > 1:
            return torch.cat([fourier, tokens[..., 1:]], dim=-1)
        return fourier

    def _pool(
        self,
        encoded: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool Transformer outputs ``[B, 1+N, D]`` → ``[B, D]`` (or 3D for cls_mean_max)."""
        cls = encoded[:, 0]
        if self.pool_mode == "cls":
            return cls
        peak = encoded[:, 1:]
        valid = (~pad_mask).unsqueeze(-1).to(dtype=peak.dtype)
        if self.pool_mode == "attn":
            # Masked additive-softmax attention pooling with a learned query.
            scores = (peak * self.pool_query.view(1, 1, -1)).sum(dim=-1)
            scores = scores / math.sqrt(float(self.d_model))
            scores = scores.masked_fill(pad_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            weights = torch.nan_to_num(weights, nan=0.0)  # all-pad rows → 0
            return (peak * weights).sum(dim=1)
        pooled = (peak * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if self.pool_mode == "mean":
            return pooled
        if self.pool_mode == "cls_mean_max":
            neg_inf = torch.finfo(peak.dtype).min
            masked = peak.masked_fill(pad_mask.unsqueeze(-1), neg_inf)
            maxed = masked.amax(dim=1)
            # Guard fully-padded rows (no valid peaks) against -inf.
            maxed = torch.where(valid.sum(dim=1) > 0, maxed, torch.zeros_like(maxed))
            return torch.cat([cls, pooled, maxed], dim=-1)
        return 0.5 * (cls + pooled)

    def _build_tokens(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``tokens [B, max_peaks, F]`` and ``pad_mask [B, max_peaks]`` (True=pad)."""
        device = pxrd_x.device
        dtype = torch.float32
        batch = int(peak_num.shape[0])
        peaks = pxrd_x.reshape(-1).to(device=device, dtype=dtype)
        intens = pxrd_y.reshape(-1).to(device=device, dtype=dtype)
        peak_num = peak_num.to(device=device).long()
        g_max = float(self._g_max.detach().cpu().item()) if self._g_max.numel() else 1.0
        g_max = max(g_max, 1e-8)
        max_n = int(peak_num.max().item()) if batch > 0 else 0
        take = min(self.max_peaks, max(max_n, 0))

        tokens = torch.zeros(batch, self.max_peaks, _token_feature_dim(self.token_mode), device=device, dtype=dtype)
        pad_mask = torch.ones(batch, self.max_peaks, dtype=torch.bool, device=device)
        if batch == 0 or max_n == 0 or peaks.numel() == 0 or take == 0:
            return tokens, pad_mask

        offsets = torch.zeros(batch + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(peak_num, dim=0)
        starts = offsets[:-1]
        idx_range = torch.arange(max_n, device=device).unsqueeze(0)
        valid = idx_range < peak_num.unsqueeze(1)
        gather = (starts.unsqueeze(1) + idx_range).clamp(min=0, max=max(peaks.numel() - 1, 0))
        tt = peaks[gather]
        inten = intens[gather]
        tt = torch.where(valid, tt, torch.full_like(tt, 1e6))  # push pads to high 2θ
        inten = torch.where(valid, inten, torch.zeros_like(inten))

        # Sort by ascending 2θ (low-angle first).
        order = torch.argsort(tt, dim=1, stable=True)
        tt_s = torch.gather(tt, 1, order)
        inten_s = torch.gather(inten, 1, order)
        valid_s = torch.gather(valid, 1, order)

        # Keep first min(max_peaks, max_n) then pad to max_peaks.
        keep = min(self.max_peaks, max_n)
        tt_keep = tt_s[:, :keep]
        inten_keep = inten_s[:, :keep]
        valid_keep = valid_s[:, :keep]
        tt_s = torch.zeros(batch, self.max_peaks, device=device, dtype=dtype)
        inten_s = torch.zeros(batch, self.max_peaks, device=device, dtype=dtype)
        valid_s = torch.zeros(batch, self.max_peaks, dtype=torch.bool, device=device)
        if keep > 0:
            tt_s[:, :keep] = tt_keep
            inten_s[:, :keep] = inten_keep
            valid_s[:, :keep] = valid_keep
        # Also clamp by each sample's peak_num (and max_peaks).
        rank_ok = torch.arange(self.max_peaks, device=device).unsqueeze(0) < peak_num.unsqueeze(1).clamp(
            max=self.max_peaks
        )
        valid_s = valid_s & rank_ok
        pad_mask = ~valid_s

        inv = inverse_d2_from_two_theta(tt_s, wavelength_angstrom=self.wavelength)
        assert isinstance(inv, torch.Tensor)
        inv = torch.where(valid_s, inv, torch.zeros_like(inv))
        g = inv / g_max

        inten_max = torch.where(valid_s, inten_s, torch.zeros_like(inten_s)).amax(dim=1, keepdim=True).clamp_min(1e-8)
        inten_n = inten_s / inten_max
        if self.intensity_transform == "sqrt":
            inten_n = torch.sqrt(inten_n.clamp(min=0.0))
        elif self.intensity_transform == "log":
            inten_n = torch.log1p(inten_n * 99.0) / float(torch.log1p(torch.tensor(99.0)))
        inten_n = torch.where(valid_s, inten_n, torch.zeros_like(inten_n))

        if self.token_mode == "pos":
            feats = g.unsqueeze(-1)
        elif self.token_mode == "pos_i":
            feats = torch.stack([g, inten_n], dim=-1)
        else:
            g_prev = torch.zeros_like(g)
            g_prev[:, 1:] = g[:, :-1]
            dg = torch.where(valid_s, g - g_prev, torch.zeros_like(g))
            # First peak Δg = 0; for padded slots stay 0.
            first = torch.zeros_like(valid_s)
            first[:, 0] = valid_s[:, 0]
            dg = torch.where(first, torch.zeros_like(dg), dg)
            n_eff = peak_num.to(dtype).clamp(min=1.0, max=float(self.max_peaks)).unsqueeze(1)
            rank = (torch.arange(self.max_peaks, device=device, dtype=dtype).unsqueeze(0) / n_eff).clamp(0.0, 1.0)
            rank = torch.where(valid_s, rank, torch.zeros_like(rank))
            feats = torch.stack([g, dg, inten_n, rank], dim=-1)

        tokens = torch.where(valid_s.unsqueeze(-1), feats, torch.zeros_like(feats))
        return tokens, pad_mask

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        tokens, pad_mask = self._build_tokens(pxrd_x, pxrd_y, peak_num)
        batch = tokens.shape[0]
        x = self.input_proj(self._featurize(tokens))  # [B, N, D]
        # Rank ids 1..N for peaks; CLS uses 0.
        rank_ids = torch.arange(1, self.max_peaks + 1, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = x + self.rank_embed(rank_ids)
        cls = self.cls_token.expand(batch, -1, -1) + self.rank_embed(
            torch.zeros(batch, 1, dtype=torch.long, device=tokens.device)
        )
        x = torch.cat([cls, x], dim=1)
        # key_padding_mask: True = ignore; CLS never padded.
        cls_pad = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
        key_padding_mask = torch.cat([cls_pad, pad_mask], dim=1)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self._pool(x, pad_mask)
        return self.out_proj(self.out_norm(pooled))
