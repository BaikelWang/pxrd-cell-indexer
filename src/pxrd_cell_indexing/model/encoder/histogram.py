"""Inverse-d² histogram MLP encoder (R1 strong baseline from D-A3).

R8: vectorized batch featurization (no Python for / .item() sync) and optional
deeper residual MLP so a 24GB GPU can actually be used.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pxrd_cell_indexing.data.peak_features import (
    PeakFeatureConfig,
    histogram_feature_dim,
    inverse_d2_from_two_theta,
    inverse_d2_max,
    normalize_intensity,
)


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: x + Dropout(Linear(GELU(Linear(LN(x)))))."""

    def __init__(self, dim: int, *, dropout: float = 0.0, expand: float = 2.0) -> None:
        super().__init__()
        inner = max(int(dim * expand), dim)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x + self.drop(h)


class InverseD2HistogramEncoder(nn.Module):
    """Bag-of-peaks encoder: inverse_d2 histogram + sorted peaks + peak count."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.output_dim = int(cfg.get("output_dim", 512))
        self.peak_feature_config = PeakFeatureConfig(
            feature_mode="inverse_d2_i",
            wavelength_angstrom=float(cfg.get("wavelength_angstrom", 1.54184)),
            intensity_transform=str(cfg.get("intensity_transform", "linear")),  # type: ignore[arg-type]
            two_theta_min_deg=float(cfg.get("two_theta_min_deg", 5.0)),
            two_theta_max_deg=float(cfg.get("two_theta_max_deg", 80.0)),
            hist_bins=int(cfg.get("hist_bins", 256)),
            sorted_peak_count=int(cfg.get("sorted_peak_count", 24)),
            hist_pool=str(cfg.get("hist_pool", "max")),  # type: ignore[arg-type]
        )
        in_dim = histogram_feature_dim(self.peak_feature_config)
        hidden = int(cfg.get("histogram_hidden_dim", 512))
        dropout = float(cfg.get("histogram_dropout", 0.0))
        # R8: num residual blocks after the stem. 0 → legacy 3-stage MLP
        # (backward-compatible with all existing checkpoints / configs).
        num_blocks = int(cfg.get("histogram_num_blocks", 0))
        self.num_blocks = num_blocks

        if num_blocks <= 0:
            # Legacy shallow MLP (≈0.7M for hidden=512).
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, self.output_dim),
            )
        else:
            # Deeper residual tower: stem → N residual blocks → project to emb.
            self.stem = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(
                [_ResidualMLPBlock(hidden, dropout=dropout) for _ in range(num_blocks)]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.output_dim),
            )
            self.net = None  # type: ignore[assignment]
        self.in_dim = in_dim
        self.hidden = hidden
        self.register_buffer(
            "_g_max",
            torch.tensor(
                inverse_d2_max(
                    wavelength_angstrom=self.peak_feature_config.wavelength_angstrom,
                    two_theta_max_deg=self.peak_feature_config.two_theta_max_deg,
                ),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _featurize_one(
        self,
        two_theta: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        """Kept for unit tests / debugging; training uses ``_featurize_batch``."""
        cfg = self.peak_feature_config
        inv = inverse_d2_from_two_theta(
            two_theta.reshape(-1),
            wavelength_angstrom=cfg.wavelength_angstrom,
        )
        assert isinstance(inv, torch.Tensor)
        g_max = float(self._g_max.item()) if self._g_max.numel() else 1.0
        hist = torch.zeros(cfg.hist_bins, device=two_theta.device, dtype=torch.float32)
        if inv.numel():
            bins = torch.clamp(
                (inv / max(g_max, 1e-8) * cfg.hist_bins).long(),
                0,
                cfg.hist_bins - 1,
            )
            inten_n = normalize_intensity(intensity.reshape(-1), transform=cfg.intensity_transform)
            assert isinstance(inten_n, torch.Tensor)
            if cfg.hist_pool == "sum":
                hist.scatter_add_(0, bins, inten_n.to(dtype=torch.float32))
            else:
                hist.scatter_reduce_(
                    0,
                    bins,
                    inten_n.to(dtype=torch.float32),
                    reduce="amax",
                    include_self=True,
                )
            max_h = hist.max()
            if float(max_h.item()) > 0:
                hist = hist / max_h
        top = torch.zeros(cfg.sorted_peak_count, device=two_theta.device, dtype=torch.float32)
        if inv.numel():
            qs, _ = torch.sort(inv)
            k = min(cfg.sorted_peak_count, int(qs.numel()))
            top[:k] = qs[:k] / max(g_max, 1e-8)
        npk = torch.tensor([inv.numel() / 50.0], device=two_theta.device, dtype=torch.float32)
        return torch.cat([hist, top, npk], dim=0)

    def _featurize_batch(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized bag-of-peaks features ``[B, in_dim]`` (no host sync loop)."""
        cfg = self.peak_feature_config
        device = pxrd_x.device
        dtype = torch.float32
        batch = int(peak_num.shape[0])
        peaks = pxrd_x.reshape(-1).to(device=device, dtype=dtype)
        intens = pxrd_y.reshape(-1).to(device=device, dtype=dtype)
        peak_num = peak_num.to(device=device).long()
        g_max = float(self._g_max.detach().cpu().item()) if self._g_max.numel() else 1.0
        g_max = max(g_max, 1e-8)

        offsets = torch.zeros(batch + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(peak_num, dim=0)
        max_n = int(peak_num.max().item()) if batch > 0 else 0
        if max_n == 0 or peaks.numel() == 0:
            return torch.zeros(batch, self.in_dim, device=device, dtype=dtype)

        starts = offsets[:-1]
        idx_range = torch.arange(max_n, device=device).unsqueeze(0)  # [1, max_n]
        valid = idx_range < peak_num.unsqueeze(1)  # [B, max_n]
        gather = (starts.unsqueeze(1) + idx_range).clamp(min=0, max=max(peaks.numel() - 1, 0))
        tt = peaks[gather]  # [B, max_n]
        inten = intens[gather]
        tt = torch.where(valid, tt, torch.zeros_like(tt))
        inten = torch.where(valid, inten, torch.zeros_like(inten))

        inv = inverse_d2_from_two_theta(tt, wavelength_angstrom=cfg.wavelength_angstrom)
        assert isinstance(inv, torch.Tensor)
        inv = torch.where(valid, inv, torch.zeros_like(inv))

        # Per-sample intensity normalization (matches _featurize_one).
        inten_max = torch.where(valid, inten, torch.zeros_like(inten)).amax(dim=1, keepdim=True).clamp_min(1e-8)
        inten_n = inten / inten_max
        if cfg.intensity_transform == "sqrt":
            inten_n = torch.sqrt(inten_n.clamp(min=0.0))
        elif cfg.intensity_transform == "log":
            inten_n = torch.log1p(inten_n * 99.0) / float(torch.log1p(torch.tensor(99.0)))
        elif cfg.intensity_transform == "none":
            inten_n = inten
        inten_n = torch.where(valid, inten_n, torch.zeros_like(inten_n))

        bins = torch.clamp((inv / g_max * cfg.hist_bins).long(), 0, cfg.hist_bins - 1)
        bins = torch.where(valid, bins, torch.zeros_like(bins))

        hist = torch.zeros(batch, cfg.hist_bins, device=device, dtype=dtype)
        flat_bins = (torch.arange(batch, device=device).unsqueeze(1) * cfg.hist_bins + bins).reshape(-1)
        flat_vals = inten_n.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_bins = flat_bins[flat_valid]
        flat_vals = flat_vals[flat_valid]
        if flat_bins.numel():
            if cfg.hist_pool == "sum":
                hist.view(-1).scatter_add_(0, flat_bins, flat_vals)
            else:
                hist.view(-1).scatter_reduce_(
                    0, flat_bins, flat_vals, reduce="amax", include_self=True
                )
        hist_max = hist.amax(dim=1, keepdim=True).clamp_min(1e-8)
        hist = hist / hist_max
        hist = torch.where(peak_num.unsqueeze(1) > 0, hist, torch.zeros_like(hist))

        # Sorted leading inverse_d2 peaks (ascending, padded).
        inv_for_sort = torch.where(valid, inv, torch.full_like(inv, float("inf")))
        qs, _ = torch.sort(inv_for_sort, dim=1)
        k = min(cfg.sorted_peak_count, max_n)
        top = torch.zeros(batch, cfg.sorted_peak_count, device=device, dtype=dtype)
        if k > 0:
            top_k = qs[:, :k] / g_max
            top_k = torch.where(torch.isfinite(top_k), top_k, torch.zeros_like(top_k))
            # Zero out slots beyond each sample's peak count.
            top_valid = torch.arange(k, device=device).unsqueeze(0) < peak_num.unsqueeze(1)
            top[:, :k] = torch.where(top_valid, top_k, torch.zeros_like(top_k))

        npk = (peak_num.to(dtype) / 50.0).unsqueeze(1)
        return torch.cat([hist, top, npk], dim=1)

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        features = self._featurize_batch(pxrd_x, pxrd_y, peak_num)
        if self.net is not None:
            return self.net(features)
        x = self.stem(features)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
