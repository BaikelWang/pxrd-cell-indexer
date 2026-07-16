"""Peak-reconstructed continuous-spectrum CNN encoders (R11b-E3 / spectrum-only).

- ``HistogramSpectrumFusionEncoder``: peak histogram + spectrum CNN, gated/concat fusion.
- ``SpectrumOnlyEncoder``: spectrum CNN alone (diagnostic; no histogram branch).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pxrd_cell_indexing.model.encoder.histogram import InverseD2HistogramEncoder


class _SpectrumCNN(nn.Module):
    """Lightweight 1D CNN over a reconstructed PXRD intensity profile."""

    def __init__(
        self,
        *,
        out_dim: int,
        channels: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, out_dim),
            nn.GELU(),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # spectrum: [B, G] → [B, 1, G]
        x = spectrum.unsqueeze(1)
        return self.proj(self.conv(x))


def _reconstruct_spectrum_from_peaks(
    pxrd_x: torch.Tensor,
    pxrd_y: torch.Tensor,
    peak_num: torch.Tensor,
    *,
    tt_grid: torch.Tensor,
    spectrum_bins: int,
    spectrum_sigma_deg: float,
) -> torch.Tensor:
    """Vectorized Gaussian stick reconstruction → ``[B, spectrum_bins]``."""
    device = pxrd_x.device
    dtype = torch.float32
    batch = int(peak_num.shape[0])
    peaks = pxrd_x.reshape(-1).to(device=device, dtype=dtype)
    intens = pxrd_y.reshape(-1).to(device=device, dtype=dtype)
    peak_num = peak_num.to(device=device).long()
    grid = tt_grid.to(device=device, dtype=dtype)

    if batch == 0:
        return torch.zeros(0, spectrum_bins, device=device, dtype=dtype)

    offsets = torch.zeros(batch + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(peak_num, dim=0)
    max_n = int(peak_num.max().item()) if batch > 0 else 0
    if max_n == 0 or peaks.numel() == 0:
        return torch.zeros(batch, spectrum_bins, device=device, dtype=dtype)

    starts = offsets[:-1]
    idx_range = torch.arange(max_n, device=device).unsqueeze(0)
    valid = idx_range < peak_num.unsqueeze(1)
    gather = (starts.unsqueeze(1) + idx_range).clamp(min=0, max=max(peaks.numel() - 1, 0))
    tt = peaks[gather]
    inten = intens[gather]
    tt = torch.where(valid, tt, torch.zeros_like(tt))
    inten = torch.where(valid, inten, torch.zeros_like(inten))

    inten_max = torch.where(valid, inten, torch.zeros_like(inten)).amax(dim=1, keepdim=True).clamp_min(1e-8)
    inten_n = torch.where(valid, inten / inten_max, torch.zeros_like(inten))

    sigma = max(spectrum_sigma_deg, 1e-3)
    dist = tt.unsqueeze(-1) - grid.view(1, 1, -1)
    kernels = torch.exp(-0.5 * (dist / sigma) ** 2)
    kernels = torch.where(valid.unsqueeze(-1), kernels, torch.zeros_like(kernels))
    spectrum = (kernels * inten_n.unsqueeze(-1)).sum(dim=1)
    spec_max = spectrum.amax(dim=1, keepdim=True).clamp_min(1e-8)
    spectrum = spectrum / spec_max
    spectrum = torch.where(peak_num.unsqueeze(1) > 0, spectrum, torch.zeros_like(spectrum))
    return spectrum


class SpectrumOnlyEncoder(nn.Module):
    """Diagnostic: peak→profile reconstruction + 1D CNN only (no histogram branch)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.output_dim = int(cfg.get("output_dim", 512))
        self.tt_min = float(cfg.get("two_theta_min_deg", 5.0))
        self.tt_max = float(cfg.get("two_theta_max_deg", 80.0))
        self.spectrum_bins = int(cfg.get("spectrum_bins", 1024))
        self.spectrum_sigma_deg = float(cfg.get("spectrum_sigma_deg", 0.15))
        channels_raw = cfg.get("spectrum_cnn_channels", (64, 128, 256))
        channels = tuple(int(c) for c in channels_raw)
        kernel = int(cfg.get("spectrum_cnn_kernel", 7))
        dropout = float(cfg.get("histogram_dropout", float(cfg.get("spectrum_dropout", 0.0))))

        self.spectrum_cnn = _SpectrumCNN(
            out_dim=self.output_dim,
            channels=channels,
            kernel_size=kernel,
            dropout=dropout,
        )
        grid = torch.linspace(self.tt_min, self.tt_max, self.spectrum_bins)
        self.register_buffer("_tt_grid", grid, persistent=False)

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        spectrum = _reconstruct_spectrum_from_peaks(
            pxrd_x,
            pxrd_y,
            peak_num,
            tt_grid=self._tt_grid,
            spectrum_bins=self.spectrum_bins,
            spectrum_sigma_deg=self.spectrum_sigma_deg,
        )
        return self.spectrum_cnn(spectrum)


class HistogramSpectrumFusionEncoder(nn.Module):
    """E3: peak histogram encoder + continuous-spectrum CNN, gated fusion."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.output_dim = int(cfg.get("output_dim", 512))
        self.tt_min = float(cfg.get("two_theta_min_deg", 5.0))
        self.tt_max = float(cfg.get("two_theta_max_deg", 80.0))
        self.spectrum_bins = int(cfg.get("spectrum_bins", 1024))
        self.spectrum_sigma_deg = float(cfg.get("spectrum_sigma_deg", 0.15))
        channels_raw = cfg.get("spectrum_cnn_channels", (64, 128, 256))
        channels = tuple(int(c) for c in channels_raw)
        kernel = int(cfg.get("spectrum_cnn_kernel", 7))
        dropout = float(cfg.get("histogram_dropout", 0.0))
        fusion_mode = str(cfg.get("fusion_mode", "gate"))
        if fusion_mode not in ("gate", "concat"):
            raise ValueError(f"fusion_mode must be gate|concat, got {fusion_mode}")
        self.fusion_mode = fusion_mode

        peak_cfg = dict(cfg)
        peak_cfg["output_dim"] = self.output_dim
        self.peak_encoder = InverseD2HistogramEncoder(peak_cfg)

        self.spectrum_cnn = _SpectrumCNN(
            out_dim=self.output_dim,
            channels=channels,
            kernel_size=kernel,
            dropout=dropout,
        )

        if fusion_mode == "gate":
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim),
                nn.Sigmoid(),
            )
            self.fusion_proj = None
        else:
            self.fusion_gate = None
            self.fusion_proj = nn.Sequential(
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim),
            )

        grid = torch.linspace(self.tt_min, self.tt_max, self.spectrum_bins)
        self.register_buffer("_tt_grid", grid, persistent=False)

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        peak_emb = self.peak_encoder(pxrd_x, pxrd_y, peak_num)
        spectrum = _reconstruct_spectrum_from_peaks(
            pxrd_x,
            pxrd_y,
            peak_num,
            tt_grid=self._tt_grid,
            spectrum_bins=self.spectrum_bins,
            spectrum_sigma_deg=self.spectrum_sigma_deg,
        )
        spec_emb = self.spectrum_cnn(spectrum)
        if self.fusion_mode == "gate":
            assert self.fusion_gate is not None
            gate = self.fusion_gate(torch.cat([peak_emb, spec_emb], dim=-1))
            return peak_emb + gate * spec_emb
        assert self.fusion_proj is not None
        return self.fusion_proj(torch.cat([peak_emb, spec_emb], dim=-1))
