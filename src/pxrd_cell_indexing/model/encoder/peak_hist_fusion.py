"""A2-T4: Peak Transformer + inverse_d2 histogram fusion encoder.

Plan (v3 §6.3): each branch → 256-d, concat → Linear + LayerNorm → 512-d.
Tests whether local peak relations (Transformer) and global peak distribution
(histogram) are complementary.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pxrd_cell_indexing.model.encoder.histogram import InverseD2HistogramEncoder
from pxrd_cell_indexing.model.encoder.peak_transformer import PeakGeometryTransformerEncoder


class PeakTransformerHistogramFusionEncoder(nn.Module):
    """T48-geom Peak Transformer ⊕ bag-of-peaks histogram, concat fusion."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.output_dim = int(cfg.get("output_dim", 512))
        branch_dim = int(cfg.get("fusion_branch_dim", 256))
        if branch_dim <= 0:
            raise ValueError(f"fusion_branch_dim must be > 0, got {branch_dim}")
        fusion_mode = str(cfg.get("fusion_mode", "concat"))
        if fusion_mode not in ("concat", "gate"):
            raise ValueError(f"fusion_mode must be concat|gate, got {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.branch_dim = branch_dim

        peak_cfg = dict(cfg)
        peak_cfg["output_dim"] = branch_dim
        self.peak_encoder = PeakGeometryTransformerEncoder(peak_cfg)

        hist_cfg = dict(cfg)
        hist_cfg["output_dim"] = branch_dim
        # Keep histogram capacity moderate unless caller overrides; T4 is about
        # complementarity, not re-running full E1c capacity inside the fusion.
        hist_cfg.setdefault("histogram_hidden_dim", 512)
        hist_cfg.setdefault("histogram_num_blocks", 0)
        self.hist_encoder = InverseD2HistogramEncoder(hist_cfg)

        if fusion_mode == "concat":
            # Plan: concat + Linear + LayerNorm (no deep MLP).
            self.fusion_proj = nn.Sequential(
                nn.Linear(branch_dim * 2, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
            self.fusion_gate = None
        else:
            self.fusion_gate = nn.Sequential(
                nn.Linear(branch_dim * 2, branch_dim),
                nn.GELU(),
                nn.Linear(branch_dim, branch_dim),
                nn.Sigmoid(),
            )
            self.fusion_proj = (
                nn.Identity()
                if branch_dim == self.output_dim
                else nn.Sequential(nn.Linear(branch_dim, self.output_dim), nn.LayerNorm(self.output_dim))
            )

    def forward(
        self,
        pxrd_x: torch.Tensor,
        pxrd_y: torch.Tensor,
        peak_num: torch.Tensor,
    ) -> torch.Tensor:
        peak_emb = self.peak_encoder(pxrd_x, pxrd_y, peak_num)
        hist_emb = self.hist_encoder(pxrd_x, pxrd_y, peak_num)
        if self.fusion_mode == "concat":
            return self.fusion_proj(torch.cat([peak_emb, hist_emb], dim=-1))
        assert self.fusion_gate is not None
        gate = self.fusion_gate(torch.cat([peak_emb, hist_emb], dim=-1))
        fused = peak_emb + gate * hist_emb
        return self.fusion_proj(fused)
