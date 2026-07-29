"""Conditional rectified-flow head over normalized gstar6.

Replaces point regression with a sampler: one forward pass gives a velocity
field, and integrating it from Gaussian noise yields K candidate cells per
pattern. Built for seed generation, where what matters is whether the correct
cell appears anywhere in the K draws (coverage@K), not Top-1 accuracy.

gstar6 is the natural space for this: six components for six degrees of freedom
(no rotational gauge, unlike a 3x3 matrix), any vector decodes to a valid cell,
and the log-diagonal makes it scale-equivariant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConditionalFlowHead"]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _AdaLNBlock(nn.Module):
    """Residual MLP block modulated by the (condition, time) vector."""

    def __init__(self, hidden: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.modulation = nn.Linear(cond_dim, hidden * 3)
        # Zero-init so each block starts as identity: stable at high depth.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        x = self.norm(h) * (1.0 + scale) + shift
        return h + gate * self.mlp(x)


class ConditionalFlowHead(nn.Module):
    """Velocity field ``v(z_t, t | embedding)`` for rectified flow on gstar6."""

    def __init__(
        self,
        embedding_dim: int = 512,
        *,
        target_dim: int = 6,
        hidden: int = 512,
        num_layers: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.time_dim = time_dim

        self.in_proj = nn.Linear(target_dim, hidden)
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(hidden, hidden, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, target_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        embedding: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond_proj(embedding) + self.time_proj(
            timestep_embedding(t, self.time_dim).to(embedding.dtype)
        )
        cond = F.silu(cond)
        h = self.in_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.out_norm(h))

    def flow_loss(self, embedding: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Rectified-flow matching loss against target ``z1`` (normalized gstar6)."""
        batch = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(batch, device=z1.device, dtype=z1.dtype)
        z_t = (1.0 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z1
        velocity = self.forward(embedding, z_t, t)
        return F.mse_loss(velocity, z1 - z0)

    @torch.no_grad()
    def sample(
        self,
        embedding: torch.Tensor,
        *,
        num_samples: int = 100,
        steps: int = 50,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate the ODE from noise. Returns ``[B, num_samples, target_dim]``."""
        batch = embedding.shape[0]
        cond = embedding.repeat_interleave(num_samples, dim=0)
        z = torch.randn(
            batch * num_samples,
            self.target_dim,
            device=embedding.device,
            dtype=embedding.dtype,
            generator=generator,
        )
        z = z * temperature
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full(
                (z.shape[0],), i * dt, device=embedding.device, dtype=embedding.dtype
            )
            z = z + dt * self.forward(cond, z, t)
        return z.view(batch, num_samples, self.target_dim)
