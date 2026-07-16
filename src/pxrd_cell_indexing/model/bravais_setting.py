"""Bravais cubic setting helpers (P/F/I angle targets)."""

from __future__ import annotations

import torch

# Primitive cubic angle settings from Bravais constraint validation.
CUBIC_SETTING_TARGETS_DEG: tuple[float, ...] = (90.0, 60.0, 109.47)
N_CUBIC_SETTINGS = len(CUBIC_SETTING_TARGETS_DEG)


def cubic_setting_idx_from_phys(lattice_phys: torch.Tensor) -> torch.Tensor:
    """Nearest cubic setting by mean angle (degrees). lattice_phys: [..., 6]."""
    mean_ang = lattice_phys[..., 3:6].mean(dim=-1)
    targets = torch.tensor(
        CUBIC_SETTING_TARGETS_DEG,
        device=lattice_phys.device,
        dtype=lattice_phys.dtype,
    )
    dist = (mean_ang.unsqueeze(-1) - targets).abs()
    return dist.argmin(dim=-1)


def pick_cubic_setting_self_consistent(
    cubic_preds_phys: torch.Tensor,
) -> torch.Tensor:
    """Pick setting head whose predicted mean angle is closest to its target.

    cubic_preds_phys: [B, 3, 6] physical lattices from the three setting heads.
    """
    mean_ang = cubic_preds_phys[..., 3:6].mean(dim=-1)  # [B, 3]
    targets = torch.tensor(
        CUBIC_SETTING_TARGETS_DEG,
        device=cubic_preds_phys.device,
        dtype=cubic_preds_phys.dtype,
    )
    dist = (mean_ang - targets).abs()
    return dist.argmin(dim=-1)
