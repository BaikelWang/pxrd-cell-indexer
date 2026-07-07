"""Lattice geometry utilities vendored from RealPXRD eval_utils."""

from __future__ import annotations

import torch


def lattice_params_to_matrix(cell: torch.Tensor) -> torch.Tensor:
    """Convert (a,b,c,alpha,beta,gamma) in degrees to a 3x3 lattice matrix."""
    lengths = cell[..., :3]
    angles = cell[..., 3:]
    if cell.ndim == 1:
        return _single_params_to_matrix(lengths, angles)
    matrices = []
    for row in cell:
        matrices.append(_single_params_to_matrix(row[:3], row[3:]))
    return torch.stack(matrices, dim=0)


def _single_params_to_matrix(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)
    val = (coses[0] * coses[1] - coses[2]) / (sins[0] * sins[1] + 1e-12)
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [lengths[0] * sins[1], torch.zeros((), device=lengths.device), lengths[0] * coses[1]]
    )
    vector_b = torch.stack(
        [
            -lengths[1] * sins[0] * torch.cos(gamma_star),
            lengths[1] * sins[0] * torch.sin(gamma_star),
            lengths[1] * coses[0],
        ]
    )
    vector_c = torch.stack(
        [torch.zeros((), device=lengths.device), torch.zeros((), device=lengths.device), lengths[2]]
    )
    return torch.stack([vector_a, vector_b, vector_c], dim=0)


def lattice_lengths_angles(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract lengths and angles from lattice matrix(es) with shape (..., 3, 3)."""
    lengths = torch.sqrt(torch.sum(matrix**2, dim=-1))
    batch_shape = matrix.shape[:-2]
    angles = torch.zeros(*batch_shape, 3, device=matrix.device, dtype=matrix.dtype)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        cos_angle = torch.sum(matrix[..., j, :] * matrix[..., k, :], dim=-1) / (
            lengths[..., j] * lengths[..., k] + 1e-12
        )
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angles[..., i] = torch.arccos(cos_angle) * 180.0 / torch.pi
    return lengths, angles
