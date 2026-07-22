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


# A3 gstar6: pack order [log L11, log L22, log L33, L21, L31, L32] of lower Cholesky
# of the reciprocal metric G* = (A Aᵀ)^{-1}. Wide clamp keeps decode finite under
# early-training noise without materially biasing well-trained predictions.
GSTAR6_LOG_DIAG_CLAMP = (-20.0, 20.0)
GSTAR6_SOLVE_JITTER = 1e-8


def direct_metric_from_lattice(cell: torch.Tensor) -> torch.Tensor:
    """Direct metric G = A Aᵀ from lattice params (a,b,c,α,β,γ) in degrees."""
    matrix = lattice_params_to_matrix(cell)
    return matrix @ matrix.transpose(-1, -2)


def reciprocal_metric_from_direct(
    g_direct: torch.Tensor,
    *,
    jitter: float = GSTAR6_SOLVE_JITTER,
) -> torch.Tensor:
    """G* = G^{-1} with a tiny diagonal jitter for numerical stability."""
    eye = torch.eye(3, device=g_direct.device, dtype=g_direct.dtype)
    # Broadcast eye to match leading batch dims when present.
    while eye.ndim < g_direct.ndim:
        eye = eye.unsqueeze(0)
    eye = eye.expand_as(g_direct)
    return torch.linalg.solve(g_direct + float(jitter) * eye, eye)


def cholesky_lower_from_spd(matrix: torch.Tensor) -> torch.Tensor:
    """Lower-triangular Cholesky factor of an SPD matrix."""
    return torch.linalg.cholesky(matrix)


def pack_gstar6_cholesky(chol_lower: torch.Tensor) -> torch.Tensor:
    """Pack lower Cholesky into 6D regression vector."""
    log_diag = torch.log(
        torch.stack(
            [
                chol_lower[..., 0, 0],
                chol_lower[..., 1, 1],
                chol_lower[..., 2, 2],
            ],
            dim=-1,
        ).clamp_min(1e-12)
    )
    off = torch.stack(
        [
            chol_lower[..., 1, 0],
            chol_lower[..., 2, 0],
            chol_lower[..., 2, 1],
        ],
        dim=-1,
    )
    return torch.cat([log_diag, off], dim=-1)


def unpack_gstar6_cholesky(
    components: torch.Tensor,
    *,
    log_diag_clamp: tuple[float, float] = GSTAR6_LOG_DIAG_CLAMP,
) -> torch.Tensor:
    """Unpack 6D gstar6 vector into a lower-triangular Cholesky factor."""
    arr = components.reshape(*components.shape[:-1], 6)
    log_diag = arr[..., :3].clamp(log_diag_clamp[0], log_diag_clamp[1])
    diag = torch.exp(log_diag)
    lower = torch.zeros(*arr.shape[:-1], 3, 3, device=arr.device, dtype=arr.dtype)
    lower[..., 0, 0] = diag[..., 0]
    lower[..., 1, 1] = diag[..., 1]
    lower[..., 2, 2] = diag[..., 2]
    lower[..., 1, 0] = arr[..., 3]
    lower[..., 2, 0] = arr[..., 4]
    lower[..., 2, 1] = arr[..., 5]
    return lower


def reciprocal_metric_from_gstar6(
    components: torch.Tensor,
    *,
    log_diag_clamp: tuple[float, float] = GSTAR6_LOG_DIAG_CLAMP,
) -> torch.Tensor:
    """Reconstruct SPD G* from packed Cholesky components."""
    lower = unpack_gstar6_cholesky(components, log_diag_clamp=log_diag_clamp)
    return lower @ lower.transpose(-1, -2)


def lattice_from_direct_metric(g_direct: torch.Tensor) -> torch.Tensor:
    """Decode (a,b,c,α,β,γ)[deg] from direct metric G."""
    lengths = torch.sqrt(torch.diagonal(g_direct, dim1=-2, dim2=-1).clamp_min(1e-18))
    # Keep lengths in a physically plausible band for XRD cells.
    lengths = lengths.clamp(0.5, 200.0)
    a = lengths[..., 0]
    b = lengths[..., 1]
    c = lengths[..., 2]
    cos_alpha = (g_direct[..., 1, 2] / (b * c + 1e-18)).clamp(-0.999999, 0.999999)
    cos_beta = (g_direct[..., 0, 2] / (a * c + 1e-18)).clamp(-0.999999, 0.999999)
    cos_gamma = (g_direct[..., 0, 1] / (a * b + 1e-18)).clamp(-0.999999, 0.999999)
    angles = torch.rad2deg(
        torch.stack(
            [torch.arccos(cos_alpha), torch.arccos(cos_beta), torch.arccos(cos_gamma)],
            dim=-1,
        )
    )
    # Avoid degenerate 0°/180° cells that hang pymatgen find_mapping.
    angles = angles.clamp(20.0, 160.0)
    out = torch.cat([lengths, angles], dim=-1)
    return torch.nan_to_num(out, nan=90.0, posinf=160.0, neginf=20.0)


def lattice_to_gstar6(
    cell: torch.Tensor,
    *,
    jitter: float = GSTAR6_SOLVE_JITTER,
) -> torch.Tensor:
    """Encode lattice params → packed reciprocal-metric Cholesky (gstar6)."""
    g_direct = direct_metric_from_lattice(cell)
    g_star = reciprocal_metric_from_direct(g_direct, jitter=jitter)
    # Symmetrize before Cholesky to damp float noise.
    g_star = 0.5 * (g_star + g_star.transpose(-1, -2))
    chol = cholesky_lower_from_spd(g_star)
    return pack_gstar6_cholesky(chol)


def gstar6_to_lattice(
    components: torch.Tensor,
    *,
    jitter: float = GSTAR6_SOLVE_JITTER,
    log_diag_clamp: tuple[float, float] = GSTAR6_LOG_DIAG_CLAMP,
) -> torch.Tensor:
    """Decode packed gstar6 → lattice params (a,b,c,α,β,γ)[deg]."""
    g_star = reciprocal_metric_from_gstar6(components, log_diag_clamp=log_diag_clamp)
    g_star = 0.5 * (g_star + g_star.transpose(-1, -2))
    g_direct = reciprocal_metric_from_direct(g_star, jitter=jitter)
    g_direct = 0.5 * (g_direct + g_direct.transpose(-1, -2))
    return lattice_from_direct_metric(g_direct)


def gstar6_min_eig(
    components: torch.Tensor,
    *,
    log_diag_clamp: tuple[float, float] = GSTAR6_LOG_DIAG_CLAMP,
) -> torch.Tensor:
    """Smallest eigenvalue of reconstructed G* (diagnostic; should be >0)."""
    g_star = reciprocal_metric_from_gstar6(components, log_diag_clamp=log_diag_clamp)
    g_star = 0.5 * (g_star + g_star.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(g_star)
    return eigvals[..., 0]
