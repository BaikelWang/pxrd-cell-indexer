"""Loss functions for lattice regression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.bravais_setting import cubic_setting_idx_from_phys

if TYPE_CHECKING:
    from pxrd_cell_indexing.data.normalization import LatticeNormalizer

LossMode = Literal[
    "baseline",
    "length_angle",
    "cs_mask",
    "cs_reweight",
    "combined",
    "strict_hinge",  # B1a: penalize only beyond ltol/atol
    "angle_heavy",  # B1b: physical length+angle with large angle_weight
    "angle_prior",  # R2: matrix6 SmoothL1 + Bravais min-over-hypotheses angle prior
    "joint_phys",  # R3: SmoothL1 + additive truth-based length/angle (small λ)
    "manifold_consistency",  # R4: SmoothL1 + self-consistency to own CS's *unambiguous*
    # symmetry-fixed angles (tet/ortho α,β→90°; hex β→90°). Unlike angle_prior this
    # is a single deterministic target per dim, not a min-over-hypotheses guess, so
    # it cannot fight a genuinely ambiguous ground truth (the failure mode that
    # doomed cubic angle_prior in R2).
    "peak_consistency",  # R6-A: SmoothL1 + soft Chamfer of obs 1/d² vs theory hkl
    "mcl",  # R10: Multiple-Choice-Learning min-over-K SmoothL1 (multi_hypothesis heads)
]

# Per-crystal-system masks on physical (a,b,c,alpha,beta,gamma) dimensions.
# 1 = include in loss, 0 = fixed by symmetry (not penalized).
# Note: these follow the project's Bravais-snap conventions in model/bravais.py.
CS_PHYS_PARAM_MASK: list[list[float]] = [
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # cubic
    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # tetragonal
    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # orthorhombic
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],  # hexagonal
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # trigonal
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # monoclinic
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # triclinic
]

# Upweight harder crystal systems (hex/trig/monoclinic/triclinic).
CS_SAMPLE_WEIGHT: list[float] = [1.0, 1.2, 1.2, 3.0, 3.0, 2.5, 2.5]


@dataclass(frozen=True)
class LossWeights:
    """Regression loss configuration."""

    regression: float = 1.0
    mode: LossMode = "baseline"
    length_weight: float = 1.0
    angle_weight: float = 1.0
    physical_weight: float = 1.0
    huber_delta: float = 5.0
    # B1a strict hinge thresholds (north-star aligned).
    hinge_ltol: float = 0.05
    hinge_atol_deg: float = 3.0
    hinge_length_weight: float = 1.0
    hinge_angle_weight: float = 1.0
    # R2: additive Bravais angle prior + optional CS classification for routing.
    angle_prior_weight: float = 0.25
    classification: float = 0.0
    setting_classification: float = 0.0
    """CE weight for cubic P/F/I setting classifier (R3)."""
    manifold_consistency_weight: float = 0.1
    """R4: weight for manifold_consistency_loss (deterministic per-CS angle prior)."""
    peak_consistency_weight: float = 0.15
    """R6-A: weight for peak_consistency_loss (obs↔theory 1/d² Chamfer)."""
    peak_consistency_scale: float = 1000.0
    """Bring squared 1/d² residuals (~1e-4) up toward SmoothL1 magnitude (~0.2)."""
    peak_consistency_max_hkl: int = 4
    """Enumerate |h|,|k|,|l| <= this (exclude 000)."""
    peak_consistency_n_lines: int = 20
    """Use the first N observed peaks (sorted by 2θ)."""
    wavelength_angstrom: float = 1.54184


# Cubic primitive settings (P / F / I) from Bravais constraint validation.
_CUBIC_ANGLE_TARGETS_DEG: tuple[float, ...] = (90.0, 60.0, 109.47)
# Hex/trig: only β=90° is stable on primitive labels (≥95%); do not force γ=120°.
_CS_CUBIC = 0
_CS_TETRAGONAL = 1
_CS_ORTHORHOMBIC = 2
_CS_HEXAGONAL = 3
_CS_TRIGONAL = 4

# R4: unambiguous per-CS symmetry-fixed angle dims (index into α,β,γ = 0,1,2),
# matching the *default*-tier Bravais snap conventions in model/bravais.py
# (primitive-cell reduced setting: tet/ortho fix α,β; hex fixes only β).
# Deliberately excludes cubic (handled by dedicated setting heads), trigonal,
# monoclinic, triclinic (ambiguous free-angle choice; A4 showed hex a=b already
# mismatches ~50% of train, so lengths are excluded here too).
_MANIFOLD_FIXED_ANGLES: dict[int, tuple[int, ...]] = {
    _CS_TETRAGONAL: (0, 1),
    _CS_ORTHORHOMBIC: (0, 1),
    _CS_HEXAGONAL: (1,),
}


def manifold_consistency_loss(
    pred_phys: torch.Tensor,
    crystal_system_idx: torch.Tensor,
    *,
    huber_delta: float = 5.0,
) -> torch.Tensor:
    """Push predicted angles toward their own CS's unambiguous fixed target (90°).

    Unlike ``bravais_angle_prior_loss`` this has exactly one deterministic target
    per (CS, angle-dim) pair -- no min-over-hypotheses, so it cannot contradict a
    genuinely ambiguous ground truth.
    """
    angles = pred_phys[..., 3:]
    device = pred_phys.device
    dtype = pred_phys.dtype
    per_sample = torch.zeros(pred_phys.shape[0], device=device, dtype=dtype)
    for cs_idx, angle_dims in _MANIFOLD_FIXED_ANGLES.items():
        mask = crystal_system_idx == cs_idx
        if not mask.any():
            continue
        sub = angles[mask][:, angle_dims]
        err = F.huber_loss(
            sub,
            torch.full_like(sub, 90.0),
            reduction="none",
            delta=huber_delta,
        ).mean(dim=-1)
        per_sample[mask] = err
    return per_sample.mean()


def mcl_min_loss(
    hyp_pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R10 Multiple-Choice-Learning: per-sample min-over-K SmoothL1.

    ``hyp_pred``: normalized candidates ``[B, K, D]``. ``target``: normalized
    truth ``[B, D]``. Only the closest hypothesis gets gradient per sample
    (winner-take-all), so the K heads can specialize instead of collapsing to
    the conditional mean a single-point regressor learns.

    Returns ``(scalar loss, winner_idx [B])``. ``winner_idx`` is detached
    (argmin has no gradient anyway) and is meant purely for usage-collapse
    diagnostics at the call site.
    """
    target_exp = target.unsqueeze(1).expand_as(hyp_pred)
    per_hyp = F.smooth_l1_loss(hyp_pred, target_exp, reduction="none").mean(dim=-1)  # [B, K]
    min_loss, winner_idx = per_hyp.min(dim=-1)
    return min_loss.mean(), winner_idx.detach()


def _build_hkl_grid(max_hkl: int = 4) -> torch.Tensor:
    """Integer hkl grid with |h|,|k|,|l| <= max_hkl, excluding 000. Shape [N, 3]."""
    vals = torch.arange(-max_hkl, max_hkl + 1, dtype=torch.float32)
    hh, kk, ll = torch.meshgrid(vals, vals, vals, indexing="ij")
    hkl = torch.stack([hh.reshape(-1), kk.reshape(-1), ll.reshape(-1)], dim=-1)
    keep = (hkl.abs().sum(dim=-1) > 0)
    return hkl[keep]


# Cached default grid (max_hkl=4 → 728 rows).
_HKL_GRID_CACHE: dict[int, torch.Tensor] = {}


def _hkl_grid(max_hkl: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if max_hkl not in _HKL_GRID_CACHE:
        _HKL_GRID_CACHE[max_hkl] = _build_hkl_grid(max_hkl)
    return _HKL_GRID_CACHE[max_hkl].to(device=device, dtype=dtype)


def reciprocal_metric_gstar(matrix: torch.Tensor) -> torch.Tensor:
    """Crystallographic G* = (A Aᵀ)⁻¹ so that 1/d² = hᵀ G* h (no 2π)."""
    # matrix: [3, 3] with rows = lattice vectors
    g = matrix @ matrix.transpose(-1, -2)
    # Stabilize tiny volumes.
    eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.solve(g + 1e-8 * eye, eye)


def inv_d2_from_hkl(gstar: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """1/d² for each hkl row. gstar [3,3], hkl [N,3] → [N]."""
    # (H G*) Hᵀ diagonal: sum_{ij} h_i G*_ij h_j
    gh = hkl @ gstar  # [N, 3]
    return (gh * hkl).sum(dim=-1).clamp_min(0.0)


def inv_d2_from_two_theta(
    two_theta_deg: torch.Tensor,
    *,
    wavelength_angstrom: float = 1.54184,
) -> torch.Tensor:
    """Convert 2θ (deg) to 1/d² = (2 sin(θ)/λ)²."""
    theta = torch.deg2rad(two_theta_deg / 2.0)
    q = 2.0 * torch.sin(theta.clamp(min=1e-8)) / float(wavelength_angstrom)
    return q * q


def _batch_lattice_params_to_matrix(cell: torch.Tensor) -> torch.Tensor:
    """Vectorized (a,b,c,alpha,beta,gamma)[deg] -> 3x3 matrix, no python loop over batch.

    Numerically identical to ``geometry.lattice_params_to_matrix`` applied row-wise,
    but avoids per-row kernel launches / host syncs so it stays cheap at large batch.
    """
    lengths = cell[..., :3]
    angles_r = torch.deg2rad(cell[..., 3:])
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)
    val = (coses[..., 0] * coses[..., 1] - coses[..., 2]) / (sins[..., 0] * sins[..., 1] + 1e-12)
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    zero = torch.zeros_like(lengths[..., 0])
    vector_a = torch.stack([lengths[..., 0] * sins[..., 1], zero, lengths[..., 0] * coses[..., 1]], dim=-1)
    vector_b = torch.stack(
        [
            -lengths[..., 1] * sins[..., 0] * torch.cos(gamma_star),
            lengths[..., 1] * sins[..., 0] * torch.sin(gamma_star),
            lengths[..., 1] * coses[..., 0],
        ],
        dim=-1,
    )
    vector_c = torch.stack([zero, zero, lengths[..., 2]], dim=-1)
    return torch.stack([vector_a, vector_b, vector_c], dim=-2)


def peak_consistency_loss(
    pred_phys: torch.Tensor,
    pxrd_x: torch.Tensor,
    peak_num: torch.Tensor,
    *,
    wavelength_angstrom: float = 1.54184,
    max_hkl: int = 4,
    n_lines: int = 20,
    scale: float = 1000.0,
) -> torch.Tensor:
    """Soft Chamfer: mean_i min_hkl (1/d²_obs,i − 1/d²_hkl)², scaled.

    One-way obs→theory only (does not punish systematically-absent theory peaks).
    ``pxrd_x`` is the packed batch peak tensor [Σn, 1] or [Σn]; ``peak_num`` [B].

    Fully vectorized over the batch (no python ``for``/``.item()`` sync loop) so
    this stays cheap even at batch sizes of several hundred/thousand — the earlier
    per-sample loop implementation serialized on the host and was the main reason
    large-batch ``peak_consistency`` runs saw low GPU utilization.
    """
    device = pred_phys.device
    dtype = pred_phys.dtype
    batch = pred_phys.shape[0]
    hkl = _hkl_grid(max_hkl, device=device, dtype=dtype)  # [N, 3]
    peaks = pxrd_x.reshape(-1).to(device=device, dtype=dtype)
    peak_num = peak_num.to(device=device).long()

    offsets = torch.zeros(batch + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(peak_num, dim=0)

    # How many of each sample's (leading) peaks we intend to use.
    eff_n = peak_num.clamp(min=0, max=int(n_lines))
    max_n = int(eff_n.max().item()) if batch > 0 else 0
    if max_n == 0 or peaks.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)

    starts = offsets[:-1]
    idx_range = torch.arange(max_n, device=device).unsqueeze(0)  # [1, max_n]
    valid_mask = idx_range < eff_n.unsqueeze(1)  # [B, max_n]
    gather_idx = (starts.unsqueeze(1) + idx_range).clamp(min=0, max=peaks.numel() - 1)
    obs_padded = peaks[gather_idx]  # [B, max_n]
    finite_mask = valid_mask & torch.isfinite(obs_padded)
    obs_padded = torch.nan_to_num(obs_padded, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    obs_inv_d2 = inv_d2_from_two_theta(obs_padded, wavelength_angstrom=wavelength_angstrom)  # [B, max_n]

    # Stabilize lattice params before geometry (avoid NaN → det=0 → solve fail).
    # Out-of-place only — inplace slice assign breaks autograd.
    lengths = torch.nan_to_num(pred_phys[..., :3], nan=5.0, posinf=50.0, neginf=1.0).clamp(
        min=0.5, max=80.0
    )
    angles = torch.nan_to_num(pred_phys[..., 3:], nan=90.0, posinf=160.0, neginf=20.0).clamp(
        min=20.0, max=160.0
    )
    safe = torch.cat([lengths, angles], dim=-1)  # [B, 6]

    matrix = _batch_lattice_params_to_matrix(safe)  # [B, 3, 3]
    vol = torch.det(matrix).abs()  # [B]
    degenerate = ~torch.isfinite(vol) | (vol < 1e-4)

    eye = torch.eye(3, device=device, dtype=dtype).expand(batch, -1, -1)
    g = matrix @ matrix.transpose(-1, -2)
    # Where degenerate, substitute identity so the batched solve never raises.
    g_safe = torch.where(degenerate.view(-1, 1, 1), eye, g)
    gstar = torch.linalg.solve(g_safe + 1e-8 * eye, eye)  # [B, 3, 3]
    bad = degenerate | ~torch.isfinite(gstar).all(dim=(-1, -2))

    # theory 1/d² per sample for every hkl row: [B, N]
    gh = torch.einsum("nj,bjk->bnk", hkl, gstar)
    theory_inv_d2 = (gh * hkl.unsqueeze(0)).sum(dim=-1).clamp_min(0.0)
    bad = bad | ~torch.isfinite(theory_inv_d2).all(dim=-1)

    # [B, max_n, N] soft Chamfer (obs -> nearest theoretical line).
    delta_sq = (obs_inv_d2.unsqueeze(-1) - theory_inv_d2.unsqueeze(1)) ** 2
    min_over_hkl = delta_sq.min(dim=-1).values  # [B, max_n]

    n_valid = finite_mask.sum(dim=1)
    denom = n_valid.clamp_min(1).to(dtype)
    per_sample_ok = (min_over_hkl * finite_mask.to(dtype)).sum(dim=1) / denom

    # Mild fixed penalty (not scale*1, which becomes 1000 and destabilizes λ≥0.15)
    # for degenerate/non-finite lattices; zero loss for samples with no usable peaks.
    per_sample = torch.where(bad, torch.full_like(per_sample_ok, 0.05), per_sample_ok)
    per_sample = torch.where(n_valid == 0, torch.zeros_like(per_sample), per_sample)

    return scale * per_sample.mean()


def bravais_angle_prior_loss(
    pred_phys: torch.Tensor,
    crystal_system_idx: torch.Tensor,
    *,
    huber_delta: float = 5.0,
) -> torch.Tensor:
    """Min-over-hypotheses angle prior; additive to matrix6 SmoothL1 (R2 P0-B).

    Does **not** reuse CS_PHYS_PARAM_MASK (conventional-cell, wrong for primitive).
    """
    angles = pred_phys[..., 3:]
    device = pred_phys.device
    dtype = pred_phys.dtype
    per_sample = torch.zeros(pred_phys.shape[0], device=device, dtype=dtype)

    cubic_mask = crystal_system_idx == _CS_CUBIC
    if cubic_mask.any():
        cubic_angles = angles[cubic_mask]
        hyp = []
        for target in _CUBIC_ANGLE_TARGETS_DEG:
            err = F.huber_loss(
                cubic_angles,
                torch.full_like(cubic_angles, target),
                reduction="none",
                delta=huber_delta,
            ).mean(dim=-1)
            hyp.append(err)
        per_sample[cubic_mask] = torch.stack(hyp, dim=-1).min(dim=-1).values

    # Hexagonal / trigonal: only β → 90° (data-stable on primitive cells).
    ht_mask = (crystal_system_idx == _CS_HEXAGONAL) | (crystal_system_idx == _CS_TRIGONAL)
    if ht_mask.any():
        beta = angles[ht_mask, 1]
        per_sample[ht_mask] = F.huber_loss(
            beta,
            torch.full_like(beta, 90.0),
            reduction="none",
            delta=huber_delta,
        )

    return per_sample.mean()


def _cs_mask_tensor(
    crystal_system_idx: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask_table = torch.tensor(CS_PHYS_PARAM_MASK, device=device, dtype=dtype)
    return mask_table[crystal_system_idx]


def _cs_weight_tensor(
    crystal_system_idx: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight_table = torch.tensor(CS_SAMPLE_WEIGHT, device=device, dtype=dtype)
    return weight_table[crystal_system_idx]


def _length_angle_physical_parts(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    *,
    param_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    huber_delta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean relative-length error and mean Huber angle error (physical space).

    Length term is |Δ|/|truth| averaged over a,b,c. Angle term is Huber(α,β,γ)
    with ``huber_delta`` in degrees. Both are scalars after batch mean (and
    optional sample_weight).
    """
    length_err = torch.abs(pred_phys[..., :3] - target_phys[..., :3])
    length_denom = torch.clamp(target_phys[..., :3].abs(), min=1e-6)
    length_err = length_err / length_denom

    angle_loss = F.huber_loss(
        pred_phys[..., 3:],
        target_phys[..., 3:],
        reduction="none",
        delta=huber_delta,
    )

    if param_mask is not None:
        length_mask = param_mask[..., :3]
        angle_mask = param_mask[..., 3:]
        length_err = length_err * length_mask
        angle_loss = angle_loss * angle_mask
        length_active = torch.clamp(length_mask.sum(dim=-1), min=1.0)
        angle_active = torch.clamp(angle_mask.sum(dim=-1), min=1.0)
        per_sample_length = length_err.sum(dim=-1) / length_active
        per_sample_angle = angle_loss.sum(dim=-1) / angle_active
    else:
        per_sample_length = length_err.mean(dim=-1)
        per_sample_angle = angle_loss.mean(dim=-1)

    if sample_weight is not None:
        per_sample_length = per_sample_length * sample_weight
        per_sample_angle = per_sample_angle * sample_weight

    return per_sample_length.mean(), per_sample_angle.mean()


def _length_angle_physical_loss(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    *,
    param_mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    length_weight: float,
    angle_weight: float,
    huber_delta: float,
) -> torch.Tensor:
    loss_length, loss_angle = _length_angle_physical_parts(
        pred_phys,
        target_phys,
        param_mask=param_mask,
        sample_weight=sample_weight,
        huber_delta=huber_delta,
    )
    return length_weight * loss_length + angle_weight * loss_angle


def _strict_hinge_physical_loss(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    *,
    ltol: float,
    atol_deg: float,
    length_weight: float,
    angle_weight: float,
) -> torch.Tensor:
    """Hinge on relative length / absolute angle beyond north-star tolerances."""
    length_rel = torch.abs(pred_phys[..., :3] - target_phys[..., :3]) / torch.clamp(
        target_phys[..., :3].abs(), min=1e-6
    )
    angle_abs = torch.abs(pred_phys[..., 3:] - target_phys[..., 3:])
    length_hinge = F.relu(length_rel - ltol).mean(dim=-1)
    angle_hinge = F.relu(angle_abs - atol_deg).mean(dim=-1)
    return (length_weight * length_hinge + angle_weight * angle_hinge).mean()


class IndexingLoss(nn.Module):
    """Lattice regression loss with optional physical-space refinements."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        *,
        normalizer: LatticeNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.normalizer = normalizer

    def forward(
        self,
        lattice_norm_pred: torch.Tensor,
        lattice_norm_target: torch.Tensor,
        *,
        lattice_phys_target: torch.Tensor | None = None,
        crystal_system_idx: torch.Tensor | None = None,
        crystal_system_logits: torch.Tensor | None = None,
        cubic_setting_logits: torch.Tensor | None = None,
        pxrd_x: torch.Tensor | None = None,
        peak_num: torch.Tensor | None = None,
        lattice_hyp_pred: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.weights.mode
        loss_reg = F.smooth_l1_loss(lattice_norm_pred, lattice_norm_target)
        mcl_winner_usage_entropy = torch.zeros(
            (), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype
        )

        if mode == "mcl":
            if lattice_hyp_pred is None:
                raise ValueError(
                    "loss mode mcl requires lattice_hyp_pred (model.multi_hypothesis=True)"
                )
            # Overwrite loss_reg with the actual MCL training objective; the
            # single-point loss above still stays available implicitly via
            # outputs["lattice_norm"] (hyp-0) metrics computed by the caller.
            loss_reg, mcl_winner_idx = mcl_min_loss(lattice_hyp_pred, lattice_norm_target)
            k = lattice_hyp_pred.shape[1]
            if crystal_system_idx is not None and k > 1:
                multi_mask = crystal_system_idx != 0  # non-cubic: heads with real K>1
                if multi_mask.any():
                    usage = torch.bincount(
                        mcl_winner_idx[multi_mask], minlength=k
                    ).to(dtype=lattice_norm_pred.dtype)
                    probs = usage / usage.sum().clamp_min(1.0)
                    mcl_winner_usage_entropy = -(
                        probs * torch.log(probs.clamp_min(1e-8))
                    ).sum() / math.log(k)

        use_physical = mode in (
            "length_angle",
            "cs_mask",
            "combined",
            "strict_hinge",
            "angle_heavy",
            "angle_prior",
            "joint_phys",
            "manifold_consistency",
            "peak_consistency",
        )
        use_mask = mode in ("cs_mask", "combined")
        use_reweight = mode in ("cs_reweight", "combined")

        loss_phys = torch.zeros((), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype)
        loss_length_phys = torch.zeros(
            (), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype
        )
        loss_angle_phys = torch.zeros(
            (), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype
        )
        # Always report physical length/angle diagnostics when denormalization is
        # available (R9/R10 primary optimization signals). These do not enter
        # loss_total unless the active mode already uses physical terms.
        if self.normalizer is not None and lattice_phys_target is not None:
            pred_phys_diag = self.normalizer.denormalize(lattice_norm_pred)
            loss_length_phys, loss_angle_phys = _length_angle_physical_parts(
                pred_phys_diag,
                lattice_phys_target,
                huber_delta=self.weights.huber_delta,
            )

        if use_physical:
            if self.normalizer is None:
                raise ValueError(f"loss mode {mode!r} requires a lattice normalizer")
            if lattice_phys_target is None and mode != "peak_consistency":
                raise ValueError(f"loss mode {mode!r} requires lattice_phys_target")
            if mode == "peak_consistency" and (pxrd_x is None or peak_num is None):
                raise ValueError("loss mode peak_consistency requires pxrd_x and peak_num")

        if use_physical:
            pred_phys = self.normalizer.denormalize(lattice_norm_pred)
            if mode == "strict_hinge":
                loss_phys = _strict_hinge_physical_loss(
                    pred_phys,
                    lattice_phys_target,
                    ltol=self.weights.hinge_ltol,
                    atol_deg=self.weights.hinge_atol_deg,
                    length_weight=self.weights.hinge_length_weight,
                    angle_weight=self.weights.hinge_angle_weight,
                )
            elif mode == "angle_prior":
                if crystal_system_idx is None:
                    raise ValueError("loss mode angle_prior requires crystal_system_idx")
                loss_phys = bravais_angle_prior_loss(
                    pred_phys,
                    crystal_system_idx,
                    huber_delta=self.weights.huber_delta,
                )
            elif mode == "joint_phys":
                # Angles are in degrees (~10–90); scale to ~O(1) so λ~0.05–0.1
                # does not drown matrix6 SmoothL1 (~0.2).
                loss_phys = _length_angle_physical_loss(
                    pred_phys,
                    lattice_phys_target,
                    param_mask=None,
                    sample_weight=None,
                    length_weight=self.weights.length_weight,
                    angle_weight=self.weights.angle_weight / 90.0,
                    huber_delta=max(self.weights.huber_delta / 90.0, 0.05),
                )
            elif mode == "manifold_consistency":
                if crystal_system_idx is None:
                    raise ValueError(
                        "loss mode manifold_consistency requires crystal_system_idx"
                    )
                loss_phys = manifold_consistency_loss(
                    pred_phys,
                    crystal_system_idx,
                    huber_delta=self.weights.huber_delta,
                )
            elif mode == "peak_consistency":
                loss_phys = peak_consistency_loss(
                    pred_phys,
                    pxrd_x,  # type: ignore[arg-type]
                    peak_num,  # type: ignore[arg-type]
                    wavelength_angstrom=self.weights.wavelength_angstrom,
                    max_hkl=self.weights.peak_consistency_max_hkl,
                    n_lines=self.weights.peak_consistency_n_lines,
                    scale=self.weights.peak_consistency_scale,
                )
            else:
                param_mask = None
                sample_weight = None
                if use_mask and crystal_system_idx is not None:
                    param_mask = _cs_mask_tensor(
                        crystal_system_idx,
                        device=pred_phys.device,
                        dtype=pred_phys.dtype,
                    )
                if use_reweight and crystal_system_idx is not None:
                    sample_weight = _cs_weight_tensor(
                        crystal_system_idx,
                        device=pred_phys.device,
                        dtype=pred_phys.dtype,
                    )
                angle_w = (
                    self.weights.angle_weight
                    if mode != "angle_heavy"
                    else max(self.weights.angle_weight, 5.0)
                )
                loss_phys = _length_angle_physical_loss(
                    pred_phys,
                    lattice_phys_target,
                    param_mask=param_mask,
                    sample_weight=sample_weight,
                    length_weight=self.weights.length_weight,
                    angle_weight=angle_w,
                    huber_delta=self.weights.huber_delta,
                )

        if mode == "baseline":
            loss_total = self.weights.regression * loss_reg
        elif mode == "angle_prior":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.angle_prior_weight * loss_phys
            )
        elif mode == "joint_phys":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.physical_weight * loss_phys
            )
        elif mode == "manifold_consistency":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.manifold_consistency_weight * loss_phys
            )
        elif mode == "peak_consistency":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.peak_consistency_weight * loss_phys
            )
        elif mode == "mcl":
            loss_total = self.weights.regression * loss_reg
        elif mode == "cs_reweight":
            if crystal_system_idx is None:
                raise ValueError("loss mode cs_reweight requires crystal_system_idx")
            per_elem = F.smooth_l1_loss(
                lattice_norm_pred,
                lattice_norm_target,
                reduction="none",
            ).mean(dim=-1)
            sample_weight = _cs_weight_tensor(
                crystal_system_idx,
                device=lattice_norm_pred.device,
                dtype=lattice_norm_pred.dtype,
            )
            loss_total = self.weights.regression * (per_elem * sample_weight).mean()
        elif mode in ("length_angle", "cs_mask", "angle_heavy"):
            loss_total = self.weights.physical_weight * loss_phys
        elif mode == "strict_hinge":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.physical_weight * loss_phys
            )
        elif mode == "combined":
            loss_total = (
                self.weights.regression * loss_reg
                + self.weights.physical_weight * loss_phys
            )
        else:
            raise ValueError(f"Unsupported loss mode: {mode!r}")

        loss_cls = torch.zeros((), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype)
        if (
            self.weights.classification > 0.0
            and crystal_system_logits is not None
            and crystal_system_idx is not None
        ):
            loss_cls = F.cross_entropy(crystal_system_logits, crystal_system_idx.long())
            loss_total = loss_total + self.weights.classification * loss_cls

        loss_setting = torch.zeros(
            (), device=lattice_norm_pred.device, dtype=lattice_norm_pred.dtype
        )
        if (
            self.weights.setting_classification > 0.0
            and cubic_setting_logits is not None
            and lattice_phys_target is not None
            and crystal_system_idx is not None
        ):
            cubic_mask = crystal_system_idx == 0
            if cubic_mask.any():
                setting_tgt = cubic_setting_idx_from_phys(lattice_phys_target[cubic_mask])
                loss_setting = F.cross_entropy(
                    cubic_setting_logits[cubic_mask], setting_tgt.long()
                )
                loss_total = loss_total + self.weights.setting_classification * loss_setting

        return {
            "loss_total": loss_total,
            "loss_reg": loss_reg,
            "loss_matrix6": loss_reg,  # alias for R9/R10 reporting
            "loss_phys": loss_phys,
            "loss_length_phys": loss_length_phys,
            "loss_angle_phys": loss_angle_phys,
            "loss_cls": loss_cls,
            "loss_setting": loss_setting,
            "mcl_winner_usage_entropy": mcl_winner_usage_entropy,
        }


def compute_best_metric_score(
    valid_metrics: dict[str, float],
    *,
    best_metric: str = "top1_lattice_match_proxy",
) -> float:
    """Select checkpoint score from validation metrics."""
    if best_metric == "composite":
        proxy = valid_metrics.get("top1_lattice_match_proxy", 0.0)
        lattice = valid_metrics.get("top1_lattice_match_rate", proxy)
        return 0.5 * proxy + 0.5 * lattice
    if best_metric == "strict_composite":
        mapping = valid_metrics.get("strict_raw_top1_lattice_match_rate", 0.0)
        elementwise = valid_metrics.get("strict_raw_top1_elementwise_rate", 0.0)
        return 0.5 * mapping + 0.5 * elementwise
    return float(valid_metrics.get(best_metric, 0.0))
