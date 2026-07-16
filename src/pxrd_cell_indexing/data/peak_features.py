"""Physical peak features for PXRD indexing (reciprocal-space geometry).

Naming convention (avoid Q ambiguity):
  - reciprocal_d = 1/d = 2 sin(θ) / λ          (Å⁻¹)
  - inverse_d2   = 1/d² = reciprocal_d²         (Å⁻²)  — indexing quadratic form
  - intensity_*  = normalized intensity channels

Legacy ``two_theta_to_q`` in ``model.fom`` returns reciprocal_d (1/d), not 1/d².
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

DEFAULT_WAVELENGTH_ANGSTROM = 1.54184
DEFAULT_TWO_THETA_MIN_DEG = 5.0
DEFAULT_TWO_THETA_MAX_DEG = 80.0

IntensityTransform = Literal["none", "linear", "sqrt", "log"]
PeakFeatureMode = Literal[
    "legacy",
    "continuous_2theta_i",
    "reciprocal_d_i",
    "inverse_d2_i",
    "inverse_d2_logi",
    "inverse_d2_only",
]


@dataclass(frozen=True)
class PeakFeatureConfig:
    """Configurable physical peak features for encoder input."""

    feature_mode: PeakFeatureMode = "legacy"
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
    intensity_transform: IntensityTransform = "linear"
    two_theta_min_deg: float = DEFAULT_TWO_THETA_MIN_DEG
    two_theta_max_deg: float = DEFAULT_TWO_THETA_MAX_DEG
    hist_bins: int = 256
    sorted_peak_count: int = 24
    hist_pool: Literal["max", "sum"] = "max"
    intensity_min: float = 5.0
    max_peaks: int | None = None


def reciprocal_d_from_two_theta(
    two_theta_deg: np.ndarray | torch.Tensor,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
) -> np.ndarray | torch.Tensor:
    """Convert 2θ degrees to reciprocal_d = 1/d (Å⁻¹)."""
    if isinstance(two_theta_deg, torch.Tensor):
        theta = torch.deg2rad(two_theta_deg.to(dtype=torch.float32) / 2.0)
        return 2.0 * torch.sin(theta.clamp(min=1e-8)) / float(wavelength_angstrom)
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64) / 2.0)
    return (2.0 * np.sin(np.clip(theta, 1e-8, None)) / float(wavelength_angstrom)).astype(np.float32)


def inverse_d2_from_two_theta(
    two_theta_deg: np.ndarray | torch.Tensor,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
) -> np.ndarray | torch.Tensor:
    """Convert 2θ degrees to inverse_d2 = 1/d² (Å⁻²)."""
    s = reciprocal_d_from_two_theta(two_theta_deg, wavelength_angstrom=wavelength_angstrom)
    return s * s


def inverse_d2_max(
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    two_theta_max_deg: float = DEFAULT_TWO_THETA_MAX_DEG,
) -> float:
    """Upper bound of inverse_d2 for the configured 2θ window."""
    return float(
        inverse_d2_from_two_theta(
            np.array([two_theta_max_deg], dtype=np.float64),
            wavelength_angstrom=wavelength_angstrom,
        )[0]
    )


def normalize_intensity(
    intensity: np.ndarray | torch.Tensor,
    *,
    transform: IntensityTransform = "linear",
) -> np.ndarray | torch.Tensor:
    """Normalize intensity to ~[0, 1] with optional dynamic-range transform."""
    if isinstance(intensity, torch.Tensor):
        x = intensity.to(dtype=torch.float32)
        max_i = torch.clamp(x.amax(), min=1e-8)
        x = x / max_i
        if transform == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        if transform == "log":
            return torch.log1p(x * 99.0) / np.log1p(99.0)
        if transform == "none":
            return intensity.to(dtype=torch.float32)
        return x
    x = np.asarray(intensity, dtype=np.float32)
    max_i = float(np.max(x)) if x.size else 0.0
    if max_i <= 0:
        return x
    x = x / max_i
    if transform == "sqrt":
        return np.sqrt(np.clip(x, 0.0, None)).astype(np.float32)
    if transform == "log":
        return (np.log1p(x * 99.0) / np.log1p(99.0)).astype(np.float32)
    if transform == "none":
        return np.asarray(intensity, dtype=np.float32)
    return x.astype(np.float32)


def peak_feature_dim(mode: PeakFeatureMode) -> int:
    """Number of channels in per-peak token features."""
    if mode == "legacy":
        return 1
    if mode == "inverse_d2_only":
        return 1
    return 2


def build_per_peak_features(
    two_theta: np.ndarray | torch.Tensor,
    intensity: np.ndarray | torch.Tensor,
    *,
    config: PeakFeatureConfig,
) -> np.ndarray | torch.Tensor:
    """Build per-peak feature matrix of shape ``[N, C]``."""
    mode = config.feature_mode
    if isinstance(two_theta, torch.Tensor):
        tt = two_theta.reshape(-1).to(dtype=torch.float32)
        inten = intensity.reshape(-1).to(dtype=torch.float32)
        if mode == "legacy":
            return inten.view(-1, 1)
        inv = inverse_d2_from_two_theta(tt, wavelength_angstrom=config.wavelength_angstrom)
        inv_norm = inv / max(inverse_d2_max(
            wavelength_angstrom=config.wavelength_angstrom,
            two_theta_max_deg=config.two_theta_max_deg,
        ), 1e-8)
        recip = reciprocal_d_from_two_theta(tt, wavelength_angstrom=config.wavelength_angstrom)
        recip_norm = recip / max(
            float(reciprocal_d_from_two_theta(
                np.array([config.two_theta_max_deg]),
                wavelength_angstrom=config.wavelength_angstrom,
            )[0]),
            1e-8,
        )
        inten_n = normalize_intensity(inten, transform=config.intensity_transform)
        tt_norm = (tt - config.two_theta_min_deg) / max(
            config.two_theta_max_deg - config.two_theta_min_deg, 1e-8
        )
        if mode == "continuous_2theta_i":
            return torch.stack([tt_norm, inten_n], dim=-1)
        if mode == "reciprocal_d_i":
            return torch.stack([recip_norm, inten_n], dim=-1)
        if mode == "inverse_d2_i":
            return torch.stack([inv_norm, inten_n], dim=-1)
        if mode == "inverse_d2_logi":
            inten_log = normalize_intensity(inten, transform="log")
            return torch.stack([inv_norm, inten_log], dim=-1)
        if mode == "inverse_d2_only":
            return inv_norm.view(-1, 1)
        raise ValueError(f"Unknown feature_mode: {mode}")

    tt = np.asarray(two_theta, dtype=np.float32).reshape(-1)
    inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if mode == "legacy":
        return inten.reshape(-1, 1)
    inv = inverse_d2_from_two_theta(tt, wavelength_angstrom=config.wavelength_angstrom)
    inv_norm = inv / max(
        inverse_d2_max(
            wavelength_angstrom=config.wavelength_angstrom,
            two_theta_max_deg=config.two_theta_max_deg,
        ),
        1e-8,
    )
    recip = reciprocal_d_from_two_theta(tt, wavelength_angstrom=config.wavelength_angstrom)
    recip_max = float(
        reciprocal_d_from_two_theta(
            np.array([config.two_theta_max_deg]),
            wavelength_angstrom=config.wavelength_angstrom,
        )[0]
    )
    recip_norm = recip / max(recip_max, 1e-8)
    inten_n = normalize_intensity(inten, transform=config.intensity_transform)
    tt_norm = (tt - config.two_theta_min_deg) / max(
        config.two_theta_max_deg - config.two_theta_min_deg, 1e-8
    )
    if mode == "continuous_2theta_i":
        return np.stack([tt_norm, inten_n], axis=-1).astype(np.float32)
    if mode == "reciprocal_d_i":
        return np.stack([recip_norm, inten_n], axis=-1).astype(np.float32)
    if mode == "inverse_d2_i":
        return np.stack([inv_norm, inten_n], axis=-1).astype(np.float32)
    if mode == "inverse_d2_logi":
        inten_log = normalize_intensity(inten, transform="log")
        return np.stack([inv_norm, inten_log], axis=-1).astype(np.float32)
    if mode == "inverse_d2_only":
        return inv_norm.reshape(-1, 1).astype(np.float32)
    raise ValueError(f"Unknown feature_mode: {mode}")


def build_inverse_d2_histogram_features(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    config: PeakFeatureConfig,
) -> np.ndarray:
    """Fixed-length bag features: hist + sorted inverse_d2 + peak count.

    Returns shape ``[hist_bins + sorted_peak_count + 1]``.
    """
    tt = np.asarray(two_theta, dtype=np.float64).reshape(-1)
    inten = np.asarray(intensity, dtype=np.float64).reshape(-1)
    inv = inverse_d2_from_two_theta(tt, wavelength_angstrom=config.wavelength_angstrom)
    inv = np.asarray(inv, dtype=np.float64)
    g_max = inverse_d2_max(
        wavelength_angstrom=config.wavelength_angstrom,
        two_theta_max_deg=config.two_theta_max_deg,
    )
    hist = np.zeros(config.hist_bins, dtype=np.float32)
    if inv.size:
        bins = np.clip((inv / max(g_max, 1e-8) * config.hist_bins).astype(int), 0, config.hist_bins - 1)
        inten_n = normalize_intensity(inten, transform=config.intensity_transform)
        inten_n = np.asarray(inten_n, dtype=np.float32)
        if config.hist_pool == "sum":
            for b, w in zip(bins, inten_n):
                hist[b] += float(w)
        else:
            for b, w in zip(bins, inten_n):
                hist[b] = max(hist[b], float(w))
        if hist.max() > 0:
            hist = hist / hist.max()
    top = np.zeros(config.sorted_peak_count, dtype=np.float32)
    if inv.size:
        qs = np.sort(inv)[: config.sorted_peak_count] / max(g_max, 1e-8)
        top[: qs.size] = qs.astype(np.float32)
    npk = np.array([inv.size / 50.0], dtype=np.float32)
    return np.concatenate([hist, top, npk]).astype(np.float32)


def histogram_feature_dim(config: PeakFeatureConfig) -> int:
    return int(config.hist_bins + config.sorted_peak_count + 1)


def padding_mask_from_peak_num(
    peak_num: torch.Tensor,
    max_peak_num: int | None = None,
    *,
    include_cls: bool = True,
) -> torch.Tensor:
    """Boolean padding mask ``[B, 1+max_N]`` or ``[B, max_N]`` from peak counts.

    True means **masked / ignore**. CLS (index 0) is never masked when include_cls.
    """
    if max_peak_num is None:
        max_peak_num = int(peak_num.max().item())
    batch = peak_num.shape[0]
    device = peak_num.device
    arange = torch.arange(max_peak_num, device=device).view(1, -1)
    peak_mask = arange >= peak_num.view(-1, 1)
    if not include_cls:
        return peak_mask
    cls = torch.zeros((batch, 1), dtype=torch.bool, device=device)
    return torch.cat([cls, peak_mask], dim=1)
