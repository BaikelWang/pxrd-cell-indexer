"""A4 robustness-curriculum spectrum perturbations (v3 §8.1 / v4 §7).

Splits the single ``xrd_augment`` toggle into auditable, independently
controllable perturbation components that mimic real-experiment error
sources, as opposed to the legacy ``augment_spectrum`` (uniform per-peak
shift + intensity noise/scale, always-on when ``xrd_augment=True``):

- ``global_zero_shift_deg``: whole-pattern 2θ zero-point error (instrument).
- ``per_peak_jitter_deg``: independent per-peak position noise (peak fitting).
- peak dropout: missing weak/overlapping peaks.
- impurity peaks: extra peaks from a secondary phase.
- intensity multiplicative noise.
- preferred orientation: random suppression of a subset of peak intensities.

Two entry points:

- :func:`apply_random_robust_perturbation` — used at *train* time; draws all
  magnitudes from configured ranges and (optionally) skips perturbation for a
  ``clean_probability`` fraction of samples (the 80/20 clean/perturb mix of
  A4-C2).
- :func:`apply_named_perturbation` — used *offline* to build frozen
  robust-valid sets (V-zero/V-jitter/V-drop/V-impurity/V-mixed); applies a
  single perturbation type at a fixed, reproducible severity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.random import Generator

PerturbName = Literal["clean", "zero", "jitter", "drop", "impurity", "mixed"]


@dataclass(frozen=True)
class RobustPerturbConfig:
    """Auditable perturbation ranges for A4 train-time augmentation."""

    clean_probability: float = 0.8
    """P(sample stays clean this epoch); A4-C2 uses 0.8 (80/20 clean/perturb)."""
    global_zero_shift_deg: float = 0.3
    """Whole-pattern shift drawn uniformly from [-x, +x]."""
    per_peak_jitter_sigma_deg: float = 0.05
    per_peak_jitter_clip_deg: float = 0.15
    peak_dropout_max_count: int = 4
    impurity_peak_max_count: int = 2
    impurity_intensity_frac_max: float = 0.2
    """Impurity peak intensity is drawn below this fraction of the pattern max."""
    intensity_noise_frac_range: tuple[float, float] = (0.05, 0.10)
    """Multiplicative per-peak intensity noise std, drawn per-sample from this range."""
    preferred_orientation_max_peak_frac: float = 0.3
    """Upper bound on the fraction of peaks subject to intensity suppression."""
    preferred_orientation_max_suppress_frac: float = 0.3
    """Upper bound on how much a suppressed peak's intensity is reduced (0-1)."""
    min_peaks_floor: int = 2
    """Never drop peaks below this count (keeps model input non-degenerate)."""


def _refilter_sort(
    two_theta: np.ndarray, intensity: np.ndarray, intensity_min: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = intensity > intensity_min
    two_theta = two_theta[mask]
    intensity = intensity[mask]
    order = np.argsort(two_theta)
    return two_theta[order].astype(np.float32), intensity[order].astype(np.float32)


def _apply_global_zero_shift(
    two_theta: np.ndarray, shift_deg: float
) -> np.ndarray:
    return two_theta + shift_deg


def _apply_per_peak_jitter(
    two_theta: np.ndarray, sigma_deg: float, clip_deg: float, rng: Generator
) -> np.ndarray:
    if sigma_deg <= 0:
        return two_theta
    jitter = rng.normal(0.0, sigma_deg, size=two_theta.shape)
    jitter = np.clip(jitter, -clip_deg, clip_deg)
    return two_theta + jitter


def _apply_dropout(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    n_drop: int,
    rng: Generator,
    min_peaks_floor: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_drop = min(n_drop, max(two_theta.shape[0] - min_peaks_floor, 0))
    if n_drop <= 0:
        return two_theta, intensity
    drop_idx = rng.choice(two_theta.shape[0], size=n_drop, replace=False)
    keep_mask = np.ones(two_theta.shape[0], dtype=bool)
    keep_mask[drop_idx] = False
    return two_theta[keep_mask], intensity[keep_mask]


def _apply_impurity(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    n_impurity: int,
    intensity_frac_max: float,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if n_impurity <= 0 or two_theta.shape[0] == 0:
        return two_theta, intensity
    lo, hi = float(two_theta.min()), float(two_theta.max())
    if hi <= lo:
        return two_theta, intensity
    max_intensity = float(intensity.max())
    imp_x = rng.uniform(lo, hi, size=n_impurity)
    imp_y = rng.uniform(0.0, intensity_frac_max * max_intensity, size=n_impurity)
    return np.concatenate([two_theta, imp_x]), np.concatenate([intensity, imp_y])


def _apply_intensity_noise(
    intensity: np.ndarray, noise_frac: float, rng: Generator
) -> np.ndarray:
    if noise_frac <= 0:
        return intensity
    factor = 1.0 + rng.normal(0.0, noise_frac, size=intensity.shape)
    factor = np.clip(factor, 0.05, None)
    return intensity * factor


def _apply_preferred_orientation(
    intensity: np.ndarray,
    max_peak_frac: float,
    max_suppress_frac: float,
    rng: Generator,
) -> np.ndarray:
    if max_peak_frac <= 0 or max_suppress_frac <= 0 or intensity.shape[0] == 0:
        return intensity
    peak_frac = rng.uniform(0.0, max_peak_frac)
    n_affected = int(round(peak_frac * intensity.shape[0]))
    if n_affected <= 0:
        return intensity
    affected = rng.choice(intensity.shape[0], size=n_affected, replace=False)
    suppress = rng.uniform(0.0, max_suppress_frac, size=n_affected)
    out = intensity.copy()
    out[affected] = out[affected] * (1.0 - suppress)
    return out


def apply_random_robust_perturbation(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    config: RobustPerturbConfig,
    rng: Generator,
    *,
    intensity_min: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-time A4 augmentation: clean w.p. ``clean_probability``, else apply
    all perturbation components at magnitudes drawn from ``config``'s ranges."""
    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if rng.random() < config.clean_probability:
        return two_theta.astype(np.float32), intensity.astype(np.float32)

    shift = rng.uniform(-config.global_zero_shift_deg, config.global_zero_shift_deg)
    two_theta = _apply_global_zero_shift(two_theta, shift)
    two_theta = _apply_per_peak_jitter(
        two_theta, config.per_peak_jitter_sigma_deg, config.per_peak_jitter_clip_deg, rng
    )
    n_drop = int(rng.integers(0, config.peak_dropout_max_count + 1))
    two_theta, intensity = _apply_dropout(
        two_theta, intensity, n_drop, rng, config.min_peaks_floor
    )
    n_impurity = int(rng.integers(0, config.impurity_peak_max_count + 1))
    two_theta, intensity = _apply_impurity(
        two_theta, intensity, n_impurity, config.impurity_intensity_frac_max, rng
    )
    noise_frac = rng.uniform(*config.intensity_noise_frac_range)
    intensity = _apply_intensity_noise(intensity, noise_frac, rng)
    intensity = _apply_preferred_orientation(
        intensity,
        config.preferred_orientation_max_peak_frac,
        config.preferred_orientation_max_suppress_frac,
        rng,
    )
    return _refilter_sort(two_theta, intensity, intensity_min)


def apply_named_perturbation(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    name: PerturbName,
    severity: float,
    rng: Generator,
    *,
    intensity_min: float = 5.0,
    min_peaks_floor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Offline/eval-time: apply a single named perturbation at a fixed severity.

    Used to build the frozen robust-valid sets (v3 §8.3 / v4 §7):
      - ``zero``: global shift, severity = ± degrees.
      - ``jitter``: per-peak Gaussian sigma (degrees), clipped to 3·severity.
      - ``drop``: severity = number of peaks dropped (rounded).
      - ``impurity``: severity = number of impurity peaks added (rounded),
        each below 20% of the pattern's max intensity.
      - ``mixed``: zero + jitter + drop + impurity combined at a fixed
        moderate severity each (severity arg is ignored; kept for API symmetry).
      - ``clean``: no-op (returns filtered/sorted input unchanged).
    """
    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    if name == "clean":
        pass
    elif name == "zero":
        two_theta = _apply_global_zero_shift(two_theta, rng.uniform(-severity, severity))
    elif name == "jitter":
        two_theta = _apply_per_peak_jitter(two_theta, severity, 3.0 * severity, rng)
    elif name == "drop":
        two_theta, intensity = _apply_dropout(
            two_theta, intensity, int(round(severity)), rng, min_peaks_floor
        )
    elif name == "impurity":
        two_theta, intensity = _apply_impurity(
            two_theta, intensity, int(round(severity)), 0.2, rng
        )
    elif name == "mixed":
        two_theta = _apply_global_zero_shift(two_theta, rng.uniform(-0.2, 0.2))
        two_theta = _apply_per_peak_jitter(two_theta, 0.05, 0.15, rng)
        two_theta, intensity = _apply_dropout(two_theta, intensity, 2, rng, min_peaks_floor)
        two_theta, intensity = _apply_impurity(two_theta, intensity, 2, 0.2, rng)
    else:
        raise ValueError(f"unknown perturbation name: {name!r}")

    return _refilter_sort(two_theta, intensity, intensity_min)
