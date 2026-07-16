"""de Wolff M(N) figure-of-merit reranking for Top-K lattice candidates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.types import LatticeCandidate

# LMDB PXRD sticks from pymatgen XRDCalculator() default (Cu Kα weighted, 241113_save_pxrd_data.py).
DEFAULT_WAVELENGTH_ANGSTROM = 1.54184
DEFAULT_TWO_THETA_MAX_DEG = 90.0
DEFAULT_MAX_HKL_CAP = 30
DEFAULT_N_LINES = 20
# Ideal simulated sticks: truth peaks match theory at ~0 with correct λ; 1e-6 from valid1400 sweep.
DEFAULT_Q_MATCH_ABS_TOL = 1e-6
DEFAULT_Q_MATCH_RTOL = 0.0
UNMATCHED_Q_DELTA = 0.5
MIN_FOM = 1e-12

FomRankingMode = Literal["heuristic", "strict_dewolff", "intensity_weighted"]


@dataclass(frozen=True)
class FomRerankConfig:
    """Configurable FOM reranking (ranking-only, does not change candidate pool)."""

    mode: FomRankingMode = "heuristic"
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
    n_lines: int = DEFAULT_N_LINES
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP
    q_match_abs_tol: float = DEFAULT_Q_MATCH_ABS_TOL
    q_match_rtol: float = DEFAULT_Q_MATCH_RTOL
    unmatched_delta: float = UNMATCHED_Q_DELTA
    collapse_variants: bool = False
    """Group scale/axis variants by base Bravais key; keep one representative per group."""
    ref_volume: float | None = None
    """If set, prefer candidates near this volume (NN prediction) instead of smaller cells."""
    max_log_volume_ratio: float | None = None
    """Hard-drop candidates with |log(V/V_ref)| above this when ``ref_volume`` is set."""
    volume_log_penalty: float = 1.0
    """Weight of |log(V/V_ref)| in the sort key when ``ref_volume`` is set."""


def _params_to_array(lattice_params: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(lattice_params, dtype=np.float64).reshape(6)


def reciprocal_metric_tensor(lattice_matrix: np.ndarray) -> np.ndarray:
    """Reciprocal metric tensor G* with G*_ij = a_i* · a_j* (2π convention)."""
    matrix = np.asarray(lattice_matrix, dtype=np.float64).reshape(3, 3)
    volume = float(np.linalg.det(matrix))
    if abs(volume) < 1e-12:
        raise ValueError("degenerate lattice matrix (zero volume)")
    a_vec, b_vec, c_vec = matrix[0], matrix[1], matrix[2]
    astar = 2.0 * np.pi * np.cross(b_vec, c_vec) / volume
    bstar = 2.0 * np.pi * np.cross(c_vec, a_vec) / volume
    cstar = 2.0 * np.pi * np.cross(a_vec, b_vec) / volume
    reciprocal = np.stack([astar, bstar, cstar], axis=0)
    return reciprocal @ reciprocal.T


def _q_from_hkl_grid(
    h_vals: np.ndarray,
    k_vals: np.ndarray,
    l_vals: np.ndarray,
    gstar: np.ndarray,
) -> np.ndarray:
    """Vectorized |g|/(2π) = 1/d for all hkl combinations (excludes 000)."""
    hh, kk, ll = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
    h_flat = hh.reshape(-1).astype(np.float64)
    k_flat = kk.reshape(-1).astype(np.float64)
    l_flat = ll.reshape(-1).astype(np.float64)
    mask = (h_flat != 0) | (k_flat != 0) | (l_flat != 0)
    h_flat = h_flat[mask]
    k_flat = k_flat[mask]
    l_flat = l_flat[mask]

    g11 = gstar[0, 0]
    g22 = gstar[1, 1]
    g33 = gstar[2, 2]
    g12 = gstar[0, 1]
    g13 = gstar[0, 2]
    g23 = gstar[1, 2]

    g_sq = (
        h_flat * h_flat * g11
        + k_flat * k_flat * g22
        + l_flat * l_flat * g33
        + 2.0 * h_flat * k_flat * g12
        + 2.0 * h_flat * l_flat * g13
        + 2.0 * k_flat * l_flat * g23
    )
    g_sq = np.maximum(g_sq, 0.0)
    return np.sqrt(g_sq) / (2.0 * np.pi)


def q_to_two_theta(q_values: np.ndarray, *, wavelength_angstrom: float) -> np.ndarray:
    """Convert Q = 1/d (Å⁻¹) to 2θ degrees."""
    arg = np.clip(wavelength_angstrom * q_values / 2.0, -1.0, 1.0)
    theta_rad = np.arcsin(arg)
    return np.rad2deg(2.0 * theta_rad)


def two_theta_to_q(two_theta_deg: np.ndarray, *, wavelength_angstrom: float) -> np.ndarray:
    """Convert 2θ degrees to Q = 1/d (Å⁻¹)."""
    theta_rad = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64) / 2.0)
    return 2.0 * np.sin(theta_rad) / wavelength_angstrom


def _hkl_limits(gstar: np.ndarray, q_max: float, *, max_hkl_cap: int) -> tuple[int, int, int]:
    """Per-axis |h|/|k|/|l| upper bounds covering reflections up to q_max."""
    diag = np.array([gstar[0, 0], gstar[1, 1], gstar[2, 2]], dtype=np.float64)
    diag = np.maximum(diag, 1e-12)
    q_target = max(q_max, 1e-6) * 2.0 * np.pi
    limits = [int(np.ceil(q_target / np.sqrt(value))) for value in diag]
    return tuple(min(max_hkl_cap, max(1, limit)) for limit in limits)


def theoretical_two_theta(
    lattice_params: Sequence[float] | np.ndarray,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG,
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP,
) -> np.ndarray:
    """Enumerate hkl reflections and return sorted unique theoretical 2θ (degrees)."""
    params = _params_to_array(lattice_params)
    matrix = lattice_params_to_matrix(torch.tensor(params, dtype=torch.float64)).numpy()
    gstar = reciprocal_metric_tensor(matrix)

    q_max = 2.0 * np.sin(np.deg2rad(two_theta_max / 2.0)) / wavelength_angstrom
    h_max, k_max, l_max = _hkl_limits(gstar, q_max, max_hkl_cap=max_hkl_cap)

    h_vals = np.arange(-h_max, h_max + 1)
    k_vals = np.arange(-k_max, k_max + 1)
    l_vals = np.arange(-l_max, l_max + 1)

    q_values = _q_from_hkl_grid(h_vals, k_vals, l_vals, gstar)
    two_theta = q_to_two_theta(q_values, wavelength_angstrom=wavelength_angstrom)
    two_theta = two_theta[two_theta <= two_theta_max + 1e-6]
    if two_theta.size == 0:
        return np.array([], dtype=np.float64)

    two_theta = np.unique(np.round(two_theta, decimals=6))
    return np.sort(two_theta)


def _q_match_tolerance(
    q_obs: float,
    *,
    q_match_abs_tol: float,
    q_match_rtol: float,
) -> float:
    rel = q_match_rtol * float(q_obs) if q_match_rtol > 0.0 else 0.0
    return max(q_match_abs_tol, rel, 1e-12)


def _match_deltas_unique(
    obs_q: np.ndarray,
    theory_q: np.ndarray,
    *,
    obs_weights: np.ndarray | None = None,
    q_match_abs_tol: float = DEFAULT_Q_MATCH_ABS_TOL,
    q_match_rtol: float = DEFAULT_Q_MATCH_RTOL,
    unmatched_delta: float = UNMATCHED_Q_DELTA,
) -> list[float]:
    """Greedy one-to-one Q matching; optional intensity-ordered assignment."""
    if obs_weights is None:
        order = np.argsort(obs_q)
    else:
        order = np.argsort(-obs_weights)

    theory_sorted = np.sort(theory_q)
    used = np.zeros(theory_sorted.shape[0], dtype=bool)
    deltas = [unmatched_delta] * obs_q.shape[0]

    for obs_idx in order:
        q_obs = float(obs_q[obs_idx])
        available_idx = np.where(~used)[0]
        if available_idx.size == 0:
            continue
        diffs = np.abs(theory_sorted[available_idx] - q_obs)
        best_local = int(np.argmin(diffs))
        best_idx = int(available_idx[best_local])
        delta = float(diffs[best_local])
        tol = _q_match_tolerance(
            q_obs,
            q_match_abs_tol=q_match_abs_tol,
            q_match_rtol=q_match_rtol,
        )
        if delta <= tol:
            used[best_idx] = True
            deltas[obs_idx] = delta
    return deltas


def _candidate_volume(lattice_params: Sequence[float] | np.ndarray) -> float:
    params = _params_to_array(lattice_params)
    matrix = lattice_params_to_matrix(torch.tensor(params, dtype=torch.float64)).numpy()
    return float(abs(np.linalg.det(matrix)))


@dataclass(frozen=True)
class _MatchStats:
    n_matched: int
    mean_delta: float
    de_wolff_m: float
    strict_dewolff_m: float
    intensity_score: float
    volume: float
    n_calc: int


def _compute_match_stats(
    observed_two_theta: Sequence[float] | np.ndarray,
    lattice_params: Sequence[float] | np.ndarray,
    *,
    observed_intensity: Sequence[float] | np.ndarray | None = None,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    n_lines: int = DEFAULT_N_LINES,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG,
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP,
    q_match_abs_tol: float = DEFAULT_Q_MATCH_ABS_TOL,
    q_match_rtol: float = DEFAULT_Q_MATCH_RTOL,
    unmatched_delta: float = UNMATCHED_Q_DELTA,
) -> _MatchStats:
    observed = np.asarray(observed_two_theta, dtype=np.float64).reshape(-1)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return _MatchStats(0, unmatched_delta, MIN_FOM, MIN_FOM, 0.0, 0.0, 0)

    sort_idx = np.argsort(observed)
    observed = observed[sort_idx]
    weights: np.ndarray | None = None
    if observed_intensity is not None:
        raw = np.asarray(observed_intensity, dtype=np.float64).reshape(-1)
        if raw.size == observed.size:
            weights = raw[sort_idx]
            total = float(weights.sum())
            if total > 0:
                weights = weights / total

    obs_max = float(np.max(observed))
    theory = theoretical_two_theta(
        lattice_params,
        wavelength_angstrom=wavelength_angstrom,
        two_theta_max=max(two_theta_max, obs_max + 0.5),
        max_hkl_cap=max_hkl_cap,
    )
    volume = _candidate_volume(lattice_params)
    if theory.size == 0:
        return _MatchStats(0, unmatched_delta, MIN_FOM, MIN_FOM, 0.0, volume, 0)

    theory_q = two_theta_to_q(theory, wavelength_angstrom=wavelength_angstrom)
    n = min(n_lines, observed.size)
    obs_q = two_theta_to_q(observed[:n], wavelength_angstrom=wavelength_angstrom)
    obs_weights = weights[:n] if weights is not None else None
    deltas = _match_deltas_unique(
        obs_q,
        theory_q,
        obs_weights=obs_weights,
        q_match_abs_tol=q_match_abs_tol,
        q_match_rtol=q_match_rtol,
        unmatched_delta=unmatched_delta,
    )
    n_matched = sum(1 for delta in deltas if delta < unmatched_delta - 1e-9)
    mean_delta = float(np.mean(deltas))
    q_n = float(obs_q[-1]) if obs_q.size else 0.0
    n_calc = max(int(np.sum(theory_q <= q_n + 1e-12)), 1)

    if mean_delta < 1e-12:
        de_wolff_m = float("inf")
        strict_dewolff_m = float("inf")
    else:
        de_wolff_m = q_n / (2.0 * n * mean_delta)
        strict_dewolff_m = q_n / (2.0 * mean_delta * n_calc)

    if obs_weights is None:
        intensity_score = n_matched / max(n, 1)
    else:
        matched_weight = sum(
            weight
            for weight, delta in zip(obs_weights, deltas, strict=False)
            if delta < unmatched_delta - 1e-9
        )
        intensity_score = float(matched_weight)

    return _MatchStats(
        n_matched=n_matched,
        mean_delta=mean_delta,
        de_wolff_m=de_wolff_m,
        strict_dewolff_m=strict_dewolff_m,
        intensity_score=intensity_score,
        volume=volume,
        n_calc=n_calc,
    )


def de_wolff_fom(
    observed_two_theta: Sequence[float] | np.ndarray,
    lattice_params: Sequence[float] | np.ndarray,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    n_lines: int = DEFAULT_N_LINES,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG,
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP,
    q_match_abs_tol: float = DEFAULT_Q_MATCH_ABS_TOL,
    q_match_rtol: float = DEFAULT_Q_MATCH_RTOL,
    unmatched_delta: float = UNMATCHED_Q_DELTA,
) -> float:
    """Simplified de Wolff M(N) = Q_N / (2·N·mean|ΔQ|)."""
    stats = _compute_match_stats(
        observed_two_theta,
        lattice_params,
        wavelength_angstrom=wavelength_angstrom,
        n_lines=n_lines,
        two_theta_max=two_theta_max,
        max_hkl_cap=max_hkl_cap,
        q_match_abs_tol=q_match_abs_tol,
        q_match_rtol=q_match_rtol,
        unmatched_delta=unmatched_delta,
    )
    return stats.de_wolff_m


def _base_bravais_key(bravais_key: str | None) -> str:
    if not bravais_key:
        return "unknown"
    return re.split(r":(?:scale|axis)", bravais_key, maxsplit=1)[0]


def collapse_scale_duplicates(
    candidates: list[LatticeCandidate],
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    config: FomRerankConfig | None = None,
    observed_intensity: Sequence[float] | np.ndarray | None = None,
) -> list[LatticeCandidate]:
    """Keep one representative per base Bravais hypothesis (smallest volume, best fit)."""
    if not candidates:
        return []
    cfg = config or FomRerankConfig()

    groups: dict[str, list[LatticeCandidate]] = {}
    for candidate in candidates:
        base = _base_bravais_key(candidate.bravais_key)
        groups.setdefault(base, []).append(candidate)

    representatives: list[LatticeCandidate] = []
    for group in groups.values():
        best: LatticeCandidate | None = None
        best_key: tuple[float, float, float] | None = None
        for candidate in group:
            stats = _compute_match_stats(
                observed_two_theta,
                [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma],
                observed_intensity=observed_intensity,
                wavelength_angstrom=cfg.wavelength_angstrom,
                n_lines=cfg.n_lines,
                two_theta_max=cfg.two_theta_max,
                max_hkl_cap=cfg.max_hkl_cap,
                q_match_abs_tol=cfg.q_match_abs_tol,
                q_match_rtol=cfg.q_match_rtol,
                unmatched_delta=cfg.unmatched_delta,
            )
            key = (-stats.n_matched, stats.volume, stats.mean_delta)
            if best is None or key < best_key:  # type: ignore[operator]
                best = candidate
                best_key = key
        if best is not None:
            representatives.append(best)
    return representatives


def _sort_key_for_mode(
    stats: _MatchStats,
    mode: FomRankingMode,
    *,
    ref_volume: float | None = None,
    volume_log_penalty: float = 1.0,
) -> tuple:
    """Ascending sort key (smaller = better rank)."""
    if ref_volume is not None and ref_volume > 1e-12 and stats.volume > 1e-12:
        vol_term = volume_log_penalty * abs(math.log(stats.volume / ref_volume))
    else:
        # Legacy: prefer smaller volume among equal peak matches (can favor half-cells).
        vol_term = stats.volume
    if mode == "strict_dewolff":
        return (-stats.n_matched, vol_term, -stats.strict_dewolff_m)
    if mode == "intensity_weighted":
        return (-stats.intensity_score, -stats.n_matched, vol_term, stats.mean_delta)
    return (-stats.n_matched, vol_term, stats.mean_delta)


def rerank_candidates_by_fom(
    candidates: list[LatticeCandidate],
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    observed_intensity: Sequence[float] | np.ndarray | None = None,
    config: FomRerankConfig | None = None,
    # Legacy keyword overrides (used when config is None).
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    n_lines: int = DEFAULT_N_LINES,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG,
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP,
    q_match_abs_tol: float = DEFAULT_Q_MATCH_ABS_TOL,
    q_match_rtol: float = DEFAULT_Q_MATCH_RTOL,
    unmatched_delta: float = UNMATCHED_Q_DELTA,
    mode: FomRankingMode | None = None,
    collapse_variants: bool | None = None,
) -> list[LatticeCandidate]:
    """Rerank candidates by peak-table FOM (ignores Bravais confidence)."""
    if config is None:
        cfg = FomRerankConfig(
            mode=mode or "heuristic",
            wavelength_angstrom=wavelength_angstrom,
            n_lines=n_lines,
            two_theta_max=two_theta_max,
            max_hkl_cap=max_hkl_cap,
            q_match_abs_tol=q_match_abs_tol,
            q_match_rtol=q_match_rtol,
            unmatched_delta=unmatched_delta,
            collapse_variants=collapse_variants if collapse_variants is not None else False,
        )
    else:
        cfg = config

    pool = list(candidates)
    if cfg.collapse_variants:
        pool = collapse_scale_duplicates(
            pool,
            observed_two_theta,
            config=cfg,
            observed_intensity=observed_intensity,
        )

    if (
        cfg.ref_volume is not None
        and cfg.ref_volume > 1e-12
        and cfg.max_log_volume_ratio is not None
    ):
        kept: list[LatticeCandidate] = []
        for candidate in pool:
            vol = _candidate_volume(
                [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]
            )
            if vol < 1e-12:
                continue
            if abs(math.log(vol / cfg.ref_volume)) <= cfg.max_log_volume_ratio:
                kept.append(candidate)
        pool = kept

    if not pool:
        return []

    scored: list[tuple[tuple, float, LatticeCandidate]] = []
    for candidate in pool:
        stats = _compute_match_stats(
            observed_two_theta,
            [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma],
            observed_intensity=observed_intensity,
            wavelength_angstrom=cfg.wavelength_angstrom,
            n_lines=cfg.n_lines,
            two_theta_max=cfg.two_theta_max,
            max_hkl_cap=cfg.max_hkl_cap,
            q_match_abs_tol=cfg.q_match_abs_tol,
            q_match_rtol=cfg.q_match_rtol,
            unmatched_delta=cfg.unmatched_delta,
        )
        sort_key = _sort_key_for_mode(
            stats,
            cfg.mode,
            ref_volume=cfg.ref_volume,
            volume_log_penalty=cfg.volume_log_penalty,
        )
        scored.append(
            (
                sort_key,
                stats.de_wolff_m,
                candidate.model_copy(update={"fom_score": stats.de_wolff_m}),
            )
        )

    scored.sort(key=lambda item: item[0])
    return [candidate for _, _, candidate in scored]


def slice_observed_two_theta(
    pxrd_x: torch.Tensor,
    peak_num: torch.Tensor,
    sample_index: int,
) -> np.ndarray:
    """Extract one sample's 2θ peak positions from a flattened BertModel batch."""
    offsets = torch.zeros(peak_num.shape[0] + 1, dtype=torch.long, device=peak_num.device)
    offsets[1:] = torch.cumsum(peak_num, dim=0)
    start = int(offsets[sample_index].item())
    end = int(offsets[sample_index + 1].item())
    peaks = pxrd_x[start:end].detach().cpu().numpy().reshape(-1)
    return peaks.astype(np.float64)


def slice_observed_intensity(
    pxrd_y: torch.Tensor,
    peak_num: torch.Tensor,
    sample_index: int,
) -> np.ndarray:
    """Extract one sample's peak intensities aligned with slice_observed_two_theta."""
    offsets = torch.zeros(peak_num.shape[0] + 1, dtype=torch.long, device=peak_num.device)
    offsets[1:] = torch.cumsum(peak_num, dim=0)
    start = int(offsets[sample_index].item())
    end = int(offsets[sample_index + 1].item())
    values = pxrd_y[start:end].detach().cpu().numpy().reshape(-1)
    return values.astype(np.float64)
