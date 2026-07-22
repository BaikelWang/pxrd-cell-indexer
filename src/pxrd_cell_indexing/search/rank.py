"""B2: deterministic peaks-only candidate ranking (v3 §12.1).

Ranks a *fixed* B1 candidate pool (this module does not generate or filter
candidates) using only cheap, non-learned features computed from the
observed peaks, the candidate's own peak-fit quality, and (optionally) the
NN's raw single-point proposal as a soft prior for volume/metric proximity.

Feature list follows v3 §12.1 exactly:
  - indexed observed peak count / matched fraction   -> ``n_matched``
  - median / mean / max |Δq|                          -> ``mean_delta`` etc.
  - unexplained observed peak penalty                 -> ``n_peaks - n_matched``
  - theoretical-peak over-density penalty              -> ``n_calc / n_matched``
  - |log(V / V_nn)|                                    -> ``vol_log_ratio``
  - CS probability                                     -> ``cs_log_prob`` (caller-supplied; a
    no-op constant when every candidate in the pool already shares one
    crystal system, as is the case for the current single-system B1 pools)
  - half-cell / super-cell flag                        -> ``supercell_flag``
  - reciprocal-metric distance to NN proposal          -> ``gstar_dist_to_nn``

Kept deliberately separate from ``model/fom.py`` (the legacy R6-C FOM used
on NN-seeded Bravais pools) so the two ranking methods can be compared
head-to-head on the same fixed pool (v3 §12.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.search.qsearch import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    _theoretical_q2_values,
    inverse_d2_from_two_theta_f64,
)

# Candidate volume ratios (vs a reference volume) consistent with a
# half-cell/super-cell relationship along one or more axes.
_SUPERCELL_RATIOS: tuple[float, ...] = (0.125, 0.25, 1.0 / 3.0, 0.5, 2.0, 3.0, 4.0, 8.0)
_SUPERCELL_LOG_TOL = 0.03  # |log(ratio_actual / ratio_nominal)| tolerance


class _CandidateLike(Protocol):
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    n_matched: int
    n_peaks: int
    volume: float


def _params6(candidate: _CandidateLike) -> list[float]:
    return [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]


def _gstar_from_params6(params6: Sequence[float]) -> np.ndarray:
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    direct_metric = matrix @ matrix.T
    return np.linalg.inv(direct_metric)


def _greedy_match_deltas(q_obs: np.ndarray, theory_q: np.ndarray) -> list[float | None]:
    """One-to-one greedy match; None for observed peaks left unmatched."""
    used = np.zeros(theory_q.shape[0], dtype=bool)
    deltas: list[float | None] = []
    for q in np.sort(q_obs):
        available = np.where(~used)[0]
        if available.size == 0:
            deltas.append(None)
            continue
        diffs = np.abs(theory_q[available] - q)
        best_local = int(np.argmin(diffs))
        used[available[best_local]] = True
        deltas.append(float(diffs[best_local]))
    return deltas


@dataclass(frozen=True)
class CandidateFeatures:
    n_matched: int
    n_peaks: int
    matched_fraction: float
    mean_delta: float
    median_delta: float
    max_delta: float
    n_calc: int
    density_ratio: float
    volume: float
    vol_log_ratio: float | None
    supercell_flag: bool
    gstar_dist_to_nn: float | None
    cs_log_prob: float


def compute_candidate_features(
    candidate: _CandidateLike,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    q_match_abs_tol: float = 1e-6,
    nn_volume: float | None = None,
    nn_gstar: np.ndarray | None = None,
    cs_log_prob: float = 0.0,
) -> CandidateFeatures:
    obs = np.asarray(observed_two_theta, dtype=np.float64).reshape(-1)
    obs = obs[np.isfinite(obs)]
    q_obs = inverse_d2_from_two_theta_f64(obs, wavelength_angstrom=wavelength_angstrom)
    gstar = _gstar_from_params6(_params6(candidate))
    q_max = float(q_obs.max()) if q_obs.size else 0.0
    theory_q = _theoretical_q2_values(gstar, q_max)
    n_calc = max(int(theory_q.size), 1)

    if theory_q.size == 0:
        deltas_raw = [None] * q_obs.size
    else:
        deltas_raw = _greedy_match_deltas(q_obs, theory_q)
    unmatched_delta = 0.5
    deltas = [d if d is not None and d <= q_match_abs_tol else unmatched_delta for d in deltas_raw]
    n_matched = sum(1 for d in deltas if d < unmatched_delta - 1e-12)
    matched_only = [d for d in deltas if d < unmatched_delta - 1e-12]
    mean_delta = float(np.mean(matched_only)) if matched_only else unmatched_delta
    median_delta = float(np.median(matched_only)) if matched_only else unmatched_delta
    max_delta = float(np.max(matched_only)) if matched_only else unmatched_delta

    vol_log_ratio: float | None = None
    supercell_flag = False
    if nn_volume is not None and nn_volume > 1e-12 and candidate.volume > 1e-12:
        ratio = candidate.volume / nn_volume
        vol_log_ratio = abs(math.log(ratio))
        for nominal in _SUPERCELL_RATIOS:
            if abs(math.log(ratio / nominal)) <= _SUPERCELL_LOG_TOL:
                supercell_flag = True
                break

    gstar_dist_to_nn: float | None = None
    if nn_gstar is not None:
        # Scale-normalized (trace-normalized) Frobenius distance: robust to
        # the two metrics living at different absolute volume scales, so it
        # measures "shape" proximity, not just size proximity (size is
        # already covered by vol_log_ratio).
        scale = max(float(np.trace(gstar)), 1e-12)
        nn_scale = max(float(np.trace(nn_gstar)), 1e-12)
        diff = gstar / scale - np.asarray(nn_gstar, dtype=np.float64) / nn_scale
        gstar_dist_to_nn = float(np.linalg.norm(diff))

    return CandidateFeatures(
        n_matched=n_matched,
        n_peaks=int(q_obs.size),
        matched_fraction=n_matched / max(int(q_obs.size), 1),
        mean_delta=mean_delta,
        median_delta=median_delta,
        max_delta=max_delta,
        n_calc=n_calc,
        density_ratio=n_calc / max(n_matched, 1),
        volume=float(candidate.volume),
        vol_log_ratio=vol_log_ratio,
        supercell_flag=supercell_flag,
        gstar_dist_to_nn=gstar_dist_to_nn,
        cs_log_prob=cs_log_prob,
    )


@dataclass(frozen=True)
class RankConfig:
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
    q_match_abs_tol: float = 1e-6
    density_ratio_weight: float = 0.02
    vol_log_ratio_weight: float = 1.0
    gstar_dist_weight: float = 5.0
    cs_log_prob_weight: float = 1.0


def _sort_key(
    features: CandidateFeatures, config: RankConfig
) -> tuple[int, int, float, float, float, float, float]:
    """Ascending key (smaller = better rank). Mirrors v3 §12.1's feature list,
    in priority order: exact matches first (dominant discriminator given the
    tight ``q_match_abs_tol``), then supercell/density red flags, then
    volume/metric proximity to the NN proposal, then raw fit quality."""
    density_term = config.density_ratio_weight * features.density_ratio
    vol_term = (
        config.vol_log_ratio_weight * features.vol_log_ratio
        if features.vol_log_ratio is not None
        else 0.0
    )
    gstar_term = (
        config.gstar_dist_weight * features.gstar_dist_to_nn
        if features.gstar_dist_to_nn is not None
        else 0.0
    )
    cs_term = -config.cs_log_prob_weight * features.cs_log_prob
    return (
        -features.n_matched,
        int(features.supercell_flag),
        density_term + vol_term + gstar_term + cs_term,
        features.mean_delta,
        features.median_delta,
        features.max_delta,
        features.volume,
    )


def rank_by_deterministic_score(
    candidates: list,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    config: RankConfig | None = None,
    nn_volume: float | None = None,
    nn_gstar: np.ndarray | None = None,
    cs_log_probs: Sequence[float] | None = None,
) -> list:
    """Rank a fixed candidate pool with the new deterministic score (v3 §12.1)."""
    cfg = config or RankConfig()
    scored = []
    for idx, cand in enumerate(candidates):
        feats = compute_candidate_features(
            cand,
            observed_two_theta,
            wavelength_angstrom=cfg.wavelength_angstrom,
            q_match_abs_tol=cfg.q_match_abs_tol,
            nn_volume=nn_volume,
            nn_gstar=nn_gstar,
            cs_log_prob=cs_log_probs[idx] if cs_log_probs is not None else 0.0,
        )
        scored.append((_sort_key(feats, cfg), cand))
    scored.sort(key=lambda pair: pair[0])
    return [cand for _, cand in scored]


def rank_by_nn_proximity(candidates: list, nn_gstar: np.ndarray) -> list:
    """Baseline "NN confidence" ranking: ignore peak fit entirely, just pick
    whichever pool candidate's reciprocal metric is closest (shape+scale) to
    the NN's own raw single-point proposal."""
    nn_gstar = np.asarray(nn_gstar, dtype=np.float64)

    def _dist(cand: _CandidateLike) -> float:
        gstar = _gstar_from_params6(_params6(cand))
        return float(np.linalg.norm(gstar - nn_gstar))

    return sorted(candidates, key=_dist)
