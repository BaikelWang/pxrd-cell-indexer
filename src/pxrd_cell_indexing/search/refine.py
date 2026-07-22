"""B3: optional iterative local refiner for B1 q-search candidates (v3 §13).

Only meaningful *after* B1 recall and B2 ranking have already passed their
Gates -- this step does not search for new hkl-assignment basins, it only
tightens an already-correct-basin candidate's reciprocal metric using *all*
its currently-matched peaks (the initial q-search exact solve only used the
system's minimal ``k`` low-angle peaks; peaks discovered later by the de
Wolff consistency check never feed back into the metric estimate).

Each step:
  1. match every observed peak to its nearest theoretical hkl line (within
     tolerance) under the *current* G*;
  2. re-solve G* by (weighted) least squares over *all* matched (hkl, q_obs)
     pairs -- ``q = hᵀG*h`` is linear in G*'s free params, so this is a
     plain overdetermined linear solve, not gradient descent;
  3. keep the step only if it stays SPD and does not lose matched peaks.

If the input peaks already match to ~float64 precision (e.g. noiseless
simulated data), refinement is expected to be a no-op -- there is no residual
for a linear least-squares refit to remove. This module still reports
per-step matched-peak count / mean residual so that a null result is visible
and auditable, rather than silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.search.qsearch import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    QSearchCandidate,
    _basis,
    _coeff_row,
    _fast_match_count,
    gstar_to_lattice_params,
    inverse_d2_from_two_theta_f64,
)


def _theoretical_hkl_grid(gstar: np.ndarray, q_max: float, *, max_hkl_cap: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Like ``qsearch._theoretical_q2_values`` but keeps the (h,k,l) labels
    (needed to build refit coefficient rows), at a smaller default hkl cap
    (refinement only needs the low-order lines actually observed, not the
    exhaustive high-order grid the de Wolff consistency check scans)."""
    diag = np.clip(np.diag(gstar), 1e-12, None)
    limits = np.clip(np.ceil(np.sqrt(max(q_max, 1e-9) / diag)).astype(int) + 1, 1, max_hkl_cap)
    h_max, k_max, l_max = (int(v) for v in limits)
    h = np.arange(-h_max, h_max + 1)
    k = np.arange(-k_max, k_max + 1)
    l = np.arange(-l_max, l_max + 1)
    hh, kk, ll = np.meshgrid(h, k, l, indexing="ij")
    hf, kf, lf = hh.reshape(-1).astype(np.float64), kk.reshape(-1).astype(np.float64), ll.reshape(-1).astype(np.float64)
    mask = (hf != 0) | (kf != 0) | (lf != 0)
    hf, kf, lf = hf[mask], kf[mask], lf[mask]
    q2 = (
        hf * hf * gstar[0, 0]
        + kf * kf * gstar[1, 1]
        + lf * lf * gstar[2, 2]
        + 2.0 * hf * kf * gstar[0, 1]
        + 2.0 * hf * lf * gstar[0, 2]
        + 2.0 * kf * lf * gstar[1, 2]
    )
    margin = max(1e-6, 1e-4 * q_max)
    keep = (q2 > 0) & (q2 <= q_max + margin)
    hkl = np.stack([hf[keep], kf[keep], lf[keep]], axis=1)
    return hkl, q2[keep]


def _match_hkl_to_peaks(
    q_obs: np.ndarray, hkl: np.ndarray, theory_q: np.ndarray, *, q_match_abs_tol: float
) -> list[tuple[np.ndarray, float] | None]:
    """Greedy 1:1 de Wolff-style match; returns, per observed peak (ascending
    q order), the matched ``(hkl, |Δq|)`` or ``None``."""
    used = np.zeros(theory_q.shape[0], dtype=bool)
    out: list[tuple[np.ndarray, float] | None] = []
    for q in np.sort(q_obs):
        available = np.where(~used)[0]
        if available.size == 0:
            out.append(None)
            continue
        diffs = np.abs(theory_q[available] - q)
        best_local = int(np.argmin(diffs))
        delta = float(diffs[best_local])
        if delta <= q_match_abs_tol:
            best_idx = int(available[best_local])
            used[best_idx] = True
            out.append((hkl[best_idx], delta))
        else:
            out.append(None)
    return out


@dataclass(frozen=True)
class RefineStepResult:
    candidate: QSearchCandidate
    n_matched: int
    mean_abs_delta: float
    is_spd: bool
    n_matched_used_in_refit: int


def refine_candidate_step(
    candidate: QSearchCandidate,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    q_match_abs_tol: float = 1e-6,
) -> RefineStepResult | None:
    """One refit step: match all peaks under the candidate's current metric,
    then re-solve G* by least squares over every matched (hkl, q) pair.
    Returns ``None`` if fewer than ``k`` (system DOF) peaks match -- can't
    refit an underdetermined system, so the candidate is returned unchanged
    by the caller in that case."""
    system = candidate.crystal_system
    basis = _basis(system)
    k = basis.shape[1]

    params6 = [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma]
    matrix = lattice_params_to_matrix(torch.tensor(params6, dtype=torch.float64)).numpy()
    gstar = np.linalg.inv(matrix @ matrix.T)

    obs = np.asarray(observed_two_theta, dtype=np.float64).reshape(-1)
    obs = obs[np.isfinite(obs)]
    q_obs = inverse_d2_from_two_theta_f64(obs, wavelength_angstrom=wavelength_angstrom)
    q_max = float(q_obs.max()) if q_obs.size else 0.0

    hkl_grid, theory_q = _theoretical_hkl_grid(gstar, q_max)
    if theory_q.size == 0:
        return None
    matches = _match_hkl_to_peaks(q_obs, hkl_grid, theory_q, q_match_abs_tol=q_match_abs_tol)
    matched = [m for m in matches if m is not None]
    if len(matched) < k:
        return None

    # ``matches`` is aligned with ``np.sort(q_obs)``; keep only the entries
    # (hkl, observed q) for peaks that actually matched.
    sorted_q_obs = np.sort(q_obs)
    matched_hkl = [m[0] for m in matches if m is not None]
    matched_q_obs = np.array(
        [q for q, m in zip(sorted_q_obs, matches, strict=True) if m is not None], dtype=np.float64
    )
    coeff_rows = np.stack([_coeff_row(int(h), int(kk), int(l)) for h, kk, l in matched_hkl])
    basis_rows = coeff_rows @ basis  # (M, k)

    try:
        params_k, *_ = np.linalg.lstsq(basis_rows, matched_q_obs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    gvec6 = basis @ params_k
    new_gstar = np.array(
        [
            [gvec6[0], gvec6[3], gvec6[4]],
            [gvec6[3], gvec6[1], gvec6[5]],
            [gvec6[4], gvec6[5], gvec6[2]],
        ]
    )
    lattice = gstar_to_lattice_params(new_gstar)
    if lattice is None:
        return RefineStepResult(
            candidate=candidate,
            n_matched=candidate.n_matched,
            mean_abs_delta=float("nan"),
            is_spd=False,
            n_matched_used_in_refit=len(matched),
        )

    a, b, c, alpha, beta, gamma = lattice
    direct = np.linalg.inv(new_gstar)
    volume = float(np.sqrt(max(np.linalg.det(direct), 0.0)))

    n_matched_new = _fast_match_count(q_obs, new_gstar, q_match_abs_tol=q_match_abs_tol)
    deltas = [d for _, d in matched]
    mean_abs_delta = float(np.mean(deltas)) if deltas else float("nan")

    new_candidate = replace(
        candidate,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        n_matched=n_matched_new,
        n_peaks=candidate.n_peaks,
        volume=volume,
    )
    return RefineStepResult(
        candidate=new_candidate,
        n_matched=n_matched_new,
        mean_abs_delta=mean_abs_delta,
        is_spd=True,
        n_matched_used_in_refit=len(matched),
    )


def iterative_refine(
    candidate: QSearchCandidate,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    max_steps: int = 3,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    q_match_abs_tol: float = 1e-6,
) -> list[QSearchCandidate]:
    """Returns ``[step0=input, step1, ..., stepN]`` (N<=max_steps). Stops
    early (repeats the last good candidate) if a step would lose matched
    peaks, go non-SPD, or become underdetermined -- refinement must never
    make the candidate *worse* (v3 §13: "若 peak score 改善但 strict 变差,
    说明 residual objective 偏好等价/伪胞,停止 refiner")."""
    trace = [candidate]
    current = candidate
    for _ in range(max_steps):
        result = refine_candidate_step(
            current,
            observed_two_theta,
            wavelength_angstrom=wavelength_angstrom,
            q_match_abs_tol=q_match_abs_tol,
        )
        if result is None or not result.is_spd or result.n_matched < current.n_matched:
            trace.append(current)
            continue
        current = result.candidate
        trace.append(current)
    return trace
