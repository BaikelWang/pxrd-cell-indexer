"""Inference-time local lattice refine by soft Q-match to observed peaks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from pxrd_cell_indexing.model.bravais import CUBIC_I_ANGLE, generate_bravais_hypotheses
from pxrd_cell_indexing.model.fom import (
    DEFAULT_MAX_HKL_CAP,
    DEFAULT_N_LINES,
    DEFAULT_TWO_THETA_MAX_DEG,
    DEFAULT_WAVELENGTH_ANGSTROM,
    theoretical_two_theta,
    two_theta_to_q,
)
from pxrd_cell_indexing.model.topk import (
    dedupe_candidates,
    filter_candidates_by_volume_vs_base,
    lattice_params_volume,
)
from pxrd_cell_indexing.types import LatticeCandidate

LatticeParams6 = np.ndarray
PackFn = Callable[[LatticeParams6], np.ndarray]
UnpackFn = Callable[[np.ndarray, LatticeParams6], LatticeParams6]


@dataclass(frozen=True)
class RefineConfig:
    """Local refine hyperparameters (inference only)."""

    max_steps: int = 40
    top_n: int = 10
    """Refine only the first ``top_n`` candidates (by current pool order)."""
    length_rel_bound: float = 0.25
    """Relative ± bound on a,b,c around the seed candidate."""
    angle_abs_bound_deg: float = 15.0
    """Absolute ± bound on angles around the seed candidate."""
    max_log_volume_ratio: float = float(np.log(2.0))
    """Reject refined params if |log(V/V_seed)| exceeds this."""
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
    n_lines: int = DEFAULT_N_LINES
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG
    max_hkl_cap: int = 20
    """Slightly smaller than FOM default for speed during optimize loops."""
    unmatched_penalty: float = 0.5
    """Soft penalty when theory has no nearby peak (Å⁻¹)."""
    ftol: float = 1e-8


@dataclass(frozen=True)
class ManifoldSpec:
    """Crystal-system / Bravais manifold with free-parameter packing."""

    key: str
    crystal_system: str
    pack: PackFn
    unpack: UnpackFn
    n_free: int


def _pack_cubic_mean(params: LatticeParams6) -> np.ndarray:
    return np.array([(params[0] + params[1] + params[2]) / 3.0], dtype=np.float64)


def _unpack_cubic_p(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    return np.array([a, a, a, 90.0, 90.0, 90.0], dtype=np.float64)


def _unpack_cubic_f(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    return np.array([a, a, a, 60.0, 60.0, 60.0], dtype=np.float64)


def _unpack_cubic_i(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    ang = CUBIC_I_ANGLE
    return np.array([a, a, a, ang, ang, ang], dtype=np.float64)


def _pack_tet(params: LatticeParams6) -> np.ndarray:
    a = (params[0] + params[1]) / 2.0
    return np.array([a, params[2]], dtype=np.float64)


def _unpack_tet(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    c = float(max(free[1], 1e-3))
    return np.array([a, a, c, 90.0, 90.0, 90.0], dtype=np.float64)


def _pack_orth(params: LatticeParams6) -> np.ndarray:
    return np.array(params[:3], dtype=np.float64)


def _unpack_orth(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a, b, c = (float(max(v, 1e-3)) for v in free[:3])
    return np.array([a, b, c, 90.0, 90.0, 90.0], dtype=np.float64)


def _pack_hex(params: LatticeParams6) -> np.ndarray:
    a = (params[0] + params[1]) / 2.0
    return np.array([a, params[2]], dtype=np.float64)


def _unpack_hex(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    c = float(max(free[1], 1e-3))
    return np.array([a, a, c, 90.0, 90.0, 120.0], dtype=np.float64)


def _pack_trig_r(params: LatticeParams6) -> np.ndarray:
    a = (params[0] + params[1]) / 2.0
    alpha = (params[3] + params[4]) / 2.0
    return np.array([a, params[2], alpha], dtype=np.float64)


def _unpack_trig_r(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a = float(max(free[0], 1e-3))
    c = float(max(free[1], 1e-3))
    alpha = float(np.clip(free[2], 1.0, 179.0))
    return np.array([a, a, c, alpha, alpha, alpha], dtype=np.float64)


def _pack_mono_beta(params: LatticeParams6) -> np.ndarray:
    return np.array([params[0], params[1], params[2], params[4]], dtype=np.float64)


def _unpack_mono_beta(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a, b, c = (float(max(v, 1e-3)) for v in free[:3])
    beta = float(np.clip(free[3], 1.0, 179.0))
    return np.array([a, b, c, 90.0, beta, 90.0], dtype=np.float64)


def _pack_mono_alpha(params: LatticeParams6) -> np.ndarray:
    return np.array([params[0], params[1], params[2], params[3]], dtype=np.float64)


def _unpack_mono_alpha(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a, b, c = (float(max(v, 1e-3)) for v in free[:3])
    alpha = float(np.clip(free[3], 1.0, 179.0))
    return np.array([a, b, c, alpha, 90.0, 90.0], dtype=np.float64)


def _pack_mono_gamma(params: LatticeParams6) -> np.ndarray:
    return np.array([params[0], params[1], params[2], params[5]], dtype=np.float64)


def _unpack_mono_gamma(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    a, b, c = (float(max(v, 1e-3)) for v in free[:3])
    gamma = float(np.clip(free[3], 1.0, 179.0))
    return np.array([a, b, c, 90.0, 90.0, gamma], dtype=np.float64)


def _pack_triclinic(params: LatticeParams6) -> np.ndarray:
    return np.asarray(params, dtype=np.float64).reshape(6).copy()


def _unpack_triclinic(free: np.ndarray, _seed: LatticeParams6) -> LatticeParams6:
    out = np.asarray(free, dtype=np.float64).reshape(6).copy()
    out[:3] = np.maximum(out[:3], 1e-3)
    out[3:] = np.clip(out[3:], 1.0, 179.0)
    return out


MANIFOLD_SPECS: dict[str, ManifoldSpec] = {
    "cubic_P": ManifoldSpec("cubic_P", "cubic", _pack_cubic_mean, _unpack_cubic_p, 1),
    "cubic_F": ManifoldSpec("cubic_F", "cubic", _pack_cubic_mean, _unpack_cubic_f, 1),
    "cubic_I": ManifoldSpec("cubic_I", "cubic", _pack_cubic_mean, _unpack_cubic_i, 1),
    "tetragonal_P": ManifoldSpec("tetragonal_P", "tetragonal", _pack_tet, _unpack_tet, 2),
    "orthorhombic_P": ManifoldSpec("orthorhombic_P", "orthorhombic", _pack_orth, _unpack_orth, 3),
    "hex_trig_P": ManifoldSpec("hex_trig_P", "hexagonal", _pack_hex, _unpack_hex, 2),
    "hex_trig_P_strict": ManifoldSpec(
        "hex_trig_P_strict", "hexagonal", _pack_hex, _unpack_hex, 2
    ),
    "trigonal_R": ManifoldSpec("trigonal_R", "trigonal", _pack_trig_r, _unpack_trig_r, 3),
    "monoclinic_P_alpha": ManifoldSpec(
        "monoclinic_P_alpha", "monoclinic", _pack_mono_alpha, _unpack_mono_alpha, 4
    ),
    "monoclinic_P_beta": ManifoldSpec(
        "monoclinic_P_beta", "monoclinic", _pack_mono_beta, _unpack_mono_beta, 4
    ),
    "monoclinic_P_gamma": ManifoldSpec(
        "monoclinic_P_gamma", "monoclinic", _pack_mono_gamma, _unpack_mono_gamma, 4
    ),
    "identity": ManifoldSpec("identity", "unknown", _pack_triclinic, _unpack_triclinic, 6),
}


def resolve_manifold(bravais_key: str | None) -> ManifoldSpec:
    """Map a Bravais / candidate key to a manifold (falls back to triclinic)."""
    if not bravais_key:
        return MANIFOLD_SPECS["identity"]
    base = bravais_key.split(":")[0]
    return MANIFOLD_SPECS.get(base, MANIFOLD_SPECS["identity"])


@dataclass(frozen=True)
class SearchConfig:
    """Multi-seed manifold local search for Top-K pool expansion."""

    max_seeds: int = 8
    """How many Bravais hypotheses (by geometric score) to refine."""
    keep_unrefined: bool = True
    """Also keep snapped seeds in the pool (before refine)."""
    bravais_set: str = "default"
    refine: RefineConfig = field(
        default_factory=lambda: RefineConfig(max_steps=30, max_hkl_cap=15, n_lines=12)
    )
    k: int = 20
    max_log_volume_ratio_vs_base: float | None = float(np.log(2.0))
    """Volume guard vs NN raw prediction; None disables."""


def _clamp_params(params: np.ndarray) -> np.ndarray:
    out = np.asarray(params, dtype=np.float64).reshape(6).copy()
    out[:3] = np.maximum(out[:3], 1e-3)
    out[3:] = np.clip(out[3:], 1.0, 179.0)
    return out


def soft_q_match_loss(
    lattice_params: Sequence[float] | np.ndarray,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    n_lines: int = DEFAULT_N_LINES,
    two_theta_max: float = DEFAULT_TWO_THETA_MAX_DEG,
    max_hkl_cap: int = DEFAULT_MAX_HKL_CAP,
    unmatched_penalty: float = 0.5,
) -> float:
    """Mean soft |ΔQ| of the first ``n_lines`` observed peaks to nearest theory Q."""
    observed = np.asarray(observed_two_theta, dtype=np.float64).reshape(-1)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return unmatched_penalty

    observed = np.sort(observed)
    n = min(n_lines, observed.size)
    obs_q = two_theta_to_q(observed[:n], wavelength_angstrom=wavelength_angstrom)

    try:
        theory = theoretical_two_theta(
            lattice_params,
            wavelength_angstrom=wavelength_angstrom,
            two_theta_max=max(two_theta_max, float(observed.max()) + 0.5),
            max_hkl_cap=max_hkl_cap,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return unmatched_penalty * 10.0

    if theory.size == 0:
        return unmatched_penalty * float(n)

    theory_q = two_theta_to_q(theory, wavelength_angstrom=wavelength_angstrom)
    # Soft nearest-neighbor: no hard tolerance; large gaps capped by unmatched_penalty.
    diffs = np.abs(obs_q[:, None] - theory_q[None, :])
    nearest = diffs.min(axis=1)
    nearest = np.minimum(nearest, unmatched_penalty)
    return float(nearest.mean())


def refine_candidate_by_q_match(
    candidate: LatticeCandidate,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    config: RefineConfig | None = None,
) -> LatticeCandidate:
    """Locally refine one candidate's six parameters against observed peak Qs."""
    cfg = config or RefineConfig()
    seed = np.array(
        [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma],
        dtype=np.float64,
    )
    seed_volume = lattice_params_volume(seed)
    if cfg.max_steps <= 0:
        return candidate

    length_lo = seed[:3] * (1.0 - cfg.length_rel_bound)
    length_hi = seed[:3] * (1.0 + cfg.length_rel_bound)
    angle_lo = np.clip(seed[3:] - cfg.angle_abs_bound_deg, 1.0, 179.0)
    angle_hi = np.clip(seed[3:] + cfg.angle_abs_bound_deg, 1.0, 179.0)
    bounds = list(
        zip(
            np.concatenate([length_lo, angle_lo]).tolist(),
            np.concatenate([length_hi, angle_hi]).tolist(),
        )
    )

    def objective(x: np.ndarray) -> float:
        params = _clamp_params(x)
        try:
            volume = lattice_params_volume(params)
        except Exception:
            return 1e3
        if volume < 1e-12 or seed_volume < 1e-12:
            return 1e3
        log_ratio = abs(math.log(volume / seed_volume))
        if log_ratio > cfg.max_log_volume_ratio:
            return 1e3 + 10.0 * log_ratio
        loss = soft_q_match_loss(
            params,
            observed_two_theta,
            wavelength_angstrom=cfg.wavelength_angstrom,
            n_lines=cfg.n_lines,
            two_theta_max=cfg.two_theta_max,
            max_hkl_cap=cfg.max_hkl_cap,
            unmatched_penalty=cfg.unmatched_penalty,
        )
        # Mild volume regularizer keeps refine near seed scale.
        return loss + 0.01 * log_ratio

    result = minimize(
        objective,
        seed,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.max_steps, "ftol": cfg.ftol},
    )
    refined = _clamp_params(result.x if result.success or result.x is not None else seed)

    # Final volume guard vs seed (hard reject → keep original).
    try:
        refined_volume = lattice_params_volume(refined)
        if (
            refined_volume < 1e-12
            or abs(math.log(refined_volume / max(seed_volume, 1e-12))) > cfg.max_log_volume_ratio
        ):
            return candidate
    except Exception:
        return candidate

    # Keep only if soft loss improved.
    seed_loss = soft_q_match_loss(
        seed,
        observed_two_theta,
        wavelength_angstrom=cfg.wavelength_angstrom,
        n_lines=cfg.n_lines,
        two_theta_max=cfg.two_theta_max,
        max_hkl_cap=cfg.max_hkl_cap,
        unmatched_penalty=cfg.unmatched_penalty,
    )
    refined_loss = soft_q_match_loss(
        refined,
        observed_two_theta,
        wavelength_angstrom=cfg.wavelength_angstrom,
        n_lines=cfg.n_lines,
        two_theta_max=cfg.two_theta_max,
        max_hkl_cap=cfg.max_hkl_cap,
        unmatched_penalty=cfg.unmatched_penalty,
    )
    if refined_loss > seed_loss * 0.999:
        return candidate

    key = candidate.bravais_key or "unknown"
    return LatticeCandidate(
        crystal_system=candidate.crystal_system,
        a=float(refined[0]),
        b=float(refined[1]),
        c=float(refined[2]),
        alpha=float(refined[3]),
        beta=float(refined[4]),
        gamma=float(refined[5]),
        confidence=candidate.confidence,
        bravais_key=f"{key}:refined",
        fom_score=candidate.fom_score,
    )


def refine_candidates(
    candidates: Sequence[LatticeCandidate],
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    config: RefineConfig | None = None,
) -> list[LatticeCandidate]:
    """Refine top-N candidates, merge with originals, dedupe, keep pool size."""
    cfg = config or RefineConfig()
    pool = list(candidates)
    if not pool or cfg.max_steps <= 0:
        return pool

    n_refine = min(cfg.top_n, len(pool))
    refined: list[LatticeCandidate] = []
    for candidate in pool[:n_refine]:
        refined.append(
            refine_candidate_by_q_match(
                candidate,
                observed_two_theta,
                config=cfg,
            )
        )

    merged = dedupe_candidates(list(pool) + refined)
    # Prefer refined (often slightly lower confidence) when near-duplicates:
    # re-sort putting :refined keys first among equals via confidence then key.
    merged = sorted(
        merged,
        key=lambda c: (
            0 if (c.bravais_key or "").endswith(":refined") else 1,
            -c.confidence,
        ),
    )
    # Restore confidence-primary order for downstream FOM, but keep refined variants.
    merged = sorted(merged, key=lambda c: c.confidence, reverse=True)
    return merged[: len(pool)]


def _free_bounds(
    free0: np.ndarray,
    seed_params: LatticeParams6,
    *,
    length_rel_bound: float,
    angle_abs_bound_deg: float,
) -> list[tuple[float, float]]:
    """Build L-BFGS bounds for a free-parameter vector packed from ``seed_params``."""
    bounds: list[tuple[float, float]] = []
    # Heuristic: values < 20 treated as lengths (Å); otherwise angles (deg).
    # Cubic a~few Å; angles are 60–120. Safe split at 20.
    for value in free0:
        if value < 20.0:
            lo = max(value * (1.0 - length_rel_bound), 1e-3)
            hi = value * (1.0 + length_rel_bound)
        else:
            lo = max(value - angle_abs_bound_deg, 1.0)
            hi = min(value + angle_abs_bound_deg, 179.0)
        bounds.append((float(lo), float(hi)))
    # Guard: if packing mixes angle into <20 (rare), still ok via clamp in unpack.
    _ = seed_params
    return bounds


def refine_candidate_on_manifold(
    candidate: LatticeCandidate,
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    config: RefineConfig | None = None,
    manifold: ManifoldSpec | None = None,
) -> LatticeCandidate:
    """Refine free parameters on a Bravais manifold; project angles/lengths via unpack."""
    cfg = config or RefineConfig()
    spec = manifold or resolve_manifold(candidate.bravais_key)
    seed = np.array(
        [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma],
        dtype=np.float64,
    )
    # Project seed onto manifold first so free0 is consistent.
    free0 = spec.pack(seed)
    seed_on_manifold = spec.unpack(free0, seed)
    seed_volume = lattice_params_volume(seed_on_manifold)
    if cfg.max_steps <= 0:
        return LatticeCandidate(
            crystal_system=spec.crystal_system,
            a=float(seed_on_manifold[0]),
            b=float(seed_on_manifold[1]),
            c=float(seed_on_manifold[2]),
            alpha=float(seed_on_manifold[3]),
            beta=float(seed_on_manifold[4]),
            gamma=float(seed_on_manifold[5]),
            confidence=candidate.confidence,
            bravais_key=candidate.bravais_key or spec.key,
            fom_score=candidate.fom_score,
        )

    bounds = _free_bounds(
        free0,
        seed_on_manifold,
        length_rel_bound=cfg.length_rel_bound,
        angle_abs_bound_deg=cfg.angle_abs_bound_deg,
    )

    def objective(x: np.ndarray) -> float:
        params = spec.unpack(x, seed_on_manifold)
        try:
            volume = lattice_params_volume(params)
        except Exception:
            return 1e3
        if volume < 1e-12 or seed_volume < 1e-12:
            return 1e3
        log_ratio = abs(math.log(volume / seed_volume))
        if log_ratio > cfg.max_log_volume_ratio:
            return 1e3 + 10.0 * log_ratio
        loss = soft_q_match_loss(
            params,
            observed_two_theta,
            wavelength_angstrom=cfg.wavelength_angstrom,
            n_lines=cfg.n_lines,
            two_theta_max=cfg.two_theta_max,
            max_hkl_cap=cfg.max_hkl_cap,
            unmatched_penalty=cfg.unmatched_penalty,
        )
        return loss + 0.01 * log_ratio

    result = minimize(
        objective,
        free0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.max_steps, "ftol": cfg.ftol},
    )
    free_opt = result.x if result.x is not None else free0
    refined = spec.unpack(free_opt, seed_on_manifold)

    try:
        refined_volume = lattice_params_volume(refined)
        if (
            refined_volume < 1e-12
            or abs(math.log(refined_volume / max(seed_volume, 1e-12))) > cfg.max_log_volume_ratio
        ):
            refined = seed_on_manifold
    except Exception:
        refined = seed_on_manifold

    seed_loss = soft_q_match_loss(
        seed_on_manifold,
        observed_two_theta,
        wavelength_angstrom=cfg.wavelength_angstrom,
        n_lines=cfg.n_lines,
        two_theta_max=cfg.two_theta_max,
        max_hkl_cap=cfg.max_hkl_cap,
        unmatched_penalty=cfg.unmatched_penalty,
    )
    refined_loss = soft_q_match_loss(
        refined,
        observed_two_theta,
        wavelength_angstrom=cfg.wavelength_angstrom,
        n_lines=cfg.n_lines,
        two_theta_max=cfg.two_theta_max,
        max_hkl_cap=cfg.max_hkl_cap,
        unmatched_penalty=cfg.unmatched_penalty,
    )
    if refined_loss > seed_loss * 0.999:
        refined = seed_on_manifold
        tag = "manifold"
    else:
        tag = "manifold_refined"

    key = candidate.bravais_key or spec.key
    # Confidence boost when peak match improved.
    conf = float(candidate.confidence)
    if refined_loss < seed_loss:
        conf = conf * (1.0 + min(1.0, (seed_loss - refined_loss) / max(seed_loss, 1e-6)))

    return LatticeCandidate(
        crystal_system=spec.crystal_system,
        a=float(refined[0]),
        b=float(refined[1]),
        c=float(refined[2]),
        alpha=float(refined[3]),
        beta=float(refined[4]),
        gamma=float(refined[5]),
        confidence=conf,
        bravais_key=f"{key}:{tag}",
        fom_score=candidate.fom_score,
    )


def build_manifold_search_candidates(
    lattice_params: torch.Tensor | np.ndarray,
    observed_two_theta_batch: Sequence[Sequence[float] | np.ndarray],
    *,
    config: SearchConfig | None = None,
) -> list[list[LatticeCandidate]]:
    """Multi-seed Bravais-manifold local search → Top-K pools (one per sample).

    For each NN-predicted lattice:
      1. generate Bravais snap seeds
      2. refine each seed on its crystal-system manifold via soft Q-match
      3. merge, volume-guard vs NN prediction, dedupe, truncate to ``k``
    """
    cfg = config or SearchConfig()
    if torch.is_tensor(lattice_params):
        rows = lattice_params.detach().cpu().to(dtype=torch.float64).numpy()
    else:
        rows = np.asarray(lattice_params, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, 6)
    if len(observed_two_theta_batch) != rows.shape[0]:
        raise ValueError(
            f"observed batch size {len(observed_two_theta_batch)} != "
            f"lattice batch {rows.shape[0]}"
        )

    pools: list[list[LatticeCandidate]] = []
    for i, row in enumerate(rows):
        observed = observed_two_theta_batch[i]
        hypotheses = generate_bravais_hypotheses(row, bravais_set=cfg.bravais_set)
        seeds = hypotheses[: cfg.max_seeds]
        candidates: list[LatticeCandidate] = []
        for hyp in seeds:
            seed_cand = LatticeCandidate(
                crystal_system=hyp.crystal_system_label or "unknown",
                a=float(hyp.snapped_params[0]),
                b=float(hyp.snapped_params[1]),
                c=float(hyp.snapped_params[2]),
                alpha=float(hyp.snapped_params[3]),
                beta=float(hyp.snapped_params[4]),
                gamma=float(hyp.snapped_params[5]),
                confidence=float(hyp.confidence),
                bravais_key=hyp.bravais_key,
            )
            if cfg.keep_unrefined:
                candidates.append(seed_cand)
            manifold = resolve_manifold(hyp.bravais_key)
            refined = refine_candidate_on_manifold(
                seed_cand,
                observed,
                config=cfg.refine,
                manifold=manifold,
            )
            candidates.append(refined)

        base_volume = lattice_params_volume(row)
        if cfg.max_log_volume_ratio_vs_base is not None:
            candidates = filter_candidates_by_volume_vs_base(
                candidates,
                base_volume=base_volume,
                max_log_volume_ratio=cfg.max_log_volume_ratio_vs_base,
            )
        merged = dedupe_candidates(candidates)
        merged = sorted(merged, key=lambda c: c.confidence, reverse=True)
        pools.append(merged[: cfg.k])
    return pools
