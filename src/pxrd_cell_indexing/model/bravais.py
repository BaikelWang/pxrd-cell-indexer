"""Bravais primitive-cell constraint hypotheses for geometry snap Top-K."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

CUBIC_I_ANGLE = 109.47122063449069
LENGTH_RTOL_REF = 0.02
ANGLE_ATOL_REF = 2.0
IDENTITY_PENALTY_SCORE = 1.0

LatticeParams = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class BravaisHypothesis:
    """One snapped lattice hypothesis with geometric fit score."""

    bravais_key: str
    crystal_system_label: str | None
    snapped_params: LatticeParams
    score: float

    @property
    def confidence(self) -> float:
        return 1.0 / (1.0 + self.score)


def _as_params(values: torch.Tensor | list[float]) -> LatticeParams:
    if torch.is_tensor(values):
        row = values.reshape(6).tolist()
    else:
        row = list(values)
    return tuple(float(v) for v in row)


def _snap_cubic_p(params: LatticeParams) -> LatticeParams:
    a, b, c, _, _, _ = params
    mean_len = (a + b + c) / 3.0
    return mean_len, mean_len, mean_len, 90.0, 90.0, 90.0


def _snap_cubic_f(params: LatticeParams) -> LatticeParams:
    a, b, c, _, _, _ = params
    mean_len = (a + b + c) / 3.0
    return mean_len, mean_len, mean_len, 60.0, 60.0, 60.0


def _snap_cubic_i(params: LatticeParams) -> LatticeParams:
    a, b, c, _, _, _ = params
    mean_len = (a + b + c) / 3.0
    angle = CUBIC_I_ANGLE
    return mean_len, mean_len, mean_len, angle, angle, angle


def _snap_tetragonal_p(params: LatticeParams) -> LatticeParams:
    a, b, c, _, _, gamma = params
    return a, b, c, 90.0, 90.0, gamma


def _snap_orthorhombic_p(params: LatticeParams) -> LatticeParams:
    a, b, c, _, _, gamma = params
    return a, b, c, 90.0, 90.0, gamma


def _snap_hex_trig_p(params: LatticeParams) -> LatticeParams:
    a, b, c, alpha, _, gamma = params
    return a, b, c, alpha, 90.0, gamma


def _snap_hex_trig_p_strict(params: LatticeParams) -> LatticeParams:
    """Stricter hex/trig: a=b, β=90°, γ=120° (only ~50% of train hex; A4 ablation)."""
    a, b, c, alpha, _, _ = params
    mean_ab = (a + b) / 2.0
    return mean_ab, mean_ab, c, alpha, 90.0, 120.0


def _snap_trigonal_r(params: LatticeParams) -> LatticeParams:
    a, b, c, alpha, beta, gamma = params
    mean_ab = (a + b) / 2.0
    mean_angle = (alpha + beta) / 2.0
    return mean_ab, mean_ab, c, mean_angle, mean_angle, gamma


def _snap_monoclinic_p_alpha(params: LatticeParams) -> LatticeParams:
    """Monoclinic-like: free α, force β=γ=90° (soft; train mono is messy)."""
    a, b, c, alpha, _, _ = params
    return a, b, c, alpha, 90.0, 90.0


def _snap_monoclinic_p_beta(params: LatticeParams) -> LatticeParams:
    """Monoclinic-like: free β, force α=γ=90°."""
    a, b, c, _, beta, _ = params
    return a, b, c, 90.0, beta, 90.0


def _snap_monoclinic_p_gamma(params: LatticeParams) -> LatticeParams:
    """Monoclinic-like: free γ, force α=β=90°."""
    a, b, c, _, _, gamma = params
    return a, b, c, 90.0, 90.0, gamma


def _snap_identity(params: LatticeParams) -> LatticeParams:
    return params


@dataclass(frozen=True)
class _BravaisConstraint:
    key: str
    crystal_system_label: str | None
    snap_fn: Callable[[LatticeParams], LatticeParams]
    length_indices: tuple[int, ...]
    angle_indices: tuple[int, ...]
    fixed_score: float | None = None
    # default = Decision A table; extended = A4 low-symmetry extras
    tier: str = "default"


BRAVAIS_CONSTRAINTS: tuple[_BravaisConstraint, ...] = (
    _BravaisConstraint("cubic_P", "cubic", _snap_cubic_p, (0, 1, 2), (3, 4, 5)),
    _BravaisConstraint("cubic_F", "cubic", _snap_cubic_f, (0, 1, 2), (3, 4, 5)),
    _BravaisConstraint("cubic_I", "cubic", _snap_cubic_i, (0, 1, 2), (3, 4, 5)),
    _BravaisConstraint("tetragonal_P", "tetragonal", _snap_tetragonal_p, (), (3, 4)),
    _BravaisConstraint("orthorhombic_P", "orthorhombic", _snap_orthorhombic_p, (), (3, 4)),
    _BravaisConstraint("hex_trig_P", "hexagonal", _snap_hex_trig_p, (), (4,)),
    _BravaisConstraint("trigonal_R", "trigonal", _snap_trigonal_r, (0, 1), (3, 4)),
    _BravaisConstraint(
        "identity",
        None,
        _snap_identity,
        (),
        (),
        fixed_score=IDENTITY_PENALTY_SCORE,
    ),
    # A4 extended (not in Decision A default table)
    _BravaisConstraint(
        "hex_trig_P_strict",
        "hexagonal",
        _snap_hex_trig_p_strict,
        (0, 1),
        (4, 5),
        tier="extended",
    ),
    _BravaisConstraint(
        "monoclinic_P_alpha",
        "monoclinic",
        _snap_monoclinic_p_alpha,
        (),
        (4, 5),
        tier="extended",
    ),
    _BravaisConstraint(
        "monoclinic_P_beta",
        "monoclinic",
        _snap_monoclinic_p_beta,
        (),
        (3, 5),
        tier="extended",
    ),
    _BravaisConstraint(
        "monoclinic_P_gamma",
        "monoclinic",
        _snap_monoclinic_p_gamma,
        (),
        (3, 4),
        tier="extended",
    ),
)


def iter_bravais_constraints(bravais_set: str = "default") -> tuple[_BravaisConstraint, ...]:
    """Return constraint table for ``default`` (Decision A) or ``extended`` (A4)."""
    key = bravais_set.strip().lower()
    if key == "default":
        return tuple(c for c in BRAVAIS_CONSTRAINTS if c.tier == "default")
    if key == "extended":
        return BRAVAIS_CONSTRAINTS
    raise ValueError(f"Unknown bravais_set {bravais_set!r}; expected default|extended")


def _length_deviation(
    raw: LatticeParams,
    snapped: LatticeParams,
    indices: tuple[int, ...],
) -> float:
    if not indices:
        return 0.0
    diffs = []
    for idx in indices:
        denom = max(abs(raw[idx]), 1e-6)
        diffs.append(abs(snapped[idx] - raw[idx]) / denom)
    return sum(diffs) / len(diffs)


def _angle_deviation(
    raw: LatticeParams,
    snapped: LatticeParams,
    indices: tuple[int, ...],
) -> float:
    if not indices:
        return 0.0
    diffs = [abs(snapped[idx] - raw[idx]) for idx in indices]
    return sum(diffs) / len(diffs)


def compute_hypothesis_score(
    raw: LatticeParams,
    snapped: LatticeParams,
    *,
    length_indices: tuple[int, ...],
    angle_indices: tuple[int, ...],
    fixed_score: float | None = None,
) -> float:
    if fixed_score is not None:
        return fixed_score
    len_dev = _length_deviation(raw, snapped, length_indices)
    ang_dev = _angle_deviation(raw, snapped, angle_indices)
    return len_dev / LENGTH_RTOL_REF + ang_dev / ANGLE_ATOL_REF


def generate_bravais_hypotheses(
    params: torch.Tensor | list[float],
    *,
    identity_penalty_score: float = IDENTITY_PENALTY_SCORE,
    bravais_set: str = "default",
) -> list[BravaisHypothesis]:
    """Generate Bravais snap hypotheses for one primitive lattice."""
    raw = _as_params(params)
    hypotheses: list[BravaisHypothesis] = []
    for constraint in iter_bravais_constraints(bravais_set):
        snapped = constraint.snap_fn(raw)
        fixed = (
            identity_penalty_score
            if constraint.key == "identity"
            else constraint.fixed_score
        )
        score = compute_hypothesis_score(
            raw,
            snapped,
            length_indices=constraint.length_indices,
            angle_indices=constraint.angle_indices,
            fixed_score=fixed,
        )
        hypotheses.append(
            BravaisHypothesis(
                bravais_key=constraint.key,
                crystal_system_label=constraint.crystal_system_label,
                snapped_params=snapped,
                score=score,
            )
        )
    hypotheses.sort(key=lambda item: item.score)
    return hypotheses


def best_hypothesis(
    params: torch.Tensor | list[float],
    *,
    identity_penalty_score: float = IDENTITY_PENALTY_SCORE,
    bravais_set: str = "default",
) -> BravaisHypothesis:
    return generate_bravais_hypotheses(
        params,
        identity_penalty_score=identity_penalty_score,
        bravais_set=bravais_set,
    )[0]
