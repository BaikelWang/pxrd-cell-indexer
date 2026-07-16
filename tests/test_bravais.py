"""Tests for Bravais snap hypotheses."""

from __future__ import annotations

import math

import torch

from pxrd_cell_indexing.model.bravais import (
    CUBIC_I_ANGLE,
    generate_bravais_hypotheses,
)


def test_cubic_p_snap_matches_constraint() -> None:
    raw = [5.0, 5.1, 4.9, 91.0, 89.0, 90.0]
    hyps = generate_bravais_hypotheses(raw)
    cubic_p = next(h for h in hyps if h.bravais_key == "cubic_P")
    a, b, c, alpha, beta, gamma = cubic_p.snapped_params
    assert math.isclose(a, b, rel_tol=1e-6)
    assert math.isclose(b, c, rel_tol=1e-6)
    assert alpha == beta == gamma == 90.0


def test_cubic_f_snap_uses_sixty_degree_angles() -> None:
    raw = [5.0, 5.0, 5.0, 60.0, 60.0, 60.0]
    hyps = generate_bravais_hypotheses(raw)
    cubic_f = next(h for h in hyps if h.bravais_key == "cubic_F")
    assert cubic_f.snapped_params[3:] == (60.0, 60.0, 60.0)
    assert cubic_f.score < 0.1


def test_cubic_i_snap_uses_body_centered_angle() -> None:
    raw = [5.0, 5.0, 5.0, CUBIC_I_ANGLE, CUBIC_I_ANGLE, CUBIC_I_ANGLE]
    hyps = generate_bravais_hypotheses(raw)
    cubic_i = next(h for h in hyps if h.bravais_key == "cubic_I")
    assert math.isclose(cubic_i.snapped_params[3], CUBIC_I_ANGLE, abs_tol=1e-3)


def test_trigonal_r_snap_averages_lengths_and_angles() -> None:
    raw = [4.0, 4.2, 6.0, 76.0, 76.4, 60.0]
    hyps = generate_bravais_hypotheses(raw)
    trig_r = next(h for h in hyps if h.bravais_key == "trigonal_R")
    a, b, _, alpha, beta, _ = trig_r.snapped_params
    assert math.isclose(a, 4.1, abs_tol=1e-6)
    assert math.isclose(b, 4.1, abs_tol=1e-6)
    assert math.isclose(alpha, 76.2, abs_tol=1e-6)
    assert math.isclose(beta, 76.2, abs_tol=1e-6)


def test_identity_has_fixed_penalty_score() -> None:
    raw = [3.0, 4.0, 5.0, 95.0, 88.0, 92.0]
    hyps = generate_bravais_hypotheses(raw, identity_penalty_score=1.0)
    identity = next(h for h in hyps if h.bravais_key == "identity")
    assert identity.snapped_params == tuple(raw)
    assert identity.score == 1.0


def test_perfect_cubic_p_ranks_first() -> None:
    raw = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    hyps = generate_bravais_hypotheses(raw)
    assert hyps[0].bravais_key == "cubic_P"
    assert hyps[0].score < hyps[-1].score


def test_generate_bravais_hypotheses_accepts_tensor() -> None:
    tensor = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    hyps = generate_bravais_hypotheses(tensor[0])
    assert len(hyps) == 8


def test_extended_bravais_set_adds_mono_and_hex_strict() -> None:
    raw = [4.0, 5.0, 6.0, 95.0, 100.0, 88.0]
    default = generate_bravais_hypotheses(raw, bravais_set="default")
    extended = generate_bravais_hypotheses(raw, bravais_set="extended")
    assert len(default) == 8
    assert len(extended) == 12
    keys = {h.bravais_key for h in extended}
    assert "monoclinic_P_beta" in keys
    assert "hex_trig_P_strict" in keys
    mono_b = next(h for h in extended if h.bravais_key == "monoclinic_P_beta")
    assert mono_b.snapped_params == (4.0, 5.0, 6.0, 90.0, 100.0, 90.0)
    hex_s = next(h for h in extended if h.bravais_key == "hex_trig_P_strict")
    assert math.isclose(hex_s.snapped_params[0], hex_s.snapped_params[1], abs_tol=1e-6)
    assert hex_s.snapped_params[4] == 90.0
    assert hex_s.snapped_params[5] == 120.0
