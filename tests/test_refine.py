"""Tests for inference-time Q-match lattice refine."""

from __future__ import annotations

import numpy as np

from pxrd_cell_indexing.model.fom import theoretical_two_theta
from pxrd_cell_indexing.model.refine import (
    RefineConfig,
    SearchConfig,
    build_manifold_search_candidates,
    refine_candidate_by_q_match,
    refine_candidate_on_manifold,
    resolve_manifold,
    soft_q_match_loss,
)
from pxrd_cell_indexing.types import LatticeCandidate


def test_soft_q_match_loss_zero_on_exact_lattice() -> None:
    truth = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    observed = theoretical_two_theta(truth)[:15]
    loss = soft_q_match_loss(truth, observed, n_lines=10)
    assert loss < 1e-5


def test_refine_moves_perturbed_cubic_toward_truth() -> None:
    truth = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
    observed = theoretical_two_theta(truth)[:20]
    seed = LatticeCandidate(
        crystal_system="cubic",
        a=5.3,
        b=5.3,
        c=5.3,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=1.0,
        bravais_key="cubic_P",
    )
    before = soft_q_match_loss(
        [seed.a, seed.b, seed.c, seed.alpha, seed.beta, seed.gamma],
        observed,
        n_lines=10,
    )
    refined = refine_candidate_by_q_match(
        seed,
        observed,
        config=RefineConfig(max_steps=60, top_n=1, length_rel_bound=0.2, angle_abs_bound_deg=5.0),
    )
    after = soft_q_match_loss(
        [refined.a, refined.b, refined.c, refined.alpha, refined.beta, refined.gamma],
        observed,
        n_lines=10,
    )
    assert after <= before + 1e-9
    # Should move lengths closer to 5.0
    assert abs(refined.a - 5.0) < abs(seed.a - 5.0)


def test_refine_steps_zero_is_noop() -> None:
    seed = LatticeCandidate(
        crystal_system="cubic",
        a=5.3,
        b=5.3,
        c=5.3,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=1.0,
        bravais_key="cubic_P",
    )
    observed = theoretical_two_theta([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])[:10]
    out = refine_candidate_by_q_match(
        seed, observed, config=RefineConfig(max_steps=0)
    )
    assert out.a == seed.a


def test_manifold_refine_keeps_tetragonal_constraints() -> None:
    truth = [4.0, 4.0, 6.0, 90.0, 90.0, 90.0]
    observed = theoretical_two_theta(truth)[:20]
    seed = LatticeCandidate(
        crystal_system="tetragonal",
        a=4.2,
        b=4.1,
        c=6.3,
        alpha=88.0,
        beta=91.0,
        gamma=89.0,
        confidence=0.5,
        bravais_key="tetragonal_P",
    )
    refined = refine_candidate_on_manifold(
        seed,
        observed,
        config=RefineConfig(max_steps=40, length_rel_bound=0.2),
        manifold=resolve_manifold("tetragonal_P"),
    )
    assert abs(refined.a - refined.b) < 1e-6
    assert abs(refined.alpha - 90.0) < 1e-6
    assert abs(refined.beta - 90.0) < 1e-6
    assert abs(refined.gamma - 90.0) < 1e-6
    assert abs(refined.a - 4.0) < abs(seed.a - 4.0)


def test_build_manifold_search_recovers_perturbed_cubic() -> None:
    truth = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
    observed = theoretical_two_theta(truth)[:25]
    # NN-like noisy cubic seed
    pred = np.array([5.25, 5.18, 5.30, 89.0, 91.0, 90.5])
    pools = build_manifold_search_candidates(
        pred,
        [observed],
        config=SearchConfig(
            max_seeds=4,
            k=10,
            bravais_set="default",
            refine=RefineConfig(max_steps=40, max_hkl_cap=12, n_lines=10),
            max_log_volume_ratio_vs_base=float(np.log(2.0)),
        ),
    )
    assert len(pools) == 1
    assert len(pools[0]) >= 1
    hit = any(
        abs(c.a - 5.0) < 0.15 and abs(c.alpha - 90.0) < 1.0
        for c in pools[0]
    )
    assert hit
