"""Tests for de Wolff FOM candidate reranking."""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core.lattice import Lattice

from pxrd_cell_indexing.model.fom import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    FomRerankConfig,
    collapse_scale_duplicates,
    de_wolff_fom,
    rerank_candidates_by_fom,
    slice_observed_intensity,
    slice_observed_two_theta,
    theoretical_two_theta,
    two_theta_to_q,
)
from pxrd_cell_indexing.types import LatticeCandidate


def _cubic_params(a: float = 5.0) -> list[float]:
    return [a, a, a, 90.0, 90.0, 90.0]


def test_theoretical_two_theta_matches_pymatgen_cubic() -> None:
    params = _cubic_params(5.0)
    theory = theoretical_two_theta(params, two_theta_max=60.0, max_hkl_cap=5)

    lat = Lattice.from_parameters(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    expected: list[float] = []
    for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0)]:
        d = lat.d_hkl(hkl)
        two_theta = np.rad2deg(2.0 * np.arcsin(DEFAULT_WAVELENGTH_ANGSTROM / (2.0 * d)))
        if two_theta <= 60.0:
            expected.append(two_theta)

    expected_arr = np.sort(np.unique(np.round(expected, decimals=4)))
    theory_rounded = np.round(theory[: len(expected_arr)], decimals=4)
    assert theory_rounded.size >= 3
    assert np.allclose(theory_rounded[:3], expected_arr[:3], atol=0.05)


def test_de_wolff_fom_self_consistency_beats_wrong_cell() -> None:
    params = _cubic_params(5.0)
    observed = theoretical_two_theta(params, two_theta_max=80.0, max_hkl_cap=8)[:15]
    assert observed.size >= 5

    good_fom = de_wolff_fom(observed, params, n_lines=10)
    bad_fom = de_wolff_fom(observed, [6.5, 6.5, 6.5, 90.0, 90.0, 90.0], n_lines=10)

    assert good_fom > bad_fom
    assert good_fom > 1.0


def test_rerank_candidates_by_fom_promotes_true_cell() -> None:
    true_params = _cubic_params(5.0)
    observed = theoretical_two_theta(true_params, two_theta_max=80.0, max_hkl_cap=8)[:12]

    true_candidate = LatticeCandidate(
        crystal_system="cubic",
        a=5.0,
        b=5.0,
        c=5.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.1,
        bravais_key="cubic_P",
    )
    decoy_candidates = [
        LatticeCandidate(
            crystal_system="cubic",
            a=6.5,
            b=6.5,
            c=6.5,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            confidence=0.9,
            bravais_key="cubic_P",
        ),
        LatticeCandidate(
            crystal_system="orthorhombic",
            a=4.0,
            b=7.0,
            c=8.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            confidence=0.8,
            bravais_key="orthorhombic_P",
        ),
    ]
    pool = decoy_candidates + [true_candidate]
    reranked = rerank_candidates_by_fom(pool, observed, n_lines=10)

    assert reranked[0].a == 5.0
    assert reranked[0].fom_score is not None
    assert reranked[0].fom_score > (reranked[1].fom_score or 0.0)


def test_slice_observed_two_theta_from_batch() -> None:
    pxrd_x = torch.tensor([[10.0], [20.0], [30.0], [15.0], [25.0]], dtype=torch.float32)
    peak_num = torch.tensor([2, 3], dtype=torch.long)

    sample0 = slice_observed_two_theta(pxrd_x, peak_num, 0)
    sample1 = slice_observed_two_theta(pxrd_x, peak_num, 1)

    assert np.allclose(sample0, [10.0, 20.0])
    assert np.allclose(sample1, [30.0, 15.0, 25.0])


def test_slice_observed_intensity_from_batch() -> None:
    pxrd_y = torch.tensor([[10.0], [20.0], [30.0], [15.0], [25.0]], dtype=torch.float32)
    peak_num = torch.tensor([2, 3], dtype=torch.long)
    assert np.allclose(slice_observed_intensity(pxrd_y, peak_num, 0), [10.0, 20.0])


def test_collapse_scale_duplicates_keeps_smallest_volume() -> None:
    params = _cubic_params(5.0)
    observed = theoretical_two_theta(params, two_theta_max=80.0, max_hkl_cap=8)[:8]
    base = LatticeCandidate(
        crystal_system="cubic",
        a=5.0,
        b=5.0,
        c=5.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.5,
        bravais_key="cubic_P",
    )
    scaled = LatticeCandidate(
        crystal_system="cubic",
        a=10.0,
        b=10.0,
        c=10.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.9,
        bravais_key="cubic_P:scale=2",
    )
    collapsed = collapse_scale_duplicates([scaled, base], observed)
    assert len(collapsed) == 1
    assert collapsed[0].a == 5.0


def test_intensity_weighted_rerank_promotes_true_cell() -> None:
    true_params = _cubic_params(5.0)
    observed = theoretical_two_theta(true_params, two_theta_max=80.0, max_hkl_cap=8)[:12]
    intensity = np.linspace(100.0, 10.0, observed.size)
    true_candidate = LatticeCandidate(
        crystal_system="cubic",
        a=5.0,
        b=5.0,
        c=5.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.1,
        bravais_key="cubic_P",
    )
    decoy = LatticeCandidate(
        crystal_system="cubic",
        a=6.5,
        b=6.5,
        c=6.5,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.9,
        bravais_key="cubic_P",
    )
    reranked = rerank_candidates_by_fom(
        [decoy, true_candidate],
        observed,
        observed_intensity=intensity,
        config=FomRerankConfig(mode="intensity_weighted"),
    )
    assert reranked[0].a == 5.0


def test_two_theta_q_roundtrip() -> None:
    q = two_theta_to_q(np.array([20.0, 40.0]), wavelength_angstrom=DEFAULT_WAVELENGTH_ANGSTROM)
    assert q[0] < q[1]
    assert np.all(q > 0)
