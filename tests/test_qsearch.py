"""B1: independent q-search unit tests (v3 §11 / v4 §6.1)."""

from __future__ import annotations

import numpy as np

from pxrd_cell_indexing.eval import lattice_match_elementwise
from pxrd_cell_indexing.model.fom import theoretical_two_theta
from pxrd_cell_indexing.search.qsearch import (
    _basis,
    _coeff_row,
    _hkl_pool,
    _tiered_hkl_pool,
    gstar_to_lattice_params,
    search_crystal_system,
)


def test_hkl_pool_dedups_sign_flip_degeneracy() -> None:
    pool = _hkl_pool(2)
    assert (1, 0, 0) in pool
    assert (-1, 0, 0) not in pool
    assert (0, 0, 0) not in pool
    # No duplicate coefficient rows.
    rows = {tuple(_coeff_row(*hkl)) for hkl in pool}
    assert len(rows) == len(pool)


def test_tiered_hkl_pool_respects_nonzero_cap() -> None:
    pool = _tiered_hkl_pool(dense_max_index=2, dense_max_nonzero=1)
    assert all((h != 0) + (k != 0) + (l != 0) <= 1 for h, k, l in pool)


def test_gstar_to_lattice_params_rejects_non_spd() -> None:
    non_spd = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert gstar_to_lattice_params(non_spd) is None


def test_gstar_to_lattice_params_roundtrips_cubic() -> None:
    a = 7.5
    gstar = np.eye(3) / (a * a)
    params = gstar_to_lattice_params(gstar)
    assert params is not None
    a_out, b_out, c_out, alpha, beta, gamma = params
    assert abs(a_out - a) < 1e-6 and abs(b_out - a) < 1e-6 and abs(c_out - a) < 1e-6
    assert abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6


def test_basis_cubic_and_orthorhombic_shapes() -> None:
    assert _basis("cubic").shape == (6, 1)
    assert _basis("orthorhombic").shape == (6, 3)
    assert _basis("triclinic").shape == (6, 6)


def test_search_cubic_recovers_true_cell_top1() -> None:
    truth = (7.5, 7.5, 7.5, 90.0, 90.0, 90.0)
    two_theta = theoretical_two_theta(truth, wavelength_angstrom=1.54184, two_theta_max=90.0)
    observed = two_theta[two_theta >= 5.0][:15]

    candidates = search_crystal_system(
        observed,
        "cubic",
        max_hkl_index=6,
        n_low_peaks=4,
        pool_budget=30,
        time_budget_s=10.0,
    )
    assert candidates, "expected at least one candidate"
    top1 = candidates[0]
    assert lattice_match_elementwise(
        [top1.a, top1.b, top1.c, top1.alpha, top1.beta, top1.gamma],
        list(truth),
        ltol=0.01,
        atol_deg=1.0,
    )


def test_search_orthorhombic_finds_true_cell_in_pool() -> None:
    import torch

    from pxrd_cell_indexing.data.canonical import canonicalize_lattice
    from pxrd_cell_indexing.geometry import lattice_params_to_matrix

    truth = (6.0, 7.5, 9.0, 90.0, 90.0, 90.0)
    two_theta = theoretical_two_theta(truth, wavelength_angstrom=1.54184, two_theta_max=90.0)
    observed = two_theta[two_theta >= 5.0][:15]

    candidates = search_crystal_system(
        observed,
        "orthorhombic",
        max_hkl_index=3,
        sparse_hkl_index=6,
        n_low_peaks=6,
        pool_budget=30,
        time_budget_s=10.0,
    )
    # Sequential axial solve may assign axes in any order; compare after Niggli
    # (same protocol as B1-S0 / B1-S1 eval scripts).
    truth_m = lattice_params_to_matrix(torch.tensor(truth, dtype=torch.float64)).numpy()
    truth_n = canonicalize_lattice(truth_m, convention="niggli").as_params6()
    hits = [
        c
        for c in candidates
        if lattice_match_elementwise(c.niggli_params6(), truth_n, ltol=0.01, atol_deg=1.0)
    ]
    assert hits, "expected true orthorhombic cell to appear in the candidate pool"


def test_basis_trigonal_variants() -> None:
    assert _basis("trigonal_hex").shape == (6, 2)
    assert _basis("trigonal_rhomb").shape == (6, 2)
    # Rhomb: equal diag and equal off-diag columns.
    br = _basis("trigonal_rhomb")
    assert np.allclose(br[:3, 0], 1.0) and np.allclose(br[3:, 0], 0.0)
    assert np.allclose(br[:3, 1], 0.0) and np.allclose(br[3:, 1], 1.0)


def test_search_trigonal_rhomb_recovers_true_cell() -> None:
    truth = (6.5, 6.5, 6.5, 75.0, 75.0, 75.0)
    two_theta = theoretical_two_theta(truth, wavelength_angstrom=1.54184, two_theta_max=90.0)
    observed = two_theta[two_theta >= 5.0][:15]
    candidates = search_crystal_system(
        observed, "trigonal", max_hkl_index=4, n_low_peaks=6, pool_budget=30, time_budget_s=20.0
    )
    hits = [
        c
        for c in candidates
        if lattice_match_elementwise([c.a, c.b, c.c, c.alpha, c.beta, c.gamma], list(truth), ltol=0.02, atol_deg=1.0)
    ]
    assert hits, "expected rhombohedral trigonal cell in the candidate pool"
