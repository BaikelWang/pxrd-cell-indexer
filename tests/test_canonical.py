"""Tests for crystallographic canonical lattice helpers (R9)."""

from __future__ import annotations

import numpy as np
import pytest

from pxrd_cell_indexing.data.canonical import (
    canonicalize_lattice,
    niggli_is_idempotent,
    params_close,
)


def test_primitive_identity() -> None:
    matrix = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    out = canonicalize_lattice(matrix, convention="primitive")
    assert out.convention == "primitive"
    assert params_close(out.as_params6(), [5.0, 5.0, 5.0, 90.0, 90.0, 90.0])


def test_niggli_idempotent_on_cubic() -> None:
    matrix = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    assert niggli_is_idempotent(matrix)


def test_niggli_idempotent_on_monoclinic_like() -> None:
    # Slightly skewed cell that Niggli should stabilize.
    matrix = [
        [6.0, 0.0, 0.0],
        [0.5, 5.0, 0.0],
        [0.2, 0.1, 7.0],
    ]
    assert niggli_is_idempotent(matrix, length_tol=1e-3, angle_tol_deg=1e-2)
    once = canonicalize_lattice(matrix, convention="niggli")
    twice = canonicalize_lattice(once.matrix, convention="niggli")
    assert params_close(once.as_params6(), twice.as_params6(), length_tol=1e-3, angle_tol_deg=1e-2)


def test_reduced_with_structure_uses_reduced_convention() -> None:
    matrix = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    out = canonicalize_lattice(
        matrix,
        convention="reduced",
        species=["Si", "Si"],
        frac_coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    assert out.convention == "reduced"
    assert out.a > 0 and out.alpha > 0


@pytest.mark.parametrize("convention", ["primitive", "niggli"])
def test_canonicalize_finite(convention: str) -> None:
    matrix = np.eye(3) * 3.2
    out = canonicalize_lattice(matrix, convention=convention)  # type: ignore[arg-type]
    assert np.isfinite(out.as_array()).all()
