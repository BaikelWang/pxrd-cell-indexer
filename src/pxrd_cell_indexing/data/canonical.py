"""Canonical lattice conventions for R9 (Niggli / fixed reduced).

Terminology note: this is crystallographic canonicalization of the unit cell,
not the matrix6 "canonical free components" in ``normalization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

CanonicalConvention = Literal["primitive", "reduced", "niggli"]


@dataclass(frozen=True)
class CanonicalLattice:
    """Six lattice parameters under a named crystallographic convention."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    convention: CanonicalConvention
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    def as_params6(self) -> list[float]:
        return [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.as_params6(), dtype=np.float64)


def _lattice_from_matrix(matrix: Sequence[Sequence[float]] | np.ndarray) -> Lattice:
    arr = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return Lattice(arr)


def _from_lattice(lattice: Lattice, convention: CanonicalConvention) -> CanonicalLattice:
    matrix = tuple(tuple(float(x) for x in row) for row in np.asarray(lattice.matrix))
    abc = lattice.abc
    angles = lattice.angles
    return CanonicalLattice(
        a=float(abc[0]),
        b=float(abc[1]),
        c=float(abc[2]),
        alpha=float(angles[0]),
        beta=float(angles[1]),
        gamma=float(angles[2]),
        convention=convention,
        matrix=matrix,  # type: ignore[arg-type]
    )


def canonicalize_lattice(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    convention: CanonicalConvention = "niggli",
    species: Sequence[str] | None = None,
    frac_coords: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> CanonicalLattice:
    """Convert a 3x3 lattice matrix to the requested crystallographic convention.

    - ``primitive``: leave matrix as-is (identity path for current jsonl labels).
    - ``reduced``: ``Structure.get_reduced_structure()`` when atoms are provided;
      otherwise fall back to ``Lattice.get_niggli_reduced_lattice()`` with a warning
      via the return convention still marked ``reduced`` only when structure-based.
    - ``niggli``: ``Lattice.get_niggli_reduced_lattice()``.
    """
    lattice = _lattice_from_matrix(matrix)
    if convention == "primitive":
        return _from_lattice(lattice, "primitive")
    if convention == "niggli":
        return _from_lattice(lattice.get_niggli_reduced_lattice(), "niggli")
    if convention == "reduced":
        if species is None or frac_coords is None:
            # Structure-free fallback: Niggli is the closest unique reduced cell.
            return _from_lattice(lattice.get_niggli_reduced_lattice(), "niggli")
        structure = Structure(
            lattice,
            list(species),
            np.asarray(frac_coords, dtype=np.float64),
            coords_are_cartesian=False,
        )
        reduced = structure.get_reduced_structure()
        return _from_lattice(reduced.lattice, "reduced")
    raise ValueError(f"unknown convention: {convention!r}")


def canonicalize_from_lmdb_entry(
    entry: dict[str, Any],
    *,
    convention: CanonicalConvention = "niggli",
) -> CanonicalLattice:
    """Canonicalize using LMDB fields ``p_lattice_matrix`` / atoms."""
    return canonicalize_lattice(
        entry["p_lattice_matrix"],
        convention=convention,
        species=entry.get("p_atom_type"),
        frac_coords=entry.get("p_atom_pos"),
    )


def params_close(
    a: Sequence[float],
    b: Sequence[float],
    *,
    length_tol: float = 1e-4,
    angle_tol_deg: float = 1e-3,
) -> bool:
    aa = np.asarray(a, dtype=np.float64).reshape(6)
    bb = np.asarray(b, dtype=np.float64).reshape(6)
    return bool(
        np.all(np.abs(aa[:3] - bb[:3]) <= length_tol)
        and np.all(np.abs(aa[3:] - bb[3:]) <= angle_tol_deg)
    )


def niggli_is_idempotent(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    length_tol: float = 1e-4,
    angle_tol_deg: float = 1e-3,
) -> bool:
    """True if applying Niggli twice yields the same six parameters."""
    once = canonicalize_lattice(matrix, convention="niggli")
    twice = canonicalize_lattice(once.matrix, convention="niggli")
    return params_close(once.as_params6(), twice.as_params6(), length_tol=length_tol, angle_tol_deg=angle_tol_deg)
