"""Alternative bases of one lattice, via unimodular integer basis changes.

A basis change ``A' = M A`` with ``M`` integer and ``|det M| = 1`` describes the
*same* lattice, so every variant satisfies the strict criterion against the
original (``find_mapping`` succeeds with ``|det| = 1``). Feeding a random variant
as the flow-matching target each step turns the otherwise deterministic
``pattern -> cell`` map into a genuinely multi-modal one, which is what stops a
well-trained flow from collapsing to a single point.

Everything is metric-based: ``G = A Aᵀ`` depends only on (a,b,c,α,β,γ), and
``G' = M G Mᵀ``, so no matrix convention needs to be fixed.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "build_unimodular_bank",
    "metric_from_lattice6",
    "lattice6_from_metric",
    "apply_basis_change",
    "sample_equivalent_lattice6",
]

# Variants whose longest edge grows beyond this factor are rejected: they are
# valid bases but poor McMaille seeds (far from reduced).
DEFAULT_MAX_LEN_RATIO = 1.35


def metric_from_lattice6(lattice6: np.ndarray) -> np.ndarray:
    """(..., 6) lattice params [deg] -> (..., 3, 3) direct metric ``G = A Aᵀ``."""
    arr = np.asarray(lattice6, dtype=np.float64)
    a, b, c = arr[..., 0], arr[..., 1], arr[..., 2]
    cos_al = np.cos(np.deg2rad(arr[..., 3]))
    cos_be = np.cos(np.deg2rad(arr[..., 4]))
    cos_ga = np.cos(np.deg2rad(arr[..., 5]))
    g = np.empty(arr.shape[:-1] + (3, 3), dtype=np.float64)
    g[..., 0, 0] = a * a
    g[..., 1, 1] = b * b
    g[..., 2, 2] = c * c
    g[..., 0, 1] = g[..., 1, 0] = a * b * cos_ga
    g[..., 0, 2] = g[..., 2, 0] = a * c * cos_be
    g[..., 1, 2] = g[..., 2, 1] = b * c * cos_al
    return g


def lattice6_from_metric(g: np.ndarray) -> np.ndarray:
    """(..., 3, 3) direct metric -> (..., 6) lattice params [deg]."""
    g = np.asarray(g, dtype=np.float64)
    a = np.sqrt(np.clip(g[..., 0, 0], 1e-18, None))
    b = np.sqrt(np.clip(g[..., 1, 1], 1e-18, None))
    c = np.sqrt(np.clip(g[..., 2, 2], 1e-18, None))
    cos_al = np.clip(g[..., 1, 2] / (b * c + 1e-18), -1.0, 1.0)
    cos_be = np.clip(g[..., 0, 2] / (a * c + 1e-18), -1.0, 1.0)
    cos_ga = np.clip(g[..., 0, 1] / (a * b + 1e-18), -1.0, 1.0)
    return np.stack(
        [
            a,
            b,
            c,
            np.rad2deg(np.arccos(cos_al)),
            np.rad2deg(np.arccos(cos_be)),
            np.rad2deg(np.arccos(cos_ga)),
        ],
        axis=-1,
    )


def apply_basis_change(lattice6: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Lattice params of basis ``M A`` given params of ``A``."""
    g = metric_from_lattice6(lattice6)
    m = np.asarray(m, dtype=np.float64)
    return lattice6_from_metric(m @ g @ m.T)


def build_unimodular_bank(*, include_shears: bool = True) -> np.ndarray:
    """Integer matrices with ``|det| = 1``; index 0 is the identity."""
    mats: list[np.ndarray] = []
    seen: set[bytes] = set()

    def push(m: np.ndarray) -> None:
        m = np.asarray(m, dtype=np.int64)
        if abs(int(round(np.linalg.det(m)))) != 1:
            return
        key = m.tobytes()
        if key in seen:
            return
        seen.add(key)
        mats.append(m)

    push(np.eye(3, dtype=np.int64))

    # Signed permutations: relabel and/or reverse axes. These never change the
    # edge-length multiset, so they always survive the length filter.
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product((1, -1), repeat=3))
    signed_perms: list[np.ndarray] = []
    for perm in perms:
        base = np.zeros((3, 3), dtype=np.int64)
        for row, col in enumerate(perm):
            base[row, col] = 1
        for sign in signs:
            signed_perms.append(np.diag(np.array(sign, dtype=np.int64)) @ base)
    for m in signed_perms:
        push(m)

    if include_shears:
        # Single elementary shears, and shears composed with a signed permutation.
        shears = []
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for sign in (1, -1):
                    s = np.eye(3, dtype=np.int64)
                    s[i, j] = sign
                    shears.append(s)
        for s in shears:
            push(s)
        for s in shears:
            for m in signed_perms:
                push(m @ s)

    return np.stack(mats, axis=0)


_BANK = build_unimodular_bank()


def sample_equivalent_lattice6(
    lattice6: np.ndarray,
    rng: np.random.Generator,
    *,
    bank: np.ndarray | None = None,
    max_len_ratio: float = DEFAULT_MAX_LEN_RATIO,
    max_tries: int = 4,
) -> np.ndarray:
    """Random alternative basis of the same lattice.

    Falls back to the input when the drawn basis change would stretch the cell
    beyond ``max_len_ratio``.
    """
    bank = _BANK if bank is None else bank
    arr = np.asarray(lattice6, dtype=np.float64).reshape(6)
    ref_max = float(np.max(arr[:3]))
    g = metric_from_lattice6(arr)
    for _ in range(max_tries):
        m = bank[rng.integers(len(bank))].astype(np.float64)
        variant = lattice6_from_metric(m @ g @ m.T)
        if not np.all(np.isfinite(variant)):
            continue
        if float(np.max(variant[:3])) > max_len_ratio * ref_max:
            continue
        if float(np.min(variant[3:])) < 20.0 or float(np.max(variant[3:])) > 160.0:
            continue
        return variant.astype(np.float32)
    return arr.astype(np.float32)
