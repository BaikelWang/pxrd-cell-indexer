"""Deterministic Top-K expansion from a single lattice prediction (arms B/C)."""

from __future__ import annotations

import itertools

import numpy as np
from pymatgen.core import Lattice


def _valid(six) -> bool:
    a, b, c, al, be, ga = [float(x) for x in six]
    if min(a, b, c) < 1.0 or max(a, b, c) > 80.0:
        return False
    if min(al, be, ga) < 20.0 or max(al, be, ga) > 160.0:
        return False
    s = al + be + ga
    if not (180.0 < s < 360.0):
        return False
    try:
        v = Lattice.from_parameters(a, b, c, al, be, ga).volume
    except Exception:
        return False
    return 8.0 < v < 20000.0


def _uniq(cands: list[list[float]], atol: float = 1e-3) -> list[list[float]]:
    out: list[list[float]] = []
    for c in cands:
        arr = np.asarray(c, dtype=float)
        if any(np.allclose(arr, np.asarray(q, dtype=float), atol=atol) for q in out):
            continue
        out.append([float(x) for x in arr.tolist()])
    return out


def expand_lattice(six: list[float], k: int = 100) -> list[list[float]]:
    """Bravais projections + axis perms + scale factors → up to K candidates.

    Order: raw first, then projections, then scaled variants.
    """
    a, b, c, al, be, ga = [float(x) for x in six[:6]]
    pool: list[list[float]] = []

    def push(p):
        if _valid(p):
            pool.append(list(p))

    push([a, b, c, al, be, ga])

    # axis permutations of lengths (keep angles sorted with axes naively)
    for pa, pb, pc in itertools.permutations([a, b, c], 3):
        push([pa, pb, pc, al, be, ga])
        push([pa, pb, pc, 90.0, 90.0, 90.0])

    m = (a + b + c) / 3.0
    push([m, m, m, 90.0, 90.0, 90.0])  # cubic
    push([a, b, c, 90.0, 90.0, 90.0])  # ortho
    for i, edges in enumerate(([a, b, c],)):
        for j in range(3):
            cc = edges[j]
            aa = float(np.mean([edges[t] for t in range(3) if t != j]))
            push([aa, aa, cc, 90.0, 90.0, 90.0])  # tetra
            push([aa, aa, cc, 90.0, 90.0, 120.0])  # hex
    for ang in (al, be, ga):
        push([a, b, c, 90.0, float(ang), 90.0])  # mono

    # length scales
    scales = (0.5, 2.0 / 3.0, 0.8, 1.25, 1.5, 2.0, 3.0)
    base = _uniq(pool)
    for p in list(base):
        for s in scales:
            push([p[0] * s, p[1] * s, p[2] * s, p[3], p[4], p[5]])
            # single-axis
            for ax in range(3):
                q = list(p)
                q[ax] *= s
                push(q)

    out = _uniq(pool)
    # prefer near-raw volume first
    try:
        v0 = Lattice.from_parameters(a, b, c, al, be, ga).volume
    except Exception:
        v0 = 1.0

    def key(p):
        try:
            v = Lattice.from_parameters(*p).volume
            return (abs(np.log(max(v, 1e-6) / max(v0, 1e-6))), -v)
        except Exception:
            return (1e9, 0)

    # keep raw first
    rest = [p for p in out if not np.allclose(p, [a, b, c, al, be, ga], atol=1e-4)]
    rest.sort(key=key)
    ordered = [[a, b, c, al, be, ga]] + rest
    ordered = _uniq(ordered)
    return ordered[:k]
