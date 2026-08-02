#!/usr/bin/env python3
"""Classical cell refinement: hkl assignment + linear LSQ on the reciprocal metric tensor.

Q = 1/d^2 = h^2*G11 + k^2*G22 + l^2*G33 + hk*(2G12) + hl*(2G13) + kl*(2G23)

is LINEAR in the six reciprocal metric tensor components, so each refinement
cycle is a plain least-squares solve. This is the standard powder-indexing
refinement (what CELREF / DICVOL / McMaille do) and is far more robust than
non-linear optimisation over (a,b,c,alpha,beta,gamma).
"""

from __future__ import annotations

import math

import numpy as np

WAVELENGTH = 1.54184  # Cu Ka, matches pymatgen XRDCalculator / MP100 simulation


# --------------------------------------------------------------- conversions
def params_to_gstar(params) -> np.ndarray | None:
    """(a,b,c,al,be,ga) -> reciprocal metric tensor G*."""
    a, b, c, al, be, ga = [float(x) for x in params[:6]]
    if min(a, b, c) <= 0:
        return None
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    # direct metric tensor
    G = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ]
    )
    try:
        if np.linalg.det(G) <= 1e-12:
            return None
        return np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None


def gstar_to_params(gstar: np.ndarray) -> list[float] | None:
    """Reciprocal metric tensor G* -> (a,b,c,al,be,ga)."""
    try:
        if not np.all(np.isfinite(gstar)):
            return None
        # must be positive definite
        if np.min(np.linalg.eigvalsh((gstar + gstar.T) / 2.0)) <= 1e-12:
            return None
        G = np.linalg.inv((gstar + gstar.T) / 2.0)
    except np.linalg.LinAlgError:
        return None
    if G[0, 0] <= 0 or G[1, 1] <= 0 or G[2, 2] <= 0:
        return None
    a, b, c = (math.sqrt(G[i, i]) for i in range(3))
    def ang(x, y, z):
        v = G[y, z] / (x)
        return math.degrees(math.acos(max(-1.0, min(1.0, v))))
    al = ang(b * c, 1, 2)
    be = ang(a * c, 0, 2)
    ga = ang(a * b, 0, 1)
    out = [a, b, c, al, be, ga]
    if not all(np.isfinite(out)):
        return None
    if min(a, b, c) < 0.5 or max(a, b, c) > 400:
        return None
    if min(al, be, ga) < 5 or max(al, be, ga) > 175:
        return None
    return out


def two_theta_to_q(tt_deg, wavelength: float = WAVELENGTH) -> np.ndarray:
    """2theta (deg) -> Q = 1/d^2."""
    tt = np.asarray(tt_deg, dtype=float)
    return (2.0 * np.sin(np.radians(tt / 2.0)) / wavelength) ** 2


def q_to_two_theta(q, wavelength: float = WAVELENGTH) -> np.ndarray:
    """Q = 1/d^2 -> 2theta (deg); NaN where unphysical."""
    q = np.asarray(q, dtype=float)
    s = wavelength * np.sqrt(np.clip(q, 0, None)) / 2.0
    out = np.full(s.shape, np.nan)
    ok = s <= 1.0
    out[ok] = 2.0 * np.degrees(np.arcsin(s[ok]))
    return out


# ------------------------------------------------------------------ hkl grid
def hkl_grid(gstar: np.ndarray, q_max: float, cap: int = 60) -> np.ndarray:
    """Reflections with Q(hkl) <= q_max (one of each +/- pair, excluding 000).

    The per-axis bound sqrt(q_max * G_ii) is the exact bounding box of the
    ellipsoid; ``cap`` only guards against pathological cells. It must stay
    generous — truncating the grid undercounts N_poss and inflates M_N, which
    lets degenerate long-axis cells win the ranking.
    """
    G = np.linalg.inv(gstar)
    lim = [
        int(min(cap, max(1, math.floor(math.sqrt(max(q_max * G[i, i], 0.0))) + 1)))
        for i in range(3)
    ]
    h = np.arange(-lim[0], lim[0] + 1)
    k = np.arange(-lim[1], lim[1] + 1)
    ll = np.arange(0, lim[2] + 1)
    H, K, L = np.meshgrid(h, k, ll, indexing="ij")
    v = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1).astype(float)
    # drop 000 and keep a half-space to avoid Friedel duplicates
    keep = ~np.all(v == 0, axis=1)
    v = v[keep]
    q = np.einsum("ni,ij,nj->n", v, gstar, v)
    return v[(q > 1e-9) & (q <= q_max)]


# --------------------------------------------------------- symmetry subspaces
# The fitted vector is p = [G11, G22, G33, 2*G12, 2*G13, 2*G23] of the
# reciprocal metric tensor. Every crystal system is a LINEAR subspace of p, so
# a constrained refinement is still an ordinary least-squares solve: fit
# (A @ B) z = y and set p = B @ z.
def _basis(cols) -> np.ndarray:
    B = np.zeros((6, len(cols)))
    for j, col in enumerate(cols):
        for i, w in col:
            B[i, j] = w
    return B


SYMMETRY_BASES: dict[str, np.ndarray] = {
    # most constrained first; ties are broken in this order
    "cubic": _basis([[(0, 1.0), (1, 1.0), (2, 1.0)]]),
    "hexagonal": _basis([[(0, 1.0), (1, 1.0), (3, 1.0)], [(2, 1.0)]]),
    "rhombohedral": _basis(
        [[(0, 1.0), (1, 1.0), (2, 1.0)], [(3, 1.0), (4, 1.0), (5, 1.0)]]
    ),
    "tetragonal": _basis([[(0, 1.0), (1, 1.0)], [(2, 1.0)]]),
    "orthorhombic": _basis([[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]]),
    "monoclinic_a": _basis([[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(5, 1.0)]]),
    "monoclinic_b": _basis([[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(4, 1.0)]]),
    "monoclinic_c": _basis([[(0, 1.0)], [(1, 1.0)], [(2, 1.0)], [(3, 1.0)]]),
    "triclinic": np.eye(6),
}

SYMMETRY_ORDER = list(SYMMETRY_BASES)


def gstar_to_vec(gstar: np.ndarray) -> np.ndarray:
    return np.array(
        [
            gstar[0, 0],
            gstar[1, 1],
            gstar[2, 2],
            2 * gstar[0, 1],
            2 * gstar[0, 2],
            2 * gstar[1, 2],
        ]
    )


def vec_to_gstar(p: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [p[0], p[3] / 2.0, p[4] / 2.0],
            [p[3] / 2.0, p[1], p[5] / 2.0],
            [p[4] / 2.0, p[5] / 2.0, p[2]],
        ]
    )


# ---------------------------------------------------------------- refinement
def refine_cell(
    params,
    obs_two_theta,
    *,
    wavelength: float = WAVELENGTH,
    n_cycles: int = 8,
    q_rtol: float = 0.04,
    min_lines: int | None = None,
    symmetry: str = "triclinic",
) -> dict:
    """Refine a cell against observed 2theta by hkl assignment + linear LSQ.

    Assignment uses a *relative* window in Q (``q_rtol``) that shrinks each
    cycle: a sloppy start still picks up lines, then the fit tightens onto them.
    ``symmetry`` restricts the fit to that crystal system's linear subspace,
    which is what stops a sparse pattern from being over-fitted by a distorted
    triclinic cell.
    """
    # a constrained system has fewer free parameters, so it needs fewer lines;
    # insisting on six would skip every sparse pattern outright
    B = SYMMETRY_BASES[symmetry]
    min_lines = max(B.shape[1], 3) if min_lines is None else min_lines
    obs = np.sort(np.asarray(obs_two_theta, dtype=float))
    obs = obs[np.isfinite(obs)]
    if obs.size < min_lines:
        return {"params": list(params[:6]), "ok": False, "reason": "too_few_peaks"}
    q_obs = two_theta_to_q(obs, wavelength)

    cur = list(params[:6])
    gstar = params_to_gstar(cur)
    if gstar is None:
        return {"params": cur, "ok": False, "reason": "bad_start"}

    best = {"params": cur, "n_used": 0, "rms": np.inf}
    for cyc in range(n_cycles):
        rtol = max(q_rtol * (0.6 ** cyc), 0.002)
        # keep the grid generous: the cell may still be off by several percent
        q_max = float(q_obs.max()) * (1.0 + 4.0 * rtol)
        v = hkl_grid(gstar, q_max)
        if v.shape[0] == 0:
            break
        q_calc = np.einsum("ni,ij,nj->n", v, gstar, v)

        rows, targets, devs = [], [], []
        for q_o in q_obs:
            j = int(np.argmin(np.abs(q_calc - q_o)))
            if abs(q_calc[j] - q_o) > rtol * q_o:
                continue
            h, k, l = v[j]
            rows.append([h * h, k * k, l * l, h * k, h * l, k * l])
            targets.append(q_o)
            devs.append(abs(q_calc[j] - q_o))
        if len(rows) < min_lines:
            break

        A = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        try:
            z, *_ = np.linalg.lstsq(A @ B, y, rcond=None)
        except np.linalg.LinAlgError:
            break
        sol = B @ z
        g = vec_to_gstar(sol)
        p_new = gstar_to_params(g)
        if p_new is None:
            break
        cur, gstar = p_new, g
        rms = float(np.sqrt(np.mean(np.square(A @ sol - y))))
        # prefer fits that explain more lines; break ties on residual
        if (len(rows), -rms) > (best["n_used"], -best["rms"]):
            best = {"params": cur, "n_used": len(rows), "rms": rms}

    return {
        "params": best["params"],
        "ok": best["n_used"] >= min_lines,
        "n_lines_used": best["n_used"],
        "rms_q": None if not np.isfinite(best["rms"]) else best["rms"],
    }


# ------------------------------------------------- symmetry-aware refinement
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def permute_params(params, perm) -> list[float] | None:
    """Relabel the axes of a cell (a pure change of basis, same lattice)."""
    g = params_to_gstar(params)
    if g is None:
        return None
    P = np.eye(3)[list(perm)]
    return gstar_to_params(P @ g @ P.T)


def candidate_symmetries(params, *, rtol: float = 0.03, atol_deg: float = 3.0):
    """(system, permuted_params) pairs whose constraints the cell nearly meets.

    Detection is deliberately loose: the input cell is an approximation, so the
    point is to nominate constraints worth *testing*, not to decide the answer.
    """
    out = []
    for perm in PERMS:
        p = permute_params(params, perm)
        if p is None:
            continue
        a, b, c, al, be, ga = p

        def eq(x, y):
            return abs(x - y) <= rtol * max(abs(x), abs(y))

        def ang(x, y):
            return abs(x - y) <= atol_deg

        ab, bc, abc = eq(a, b), eq(b, c), eq(a, b) and eq(b, c)
        ortho = ang(al, 90) and ang(be, 90) and ang(ga, 90)
        if abc and ortho:
            out.append(("cubic", p))
        if abc and ang(al, be) and ang(be, ga):
            out.append(("rhombohedral", p))
        if ab and ang(al, 90) and ang(be, 90) and ang(ga, 120):
            out.append(("hexagonal", p))
        if ab and ortho:
            out.append(("tetragonal", p))
        if ortho:
            out.append(("orthorhombic", p))
        if ang(be, 90) and ang(ga, 90):
            out.append(("monoclinic_a", p))
        if ang(al, 90) and ang(ga, 90):
            out.append(("monoclinic_b", p))
        if ang(al, 90) and ang(be, 90):
            out.append(("monoclinic_c", p))
    # keep the most constrained variant of each system, deduped
    seen, uniq = set(), []
    for sysname, p in sorted(out, key=lambda x: SYMMETRY_ORDER.index(x[0])):
        k = (sysname, tuple(round(v, 3) for v in p))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((sysname, p))
    return uniq


def refine_symmetry_aware(
    params,
    obs_two_theta,
    *,
    wavelength: float = WAVELENGTH,
    max_trials: int = 8,
    **kw,
) -> dict:
    """Refine freely, then try to snap onto the highest symmetry that still fits.

    A constrained fit is accepted when it indexes at least as many lines as the
    free fit; among those, the most symmetric wins. This is what keeps a sparse
    pattern from being over-fitted by a distorted triclinic cell.
    """
    free = refine_cell(params, obs_two_theta, wavelength=wavelength,
                       symmetry="triclinic", **kw)
    best_p = free["params"]
    best_st = fom_stats(best_p, obs_two_theta, wavelength=wavelength)
    best_sys = "triclinic"

    trials = candidate_symmetries(best_p)[:max_trials]
    for sysname, p0 in trials:
        if SYMMETRY_ORDER.index(sysname) >= SYMMETRY_ORDER.index(best_sys):
            continue
        r = refine_cell(p0, obs_two_theta, wavelength=wavelength,
                        symmetry=sysname, **kw)
        if not r["ok"]:
            continue
        st = fom_stats(r["params"], obs_two_theta, wavelength=wavelength)
        if st["n_indexed"] >= best_st["n_indexed"]:
            best_p, best_st, best_sys = r["params"], st, sysname
    return {"params": best_p, "stats": best_st, "system": best_sys,
            "free_params": free["params"]}


def symmetrize_and_refine(
    params,
    obs_two_theta,
    *,
    wavelength: float = WAVELENGTH,
    symprecs=(0.05, 0.1, 0.25),
    **kw,
) -> dict:
    """Snap a near-symmetric cell onto an exact one, then refine under constraint.

    A cell that is half a degree off an ideal rhombohedron splits every
    coincident reflection, so N_poss explodes and M_N collapses even though the
    cell is essentially right. spglib recovers the exact symmetry (right shape,
    approximate scale) and the constrained fit then puts the scale back.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    best_p = list(params[:6])
    best_st = fom_stats(best_p, obs_two_theta, wavelength=wavelength)
    best_sys = "as_is"

    def consider(p, tag):
        nonlocal best_p, best_st, best_sys
        st = fom_stats(p, obs_two_theta, wavelength=wavelength)
        if (st["coverage"], st["m_n"]) > (best_st["coverage"], best_st["m_n"]):
            best_p, best_st, best_sys = p, st, tag

    for symprec in symprecs:
        try:
            s = Structure(Lattice.from_parameters(*params[:6]), ["H"], [[0, 0, 0]])
            lat = SpacegroupAnalyzer(s, symprec=symprec)
            snapped_lat = lat.get_primitive_standard_structure().lattice
        except Exception:
            continue
        snapped = [
            snapped_lat.a, snapped_lat.b, snapped_lat.c,
            snapped_lat.alpha, snapped_lat.beta, snapped_lat.gamma,
        ]
        for sysname, p0 in candidate_symmetries(snapped):
            r = refine_cell(p0, obs_two_theta, wavelength=wavelength,
                            symmetry=sysname, **kw)
            if r["ok"]:
                consider(r["params"], sysname)

    return {"params": best_p, "stats": best_st, "system": best_sys}


def coverage(params, obs_two_theta, *, wavelength: float = WAVELENGTH,
             tol_deg: float = 0.05, n_lines: int = 20) -> float:
    """Fraction of the first ``n_lines`` observed peaks explained by the cell."""
    st = fom_stats(params, obs_two_theta, wavelength=wavelength,
                   tol_deg=tol_deg, n_lines=n_lines)
    return st["coverage"]


def fom_stats(params, obs_two_theta, *, wavelength: float = WAVELENGTH,
              tol_deg: float = 0.05, n_lines: int = 20) -> dict:
    """Coverage plus de Wolff M_N.

    ``M_N = Q_N / (2 * mean|dQ| * N_poss)``. The ``N_poss`` term is what stops
    a supercell from winning: a bigger cell explains every peak but generates
    far more possible reflections, so its M_N collapses.
    """
    empty = {"coverage": 0.0, "m_n": 0.0, "n_poss": 0, "n_indexed": 0,
             "mean_dq": float("inf")}
    gstar = params_to_gstar(params)
    if gstar is None:
        return empty
    obs = np.sort(np.asarray(obs_two_theta, dtype=float))
    obs = obs[np.isfinite(obs)][:n_lines]
    if obs.size == 0:
        return empty

    q_obs = two_theta_to_q(obs, wavelength)
    q_n = float(q_obs.max())
    v = hkl_grid(gstar, q_n * 1.02)
    if v.shape[0] == 0:
        return empty
    q_calc = np.einsum("ni,ij,nj->n", v, gstar, v)
    tt_calc = q_to_two_theta(q_calc, wavelength)
    ok = np.isfinite(tt_calc)
    q_calc, tt_calc = q_calc[ok], tt_calc[ok]
    if q_calc.size == 0:
        return empty

    # distinct calculated lines up to Q_N (de Wolff N_poss)
    n_poss = int(np.unique(np.round(q_calc[q_calc <= q_n], 6)).size)
    if n_poss == 0:
        return empty

    dtt = np.array([np.min(np.abs(tt_calc - t)) for t in obs])
    dq = np.array([np.min(np.abs(q_calc - q)) for q in q_obs])
    n_indexed = int(np.sum(dtt <= tol_deg))
    mean_dq = float(np.mean(dq))
    m_n = q_n / (2.0 * mean_dq * n_poss) if mean_dq > 1e-12 else float("inf")
    return {
        "coverage": float(n_indexed / obs.size),
        "m_n": float(m_n),
        "n_poss": n_poss,
        "n_indexed": n_indexed,
        "mean_dq": mean_dq,
    }
