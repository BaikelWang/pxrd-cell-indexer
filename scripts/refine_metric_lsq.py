#!/usr/bin/env python3
"""Constrained least-squares refinement of a cell against observed 2-theta peaks.

McMaille's Rp gate needs roughly 0.2% cell accuracy; flow seeds land near 1%.
Nelder-Mead straight on the 2-theta residual does not close that gap because the
residual is riddled with local minima created by hkl assignments flipping between
steps.

Freezing the hkl assignment removes exactly that pathology. With hkl fixed,

    Q = 1/d^2 = h^2 A + k^2 B + l^2 C + kl D + hl E + hk F

is *linear* in the reciprocal metric tensor components (A..F), and every crystal
system constraint is also linear in that same basis (cubic is A=B=C with the
cross terms zero, hexagonal is A=B=F with C free, and so on). So each iteration
is an ordinary constrained linear least squares with a closed-form solution, and
the only nonlinearity left is the outer re-assignment loop.
"""

from __future__ import annotations

import numpy as np

IFI_CUBIC, IFI_HEX, IFI_TETRA, IFI_ORTHO, IFI_MONO, IFI_TRI, IFI_RHOMB = 1, 2, 3, 4, 5, 6, 7

# Columns are the free parameters, rows are (A, B, C, D, E, F).
_CONSTRAINT = {
    IFI_CUBIC: np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]),
    IFI_RHOMB: np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    ),
    IFI_TETRA: np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ),
    # gamma=120 makes the hk cross term equal the a* term.
    IFI_HEX: np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    ),
    IFI_ORTHO: np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ),
    # beta-unique: alpha*=gamma*=90 kills D and F, E stays free.
    IFI_MONO: np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),
    IFI_TRI: np.eye(6),
}


def cell_to_gstar(cell6) -> np.ndarray:
    """Cell parameters to the (A, B, C, D, E, F) reciprocal metric vector."""
    a, b, c, al, be, ga = [float(x) for x in cell6]
    ca, cb, cg = (np.cos(np.radians(x)) for x in (al, be, ga))
    sa, sb, sg = (np.sin(np.radians(x)) for x in (al, be, ga))
    g = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ]
    )
    gs = np.linalg.inv(g)
    return np.array([gs[0, 0], gs[1, 1], gs[2, 2], 2 * gs[1, 2], 2 * gs[0, 2], 2 * gs[0, 1]])


def gstar_to_cell(p) -> list[float] | None:
    """Inverse of :func:`cell_to_gstar`; None when the tensor is not a lattice."""
    A, B, C, D, E, F = [float(x) for x in p]
    gs = np.array([[A, F / 2, E / 2], [F / 2, B, D / 2], [E / 2, D / 2, C]])
    try:
        if not np.all(np.linalg.eigvalsh(gs) > 1e-12):
            return None
        g = np.linalg.inv(gs)
    except np.linalg.LinAlgError:
        return None
    diag = np.diag(g)
    if not np.all(diag > 1e-12):
        return None
    a, b, c = np.sqrt(diag)
    cos = [g[1, 2] / (b * c), g[0, 2] / (a * c), g[0, 1] / (a * b)]
    if any(abs(x) >= 1.0 for x in cos):
        return None
    al, be, ga = (float(np.degrees(np.arccos(x))) for x in cos)
    return [float(a), float(b), float(c), al, be, ga]


def _hkl_grid(p, q_max: float, hkl_cap: int = 14) -> np.ndarray:
    """Reflections whose Q stays under ``q_max``, deduped by |hkl| sign pair."""
    A, B, C = max(p[0], 1e-9), max(p[1], 1e-9), max(p[2], 1e-9)
    nh = min(hkl_cap, max(1, int(np.sqrt(q_max / A)) + 1))
    nk = min(hkl_cap, max(1, int(np.sqrt(q_max / B)) + 1))
    nl = min(hkl_cap, max(1, int(np.sqrt(q_max / C)) + 1))
    h = np.arange(0, nh + 1)
    k = np.arange(-nk, nk + 1)
    l = np.arange(-nl, nl + 1)
    grid = np.stack(np.meshgrid(h, k, l, indexing="ij"), axis=-1).reshape(-1, 3)
    # Friedel pairs are redundant; keep one half space.
    keep = (grid[:, 0] > 0) | ((grid[:, 0] == 0) & ((grid[:, 1] > 0) | ((grid[:, 1] == 0) & (grid[:, 2] > 0))))
    return grid[keep]


def _design(hkl: np.ndarray) -> np.ndarray:
    """Rows of the Q = M . (A..F) design matrix."""
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    return np.stack([h * h, k * k, l * l, k * l, h * l, h * k], axis=1).astype(float)


def two_theta_to_q(two_theta, wavelength: float) -> np.ndarray:
    return (2.0 * np.sin(np.radians(np.asarray(two_theta, float)) / 2.0) / wavelength) ** 2


def q_to_two_theta(q, wavelength: float) -> np.ndarray:
    s = np.clip(np.sqrt(np.maximum(np.asarray(q, float), 0.0)) * wavelength / 2.0, -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(s))


def _residual(cell6, two_theta_obs, wavelength, tol_2th) -> tuple[float, int]:
    """Mean |d2theta| over assigned peaks, and how many peaks got assigned."""
    p = cell_to_gstar(cell6)
    q_obs = two_theta_to_q(two_theta_obs, wavelength)
    hkl = _hkl_grid(p, float(q_obs.max()) * 1.10)
    if len(hkl) == 0:
        return float("inf"), 0
    q_calc = _design(hkl) @ p
    tt_calc = q_to_two_theta(q_calc, wavelength)
    d = np.abs(np.asarray(two_theta_obs, float)[:, None] - tt_calc[None, :])
    best = d.min(axis=1)
    ok = best <= tol_2th
    if not ok.any():
        return float("inf"), 0
    return float(best[ok].mean()), int(ok.sum())


def refine_cell(
    cell6,
    ifi: int,
    two_theta_obs,
    wavelength: float,
    *,
    tol_start: float = 1.5,
    tol_end: float = 0.15,
    tol_eval: float = 0.30,
    n_iter: int = 8,
    max_drift: float = 0.06,
    min_peaks: int = 6,
) -> tuple[list[float], dict]:
    """Refine ``cell6`` under the symmetry of ``ifi`` against observed peaks.

    The matching tolerance is annealed from ``tol_start`` down to ``tol_end``. A
    1% cell error already displaces a 60-degree line by roughly 0.66 degrees, so
    opening on a tight window silently mis-assigns every high-angle peak and the
    fit locks onto the wrong solution; tightening later then trims the outliers.

    ``max_drift`` is a trust region on the relative edge change: the point is to
    sharpen a seed the flow model already placed correctly, not to let the fit
    wander off to a different lattice. Returns the original cell unchanged if the
    refinement fails any guard.
    """
    start = [float(x) for x in list(cell6)[:6]]
    info = {"ok": False, "reason": "", "n_indexed": 0, "iters": 0}
    T = _CONSTRAINT.get(int(ifi))
    if T is None:
        info["reason"] = "unknown ifi"
        return start, info

    tt_obs = np.asarray(two_theta_obs, float)
    tt_obs = tt_obs[np.isfinite(tt_obs) & (tt_obs > 0)]
    if tt_obs.size < min_peaks:
        info["reason"] = "too few peaks"
        return start, info
    q_obs = two_theta_to_q(tt_obs, wavelength)
    q_max = float(q_obs.max()) * 1.10

    r0, n0 = _residual(start, tt_obs, wavelength, tol_eval)
    p = cell_to_gstar(start)
    cur = start
    n_assigned = 0

    for it in range(n_iter):
        # Geometric anneal of the assignment window.
        f = it / max(n_iter - 1, 1)
        tol = tol_start * (tol_end / tol_start) ** f
        hkl = _hkl_grid(p, q_max)
        if len(hkl) == 0:
            info["reason"] = "no reflections"
            break
        M_all = _design(hkl)
        q_calc = M_all @ p
        tt_calc = q_to_two_theta(q_calc, wavelength)

        # Freeze the assignment: nearest calculated line per observed peak.
        d = np.abs(tt_obs[:, None] - tt_calc[None, :])
        j = d.argmin(axis=1)
        ok = d[np.arange(len(tt_obs)), j] <= tol
        n_assigned = int(ok.sum())
        if n_assigned < max(min_peaks, T.shape[1] + 2):
            info["reason"] = f"only {n_assigned} peaks indexed"
            break

        # Linear LSQ in the constrained subspace, then lift back to (A..F).
        M = M_all[j[ok]] @ T
        try:
            x, *_ = np.linalg.lstsq(M, q_obs[ok], rcond=None)
        except np.linalg.LinAlgError:
            info["reason"] = "lstsq failed"
            break
        p_new = T @ x
        cell_new = gstar_to_cell(p_new)
        if cell_new is None:
            info["reason"] = "degenerate tensor"
            break
        p, cur = p_new, cell_new
        info["iters"] = it + 1

    drift = max(abs(cur[i] - start[i]) / max(start[i], 1e-9) for i in range(3))
    drift = max(drift, max(abs(cur[i] - start[i]) for i in range(3, 6)) / 90.0)
    if drift > max_drift:
        info["reason"] = f"drift {drift:.3f} exceeds trust region"
        return start, info

    r1, n1 = _residual(cur, tt_obs, wavelength, tol_eval)
    # Accept only a real improvement: more peaks indexed, or a tighter fit.
    if n1 < n0 or not np.isfinite(r1) or r1 > r0:
        info["reason"] = "no improvement"
        return start, info

    info.update(
        ok=True,
        n_indexed=n1,
        resid_before=r0,
        resid_after=r1,
        drift=drift,
        n_assigned=n_assigned,
    )
    return cur, info


# Whether a seed converges depends on getting the first hkl assignment right, so
# outcomes are bimodal: exact, or stuck. Trying a few annealing schedules and
# keeping the tightest fit recovers a slice of the stuck ones.
TOL_LADDERS = ((1.5, 0.15), (0.8, 0.10), (2.5, 0.25), (0.4, 0.08))


def refine_cell_multi(cell6, ifi: int, two_theta_obs, wavelength: float, **kw):
    """:func:`refine_cell` over several tolerance ladders, best fit wins."""
    best, best_score = [float(x) for x in list(cell6)[:6]], None
    best_info = {"ok": False, "reason": "all ladders rejected"}
    for tol_start, tol_end in TOL_LADDERS:
        cell, info = refine_cell(
            cell6, ifi, two_theta_obs, wavelength, tol_start=tol_start, tol_end=tol_end, **kw
        )
        if not info.get("ok"):
            continue
        # Reward a tight residual spread over many indexed lines.
        score = info["resid_after"] / max(info["n_indexed"], 1)
        if best_score is None or score < best_score:
            best, best_score, best_info = cell, score, info
    return best, best_info


def read_dat_peaks(path) -> tuple[float, list[float]]:
    """Wavelength and the 2-theta column from a McMaille ``.dat`` input."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    wavelength, peaks, seen_wl, in_peaks = 1.54056, [], False, False
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("!"):
            # Peaks start after the "2-theta" banner; the lines before it are
            # McMaille control parameters that also parse as bare numbers.
            if "2-theta" in s.lower():
                in_peaks = True
            continue
        try:
            vals = [float(x) for x in s.split()]
        except ValueError:
            continue
        if not seen_wl:
            wavelength = vals[0]
            seen_wl = True
            continue
        if in_peaks and vals and 0.0 < vals[0] < 180.0:
            peaks.append(vals[0])
    return wavelength, peaks
