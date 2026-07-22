"""Independent q-search: peaks → candidate unit cells, no NN initial value.

Algorithm (Ito/dichotomy-style direct method, v3 §11.1):

    observed 2θ peaks
      -> convert to q = 1/d² (Å⁻²), sort ascending
      -> per crystal system, assign small-integer hkl to the ``k`` lowest
         peaks (k = system's reciprocal-metric DOF)
      -> 1/d²_hkl = hᵀ G* h is *linear* in G*'s 6 independent components, so
         each (peak, hkl) pair gives one linear equation; k equations solve
         exactly for the system's free metric parameters
      -> reconstruct full G*, require SPD (physical cell)
      -> validate against *all* observed peaks (de Wolff-style greedy 1/d²
         matching, pure numpy); keep only high-consistency solutions
      -> Niggli-reduce, dedup, per-system candidate budget

This is intentionally independent of any NN prediction/initial value; the
model can later contribute only a soft prior (CS search budget, volume
range) per v3 §11.1, not a hard filter.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from pxrd_cell_indexing.data.canonical import canonicalize_lattice
from pxrd_cell_indexing.geometry import lattice_params_to_matrix

DEFAULT_WAVELENGTH_ANGSTROM = 1.54184

# Reciprocal-metric DOF per crystal system (free params in the linear solve).
# monoclinic_{a,b,c}: internal sub-variants for the 3 possible unique axes
# (see `_basis` below) -- `search_crystal_system("monoclinic", ...)` tries
# all 3 and merges, since peak positions alone can't tell which axis a given
# (possibly Niggli-reduced) cell put its oblique angle on.
# cubic_{p,f,i}: valid1400/MP-style labels store *Niggli-reduced* (i.e.
# primitive-cell) lattice params -- an F- or I-centered cubic Bravais lattice
# Niggli-reduces to a rhombohedral-shaped primitive cell (alpha=beta=gamma=60
# deg for F, ~109.47 deg for I), not the conventional 90/90/90 orthogonal
# cell, so its reciprocal metric has nonzero off-diagonal terms even though
# the crystal system is still "cubic". `search_crystal_system("cubic", ...)`
# tries all 3 centerings and merges.
CRYSTAL_SYSTEM_DOF: dict[str, int] = {
    "cubic": 1,
    "cubic_p": 1,
    "cubic_f": 1,
    "cubic_i": 1,
    "tetragonal": 2,
    "hexagonal": 2,
    "trigonal": 2,
    "trigonal_hex": 2,
    "trigonal_rhomb": 2,
    "orthorhombic": 3,
    "monoclinic": 4,
    "monoclinic_a": 4,
    "monoclinic_b": 4,
    "monoclinic_c": 4,
    "triclinic": 6,
}
_MONOCLINIC_VARIANTS = ("monoclinic_a", "monoclinic_b", "monoclinic_c")
_CUBIC_VARIANTS = ("cubic_p", "cubic_f", "cubic_i")
# trigonal_hex: hexagonal axes (a=b, γ=120°); trigonal_rhomb: free rhombohedral
# primitive (a=b=c, α=β=γ) -- valid1400 Niggli labels use both images.
_TRIGONAL_VARIANTS = ("trigonal_hex", "trigonal_rhomb")
# Systems that use sequential (axial → off-diagonal) solve instead of joint M^k.
_SEQUENTIAL_SYSTEMS = frozenset(
    {"orthorhombic", "monoclinic_a", "monoclinic_b", "monoclinic_c", "triclinic"}
)

# Per-system (6, k) basis mapping free params x -> full G* vector
# [G11, G22, G33, G12, G13, G23] = Basis @ x.
def _basis(system: str) -> np.ndarray:
    if system in ("cubic", "cubic_p"):
        # Primitive (P) cubic cell on its own orthogonal axes: G11=G22=G33=A,
        # cross=0. A = 1/a^2, a = cell edge = the reciprocal-solve's own "a".
        return np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
    if system == "cubic_f":
        # Niggli-reduced primitive cell of an F-centered cubic Bravais
        # lattice: primitive vectors at 60 deg to each other, length a_p.
        # G* = (1/a_p^2) * [[1.5,-.5,-.5],[-.5,1.5,-.5],[-.5,-.5,1.5]].
        return np.array([[1.5], [1.5], [1.5], [-0.5], [-0.5], [-0.5]])
    if system == "cubic_i":
        # Niggli-reduced primitive cell of an I-centered cubic Bravais
        # lattice: primitive vectors at arccos(-1/3)~=109.47 deg, length a_p.
        # G* = (1/a_p^2) * [[1.5,.75,.75],[.75,1.5,.75],[.75,.75,1.5]].
        return np.array([[1.5], [1.5], [1.5], [0.75], [0.75], [0.75]])
    if system == "tetragonal":
        # G11=G22=A, G33=C, cross=0.
        return np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
    if system in ("hexagonal", "trigonal", "trigonal_hex"):
        # Standard hex axes (a=b, γ=120°, α=β=90°): G11=G22=4A/3, G12=2A/3, G33=C.
        return np.array(
            [
                [4.0 / 3.0, 0.0],
                [4.0 / 3.0, 0.0],
                [0.0, 1.0],
                [2.0 / 3.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    if system == "trigonal_rhomb":
        # Rhombohedral primitive: G11=G22=G33=A, G12=G13=G23=B (2 free params).
        # Covers general rhombohedral cells; cubic_f/i are the special cases
        # B/A = -1/3 and +1/2.
        return np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
    if system == "orthorhombic":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
    if system in ("monoclinic", "monoclinic_b"):
        # b-axis unique (standard) convention: free G11, G22, G33, G13;
        # G12=G23=0.
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    if system == "monoclinic_a":
        # a-axis unique: free G11, G22, G33, G23; G12=G13=0.
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    if system == "monoclinic_c":
        # c-axis unique: free G11, G22, G33, G12; G13=G23=0.
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    if system == "triclinic":
        return np.eye(6)
    raise ValueError(f"unknown crystal system: {system!r}")


def _coeff_row(h: int, k: int, l: int) -> np.ndarray:
    """Row of the linear system for peak hkl: q = row · [G11,G22,G33,G12,G13,G23]."""
    return np.array([h * h, k * k, l * l, 2 * h * k, 2 * h * l, 2 * k * l], dtype=np.float64)


def _hkl_pool(max_index: int, max_nonzero: int = 3) -> list[tuple[int, int, int]]:
    """Small-integer hkl within a box, deduped by the (h,k,l)~(-h,-k,-l) degeneracy
    (coeff row is degree-2 homogeneous, so both signs give an identical equation).

    ``max_nonzero`` caps how many of (h,k,l) may be nonzero -- axial/zone
    reflections (1-2 nonzero indices) are statistically the lowest-angle
    lines, so a "sparse" pool can afford a much larger index bound at the
    same combinatorial cost as a small "dense" (3-nonzero) pool.
    """
    pool: list[tuple[int, int, int]] = []
    rng = range(-max_index, max_index + 1)
    for h, k, l in itertools.product(rng, rng, rng):
        if h == 0 and k == 0 and l == 0:
            continue
        if (h != 0) + (k != 0) + (l != 0) > max_nonzero:
            continue
        # Canonical representative: first nonzero index positive.
        first_nonzero = h if h != 0 else (k if k != 0 else l)
        if first_nonzero < 0:
            continue
        pool.append((h, k, l))
    return pool


def _tiered_hkl_pool(
    *,
    dense_max_index: int,
    dense_max_nonzero: int = 3,
    sparse_max_index: int | None = None,
    sparse_max_nonzero: int = 2,
) -> list[tuple[int, int, int]]:
    """Sparse (axial/zone, large index bound) tier first, then dense (small index
    bound, up to ``dense_max_nonzero`` simultaneously nonzero) tier -- coarse-to-fine
    trial order at a fraction of the cost of a uniform dense box (v3 §11.1 "枚举低阶
    hkl ... coarse-to-fine"). Capping ``dense_max_nonzero`` below 3 is what keeps
    3+ DOF systems (orthorhombic/monoclinic/triclinic) inside the full-enumeration
    budget instead of falling back to low-yield random sampling."""
    dense = _hkl_pool(dense_max_index, max_nonzero=dense_max_nonzero)
    if sparse_max_index is None or sparse_max_index <= dense_max_index:
        return dense
    sparse = _hkl_pool(sparse_max_index, max_nonzero=sparse_max_nonzero)
    sparse_set = set(sparse)
    return sparse + [hkl for hkl in dense if hkl not in sparse_set]


def _gvec_to_matrix(gvec6: np.ndarray) -> np.ndarray:
    g11, g22, g33, g12, g13, g23 = gvec6
    return np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]], dtype=np.float64)


def inverse_d2_from_two_theta_f64(
    two_theta_deg: np.ndarray, *, wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
) -> np.ndarray:
    """1/d² (Å⁻²) in float64 -- ``data.peak_features.inverse_d2_from_two_theta`` casts
    to float32, which is precise enough for NN training but introduces ~1e-7 relative
    noise that can flip a peak's de Wolff match at the 1e-6 exact-match tolerance used
    here (ideal/synthetic peaks should match to numerical-noise precision, not ~1e-7)."""
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64) / 2.0)
    s = 2.0 * np.sin(np.clip(theta, 1e-8, None)) / float(wavelength_angstrom)
    return s * s


def gstar_to_lattice_params(gstar: np.ndarray) -> tuple[float, float, float, float, float, float] | None:
    """G* (no-2π reciprocal metric) -> (a,b,c,α,β,γ)[deg]; None if not SPD.

    Pure numpy (no torch): this is the innermost hot-path call of the search
    (once per trial that survives the cheap cond-number filter), and per-call
    torch tensor/dispatch overhead (~0.1-0.5ms) dominates runtime at scale.
    """
    eigvals = np.linalg.eigvalsh(gstar)
    if eigvals[0] <= 1e-10:
        return None
    try:
        g_direct = np.linalg.inv(gstar)
    except np.linalg.LinAlgError:
        return None
    lengths = np.sqrt(np.clip(np.diag(g_direct), 1e-18, None))
    a, b, c = (float(v) for v in lengths)
    if not (0.5 < a < 100 and 0.5 < b < 100 and 0.5 < c < 100):
        return None
    cos_alpha = float(np.clip(g_direct[1, 2] / (b * c + 1e-18), -0.999999, 0.999999))
    cos_beta = float(np.clip(g_direct[0, 2] / (a * c + 1e-18), -0.999999, 0.999999))
    cos_gamma = float(np.clip(g_direct[0, 1] / (a * b + 1e-18), -0.999999, 0.999999))
    alpha = float(np.degrees(np.arccos(cos_alpha)))
    beta = float(np.degrees(np.arccos(cos_beta)))
    gamma = float(np.degrees(np.arccos(cos_gamma)))
    if not (20.0 < alpha < 160.0 and 20.0 < beta < 160.0 and 20.0 < gamma < 160.0):
        return None
    return a, b, c, alpha, beta, gamma


def _theoretical_q2_values(gstar: np.ndarray, q_max: float, *, max_hkl_cap: int = 40) -> np.ndarray:
    """All theoretical 1/d² values up to ``q_max`` for reciprocal metric ``gstar``
    (no-2π convention, so directly comparable to ``inverse_d2_from_two_theta_f64``).
    Pure numpy; avoids the torch round-trip inside ``model.fom.theoretical_two_theta``."""
    diag = np.clip(np.diag(gstar), 1e-12, None)
    # +1 index margin absorbs the ceil/float-noise gap between this grid's own
    # arithmetic and whatever path (e.g. matrix round-trip) produced ``q_max``.
    limits = np.clip(np.ceil(np.sqrt(max(q_max, 1e-9) / diag)).astype(int) + 1, 1, max_hkl_cap)
    h_max, k_max, l_max = (int(v) for v in limits)
    h = np.arange(-h_max, h_max + 1)
    k = np.arange(-k_max, k_max + 1)
    l = np.arange(-l_max, l_max + 1)
    hh, kk, ll = np.meshgrid(h, k, l, indexing="ij")
    hf, kf, lf = hh.reshape(-1).astype(np.float64), kk.reshape(-1).astype(np.float64), ll.reshape(-1).astype(np.float64)
    mask = (hf != 0) | (kf != 0) | (lf != 0)
    hf, kf, lf = hf[mask], kf[mask], lf[mask]
    q2 = (
        hf * hf * gstar[0, 0]
        + kf * kf * gstar[1, 1]
        + lf * lf * gstar[2, 2]
        + 2.0 * hf * kf * gstar[0, 1]
        + 2.0 * hf * lf * gstar[0, 2]
        + 2.0 * kf * lf * gstar[1, 2]
    )
    margin = max(1e-6, 1e-4 * q_max)
    q2 = q2[(q2 > 0) & (q2 <= q_max + margin)]
    return np.sort(np.unique(np.round(q2, 10)))


def _approx_match_counts(
    gvec_batch: np.ndarray,
    confirm_coeff_rows: np.ndarray,
    q_obs: np.ndarray,
    *,
    q_match_abs_tol: float,
    candidate_chunk: int = 2_000,
    deadline: float | None = None,
) -> np.ndarray:
    """Cheap vectorized *pre-filter* match count: projects every candidate's
    ``gvec`` onto a fixed hkl grid (``confirm_coeff_rows``, shape (M, 6)) via a
    single matmul, then does a nearest-value lookup per observed peak with
    numpy broadcasting -- no per-candidate Python loop.

    This approximates (but does not replace) ``_fast_match_count``'s greedy
    1:1 de Wolff matching: with ``q_match_abs_tol`` this tight, two pool hkls
    landing within tol of the same observed peak is exceedingly rare, so the
    "nearest hkl in the grid" count is a very close upper-bound proxy. Used
    only to cheaply discard the bulk of spurious hkl-assignment solutions
    before paying for the exact, per-candidate ``_fast_match_count`` call.
    """
    n = gvec_batch.shape[0]
    M = confirm_coeff_rows.shape[0]
    # Bound each chunk's (M, chunk) intermediate to a roughly fixed element
    # budget regardless of how wide the confirm grid is, so a caller with a
    # big M (e.g. a generous confirm_max_index) can't blow past its time
    # budget inside a single un-interruptible chunk.
    chunk = max(1, min(candidate_chunk, (2_000_000 // max(M, 1)) + 1))
    counts = np.zeros(n, dtype=np.int32)
    for start in range(0, n, chunk):
        if deadline is not None and time.monotonic() > deadline:
            # Unprocessed candidates are treated as non-matches (safe: they
            # just won't be promoted by this filter, never accepted).
            break
        end = min(start + chunk, n)
        pred = confirm_coeff_rows @ gvec_batch[start:end].T  # (M, chunk)
        chunk_counts = np.zeros(end - start, dtype=np.int32)
        for q in q_obs:
            min_diff = np.min(np.abs(pred - q), axis=0)  # (chunk,)
            chunk_counts += (min_diff <= q_match_abs_tol).astype(np.int32)
        counts[start:end] = chunk_counts
    return counts


def _fast_match_count(
    q_obs: np.ndarray,
    gstar: np.ndarray,
    *,
    q_match_abs_tol: float = 1e-6,
) -> int:
    """Greedy one-to-one match count of observed 1/d² against theoretical lines
    for ``gstar``; the de Wolff-style consistency filter, computed entirely in
    numpy (no torch, no lattice-param round-trip)."""
    q_max = float(q_obs.max())
    theory = _theoretical_q2_values(gstar, q_max)
    if theory.size == 0:
        return 0
    used = np.zeros(theory.shape[0], dtype=bool)
    n_matched = 0
    for q in np.sort(q_obs):
        available = np.where(~used)[0]
        if available.size == 0:
            break
        diffs = np.abs(theory[available] - q)
        best_local = int(np.argmin(diffs))
        if diffs[best_local] <= q_match_abs_tol:
            used[available[best_local]] = True
            n_matched += 1
    return n_matched


def _axial_index_pool(max_index: int) -> list[int]:
    """Positive axial Miller indices 1..max_index."""
    return list(range(1, max_index + 1))


def _zone_hkl_pool(max_index: int, which: str) -> list[tuple[int, int, int]]:
    """Two-nonzero hkl that isolate one off-diagonal G* component."""
    pool: list[tuple[int, int, int]] = []
    rng = range(1, max_index + 1)
    if which == "g12":
        for h, k in itertools.product(rng, rng):
            pool.append((h, k, 0))
    elif which == "g13":
        for h, l in itertools.product(rng, rng):
            pool.append((h, 0, l))
    elif which == "g23":
        for k, l in itertools.product(rng, rng):
            pool.append((0, k, l))
    else:
        raise ValueError(which)
    return pool


def _offdiag_from_peak(
    q: float,
    hkl: tuple[int, int, int],
    g11: float,
    g22: float,
    g33: float,
    which: str,
) -> float | None:
    """Solve one off-diagonal given diagonals: q = hᵀG*h."""
    h, k, l = hkl
    residual = q - h * h * g11 - k * k * g22 - l * l * g33
    if which == "g12":
        denom = 2.0 * h * k
    elif which == "g13":
        denom = 2.0 * h * l
    elif which == "g23":
        denom = 2.0 * k * l
    else:
        raise ValueError(which)
    if abs(denom) < 1e-12:
        return None
    return residual / denom


def _gvec_to_candidate_if_ok(
    gvec: np.ndarray,
    *,
    system: str,
    q_obs: np.ndarray,
    n_peaks_total: int,
    min_matched: int,
    q_match_abs_tol: float,
    hkl_used: tuple[tuple[int, int, int], ...],
    seen: set[tuple[float, ...]],
) -> QSearchCandidate | None:
    """SPD + length/angle gates + exact match filter → candidate or None."""
    gstar = np.array(
        [
            [gvec[0], gvec[3], gvec[4]],
            [gvec[3], gvec[1], gvec[5]],
            [gvec[4], gvec[5], gvec[2]],
        ],
        dtype=np.float64,
    )
    params = gstar_to_lattice_params(gstar)
    if params is None:
        return None
    rounded = tuple(round(v, 4) for v in params)
    if rounded in seen:
        return None
    n_matched = _fast_match_count(q_obs, gstar, q_match_abs_tol=q_match_abs_tol)
    if n_matched < min_matched:
        return None
    seen.add(rounded)
    volume = float(1.0 / np.sqrt(max(np.linalg.det(gstar), 1e-30)))
    return QSearchCandidate(
        crystal_system=system,
        a=params[0],
        b=params[1],
        c=params[2],
        alpha=params[3],
        beta=params[4],
        gamma=params[5],
        n_matched=n_matched,
        n_peaks=n_peaks_total,
        volume=volume,
        hkl_used=hkl_used,
    )


def _search_sequential(
    observed_two_theta: Sequence[float] | np.ndarray,
    system: str,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    axial_max_index: int = 8,
    zone_max_index: int = 4,
    n_low_peaks: int | None = None,
    match_fraction_min: float = 0.95,
    q_match_abs_tol: float = 1e-6,
    pool_budget: int = 30,
    time_budget_s: float = 25.0,
    confirm_max_index: int | None = None,
) -> list[QSearchCandidate]:
    """Sequential axial→off-diagonal solve for ortho / monoclinic_* / triclinic.

    Replaces joint ``M^k`` enumeration (v4 §6.4 / B1-S0 §8.2):
      1. independently solve G11/G22/G33 from axial (h00)/(0k0)/(00l) peaks;
      2. solve each needed off-diagonal from one zone peak (known diagonals).
    Combinatorial cost ~ ``M_axis^3 + M_zone`` instead of ``M_dense^k``.
    """
    two_theta = np.asarray(observed_two_theta, dtype=np.float64)
    two_theta = np.sort(two_theta[np.isfinite(two_theta)])
    if two_theta.shape[0] < 3:
        return []
    q_obs = inverse_d2_from_two_theta_f64(two_theta, wavelength_angstrom=wavelength_angstrom)
    n_peaks_total = int(two_theta.shape[0])
    if n_low_peaks is None:
        n_low_peaks = min(n_peaks_total, 8)
    n_low_peaks = max(3, min(n_low_peaks, n_peaks_total))
    # Axial (h00)/(0k0)/(00l) for the *short* real-space axis can sit at
    # relatively high 2θ; a tight n_low_peaks window drops them and yields
    # zero candidates on anisotropic cells. Use a wider window for diagonals.
    n_axial_peaks = max(n_low_peaks, min(n_peaks_total, 12))
    q_low = q_obs[:n_low_peaks]
    q_axial = q_obs[:n_axial_peaks]

    axial_idx = _axial_index_pool(axial_max_index)
    # Per-axis candidate list: (peak_i, miller, G_ii)
    g11_opts = [(i, h, float(q_axial[i] / (h * h))) for i in range(n_axial_peaks) for h in axial_idx]
    g22_opts = [(i, k, float(q_axial[i] / (k * k))) for i in range(n_axial_peaks) for k in axial_idx]
    g33_opts = [(i, l, float(q_axial[i] / (l * l))) for i in range(n_axial_peaks) for l in axial_idx]

    offdiag_needed: list[str]
    if system == "orthorhombic":
        offdiag_needed = []
    elif system == "monoclinic_b":
        offdiag_needed = ["g13"]
    elif system == "monoclinic_a":
        offdiag_needed = ["g23"]
    elif system == "monoclinic_c":
        offdiag_needed = ["g12"]
    elif system == "triclinic":
        offdiag_needed = ["g12", "g13", "g23"]
    else:
        raise ValueError(f"sequential solve not defined for {system!r}")

    zone_pools = {which: _zone_hkl_pool(zone_max_index, which) for which in offdiag_needed}

    if confirm_max_index is None:
        # Keep the confirm grid modest: sequential already enumerates many
        # diagonal seeds; a huge confirm matmul dominates wall time.
        confirm_max_index = min(12, max(axial_max_index, zone_max_index) + 3)
    confirm_pool_arr = np.array(_hkl_pool(confirm_max_index, max_nonzero=3), dtype=np.float64)
    confirm_coeff_rows = np.stack(
        [
            confirm_pool_arr[:, 0] ** 2,
            confirm_pool_arr[:, 1] ** 2,
            confirm_pool_arr[:, 2] ** 2,
            2 * confirm_pool_arr[:, 0] * confirm_pool_arr[:, 1],
            2 * confirm_pool_arr[:, 0] * confirm_pool_arr[:, 2],
            2 * confirm_pool_arr[:, 1] * confirm_pool_arr[:, 2],
        ],
        axis=1,
    )

    min_matched = int(np.ceil(match_fraction_min * n_peaks_total))
    collect_budget = max(pool_budget * 10, 300)
    seen: set[tuple[float, ...]] = set()
    candidates: list[QSearchCandidate] = []
    start = time.monotonic()
    deadline = start + time_budget_s

    # Prefer small Miller indices (true low-angle axials are usually |h|<=3).
    g11_opts.sort(key=lambda t: (t[1], t[0]))
    g22_opts.sort(key=lambda t: (t[1], t[0]))
    g33_opts.sort(key=lambda t: (t[1], t[0]))

    def _dedupe_diag_opts(
        opts: list[tuple[int, int, float]], cap: int
    ) -> list[tuple[int, int, float]]:
        """Keep first-seen Gii clusters; preserves low-Miller priority."""
        out: list[tuple[int, int, float]] = []
        for item in opts:
            gii = item[2]
            if any(abs(gii - u[2]) <= max(1e-8, 1e-5 * abs(u[2])) for u in out):
                continue
            out.append(item)
            if len(out) >= cap:
                break
        return out

    diag_cap = 30 if system == "triclinic" else 80
    g11_opts = _dedupe_diag_opts(g11_opts, diag_cap)
    g22_opts = _dedupe_diag_opts(g22_opts, diag_cap)
    g33_opts = _dedupe_diag_opts(g33_opts, diag_cap)

    # Build candidate gvecs in chunks, then one vectorized approx_match pass.
    batch_gvecs: list[np.ndarray] = []
    batch_meta: list[tuple[tuple[int, int, int], ...]] = []
    batch_size = 2_000
    max_diag_seeds = 80_000
    diag_seeds = 0

    def _flush_batch() -> None:
        nonlocal batch_gvecs, batch_meta, candidates
        if not batch_gvecs or time.monotonic() > deadline:
            batch_gvecs, batch_meta = [], []
            return
        gmat = np.stack(batch_gvecs, axis=0)
        approx = _approx_match_counts(
            gmat, confirm_coeff_rows, q_obs, q_match_abs_tol=q_match_abs_tol, deadline=deadline
        )
        order = np.argsort(-approx)
        for j in order:
            if approx[j] < min_matched or len(candidates) >= collect_budget:
                if approx[j] < min_matched:
                    continue
                break
            cand = _gvec_to_candidate_if_ok(
                gmat[j],
                system=system,
                q_obs=q_obs,
                n_peaks_total=n_peaks_total,
                min_matched=min_matched,
                q_match_abs_tol=q_match_abs_tol,
                hkl_used=batch_meta[j],
                seen=seen,
            )
            if cand is not None:
                candidates.append(cand)
        batch_gvecs, batch_meta = [], []

    def _has_perfect() -> bool:
        # Triclinic powder patterns can admit multiple 100%-match G* frames;
        # never short-circuit mid-search so alternate frames stay in the pool.
        if system == "triclinic":
            return False
        return any(c.n_matched >= n_peaks_total for c in candidates)

    def _iter_diag_seeds():
        """Yield ((i,h,G11),(j,k,G22),(m,l,G33)) seeds.

        Triclinic pass-1: triples that include the lowest-q peak as an axial
        (100)/(010)/(001) with G11≥G22≥G33 — C(N-1,2) seeds.
        Pass-2 (only if pass-1 found nothing): general low-Miller product.
        """
        if system != "triclinic":
            return itertools.product(g11_opts, g22_opts, g33_opts)

        def _pass1():
            # Lowest-q peak is an axial reflection on noiseless triclinic
            # synthetics → C(N-1,2) instead of C(N,3).
            for rest in itertools.combinations(range(1, n_peaks_total), 2):
                triple = (0, rest[0], rest[1])
                ordered = sorted(triple, key=lambda idx: -float(q_obs[idx]))
                i1, i2, i3 = ordered
                yield (
                    (i1, 1, float(q_obs[i1])),
                    (i2, 1, float(q_obs[i2])),
                    (i3, 1, float(q_obs[i3])),
                )

        def _pass2():
            for seed in itertools.product(g11_opts, g22_opts, g33_opts):
                if seed[0][1] == seed[1][1] == seed[2][1] == 1:
                    continue
                yield seed

        # Materialize pass-1; keep pass-2 lazy and only run if pass-1 is empty.
        return list(_pass1()), _pass2()

    tric_pass2_iter = None
    if system == "triclinic":
        diag_seed_iter, tric_pass2_iter = _iter_diag_seeds()
    else:
        diag_seed_iter = _iter_diag_seeds()

    def _consume_diag_seed(i1, h, g11, i2, k, g22, i3, l, g33) -> bool:
        """Process one axial seed. Returns False if search should stop."""
        nonlocal diag_seeds, batch_gvecs, batch_meta
        if len(candidates) >= collect_budget or time.monotonic() > deadline or _has_perfect():
            return False
        if len({i1, i2, i3}) < 3:
            return True
        if g11 <= 1e-12 or g22 <= 1e-12 or g33 <= 1e-12:
            return True
        # Triclinic: only keep one axis labeling (G11≥G22≥G33) to cut the
        # 3! equivalent permutations; peak-match is labeling-invariant once
        # off-diagonals are solved in the same frame.
        if system == "triclinic" and (g11 + 1e-12 < g22 or g22 + 1e-12 < g33):
            return True
        diag_seeds += 1
        if diag_seeds > max_diag_seeds:
            return False

        a_est, b_est, c_est = 1.0 / np.sqrt(g11), 1.0 / np.sqrt(g22), 1.0 / np.sqrt(g33)
        if not (0.5 < a_est < 100 and 0.5 < b_est < 100 and 0.5 < c_est < 100):
            return True

        axial_hkl = ((h, 0, 0), (0, k, 0), (0, 0, l))

        if not offdiag_needed:
            batch_gvecs.append(np.array([g11, g22, g33, 0.0, 0.0, 0.0], dtype=np.float64))
            batch_meta.append(axial_hkl)
            if len(batch_gvecs) >= batch_size:
                _flush_batch()
            return True

        # Zone / off-diagonal peaks may sit *above* the axial window
        # (e.g. triclinic (110) after several (00l) harmonics). Use all
        # observed peaks except the three axial assignments.
        unused_peaks = [i for i in range(n_peaks_total) if i not in (i1, i2, i3)]
        per_off: dict[str, list[tuple[float, tuple[int, int, int]]]] = {w: [] for w in offdiag_needed}
        # Prefer fundamentals (110)/(101)/(011) from every unused peak.
        # Do NOT rank by |Gij| — that drops large true couplings on oblique cells.
        per_off_cap = 14 if len(offdiag_needed) >= 3 else 12
        fund_hkl = {"g12": (1, 1, 0), "g13": (1, 0, 1), "g23": (0, 1, 1)}
        for which in offdiag_needed:
            # Always try the near-orthogonal / zero-coupling seed.
            per_off[which].append((0.0, (0, 0, 0)))
            scored: list[tuple[int, int, float, tuple[int, int, int]]] = []
            # Fundamentals first (every unused peak), then higher-zone fillers.
            hkl_order = []
            if which in fund_hkl:
                hkl_order.append(fund_hkl[which])
            hkl_order.extend(zhkl for zhkl in zone_pools[which] if zhkl not in hkl_order)
            for zhkl in hkl_order:
                for pi in unused_peaks:
                    val = _offdiag_from_peak(float(q_obs[pi]), zhkl, g11, g22, g33, which)
                    if val is None or not np.isfinite(val):
                        continue
                    if which == "g12" and abs(val) >= np.sqrt(g11 * g22):
                        continue
                    if which == "g13" and abs(val) >= np.sqrt(g11 * g33):
                        continue
                    if which == "g23" and abs(val) >= np.sqrt(g22 * g33):
                        continue
                    miller_sum = abs(zhkl[0]) + abs(zhkl[1]) + abs(zhkl[2])
                    scored.append((miller_sum, pi, float(val), zhkl))
            scored.sort(key=lambda t: (t[0], t[1]))
            uniq: list[tuple[float, tuple[int, int, int]]] = list(per_off[which])
            for _ms, _pi, val, zhkl in scored:
                if any(abs(val - u[0]) <= 1e-7 for u in uniq):
                    continue
                uniq.append((val, zhkl))
                if len(uniq) >= per_off_cap:
                    break
            per_off[which] = uniq

        if any(len(per_off[w]) == 0 for w in offdiag_needed):
            return True

        # Triclinic: beam-expand off-diagonals (full 14³ product is too slow).
        # Pass-1 axial triples are few (~C(N-1,2)), so beam cost stays modest.
        if len(offdiag_needed) >= 3:
            beam: list[tuple[np.ndarray, tuple[tuple[int, int, int], ...]]] = [
                (np.array([g11, g22, g33, 0.0, 0.0, 0.0], dtype=np.float64), axial_hkl)
            ]
            beam_width = 32
            for which in offdiag_needed:
                if time.monotonic() > deadline or not beam:
                    beam = []
                    break
                nxt: list[tuple[np.ndarray, tuple[tuple[int, int, int], ...]]] = []
                for gvec0, meta0 in beam:
                    for val, zhkl in per_off[which]:
                        gvec = gvec0.copy()
                        if which == "g12":
                            gvec[3] = val
                        elif which == "g13":
                            gvec[4] = val
                        else:
                            gvec[5] = val
                        meta = meta0 if zhkl == (0, 0, 0) else meta0 + (zhkl,)
                        nxt.append((gvec, meta))
                gmat = np.stack([t[0] for t in nxt], axis=0)
                approx = _approx_match_counts(
                    gmat, confirm_coeff_rows, q_obs, q_match_abs_tol=q_match_abs_tol, deadline=deadline
                )
                order = np.argsort(-approx)[:beam_width]
                beam = [(nxt[j][0], nxt[j][1]) for j in order if approx[j] >= max(1, min_matched // 2)]
            for gvec, meta in beam:
                batch_gvecs.append(gvec)
                batch_meta.append(meta)
            if len(batch_gvecs) >= batch_size:
                _flush_batch()
            return True

        lists = [per_off[w] for w in offdiag_needed]
        for combo in itertools.product(*lists):
            if time.monotonic() > deadline:
                break
            g12 = g13 = g23 = 0.0
            extra_hkl: list[tuple[int, int, int]] = []
            for which, (val, zhkl) in zip(offdiag_needed, combo, strict=True):
                if which == "g12":
                    g12 = val
                elif which == "g13":
                    g13 = val
                else:
                    g23 = val
                if zhkl != (0, 0, 0):
                    extra_hkl.append(zhkl)
            batch_gvecs.append(np.array([g11, g22, g33, g12, g13, g23], dtype=np.float64))
            batch_meta.append(axial_hkl + tuple(extra_hkl))
            if len(batch_gvecs) >= batch_size:
                _flush_batch()
                if len(candidates) >= collect_budget:
                    return False
        return True

    for (i1, h, g11), (i2, k, g22), (i3, l, g33) in diag_seed_iter:
        if not _consume_diag_seed(i1, h, g11, i2, k, g22, i3, l, g33):
            break
    _flush_batch()
    # Triclinic pass-2 only if pass-1 produced no keepers.
    if system == "triclinic" and not candidates and tric_pass2_iter is not None:
        for (i1, h, g11), (i2, k, g22), (i3, l, g33) in tric_pass2_iter:
            if not _consume_diag_seed(i1, h, g11, i2, k, g22, i3, l, g33):
                break
        _flush_batch()

    candidates.sort(key=lambda c: (-c.n_matched, c.volume))
    return candidates[:pool_budget]


@dataclass
class QSearchCandidate:
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    n_matched: int
    n_peaks: int
    volume: float
    hkl_used: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)

    def as_params6(self) -> list[float]:
        return [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]

    def niggli_params6(self) -> list[float]:
        matrix = lattice_params_to_matrix(torch.tensor(self.as_params6(), dtype=torch.float64)).numpy()
        canon = canonicalize_lattice(matrix, convention="niggli")
        return canon.as_params6()


def search_crystal_system(
    observed_two_theta: Sequence[float] | np.ndarray,
    system: str,
    *,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    max_hkl_index: int = 3,
    dense_max_nonzero: int = 3,
    sparse_hkl_index: int | None = None,
    sparse_max_nonzero: int = 2,
    n_low_peaks: int | None = None,
    match_fraction_min: float = 0.95,
    q_match_abs_tol: float = 1e-6,
    max_combo_trials: int = 20_000_000,
    pool_budget: int = 50,
    time_budget_s: float = 20.0,
    confirm_max_index: int | None = None,
) -> list[QSearchCandidate]:
    """Independent candidate search for one crystal system (no NN dependence).

    ``n_low_peaks``: size of the low-angle peak subset to draw the ``k``
    hkl-assigned peaks from (defaults to ``k + 3``, capped by available peaks).
    ``match_fraction_min``: fraction of *all* observed peaks a candidate must
    explain (de Wolff-style match) to be kept -- this is the consistency
    filter that rejects spurious hkl assignments.
    """
    if system in ("monoclinic", "cubic", "trigonal"):
        # Peaks alone can't reveal which Bravais-centering/unique-axis /
        # hex-vs-rhomb setting a (possibly Niggli-reduced) cell used -- try
        # all internal sub-variants, splitting the time budget, and merge.
        if system == "monoclinic":
            variants = _MONOCLINIC_VARIANTS
        elif system == "cubic":
            variants = _CUBIC_VARIANTS
        else:
            variants = _TRIGONAL_VARIANTS
        per_variant_budget = time_budget_s / len(variants)
        merged: list[QSearchCandidate] = []
        deadline_all = time.monotonic() + time_budget_s
        for variant in variants:
            remaining = deadline_all - time.monotonic()
            if remaining <= 0:
                break
            variant_candidates = search_crystal_system(
                observed_two_theta,
                variant,
                wavelength_angstrom=wavelength_angstrom,
                max_hkl_index=max_hkl_index,
                dense_max_nonzero=dense_max_nonzero,
                sparse_hkl_index=sparse_hkl_index,
                sparse_max_nonzero=sparse_max_nonzero,
                n_low_peaks=n_low_peaks,
                match_fraction_min=match_fraction_min,
                q_match_abs_tol=q_match_abs_tol,
                max_combo_trials=max_combo_trials,
                pool_budget=pool_budget,
                time_budget_s=min(per_variant_budget, remaining),
                confirm_max_index=confirm_max_index,
            )
            for cand in variant_candidates:
                cand.crystal_system = system
            merged.extend(variant_candidates)
        merged.sort(key=lambda c: (-c.n_matched, c.volume))
        return merged[:pool_budget]

    if system in _SEQUENTIAL_SYSTEMS:
        return _search_sequential(
            observed_two_theta,
            system,
            wavelength_angstrom=wavelength_angstrom,
            axial_max_index=max(sparse_hkl_index or 0, 8),
            zone_max_index=max(max_hkl_index, 4),
            n_low_peaks=n_low_peaks,
            match_fraction_min=match_fraction_min,
            q_match_abs_tol=q_match_abs_tol,
            pool_budget=pool_budget,
            time_budget_s=time_budget_s,
            confirm_max_index=confirm_max_index,
        )

    k = CRYSTAL_SYSTEM_DOF[system]
    basis = _basis(system)

    two_theta = np.asarray(observed_two_theta, dtype=np.float64)
    two_theta = np.sort(two_theta[np.isfinite(two_theta)])
    if two_theta.shape[0] < k:
        return []
    q_obs = inverse_d2_from_two_theta_f64(two_theta, wavelength_angstrom=wavelength_angstrom)

    if n_low_peaks is None:
        n_low_peaks = min(two_theta.shape[0], k + 3)
    n_low_peaks = max(k, min(n_low_peaks, two_theta.shape[0]))

    hkl_pool = _tiered_hkl_pool(
        dense_max_index=max_hkl_index,
        dense_max_nonzero=dense_max_nonzero,
        sparse_max_index=sparse_hkl_index,
        sparse_max_nonzero=sparse_max_nonzero,
    )
    pool_arr = np.array(hkl_pool, dtype=np.float64)  # (M, 3)
    # Row of the linear system per pool hkl, already projected onto the free
    # params (basis): shape (M, k).
    coeff_rows = np.stack(
        [pool_arr[:, 0] ** 2, pool_arr[:, 1] ** 2, pool_arr[:, 2] ** 2,
         2 * pool_arr[:, 0] * pool_arr[:, 1], 2 * pool_arr[:, 0] * pool_arr[:, 2], 2 * pool_arr[:, 1] * pool_arr[:, 2]],
        axis=1,
    )  # (M, 6)
    basis_rows = coeff_rows @ basis  # (M, k)

    # Fixed, generous grid (independent of any one candidate's cell size) used
    # only for the cheap vectorized pre-filter below -- deliberately wider
    # than the solving pool so it doesn't reject true cells whose higher-angle
    # peaks need larger/denser hkl than the (combinatorially-constrained)
    # solving pool covers.
    if confirm_max_index is None:
        confirm_max_index = min(16, max(max_hkl_index, sparse_hkl_index or 0) + 6)
    confirm_pool_arr = np.array(_hkl_pool(confirm_max_index, max_nonzero=3), dtype=np.float64)
    confirm_coeff_rows = np.stack(
        [
            confirm_pool_arr[:, 0] ** 2, confirm_pool_arr[:, 1] ** 2, confirm_pool_arr[:, 2] ** 2,
            2 * confirm_pool_arr[:, 0] * confirm_pool_arr[:, 1],
            2 * confirm_pool_arr[:, 0] * confirm_pool_arr[:, 2],
            2 * confirm_pool_arr[:, 1] * confirm_pool_arr[:, 2],
        ],
        axis=1,
    )  # (M_confirm, 6)

    peak_combos = list(itertools.combinations(range(n_low_peaks), k))
    peak_combos.sort(key=sum)  # prefer combos using the lowest-index peaks first

    seen_lattices: set[tuple[float, float, float, float, float, float]] = set()
    candidates: list[QSearchCandidate] = []
    n_peaks_total = two_theta.shape[0]
    min_matched = int(np.ceil(match_fraction_min * n_peaks_total))
    M = len(hkl_pool)

    # Many observed-peak subsets are also satisfied by integer super-/sub-cell
    # duplicates of the true cell (e.g. b'=5b reproduces every b-axis line via
    # a 5x larger hkl) -- these have identical n_matched but larger volume, so
    # the final (-n_matched, volume) sort *would* rank the true minimal cell
    # first, but only if it was actually collected. Collect well past the
    # requested ``pool_budget`` internally so cheap high-volume duplicates
    # can't crowd out the true cell before it's ever found, then trim to
    # ``pool_budget`` after the final sort.
    collect_budget = max(pool_budget * 10, 300)

    start = time.monotonic()
    trials = 0
    for peak_idx in peak_combos:
        if len(candidates) >= collect_budget or time.monotonic() - start > time_budget_s:
            break
        rhs_single = q_obs[list(peak_idx)]  # (k,)
        # A few low-angle peaks *not* used to solve this combo -- stage-A
        # confirmation set (see below).
        confirm_stage_a_peaks = q_obs[[i for i in range(n_low_peaks) if i not in peak_idx]][:4]

        # All M^k hkl-index assignments for this peak combo, batched & chunked.
        index_grids = np.meshgrid(*[np.arange(M) for _ in range(k)], indexing="ij")
        flat_indices = [g.reshape(-1) for g in index_grids]
        n_total = flat_indices[0].shape[0]
        chunk_size = max(1, min(n_total, max(1, max_combo_trials // max(1, len(peak_combos)))))
        # Cap per-chunk element budget (~ chunk_size * k^2 matrix entries)
        # so a single un-interruptible det/solve/eigvalsh/inv batch can't
        # itself blow past the time budget for larger-k systems.
        chunk_size = min(chunk_size, max(1, 4_000_000 // max(k * k, 1)))

        for chunk_start in range(0, n_total, chunk_size):
            if time.monotonic() - start > time_budget_s or len(candidates) >= collect_budget:
                break
            chunk_end = min(chunk_start + chunk_size, n_total)
            trials += chunk_end - chunk_start
            idx_chunk = [fi[chunk_start:chunk_end] for fi in flat_indices]

            # A[:, p, :] = basis_rows[idx_chunk[p]] -> batched (n, k, k) system.
            A = np.stack([basis_rows[idx_chunk[p]] for p in range(k)], axis=1)
            rhs = np.broadcast_to(rhs_single, (A.shape[0], k))

            if k == 1:
                valid = np.abs(A[:, 0, 0]) > 1e-12
            else:
                det = np.linalg.det(A)
                scale = np.max(np.abs(A), axis=(1, 2)) ** k
                valid = np.isfinite(det) & (np.abs(det) > 1e-9 * np.maximum(scale, 1e-30))
            if not np.any(valid):
                continue

            A_v = A[valid]
            rhs_v = rhs[valid]
            hkl_idx_v = [idx[valid] for idx in idx_chunk]
            try:
                x = np.linalg.solve(A_v, rhs_v) if k > 1 else (rhs_v[:, 0] / A_v[:, 0, 0])[:, None]
            except np.linalg.LinAlgError:
                continue

            gvec = x @ basis.T  # (n_valid, 6)
            gstar_batch = np.zeros((gvec.shape[0], 3, 3), dtype=np.float64)
            gstar_batch[:, 0, 0] = gvec[:, 0]
            gstar_batch[:, 1, 1] = gvec[:, 1]
            gstar_batch[:, 2, 2] = gvec[:, 2]
            gstar_batch[:, 0, 1] = gstar_batch[:, 1, 0] = gvec[:, 3]
            gstar_batch[:, 0, 2] = gstar_batch[:, 2, 0] = gvec[:, 4]
            gstar_batch[:, 1, 2] = gstar_batch[:, 2, 1] = gvec[:, 5]

            eigvals = np.linalg.eigvalsh(gstar_batch)
            spd_mask = eigvals[:, 0] > 1e-10
            if not np.any(spd_mask):
                continue
            gstar_spd = gstar_batch[spd_mask]
            gvec_spd = gvec[spd_mask]
            hkl_idx_spd = [idx[spd_mask] for idx in hkl_idx_v]

            g_direct = np.linalg.inv(gstar_spd)
            lengths = np.sqrt(np.clip(np.diagonal(g_direct, axis1=1, axis2=2), 1e-18, None))
            a_arr, b_arr, c_arr = lengths[:, 0], lengths[:, 1], lengths[:, 2]
            len_ok = (a_arr > 0.5) & (a_arr < 100) & (b_arr > 0.5) & (b_arr < 100) & (c_arr > 0.5) & (c_arr < 100)

            cos_alpha = np.clip(g_direct[:, 1, 2] / (b_arr * c_arr + 1e-18), -0.999999, 0.999999)
            cos_beta = np.clip(g_direct[:, 0, 2] / (a_arr * c_arr + 1e-18), -0.999999, 0.999999)
            cos_gamma = np.clip(g_direct[:, 0, 1] / (a_arr * b_arr + 1e-18), -0.999999, 0.999999)
            alpha_arr = np.degrees(np.arccos(cos_alpha))
            beta_arr = np.degrees(np.arccos(cos_beta))
            gamma_arr = np.degrees(np.arccos(cos_gamma))
            ang_ok = (
                (alpha_arr > 20) & (alpha_arr < 160)
                & (beta_arr > 20) & (beta_arr < 160)
                & (gamma_arr > 20) & (gamma_arr < 160)
            )
            keep = len_ok & ang_ok
            if not np.any(keep):
                continue

            survivors = np.where(keep)[0]
            if survivors.size == 0:
                continue

            # Stage-A: near-free coarse filter. SPD+bounds alone typically
            # leaves 1e4-1e5 "survivors" per peak-pair (see B1-S0 profiling),
            # far too many to run the (M_confirm x N) stage-B check on
            # directly. Reuse the small *solving* pool (already computed,
            # M ~= a few hundred) against a handful of extra low-angle peaks
            # not used in this solve -- true cells match these almost always;
            # wrong hkl assignments rarely match even one by coincidence at
            # 1e-6 tol, so this alone prunes >99% of survivors for ~1% of the
            # cost of the wide stage-B grid.
            deadline = start + time_budget_s
            if confirm_stage_a_peaks.size > 0:
                stage_a_counts = _approx_match_counts(
                    gvec_spd[survivors],
                    coeff_rows,
                    confirm_stage_a_peaks,
                    q_match_abs_tol=q_match_abs_tol,
                    deadline=deadline,
                )
                survivors = survivors[stage_a_counts >= confirm_stage_a_peaks.size]
                if survivors.size == 0:
                    continue
            if time.monotonic() > deadline:
                break

            # Stage-B: wider grid, all observed peaks, exact threshold --
            # still cheap now that stage-A has cut survivors by >=100x.
            approx_counts = _approx_match_counts(
                gvec_spd[survivors], confirm_coeff_rows, q_obs, q_match_abs_tol=q_match_abs_tol, deadline=deadline
            )
            promising = np.where(approx_counts >= min_matched)[0]
            if promising.size == 0:
                continue
            # Best-approx-match first, so the collect_budget/time_budget cutoff
            # below keeps the most promising candidates.
            promising = promising[np.argsort(-approx_counts[promising])]
            survivors = survivors[promising]

            for i in survivors:
                if len(candidates) >= collect_budget or time.monotonic() - start > time_budget_s:
                    break
                gstar = gstar_spd[i]
                lattice_params = (
                    float(a_arr[i]), float(b_arr[i]), float(c_arr[i]),
                    float(alpha_arr[i]), float(beta_arr[i]), float(gamma_arr[i]),
                )
                rounded = tuple(round(v, 4) for v in lattice_params)
                if rounded in seen_lattices:
                    continue
                n_matched = _fast_match_count(q_obs, gstar, q_match_abs_tol=q_match_abs_tol)
                if n_matched < min_matched:
                    continue
                seen_lattices.add(rounded)
                volume = float(1.0 / np.sqrt(max(np.linalg.det(gstar), 1e-30)))
                hkl_used = tuple(tuple(int(v) for v in hkl_pool[hkl_idx_spd[p][i]]) for p in range(k))
                candidates.append(
                    QSearchCandidate(
                        crystal_system=system,
                        a=lattice_params[0],
                        b=lattice_params[1],
                        c=lattice_params[2],
                        alpha=lattice_params[3],
                        beta=lattice_params[4],
                        gamma=lattice_params[5],
                        n_matched=n_matched,
                        n_peaks=n_peaks_total,
                        volume=volume,
                        hkl_used=hkl_used,
                    )
                )
            if trials >= max_combo_trials:
                break
        if trials >= max_combo_trials:
            break

    candidates.sort(key=lambda c: (-c.n_matched, c.volume))
    return candidates[:pool_budget]


DEFAULT_SEARCH_KWARGS: dict[str, dict] = {
    "cubic": dict(max_hkl_index=6, n_low_peaks=4, pool_budget=30, time_budget_s=5.0),
    "tetragonal": dict(max_hkl_index=4, n_low_peaks=6, pool_budget=30, time_budget_s=15.0),
    "hexagonal": dict(max_hkl_index=4, n_low_peaks=6, pool_budget=30, time_budget_s=15.0),
    # Split across trigonal_hex / trigonal_rhomb (~10s each).
    "trigonal": dict(max_hkl_index=4, n_low_peaks=6, pool_budget=30, time_budget_s=20.0),
    # Sequential path: sparse_hkl_index → axial_max_index, max_hkl_index → zone_max_index.
    "orthorhombic": dict(
        max_hkl_index=3,
        dense_max_nonzero=2,
        sparse_hkl_index=6,
        sparse_max_nonzero=1,
        n_low_peaks=6,
        pool_budget=30,
        time_budget_s=10.0,
    ),
    "monoclinic": dict(
        max_hkl_index=3,
        dense_max_nonzero=2,
        sparse_hkl_index=6,
        sparse_max_nonzero=1,
        n_low_peaks=7,
        pool_budget=30,
        # Split 3-way across monoclinic_a/b/c (~12s/variant).
        time_budget_s=36.0,
    ),
    "triclinic": dict(
        max_hkl_index=4,
        dense_max_nonzero=1,
        sparse_hkl_index=6,
        sparse_max_nonzero=1,
        n_low_peaks=12,
        pool_budget=30,
        time_budget_s=20.0,
        match_fraction_min=0.80,
    ),
}


def search_all_systems(
    observed_two_theta: Sequence[float] | np.ndarray,
    *,
    systems: Sequence[str] | None = None,
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    match_fraction_min: float = 0.95,
    overrides: dict[str, dict] | None = None,
) -> dict[str, list[QSearchCandidate]]:
    """Run the independent q-search across crystal systems, return per-system pools."""
    systems = systems or list(CRYSTAL_SYSTEM_DOF.keys())
    overrides = overrides or {}
    out: dict[str, list[QSearchCandidate]] = {}
    for system in systems:
        kwargs = {"match_fraction_min": match_fraction_min}
        kwargs.update(DEFAULT_SEARCH_KWARGS.get(system, {}))
        kwargs.update(overrides.get(system, {}))
        out[system] = search_crystal_system(
            observed_two_theta,
            system,
            wavelength_angstrom=wavelength_angstrom,
            **kwargs,
        )
    return out
