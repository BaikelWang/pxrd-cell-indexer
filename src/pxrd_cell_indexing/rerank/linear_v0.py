"""V0 linear reranker over seeded-McMaille ``.allcells`` rows.

score = w1·McM20_pct + w2·nn_dist_pct − w3·vol_dev

Default weights are the untuned equal-weight point (1, 1, 0.25) locked after
the 2026-07-31 freeze eval. See ``docs/实验记录/20260731-Reranker-V0训练与冻结评测.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DEFAULT_W = (1.0, 1.0, 0.25)

ALLCELLS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)


def parse_allcells_full(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = ALLCELLS_ROW.match(line)
        if not m:
            continue
        g = m.groups()
        out.append(
            {
                "seed_src": int(g[1]),
                "stage": int(g[2]),
                "n_indexed": int(g[3]),
                "McM20": float(g[4]),
                "volume": float(g[5]),
                "Rp": float(g[6]),
                "params": [float(g[i]) for i in range(7, 13)],
                "bravais": g[13],
            }
        )
    return out


def read_seeds(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 6))
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:6]])
        except ValueError:
            continue
    return np.asarray(rows) if rows else np.zeros((0, 6))


def _gstar6(params) -> np.ndarray:
    from pymatgen.core import Lattice

    lat = Lattice.from_parameters(*[float(x) for x in params[:6]])
    rm = lat.reciprocal_lattice.matrix
    g = rm @ rm.T
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 2], g[0, 2], g[0, 1]], dtype=float)


def _volume(params) -> float:
    from pymatgen.core import Lattice

    return float(Lattice.from_parameters(*[float(x) for x in params[:6]]).volume)


def _pct_rank(values: np.ndarray, higher_better: bool = True) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size <= 1:
        return np.ones_like(v)
    finite = np.isfinite(v)
    fill = np.nanmin(v[finite]) if finite.any() else 0.0
    v = np.where(finite, v, fill)
    order = np.argsort(-v if higher_better else v)
    out = np.empty_like(v)
    out[order] = np.linspace(1.0, 0.0, v.size)
    return out


def order_allcells(
    allcells_path: Path,
    seed_path: Path | None = None,
    *,
    w: tuple[float, float, float] = DEFAULT_W,
) -> list[list[float]]:
    """Return candidate lattices reordered by the V0 linear score.

    If ``seed_path`` is omitted, looks for ``<stem>.seed`` next to ``.allcells``.
    """
    rows = sorted(parse_allcells_full(allcells_path), key=lambda r: -r["McM20"])
    if not rows:
        return []
    if seed_path is None:
        stem = allcells_path.name.replace(".allcells", "")
        seed_path = allcells_path.with_name(f"{stem}.seed")
    seeds = read_seeds(seed_path)
    seed_g = np.asarray([_gstar6(s) for s in seeds]) if seeds.size else np.zeros((0, 6))
    v_nn = float(np.median([_volume(s) for s in seeds])) if seeds.size else 0.0

    mcm = np.asarray([r["McM20"] for r in rows], dtype=float)
    nn_dist = np.zeros(len(rows), dtype=float)
    vol_dev = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        vol = r["volume"] if r["volume"] > 0 else max(_volume(r["params"]), 1e-6)
        vol_dev[i] = min(abs(np.log(vol / v_nn)) if v_nn > 0 else 0.0, 3.0) / 3.0
        if seed_g.size:
            nn_dist[i] = float(np.min(np.linalg.norm(seed_g - _gstar6(r["params"]), axis=1)))

    p_mcm = _pct_rank(mcm)
    p_nn = _pct_rank(nn_dist, higher_better=False)
    w1, w2, w3 = w
    scores = w1 * p_mcm + w2 * p_nn - w3 * vol_dev
    order = np.argsort(-scores)
    return [rows[int(i)]["params"] for i in order]
