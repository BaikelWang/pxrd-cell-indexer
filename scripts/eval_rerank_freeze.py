#!/usr/bin/env python3
"""P3 freeze-eval: apply a locked reranker once to frozen MP100/CNRS pools.

Loads ``model.json`` produced by ``train_rerank_v0.py``. No weight search,
no early stopping, no threshold tuning — just score and report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402

CNRS_DIR = Path("/nanolab/users/wyx/CNRS")
KS = [1, 5, 10, 20]

ALLCELLS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)

POOLS = {
    "cnrs_k1000": (
        "cnrs",
        ROOT / "results/flow_seedgen/cnrs_e2e_multithresh_v4_k1000/indexer_union",
    ),
    "cnrs_k100": (
        "cnrs",
        ROOT / "results/flow_seedgen/cnrs_e2e_multithresh_v4/indexer_union",
    ),
    "mp100_k1000": (
        "mp100",
        ROOT / "third_party/McMaille/run_lab/mp100_reseed_v4_k1000_native_lsq",
    ),
    "mp100_k100": (
        "mp100",
        ROOT / "third_party/McMaille/run_lab/mp100_reseed_v4_k100_native_lsq",
    ),
}


def parse_allcells_full(path: Path) -> list[dict]:
    out = []
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
    rows = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append([float(x) for x in parts[:6]])
        except ValueError:
            continue
    return np.asarray(rows) if rows else np.zeros((0, 6))


def gstar6(params) -> np.ndarray:
    from pymatgen.core import Lattice

    lat = Lattice.from_parameters(*[float(x) for x in params[:6]])
    rm = lat.reciprocal_lattice.matrix
    g = rm @ rm.T
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[1, 2], g[0, 2], g[0, 1]], dtype=float)


def cell_volume(params) -> float:
    from pymatgen.core import Lattice

    return float(Lattice.from_parameters(*[float(x) for x in params[:6]]).volume)


def pct_rank(values: np.ndarray, higher_better: bool = True) -> np.ndarray:
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


def truth_prim(dataset: str, sid: str):
    cif = CIF_DIR / f"{sid}.cif" if dataset == "mp100" else CNRS_DIR / f"{sid}_sg.cif"
    return truth_cells(cif)["prim"]


def build_features(rows: list[dict], seeds: np.ndarray) -> list[dict]:
    seed_g = np.asarray([gstar6(s) for s in seeds]) if seeds.size else np.zeros((0, 6))
    v_nn = float(np.median([cell_volume(s) for s in seeds])) if seeds.size else 0.0
    feats = []
    for r in rows:
        vol = r["volume"] if r["volume"] > 0 else max(cell_volume(r["params"]), 1e-6)
        vol_dev = abs(np.log(vol / v_nn)) if v_nn > 0 else 0.0
        nn_dist = (
            float(np.min(np.linalg.norm(seed_g - gstar6(r["params"]), axis=1)))
            if seed_g.size
            else 0.0
        )
        feats.append(
            {
                **r,
                "nn_dist": nn_dist,
                "vol_dev": float(min(vol_dev, 3.0) / 3.0),
            }
        )
    if feats:
        p_mcm = pct_rank(np.asarray([f["McM20"] for f in feats]))
        p_nn = pct_rank(np.asarray([f["nn_dist"] for f in feats]), False)
        for i, f in enumerate(feats):
            f["p_mcm"] = float(p_mcm[i])
            f["p_nn_dist"] = float(p_nn[i])
    return feats


def make_scorer(model: dict):
    cfg = model["winner"]["config"]
    if cfg["type"] == "mcm20":
        return lambda feats: np.asarray([f["McM20"] for f in feats], dtype=float)
    if cfg["type"] == "linear":
        w1, w2, w3 = cfg["w"]

        def score(feats):
            return np.asarray(
                [w1 * f["p_mcm"] + w2 * f["p_nn_dist"] - w3 * f["vol_dev"] for f in feats],
                dtype=float,
            )

        return score
    if cfg["type"] == "xgbranker":
        import xgboost as xgb

        booster = xgb.XGBRanker()
        booster.load_model(cfg["path"])
        names = cfg["feature_names"]

        def score(feats):
            X = np.asarray([[f[n] for n in names] for f in feats], dtype=float)
            return booster.predict(X)

        return score
    raise ValueError(cfg["type"])


def eval_sid(job) -> dict:
    sid, run_dir, dataset, scorer_cfg = job
    d = Path(run_dir) / sid
    stem = sid.replace("-", "_")
    allc = d / f"{stem}.allcells"
    empty = {"lib": False, "first": None}
    res = {"sample_id": sid, "n_pool": 0, "mcm20": dict(empty), "rerank": dict(empty)}
    if not allc.exists():
        return res
    rows = sorted(parse_allcells_full(allc), key=lambda r: -r["McM20"])
    if not rows:
        return res
    seeds = read_seeds(d / f"{stem}.seed") if (d / f"{stem}.seed").exists() else np.zeros((0, 6))
    feats = build_features(rows, seeds)
    truth = truth_prim(dataset, sid)
    is_hit = [bool(l4(r["params"], truth)[1]) for r in rows]

    def first_of(order):
        return next((i for i, j in enumerate(order, 1) if is_hit[j]), None)

    mcm_order = list(range(len(rows)))
    model = {"winner": {"config": scorer_cfg}}
    scorer = make_scorer(model)
    scores = scorer(feats)
    rr_order = list(np.argsort(-np.asarray(scores, dtype=float)))
    assert sorted(rr_order) == list(range(len(rows)))

    res["n_pool"] = len(rows)
    res["mcm20"] = {"lib": any(is_hit), "first": first_of(mcm_order)}
    res["rerank"] = {"lib": any(is_hit), "first": first_of(rr_order)}
    return res


def summarize(rows: list[dict], key: str) -> dict:
    n = max(len(rows), 1)
    firsts = [r[key]["first"] for r in rows]
    return {
        "lib_strict": sum(1 for r in rows if r[key]["lib"]) / n,
        "topk_strict": {str(k): sum(1 for f in firsts if f and f <= k) / n for k in KS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--pools", nargs="*", default=list(POOLS))
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text())
    scorer_cfg = model["winner"]["config"]
    print(f"locked winner: {model['winner']['name']}  cfg={scorer_cfg}", flush=True)

    report = {"model": str(args.model), "winner": model["winner"], "pools": {}}
    for name in args.pools:
        dataset, run_dir = POOLS[name]
        prefix = "mp-" if dataset == "mp100" else ""
        sids = sorted(
            p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
        )
        if dataset == "cnrs":
            sids = [s for s in sids if s.isdigit()]
        jobs = [(s, str(run_dir), dataset, scorer_cfg) for s in sids]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(eval_sid, j) for j in jobs]
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: r["sample_id"])
        base = summarize(rows, "mcm20")
        rr = summarize(rows, "rerank")
        assert abs(base["lib_strict"] - rr["lib_strict"]) < 1e-12
        report["pools"][name] = {
            "n": len(rows),
            "mcm20": base,
            "rerank": rr,
            "delta_top1": rr["topk_strict"]["1"] - base["topk_strict"]["1"],
        }
        print(
            f"{name:<14}  mcm20={base['topk_strict']['1']:.1%}  "
            f"rerank={rr['topk_strict']['1']:.1%}  "
            f"Δ={rr['topk_strict']['1']-base['topk_strict']['1']:+.1%}  "
            f"lib={rr['lib_strict']:.1%}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
