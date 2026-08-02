#!/usr/bin/env python3
"""Train / select V0 reranker on synthetic rr_train / rr_dev only (P2).

Never reads MP100 or CNRS. Two arms:
  1. Linear combo  score = w1·p_mcm + w2·p_nn_dist − w3·vol_dev
     Weights chosen by Top-1 on rr_dev (the only legitimate place to tune).
  2. XGBRanker pairwise on the same three (+ a few raw) features.

The winner on rr_dev is written to ``--out/model.json``. Freeze-eval (P3) is a
separate script that loads this file once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="dir with rr_train/rr_dev.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def top1_rate(samples: list[dict], scorer) -> float:
    hit = 0
    n = 0
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue
        n += 1
        scores = scorer(cands)
        j = int(np.argmax(scores))
        if cands[j]["is_hit"]:
            hit += 1
    return hit / max(n, 1)


def mrr(samples: list[dict], scorer) -> float:
    total = 0.0
    n = 0
    for s in samples:
        cands = s["candidates"]
        if not cands or not any(c["is_hit"] for c in cands):
            continue
        n += 1
        scores = scorer(cands)
        order = np.argsort(-scores)
        for rank, idx in enumerate(order, 1):
            if cands[int(idx)]["is_hit"]:
                total += 1.0 / rank
                break
    return total / max(n, 1)


def lib_rate(samples: list[dict]) -> float:
    return sum(1 for s in samples if s["lib_strict"]) / max(len(samples), 1)


def mcm20_top1(samples: list[dict]) -> float:
    return sum(1 for s in samples if s.get("first_mcm20") == 1) / max(len(samples), 1)


def linear_scorer(w1, w2, w3):
    def score(cands):
        return np.asarray(
            [
                w1 * c["p_mcm"] + w2 * c["p_nn_dist"] - w3 * c["vol_dev"]
                for c in cands
            ],
            dtype=float,
        )

    return score


def select_linear(dev: list[dict]) -> dict:
    # Require both McM20 and NN-distance terms (P0 showed both matter;
    # grids that zero either tend to overfit easy synthetic subsets).
    grid = [
        (a, b, c)
        for a in (0.25, 0.5, 0.75, 1.0)
        for b in (0.25, 0.5, 0.75, 1.0)
        for c in (0.0, 0.25, 0.5)
    ]
    best = None
    rows = []
    for w in grid:
        sc = linear_scorer(*w)
        t1 = top1_rate(dev, sc)
        mr = mrr(dev, sc)
        rows.append({"w": w, "top1": t1, "mrr": mr})
        if best is None or (t1, mr) > (best["top1"], best["mrr"]):
            best = {"w": w, "top1": t1, "mrr": mr}
    return {"best": best, "grid": rows}


FEATURE_NAMES = ["p_mcm", "p_nn_dist", "vol_dev", "McM20", "Rp", "n_indexed", "nn_dist"]


def build_rank_matrix(samples: list[dict]):
    Xs, ys, groups = [], [], []
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue
        X = np.asarray([[c[f] for f in FEATURE_NAMES] for c in cands], dtype=float)
        y = np.asarray([int(c["is_hit"]) for c in cands], dtype=int)
        # Skip groups with no positive — XGBRanker can't learn from them.
        if y.sum() == 0:
            continue
        Xs.append(X)
        ys.append(y)
        groups.append(len(cands))
    if not Xs:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), []
    return np.vstack(Xs), np.concatenate(ys), groups


def train_xgb(train: list[dict], dev: list[dict]) -> dict:
    import xgboost as xgb

    Xtr, ytr, gtr = build_rank_matrix(train)
    Xdv, ydv, gdv = build_rank_matrix(dev)
    if Xtr.size == 0 or Xdv.size == 0:
        return {"ok": False, "reason": "empty matrix"}

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=8,
        tree_method="hist",
        early_stopping_rounds=40,
    )
    model.fit(
        Xtr,
        ytr,
        group=gtr,
        eval_set=[(Xdv, ydv)],
        eval_group=[gdv],
        verbose=False,
    )

    def scorer(cands):
        X = np.asarray([[c[f] for f in FEATURE_NAMES] for c in cands], dtype=float)
        return model.predict(X)

    return {
        "ok": True,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "top1_dev": top1_rate(dev, scorer),
        "mrr_dev": mrr(dev, scorer),
        "feature_names": FEATURE_NAMES,
        "booster": model,
    }


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(data / "rr_train.jsonl")
    dev = load_jsonl(data / "rr_dev.jsonl")
    print(f"train={len(train)}  dev={len(dev)}", flush=True)
    print(
        f"baselines  train top1_mcm20={mcm20_top1(train):.1%} lib={lib_rate(train):.1%}  "
        f"dev top1_mcm20={mcm20_top1(dev):.1%} lib={lib_rate(dev):.1%}",
        flush=True,
    )

    print("==== linear weight search on rr_dev ====", flush=True)
    lin = select_linear(dev)
    w1, w2, w3 = lin["best"]["w"]
    print(
        f"best linear w=({w1}, {w2}, {w3})  "
        f"dev top1={lin['best']['top1']:.1%} mrr={lin['best']['mrr']:.3f}",
        flush=True,
    )
    # Also report the untuned equal-weight reference for honesty.
    eq = linear_scorer(1.0, 1.0, 0.25)
    print(
        f"equal-weight (1,1,0.25)  "
        f"dev top1={top1_rate(dev, eq):.1%} mrr={mrr(dev, eq):.3f}",
        flush=True,
    )

    print("==== XGBRanker ====", flush=True)
    xgb_res = train_xgb(train, dev)
    booster_path = None
    if xgb_res.get("ok"):
        booster_path = str(out / "xgbranker.json")
        xgb_res["booster"].save_model(booster_path)
        print(
            f"xgb  iter={xgb_res['best_iteration']}  "
            f"dev top1={xgb_res['top1_dev']:.1%} mrr={xgb_res['mrr_dev']:.3f}",
            flush=True,
        )
        # Compare on train too (sanity, not for selection).
        def xgb_sc(cands, m=xgb_res["booster"]):
            X = np.asarray([[c[f] for f in FEATURE_NAMES] for c in cands], dtype=float)
            return m.predict(X)

        print(
            f"xgb  train top1={top1_rate(train, xgb_sc):.1%}  "
            f"linear train top1={top1_rate(train, linear_scorer(w1,w2,w3)):.1%}",
            flush=True,
        )
    else:
        print(f"xgb skipped: {xgb_res.get('reason')}", flush=True)

    # Winner on rr_dev Top-1 (then MRR).
    candidates = [
        {
            "name": "linear",
            "top1": lin["best"]["top1"],
            "mrr": lin["best"]["mrr"],
            "config": {"type": "linear", "w": [w1, w2, w3]},
        },
        {
            "name": "equal_weight",
            "top1": top1_rate(dev, eq),
            "mrr": mrr(dev, eq),
            "config": {"type": "linear", "w": [1.0, 1.0, 0.25]},
        },
        {
            "name": "mcm20",
            "top1": mcm20_top1(dev),
            "mrr": mrr(dev, lambda c: np.asarray([x["McM20"] for x in c])),
            "config": {"type": "mcm20"},
        },
    ]
    if xgb_res.get("ok"):
        candidates.append(
            {
                "name": "xgbranker",
                "top1": xgb_res["top1_dev"],
                "mrr": xgb_res["mrr_dev"],
                "config": {
                    "type": "xgbranker",
                    "path": booster_path,
                    "feature_names": FEATURE_NAMES,
                    "best_iteration": xgb_res["best_iteration"],
                },
            }
        )
    winner = max(candidates, key=lambda c: (c["top1"], c["mrr"]))
    print(
        f"WINNER on rr_dev: {winner['name']}  "
        f"top1={winner['top1']:.1%} mrr={winner['mrr']:.3f}",
        flush=True,
    )

    model = {
        "protocol": {
            "data": str(data),
            "n_train": len(train),
            "n_dev": len(dev),
            "selection_metric": "rr_dev top1 then mrr",
            "forbidden": ["MP100", "CNRS"],
        },
        "baselines": {
            "dev_mcm20_top1": mcm20_top1(dev),
            "dev_lib": lib_rate(dev),
            "train_mcm20_top1": mcm20_top1(train),
            "train_lib": lib_rate(train),
        },
        "linear_search": {
            "best": lin["best"],
            "equal_weight": {"w": [1.0, 1.0, 0.25], "top1": top1_rate(dev, eq)},
        },
        "xgb": {
            k: v
            for k, v in xgb_res.items()
            if k != "booster"
        },
        "candidates": [
            {k: v for k, v in c.items() if k != "config"} | {"config": c["config"]}
            for c in candidates
        ],
        "winner": winner,
    }
    (out / "model.json").write_text(json.dumps(model, indent=2))
    (out / "linear_grid.json").write_text(json.dumps(lin["grid"], indent=2))
    print(f"wrote {out / 'model.json'}", flush=True)


if __name__ == "__main__":
    main()
