#!/usr/bin/env python3
"""Evaluate a trained FT arm on MP100 under the project strict criterion."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice

from .common import (
    MP100_DIR,
    PROJECT,
    LatticeNormalizer,
    load_bert_from_ckpt,
    load_cspflow_from_ckpt,
    matrix_to_six,
)
from .models import ArmAModel, ArmBModel, ArmCModel
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
from remeasure_l4_prim_vs_conv import CIF_DIR, l4, truth_cells  # noqa: E402
from pxrd_cell_indexing.model.topk import (  # noqa: E402
    EXTENDED_LENGTH_SCALE_FACTORS,
    TopKConfig,
    build_top_k_candidates,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--infer-timesteps", type=int, default=200)
    return ap.parse_args()


def load_model(arm: str, ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = LatticeNormalizer.from_dict(ck["normalizer"])
    if arm == "A":
        bundle, hp, _, _ = load_cspflow_from_ckpt(device)
        model = ArmAModel(bundle, timesteps=hp["timesteps"]).to(device)
    elif arm == "B":
        enc, _, _ = load_bert_from_ckpt(device, continuous_pos=False)
        model = ArmBModel(enc, freeze_encoder=True).to(device)
    else:
        enc, _, _ = load_bert_from_ckpt(device, continuous_pos=True)
        model = ArmCModel(enc).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    return model, normalizer


def mp100_peaks(sid: str):
    """Ideal simulated peaks from CIF (same style as A2 eval)."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    s = __import__("pymatgen.core", fromlist=["Structure"]).Structure.from_file(
        str(CIF_DIR / f"{sid}.cif")
    )
    calc = XRDCalculator(wavelength="CuKa")
    pat = calc.get_pattern(s)
    xs, ys = [], []
    for x, y in zip(pat.x, pat.y):
        if y > 5:
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        xs, ys = [float(pat.x[0])], [100.0]
    px = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    py = torch.tensor(ys, dtype=torch.float32).view(-1, 1)
    return px, py, torch.tensor([len(xs)], dtype=torch.long)


@torch.no_grad()
def predict_pool(arm, model, normalizer, sid, top_k, device, infer_timesteps):
    px, py, pn = mp100_peaks(sid)
    px, py, pn = px.to(device), py.to(device), pn.to(device)
    if arm == "A":
        mats = model.sample_lattices(
            px, py, pn, num_evals=top_k, infer_timesteps=infer_timesteps
        )
        cands = []
        for i in range(mats.size(0)):
            try:
                cands.append(matrix_to_six(mats[i].cpu().numpy()))
            except Exception:
                continue
        raw = cands[0] if cands else [5, 5, 5, 90, 90, 90]
        return raw, cands[:top_k]
    pred_n = model(px, py, pn)[0].cpu()
    raw = normalizer.decode(pred_n.numpy())
    cfg = TopKConfig(k=top_k, length_scale_factors=EXTENDED_LENGTH_SCALE_FACTORS)
    cands_obj = build_top_k_candidates(
        torch.tensor(raw, dtype=torch.float32).view(1, 6),
        k=top_k,
        config=cfg,
    )[0]
    cands = [[c.a, c.b, c.c, c.alpha, c.beta, c.gamma] for c in cands_obj]
    return raw, cands


def score_pool(cands, truth_six):
    flags = [l4(c, truth_six) for c in cands]
    dets = [f[2] for f in flags if f[2] is not None]
    vols = []
    for c in cands:
        try:
            vols.append(Lattice.from_parameters(*c).volume)
        except Exception:
            pass
    tv = Lattice.from_parameters(*truth_six).volume
    return {
        "n": len(cands),
        "lib_loose": any(f[0] for f in flags),
        "lib_strict": any(f[1] for f in flags),
        "topk_strict": {
            str(k): any(f[1] for f in flags[:k]) for k in (1, 5, 10, 20, 50, 100)
        },
        "topk_loose": {
            str(k): any(f[0] for f in flags[:k]) for k in (1, 5, 10, 20, 50, 100)
        },
        "best_det": min(dets) if dets else None,
        "med_vol_ratio": (
            float(np.median([v / tv for v in vols])) if vols and tv > 0 else None
        ),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, normalizer = load_model(args.arm, args.ckpt, device)

    sids = sorted(p.stem for p in MP100_DIR.glob("mp-*.cif"))
    print(f"eval arm{args.arm} n={len(sids)} ckpt={args.ckpt}", flush=True)

    per = []
    for i, sid in enumerate(sids, 1):
        t = truth_cells(CIF_DIR / f"{sid}.cif")
        raw, cands = predict_pool(
            args.arm, model, normalizer, sid, args.top_k, device, args.infer_timesteps
        )
        row = {
            "sample_id": sid,
            "system": t["system"],
            "raw_pred": raw,
            "n_cands": len(cands),
            "prim": score_pool(cands, t["prim"]),
            "conv": score_pool(cands, t["conv"]),
            "candidates": cands,
        }
        per.append(row)
        if i % 10 == 0:
            print(f"{i}/{len(sids)}", flush=True)

    n = len(per)

    def rate(tag, key):
        return sum(1 for r in per if r[tag][key]) / n

    summary = {
        "arm": args.arm,
        "ckpt": str(args.ckpt),
        "n_samples": n,
        "top_k": args.top_k,
        "prim": {
            "lib_loose": rate("prim", "lib_loose"),
            "lib_strict": rate("prim", "lib_strict"),
            "topk_strict": {
                k: sum(r["prim"]["topk_strict"][k] for r in per) / n
                for k in ("1", "5", "10", "20", "50", "100")
            },
            "best_det_median": float(
                np.median(
                    [r["prim"]["best_det"] for r in per if r["prim"]["best_det"] is not None]
                )
            )
            if any(r["prim"]["best_det"] is not None for r in per)
            else None,
            "med_vol_ratio_median": float(
                np.median(
                    [
                        r["prim"]["med_vol_ratio"]
                        for r in per
                        if r["prim"]["med_vol_ratio"] is not None
                    ]
                )
            )
            if any(r["prim"]["med_vol_ratio"] is not None for r in per)
            else None,
        },
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # store pool without dumping huge candidates in summary-only? keep them for analysis
    args.out.write_text(
        json.dumps({"summary": summary, "per_sample": per}, indent=2)
    )
    # also write pool-only compact
    pool_path = args.out.with_name(args.out.stem.replace("strict_eval", "pool_k100") + ".json")
    if "strict_eval" not in args.out.stem:
        pool_path = args.out.with_name(args.out.stem + "_pool.json")
    pool_path.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "per_sample": {
                    r["sample_id"]: {
                        "raw_pred": r["raw_pred"],
                        "candidates": r["candidates"],
                    }
                    for r in per
                },
            }
        )
    )
    print(f"wrote {args.out}")
    print(f"wrote {pool_path}")


if __name__ == "__main__":
    main()
