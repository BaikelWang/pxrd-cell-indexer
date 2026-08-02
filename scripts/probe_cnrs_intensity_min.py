#!/usr/bin/env python3
"""Sweep the CNRS eval intensity threshold and measure raw seed-pool recall.

``diagnose_cnrs_peak_protocol.py`` shows the picker at ``I>=5`` hands the model
roughly half the peaks it saw in training on low-symmetry patterns. This checks
whether that is actually what costs the accuracy: it only re-samples the flow
seed pool at several thresholds and scores primitive L4 on the raw pool, so no
McMaille run is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "McMaille" / "run_lab"))

from eval_cnrs_seedpool import pick_peaks_paperlike  # noqa: E402
from remeasure_l4_prim_vs_conv import l4, truth_cells  # noqa: E402


def load_items(cnrs: Path, imin: float, max_peaks_nn: int) -> list[dict]:
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    items = []
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        csv_path, cif_path = cnrs / f"{sid}.csv", cnrs / f"{sid}_sg.cif"
        if not csv_path.exists() or not cif_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tt, ii, _ = pick_peaks_paperlike(
            df["two_theta_deg"].to_numpy(), df["intensity"].to_numpy()
        )
        keep = ii >= imin
        tt, ii = tt[keep], ii[keep]
        if len(tt) < 3:
            continue
        if len(tt) > max_peaks_nn:
            top = np.argsort(-ii)[:max_peaks_nn]
            tt, ii = tt[top], ii[top]
            order = np.argsort(tt)
            tt, ii = tt[order], ii[order]
        try:
            truth = truth_cells(cif_path)
        except Exception:
            continue
        items.append(
            {"sample_id": sid, "tt": tt.astype(float), "ii": ii.astype(float),
             "prim": truth["prim"], "system": truth["system"], "n_peaks": len(tt)}
        )
    return items


def sample_pool(model, normalizer_mean, normalizer_std, items, k, steps, seed, device):
    from pxrd_cell_indexing.geometry import gstar6_to_lattice

    per = {}
    with torch.no_grad():
        for it in items:
            x = torch.tensor(it["tt"], dtype=torch.float32).view(-1, 1).to(device)
            y = torch.tensor(it["ii"], dtype=torch.float32).view(-1, 1).to(device)
            n = torch.tensor([len(it["tt"])], dtype=torch.long).to(device)
            emb = model.encode(x, y, n)
            gen = torch.Generator(device=device).manual_seed(seed)
            z = model.sample(emb, num_samples=k, steps=steps, generator=gen)[0]
            cells = gstar6_to_lattice(normalizer_std * z + normalizer_mean).cpu().numpy()
            per[it["sample_id"]] = [[float(v) for v in r.tolist()] for r in cells]
    return per


def score(items, pool) -> dict:
    by_sys = defaultdict(lambda: [0, 0, 0])
    rows = []
    for it in items:
        cands = pool[it["sample_id"]]
        strict = any(l4(c, it["prim"])[1] for c in cands)
        loose = any(l4(c, it["prim"])[0] for c in cands)
        b = by_sys[it["system"]]
        b[0] += 1
        b[1] += strict
        b[2] += loose
        rows.append({"sample_id": it["sample_id"], "system": it["system"],
                     "n_peaks": it["n_peaks"], "strict": strict, "loose": loose})
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "strict": sum(r["strict"] for r in rows) / n,
        "loose": sum(r["loose"] for r in rows) / n,
        "mean_peaks": float(np.mean([r["n_peaks"] for r in rows])),
        "by_system": {s: {"n": v[0], "strict": v[1] / v[0], "loose": v[2] / v[0]}
                      for s, v in sorted(by_sys.items())},
        "per_sample": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--ckpt", default="results/flow_seedgen/pxrd_indexer_full6m_v2/best.pt")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json")
    ap.add_argument("--imins", default="5.0,2.0,1.0,0.5")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-peaks-nn", type=int, default=48)
    ap.add_argument("--out", default="results/flow_seedgen/cnrs_imin_sweep.json")
    args = ap.parse_args()

    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from train_flow_seedgen import SeedGenerator
    import argparse as _ap

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(str(ROOT / args.stats))
    ck = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    model = SeedGenerator(_ap.Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)
    print(f"ckpt={args.ckpt} epoch={ck.get('epoch')} K={args.k} device={device}", flush=True)

    cnrs = Path(args.cnrs_dir)
    out = {}
    for imin in [float(x) for x in args.imins.split(",")]:
        t0 = time.time()
        items = load_items(cnrs, imin, args.max_peaks_nn)
        pool = sample_pool(model, mean, std, items, args.k, args.sample_steps, args.seed, device)
        res = score(items, pool)
        out[str(imin)] = res
        print(
            f"I>={imin:<4} n={res['n']:3d} meanPeaks={res['mean_peaks']:5.1f} "
            f"raw-pool L4-strict={res['strict']:.1%} loose={res['loose']:.1%}  "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
        for s, v in res["by_system"].items():
            print(f"      {s:14s} n={v['n']:3d} strict={v['strict']:6.1%} loose={v['loose']:6.1%}")

    (ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
