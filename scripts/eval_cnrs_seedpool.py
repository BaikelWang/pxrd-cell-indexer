#!/usr/bin/env python3
"""PXRD-indexer seed-pool eval on the local CNRS experimental set.

Data: ``/nanolab/users/wyx/CNRS`` — continuous CuKa-mapped profiles + SG CIFs
(same 126 patterns as tkh ``cnrs_recon126``; peak positions match after the
paperlike pick). This is a *seed generator* eval only (no McMaille), reporting:

* L4-strict library@K (``ltol=0.05``, ``|det-1|<0.25``) vs primitive / conventional
* usable library@K = best aligned length error < 1%

Peak picking follows the tkh external-real protocol (poly-5 BG, 1% prominence,
0.08° min separation, 5–80°, Imax=100) so results are comparable to their LMDB.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, savgol_filter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

THRESHOLDS = (0.002, 0.005, 0.01, 0.02, 0.05)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--ckpt", default="results/flow_seedgen/full6m_equiv_off/best.pt")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-peaks", type=int, default=48)
    ap.add_argument("--intensity-min", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 126")
    ap.add_argument("--out", default="results/flow_seedgen/cnrs126_seedpool_k100.json")
    return ap.parse_args()


def pick_peaks_paperlike(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    tmin: float = 5.0,
    tmax: float = 80.0,
    poly_order: int = 5,
    prom_frac: float = 0.01,
    min_distance_deg: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mask = (two_theta >= tmin) & (two_theta <= tmax) & np.isfinite(intensity) & (intensity > 0)
    tt = two_theta[mask].astype(np.float64)
    ii = intensity[mask].astype(np.float64)
    if len(tt) < 20:
        return np.zeros(0), np.zeros(0), {"n_found": 0, "reason": "too_few_points"}

    coef = np.polyfit(tt, ii, deg=poly_order)
    corr = np.clip(ii - np.polyval(coef, tt), 0.0, None)
    smooth = savgol_filter(corr, 11, 3) if len(corr) >= 11 else corr
    height = max(1.0, prom_frac * float(np.percentile(smooth, 99)))
    step = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.03
    distance = max(2, int(round(min_distance_deg / max(step, 1e-6))))
    idx, _ = find_peaks(smooth, height=height, prominence=height, distance=distance)
    peak_tt = tt[idx]
    peak_i = corr[idx]
    order = np.argsort(peak_tt)
    peak_tt, peak_i = peak_tt[order], peak_i[order]
    if peak_i.size and peak_i.max() > 0:
        peak_i = peak_i * (100.0 / peak_i.max())
    return peak_tt, peak_i, {
        "n_found": int(len(peak_tt)),
        "height": height,
        "distance": distance,
        "poly_order": poly_order,
    }


def truth_from_cif(cif: Path) -> dict:
    from remeasure_l4_prim_vs_conv import truth_cells

    return truth_cells(cif)


def score_one(payload):
    from diagnose_mcmaille_value import cell_err
    from remeasure_l4_prim_vs_conv import l4

    sid, cells, prim, conv = payload
    out = {"sample_id": sid, "prim": {}, "conv": {}}
    for label, truth in (("prim", prim), ("conv", conv)):
        best = None
        first = {t: None for t in THRESHOLDS}
        l4_hit_rank = None
        for rank, row in enumerate(cells, 1):
            if not np.isfinite(row).all():
                continue
            loose, strict, _ = l4(row.tolist(), truth)
            if strict and l4_hit_rank is None:
                l4_hit_rank = rank
            if not strict:
                continue
            err, _ = cell_err(row.tolist(), truth)
            if not np.isfinite(err):
                continue
            if best is None or err < best:
                best = float(err)
            for t in THRESHOLDS:
                if first[t] is None and err < t:
                    first[t] = rank
        out[label] = {
            "best_err": best,
            "l4_strict_rank": l4_hit_rank,
            "first_hit": {str(t): first[t] for t in THRESHOLDS},
        }
    return out


def summarize(rows: list[dict], label: str, k: int) -> dict:
    n = len(rows)
    best = [r[label]["best_err"] for r in rows]
    hits = [e for e in best if e is not None]

    def rate_at(tol: float) -> float:
        return sum(1 for e in hits if e < tol) / max(n, 1)

    def cov_at(tol: float, kk: int) -> float:
        c = 0
        for r in rows:
            fr = r[label]["first_hit"].get(str(tol))
            if fr is not None and fr <= kk:
                c += 1
        return c / max(n, 1)

    l4_lib = sum(1 for r in rows if r[label]["l4_strict_rank"] is not None) / max(n, 1)
    l4_top1 = sum(1 for r in rows if r[label]["l4_strict_rank"] == 1) / max(n, 1)
    l4_top20 = sum(
        1 for r in rows if r[label]["l4_strict_rank"] is not None and r[label]["l4_strict_rank"] <= 20
    ) / max(n, 1)
    return {
        "n": n,
        "l4_strict": {"top1": l4_top1, "top20": l4_top20, "library": l4_lib},
        "library_at_threshold": {str(t): rate_at(t) for t in THRESHOLDS},
        "coverage_1pct": {str(kk): cov_at(0.01, kk) for kk in (1, 5, 20, 50, k)},
        "median_best_err": float(np.median(hits)) if hits else None,
        "frac_any_l4": l4_lib,
    }


def main() -> None:
    args = parse_args()
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from pxrd_cell_indexing.geometry import gstar6_to_lattice
    from train_flow_seedgen import SeedGenerator

    cnrs = Path(args.cnrs_dir)
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(args.stats)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SeedGenerator(argparse.Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    print(
        f"ckpt={args.ckpt} epoch={ck.get('epoch')} device={device} "
        f"K={args.k} steps={args.sample_steps} n={len(manifest)}",
        flush=True,
    )

    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)

    items = []
    peak_meta = {}
    t_peak0 = time.time()
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        csv_path = cnrs / f"{sid}.csv"
        cif_path = cnrs / f"{sid}_sg.cif"
        if not csv_path.exists() or not cif_path.exists():
            print(f"skip missing {sid}", flush=True)
            continue
        df = pd.read_csv(csv_path)
        tt, ii, meta = pick_peaks_paperlike(
            df["two_theta_deg"].to_numpy(), df["intensity"].to_numpy()
        )
        keep = ii >= args.intensity_min
        tt, ii = tt[keep], ii[keep]
        if len(tt) > args.max_peaks:
            # strongest max_peaks, then re-sort by 2θ (training default)
            top = np.argsort(-ii)[: args.max_peaks]
            tt, ii = tt[top], ii[top]
            order = np.argsort(tt)
            tt, ii = tt[order], ii[order]
        meta["n_after_imin"] = int(len(tt))
        peak_meta[sid] = meta
        if len(tt) < 3:
            print(f"warn {sid}: only {len(tt)} peaks", flush=True)
            continue
        try:
            truth = truth_from_cif(cif_path)
        except Exception as e:
            print(f"skip cif {sid}: {e}", flush=True)
            continue
        items.append(
            {
                "sample_id": sid,
                "formula": row["formula"],
                "true_sg": row["true_sg"],
                "pxrd_x": torch.tensor(tt, dtype=torch.float32).view(-1, 1),
                "pxrd_y": torch.tensor(ii, dtype=torch.float32).view(-1, 1),
                "peak_num": torch.tensor([len(tt)], dtype=torch.long),
                "prim": truth["prim"],
                "conv": truth["conv"],
                "system": truth["system"],
            }
        )
    print(f"prepared {len(items)} samples in {time.time()-t_peak0:.1f}s", flush=True)
    n_peaks = [int(it["peak_num"][0]) for it in items]
    print(
        f"peaks after I>={args.intensity_min}: "
        f"p50={np.median(n_peaks):.0f} min={min(n_peaks)} max={max(n_peaks)}",
        flush=True,
    )

    pools = {}
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items, 1):
            emb = model.encode(
                it["pxrd_x"].to(device),
                it["pxrd_y"].to(device),
                it["peak_num"].to(device),
            )
            gen = torch.Generator(device=device).manual_seed(args.seed)
            z = model.sample(
                emb, num_samples=args.k, steps=args.sample_steps, generator=gen
            )[0]
            cells = gstar6_to_lattice(std * z + mean).cpu().numpy()
            pools[it["sample_id"]] = cells
            if i % 20 == 0 or i == len(items):
                print(
                    f"sampled {i}/{len(items)}  elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

    payloads = [
        (it["sample_id"], pools[it["sample_id"]], it["prim"], it["conv"]) for it in items
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(score_one, p) for p in payloads]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["sample_id"])

    # attach crystal system for breakdown
    sys_of = {it["sample_id"]: it["system"] for it in items}
    for r in rows:
        r["system"] = sys_of[r["sample_id"]]

    summary = {
        "ckpt": args.ckpt,
        "cnrs_dir": str(cnrs),
        "k": args.k,
        "sample_steps": args.sample_steps,
        "seed": args.seed,
        "peak_protocol": "tkh paperlike (poly5, prom1%, 0.08deg, 5-80, Imax=100)",
        "intensity_min": args.intensity_min,
        "max_peaks": args.max_peaks,
        "n": len(rows),
        "prim": summarize(rows, "prim", args.k),
        "conv": summarize(rows, "conv", args.k),
        "by_system_prim_1pct": {},
    }
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["system"]].append(r)
    for sys, rs in sorted(buckets.items()):
        s = summarize(rs, "prim", args.k)
        summary["by_system_prim_1pct"][sys] = {
            "n": s["n"],
            "library_1pct": s["library_at_threshold"]["0.01"],
            "library_l4": s["l4_strict"]["library"],
            "top1_l4": s["l4_strict"]["top1"],
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_sample": rows, "peak_meta": peak_meta}, indent=2))

    print("==== CNRS seed-pool (primitive) ====", flush=True)
    p = summary["prim"]
    print(
        f"L4-strict  @1={p['l4_strict']['top1']:.0%}  @20={p['l4_strict']['top20']:.0%}  "
        f"lib={p['l4_strict']['library']:.0%}",
        flush=True,
    )
    print(
        f"<1% usable lib={p['library_at_threshold']['0.01']:.0%}  "
        f"<0.2%={p['library_at_threshold']['0.002']:.0%}  "
        f"<5%={p['library_at_threshold']['0.05']:.0%}  "
        f"median_best={p['median_best_err']}",
        flush=True,
    )
    print("==== vs conventional truth ====", flush=True)
    c = summary["conv"]
    print(
        f"L4-strict  @1={c['l4_strict']['top1']:.0%}  @20={c['l4_strict']['top20']:.0%}  "
        f"lib={c['l4_strict']['library']:.0%}  <1%={c['library_at_threshold']['0.01']:.0%}",
        flush=True,
    )
    print("by crystal system (prim <1% library):", flush=True)
    for sys, s in summary["by_system_prim_1pct"].items():
        print(
            f"  {sys:12s} n={s['n']:3d}  <1%={s['library_1pct']:.0%}  "
            f"L4lib={s['library_l4']:.0%}  L4@1={s['top1_l4']:.0%}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
