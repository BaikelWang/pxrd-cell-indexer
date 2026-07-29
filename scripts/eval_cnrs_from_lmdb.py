#!/usr/bin/env python3
"""CNRS seed-pool eval using tkh's CuKa-converted peak LMDB (all 126)."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cnrs_seedpool import THRESHOLDS, score_one, summarize  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lmdb",
        default="/nanolab/tkh/pxrd_mof_real_eval/lmdb/cnrs_recon126_test.lmdb",
    )
    ap.add_argument("--ckpt", default="results/flow_seedgen/full6m_equiv_off/best.pt")
    ap.add_argument("--stats", default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--intensity-min", type=float, default=5.0)
    ap.add_argument("--max-peaks", type=int, default=48)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="results/flow_seedgen/cnrs126_lmdb_seedpool_k100.json")
    return ap.parse_args()


def truth_from_structure(atom_types, positions, lattice_matrix) -> dict:
    species = []
    for a in atom_types:
        # LMDB may store 'Sn' or atomic numbers
        species.append(a if isinstance(a, str) else int(a))
    # LMDB stores fractional coordinates in [0, 1].
    st = Structure(Lattice(lattice_matrix), species, positions, coords_are_cartesian=False)
    ana = SpacegroupAnalyzer(st, symprec=0.1)
    prim = ana.get_primitive_standard_structure().lattice
    conv = ana.get_conventional_standard_structure().lattice
    return {
        "prim": [prim.a, prim.b, prim.c, prim.alpha, prim.beta, prim.gamma],
        "conv": [conv.a, conv.b, conv.c, conv.alpha, conv.beta, conv.gamma],
        "system": ana.get_crystal_system(),
    }


def main() -> None:
    args = parse_args()
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from pxrd_cell_indexing.geometry import gstar6_to_lattice
    from train_flow_seedgen import SeedGenerator

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(args.stats)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SeedGenerator(argparse.Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False, readahead=False)
    items = []
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in range(n):
            d = pickle.loads(gzip.decompress(txn.get(i.to_bytes(4, "big"))))
            x = np.asarray(d["pxrd_x"], dtype=np.float64)
            y = np.asarray(d["pxrd_y"], dtype=np.float64)
            if y.max() > 0:
                y = y * (100.0 / y.max())
            keep = y >= args.intensity_min
            x, y = x[keep], y[keep]
            if len(x) > args.max_peaks:
                top = np.argsort(-y)[: args.max_peaks]
                x, y = x[top], y[top]
                order = np.argsort(x)
                x, y = x[order], y[order]
            if len(x) < 3:
                continue
            truth = truth_from_structure(d["p_atom_type"], d["p_atom_pos"], d["p_lattice_matrix"])
            wave = (d.get("external_metadata") or {}).get("peak_picking", {}).get("wavelength")
            items.append(
                {
                    "sample_id": d["sample_id"],
                    "wavelength": wave,
                    "n_peaks": len(x),
                    "pxrd_x": torch.tensor(x, dtype=torch.float32).view(-1, 1),
                    "pxrd_y": torch.tensor(y, dtype=torch.float32).view(-1, 1),
                    "peak_num": torch.tensor([len(x)], dtype=torch.long),
                    **truth,
                }
            )
    print(f"loaded {len(items)}/{n} from LMDB; device={device} K={args.k}", flush=True)
    print(
        f"peaks: p50={np.median([it['n_peaks'] for it in items]):.0f} "
        f"min={min(it['n_peaks'] for it in items)} max={max(it['n_peaks'] for it in items)}",
        flush=True,
    )

    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)
    pools = {}
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items, 1):
            emb = model.encode(
                it["pxrd_x"].to(device), it["pxrd_y"].to(device), it["peak_num"].to(device)
            )
            gen = torch.Generator(device=device).manual_seed(args.seed)
            z = model.sample(emb, num_samples=args.k, steps=args.sample_steps, generator=gen)[0]
            pools[it["sample_id"]] = gstar6_to_lattice(std * z + mean).cpu().numpy()
            if i % 20 == 0 or i == len(items):
                print(f"sampled {i}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    payloads = [(it["sample_id"], pools[it["sample_id"]], it["prim"], it["conv"]) for it in items]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(score_one, p) for p in payloads]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["sample_id"])
    sys_of = {it["sample_id"]: it["system"] for it in items}
    wave_of = {it["sample_id"]: it["wavelength"] for it in items}
    for r in rows:
        r["system"] = sys_of[r["sample_id"]]
        r["wavelength"] = wave_of[r["sample_id"]]

    from collections import defaultdict

    summary = {
        "source": "tkh cnrs_recon126 LMDB (CuKa-converted peaks)",
        "ckpt": args.ckpt,
        "k": args.k,
        "n": len(rows),
        "prim": summarize(rows, "prim", args.k),
        "conv": summarize(rows, "conv", args.k),
        "by_system_prim_1pct": {},
        "by_wavelength_prim_1pct": {},
    }
    for key, field in (("by_system_prim_1pct", "system"), ("by_wavelength_prim_1pct", "wavelength")):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r.get(field) or "unknown"].append(r)
        for name, rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            s = summarize(rs, "prim", args.k)
            summary[key][str(name)] = {
                "n": s["n"],
                "library_1pct": s["library_at_threshold"]["0.01"],
                "library_l4": s["l4_strict"]["library"],
                "top1_l4": s["l4_strict"]["top1"],
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_sample": rows}, indent=2))
    p, c = summary["prim"], summary["conv"]
    print("==== CNRS LMDB seed-pool (primitive) ====", flush=True)
    print(
        f"L4 @1={p['l4_strict']['top1']:.0%} @20={p['l4_strict']['top20']:.0%} "
        f"lib={p['l4_strict']['library']:.0%} | <1%={p['library_at_threshold']['0.01']:.0%} "
        f"<0.2%={p['library_at_threshold']['0.002']:.0%}",
        flush=True,
    )
    print(
        f"conv L4 lib={c['l4_strict']['library']:.0%} <1%={c['library_at_threshold']['0.01']:.0%}",
        flush=True,
    )
    print("by system:", flush=True)
    for sys, s in summary["by_system_prim_1pct"].items():
        print(
            f"  {sys:12s} n={s['n']:3d} <1%={s['library_1pct']:.0%} "
            f"L4lib={s['library_l4']:.0%} @1={s['top1_l4']:.0%}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
