#!/usr/bin/env python3
"""Build synthetic seeded-McMaille libraries for reranker training (P1).

Samples from the train LMDB / jsonl ONLY — never MP100 or CNRS.
Pipeline mirrors the product indexer:
  peaks (I≥5) → flow seeds (K) → symmetrize+LSQ → seeded McMaille (policy=1,
  MCM_NHKL_CAP=400) → .allcells → L4-strict labels + NN-prior features.

Stages (each skippable once its artifact exists):
  1. split   → split.json  (rr_train / rr_dev lmdb keys)
  2. items   → items.jsonl + dat_seeded/<sid>/<stem>.dat
  3. pool    → pool_k{K}.json
  4. reseed  → mc_k{K}/<sid>/*.allcells
  5. extract → rr_train.jsonl / rr_dev.jsonl  (per-sample feature rows)

Example:
  python3 scripts/build_rerank_dataset.py \\
    --out data/rerank/v0_k1000 --n-train 4000 --n-dev 1000 --k 1000 \\
    --device cuda --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_LAB = ROOT / "third_party" / "McMaille" / "run_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RUN_LAB))

from full_pipeline_mcmaille import write_mcmaille_dat  # noqa: E402
from remeasure_l4_prim_vs_conv import l4  # noqa: E402
from run_mp100_reseed_nn import _seed_physically_valid  # noqa: E402
from run_mp100_reseed_nn import run_one as reseed_run_one  # noqa: E402

WAVELENGTH = 1.5406
ZEROPOINT = 0.0
MAX_MCM_PEAKS = 20
MAX_PEAKS_NN = 48
INTENSITY_MIN = 5.0
MIN_PEAKS = 8

ALLCELLS_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--jsonl",
        default=ROOT / "data/processed/train100k_niggli_seed42.jsonl",
        type=Path,
    )
    ap.add_argument(
        "--lmdb",
        default=Path("/nanolab/users/wyx/alex_aflow_oqmd_mp/datasets/pxrd_241113_train.lmdb"),
        type=Path,
    )
    ap.add_argument(
        "--ckpt",
        default="results/flow_seedgen/pxrd_indexer_full6m_v4_wide_lr2e3/best.pt",
    )
    ap.add_argument(
        "--stats",
        default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json",
    )
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-items", action="store_true")
    ap.add_argument("--skip-pool", action="store_true")
    ap.add_argument("--skip-reseed", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument(
        "--perturb",
        action="store_true",
        help="Apply CNRS-like peak noise (2θ jitter, drop/inject peaks, Imin jitter)",
    )
    ap.add_argument(
        "--refine",
        choices=["lsq", "off"],
        default="lsq",
        help="Seed refine before McMaille (off is ~5× faster; use for perturbed train)",
    )
    ap.add_argument(
        "--resume-reseed",
        action="store_true",
        help="Skip samples that already have .allcells under mc_k{K}/",
    )
    return ap.parse_args()


def sid_of(lmdb_key: str) -> str:
    """Filesystem-safe sample id; keep key recoverable."""
    return f"s{lmdb_key}"


def make_split(args) -> dict:
    """Crystal-system stratified rr_train / rr_dev from the train jsonl."""
    rng = np.random.default_rng(args.seed)
    by_cs: dict[str, list[dict]] = defaultdict(list)
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("peak_num_filtered", 0)) < MIN_PEAKS:
                continue
            by_cs[r["crystal_system"]].append(r)

    n_total = args.n_train + args.n_dev
    systems = sorted(by_cs)
    # Round-robin per system so each cs gets ~equal count.
    per_cs = n_total // max(len(systems), 1)
    extras = n_total - per_cs * len(systems)
    train, dev = [], []
    for i, cs in enumerate(systems):
        want = per_cs + (1 if i < extras else 0)
        pool = by_cs[cs]
        if len(pool) < want:
            raise RuntimeError(f"{cs}: need {want}, have {len(pool)}")
        idx = rng.choice(len(pool), size=want, replace=False)
        chosen = [pool[j] for j in idx]
        rng.shuffle(chosen)
        n_dev_cs = max(1, int(round(want * args.n_dev / n_total)))
        n_dev_cs = min(n_dev_cs, want - 1)
        dev.extend(chosen[:n_dev_cs])
        train.extend(chosen[n_dev_cs:])

    rng.shuffle(train)
    rng.shuffle(dev)
    # Trim to exact counts after stratification rounding.
    train, dev = train[: args.n_train], dev[: args.n_dev]
    split = {
        "seed": args.seed,
        "jsonl": str(args.jsonl),
        "lmdb": str(args.lmdb),
        "min_peaks": MIN_PEAKS,
        "n_train": len(train),
        "n_dev": len(dev),
        "rr_train": [
            {
                "lmdb_key": r["lmdb_key"],
                "sample_id": sid_of(r["lmdb_key"]),
                "crystal_system": r["crystal_system"],
                "truth": [
                    float(r["lattice_a"]),
                    float(r["lattice_b"]),
                    float(r["lattice_c"]),
                    float(r["lattice_alpha"]),
                    float(r["lattice_beta"]),
                    float(r["lattice_gamma"]),
                ],
            }
            for r in train
        ],
        "rr_dev": [
            {
                "lmdb_key": r["lmdb_key"],
                "sample_id": sid_of(r["lmdb_key"]),
                "crystal_system": r["crystal_system"],
                "truth": [
                    float(r["lattice_a"]),
                    float(r["lattice_b"]),
                    float(r["lattice_c"]),
                    float(r["lattice_alpha"]),
                    float(r["lattice_beta"]),
                    float(r["lattice_gamma"]),
                ],
            }
            for r in dev
        ],
    }
    return split


def load_items(split: dict, args) -> list[dict]:
    from pxrd_cell_indexing.data.dataset import (
        PeakFilterConfig,
        PXRDDataset,
        PXRDDatasetConfig,
    )

    # Build a temporary jsonl covering just our keys so PXRDDataset can index them.
    tmp = args.out / "_tmp_split_meta.jsonl"
    meta_by_key = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            meta_by_key[r["lmdb_key"]] = r
    keys = [r["lmdb_key"] for r in split["rr_train"] + split["rr_dev"]]
    with open(tmp, "w") as f:
        for k in keys:
            f.write(json.dumps(meta_by_key[k]) + "\n")

    ds = PXRDDataset(
        PXRDDatasetConfig(
            lmdb_path=Path(args.lmdb),
            sample_list_path=tmp,
            peak_filter=PeakFilterConfig(intensity_min=INTENSITY_MIN, max_peaks=None),
            xrd_augment=False,
        )
    )
    by_key = {rec["lmdb_key"]: i for i, rec in enumerate(ds.records)}
    train_ids = {x["sample_id"] for x in split["rr_train"]}

    items = []
    for r in split["rr_train"] + split["rr_dev"]:
        s = ds[by_key[r["lmdb_key"]]]
        tt = np.asarray(s["two_theta"], dtype=float)
        ii = np.asarray(s["intensity"], dtype=float)
        if tt.size < MIN_PEAKS:
            continue
        tt_nn, ii_nn = tt.copy(), ii.copy()
        if len(tt_nn) > MAX_PEAKS_NN:
            top = np.argsort(-ii_nn)[:MAX_PEAKS_NN]
            tt_nn, ii_nn = tt_nn[top], ii_nn[top]
            order = np.argsort(tt_nn)
            tt_nn, ii_nn = tt_nn[order], ii_nn[order]
        order = np.argsort(tt)
        tt_mcm = tt[order][:MAX_MCM_PEAKS]
        ii_mcm = np.maximum(ii[order][:MAX_MCM_PEAKS], 1.0)
        items.append(
            {
                "sample_id": r["sample_id"],
                "lmdb_key": r["lmdb_key"],
                "split": "rr_train" if r["sample_id"] in train_ids else "rr_dev",
                "crystal_system": r["crystal_system"],
                "truth": r["truth"],
                "tt_nn": tt_nn.astype(float).tolist(),
                "ii_nn": ii_nn.astype(float).tolist(),
                "tt_mcm": tt_mcm.astype(float).tolist(),
                "ii_mcm": ii_mcm.astype(float).tolist(),
            }
        )
    return items


def perturb_peaks(
    tt: np.ndarray,
    ii: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """CNRS-like observation noise for harder ranking supervision."""
    tt = tt.astype(float).copy()
    ii = ii.astype(float).copy()
    if tt.size == 0:
        return tt, ii
    # Zero-point + per-peak jitter (degrees).
    tt = tt + rng.normal(0.0, 0.03) + rng.normal(0.0, 0.02, size=tt.shape)
    # Randomly drop 0–20% of weaker peaks.
    if tt.size >= 10:
        keep = ii >= np.quantile(ii, rng.uniform(0.0, 0.2))
        if keep.sum() >= MIN_PEAKS:
            tt, ii = tt[keep], ii[keep]
    # Inject 0–2 weak false peaks.
    n_fake = int(rng.integers(0, 3))
    if n_fake and tt.size:
        lo, hi = float(tt.min()), float(min(tt.max(), 60.0))
        if hi > lo + 1:
            fake_tt = rng.uniform(lo, hi, size=n_fake)
            fake_ii = rng.uniform(1.0, max(float(np.median(ii)) * 0.3, 2.0), size=n_fake)
            tt = np.concatenate([tt, fake_tt])
            ii = np.concatenate([ii, fake_ii])
            order = np.argsort(tt)
            tt, ii = tt[order], ii[order]
    ii = np.maximum(ii, 1.0)
    return tt, ii


def write_dats(items: list[dict], dat_dir: Path, *, perturb: bool, seed: int) -> None:
    dat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for it in items:
        sid = it["sample_id"]
        stem = sid.replace("-", "_")
        work = dat_dir / sid
        work.mkdir(parents=True, exist_ok=True)
        tt = np.asarray(it["tt_mcm"], dtype=float)
        ii = np.asarray(it["ii_mcm"], dtype=float)
        if perturb:
            tt, ii = perturb_peaks(tt, ii, rng)
            # Keep a copy of the perturbed peaks used for McMaille (NN still sees clean).
            it["tt_mcm_used"] = tt.tolist()
            it["ii_mcm_used"] = ii.tolist()
        write_mcmaille_dat(
            work / f"{stem}.dat",
            title=sid,
            wavelength=WAVELENGTH,
            zeropoint=ZEROPOINT,
            ngrid=5,
            peak_tt=tt,
            peak_i=ii,
        )


def sample_pool(items: list[dict], args, pool_path: Path) -> dict:
    import argparse as ap
    import torch
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from pxrd_cell_indexing.geometry import gstar6_to_lattice
    from train_flow_seedgen import SeedGenerator

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(str(ROOT / args.stats))
    ck = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    model = SeedGenerator(ap.Namespace(**ck["args"])).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    mean = torch.tensor(normalizer.component_mean, device=device)
    std = torch.tensor(normalizer.component_std, device=device)

    print(
        f"sampling seeds: ckpt={args.ckpt} epoch={ck.get('epoch')} "
        f"device={device} K={args.k} n={len(items)}",
        flush=True,
    )
    per = {}
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(items, 1):
            x = torch.tensor(it["tt_nn"], dtype=torch.float32).view(-1, 1).to(device)
            y = torch.tensor(it["ii_nn"], dtype=torch.float32).view(-1, 1).to(device)
            n = torch.tensor([len(it["tt_nn"])], dtype=torch.long).to(device)
            emb = model.encode(x, y, n)
            gen = torch.Generator(device=device).manual_seed(args.seed + i)
            z = model.sample(
                emb, num_samples=args.k, steps=args.sample_steps, generator=gen
            )[0]
            cells = gstar6_to_lattice(std * z + mean).cpu().numpy()
            cands = [[float(v) for v in row.tolist()] for row in cells]
            per[it["sample_id"]] = {"raw_pred": cands[0], "candidates": cands}
            if i % 50 == 0 or i == len(items):
                print(f"  sampled {i}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    payload = {
        "summary": {
            "ckpt": args.ckpt,
            "epoch": ck.get("epoch"),
            "top_k": args.k,
            "sample_steps": args.sample_steps,
            "seed": args.seed,
            "n_samples": len(per),
            "wavelength": WAVELENGTH,
            "source": "synthetic_train_lmdb",
        },
        "per_sample": per,
    }
    pool_path.write_text(json.dumps(payload))
    return payload


def run_reseed(pool: dict, dat_dir: Path, out_dir: Path, args) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    skipped = 0
    for sid, rec in sorted(pool["per_sample"].items()):
        stem = sid.replace("-", "_")
        if args.resume_reseed and (out_dir / sid / f"{stem}.allcells").exists():
            skipped += 1
            continue
        seeds = [s for s in rec["candidates"][: args.k] if _seed_physically_valid(s)]
        seed_opts = {
            "symmetrize": True,
            "keep_original": True,
            "ltol": 0.01,
            "atol": 1.0,
            "policy": 1,
            "refine": args.refine == "lsq",
        }
        jobs.append((sid, dat_dir, out_dir, seeds, args.timeout, seed_opts))

    os.environ["MCM_NHKL_CAP"] = "400"
    print(
        f"seeded McMaille: n={len(jobs)} skip={skipped} workers={args.workers} "
        f"K={args.k} refine={args.refine} MCM_NHKL_CAP=400",
        flush=True,
    )
    if not jobs:
        return []
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(reseed_run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 50 == 0 or i == len(jobs):
                print(f"  seeded {i}/{len(jobs)}  elapsed={time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["sample_id"])
    (out_dir / "summary.json").write_text(
        json.dumps({"n": len(results), "per_sample": results}, indent=2)
    )
    return results


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
    # Reciprocal metric from reciprocal lattice matrix.
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


def extract_one(job) -> dict | None:
    sid, mc_dir, truth, split, cs = job
    d = Path(mc_dir) / sid
    stem = sid.replace("-", "_")
    allc = d / f"{stem}.allcells"
    if not allc.exists():
        return None
    rows = sorted(parse_allcells_full(allc), key=lambda r: -r["McM20"])
    seeds = read_seeds(d / f"{stem}.seed") if (d / f"{stem}.seed").exists() else np.zeros((0, 6))
    seed_g = np.asarray([gstar6(s) for s in seeds]) if seeds.size else np.zeros((0, 6))
    v_nn = float(np.median([cell_volume(s) for s in seeds])) if seeds.size else 0.0

    cands = []
    for r in rows:
        vol = r["volume"] if r["volume"] > 0 else max(cell_volume(r["params"]), 1e-6)
        vol_dev = abs(np.log(vol / v_nn)) if v_nn > 0 else 0.0
        if seed_g.size:
            nn_dist = float(np.min(np.linalg.norm(seed_g - gstar6(r["params"]), axis=1)))
        else:
            nn_dist = 0.0
        hit = bool(l4(r["params"], truth)[1])
        cands.append(
            {
                "seed_src": r["seed_src"],
                "stage": r["stage"],
                "n_indexed": r["n_indexed"],
                "McM20": r["McM20"],
                "volume": vol,
                "Rp": r["Rp"],
                "params": r["params"],
                "bravais": r["bravais"],
                "nn_dist": nn_dist,
                "vol_dev": float(min(vol_dev, 3.0) / 3.0),
                "is_hit": hit,
            }
        )

    if cands:
        mcm = np.asarray([c["McM20"] for c in cands])
        nn = np.asarray([c["nn_dist"] for c in cands])
        p_mcm = pct_rank(mcm)
        p_nn = pct_rank(nn, higher_better=False)
        for i, c in enumerate(cands):
            c["p_mcm"] = float(p_mcm[i])
            c["p_nn_dist"] = float(p_nn[i])

    first = next((i for i, c in enumerate(cands, 1) if c["is_hit"]), None)
    return {
        "sample_id": sid,
        "split": split,
        "crystal_system": cs,
        "truth": truth,
        "n_pool": len(cands),
        "lib_strict": first is not None,
        "first_mcm20": first,
        "candidates": cands,
    }


def extract_all(items: list[dict], mc_dir: Path, args) -> None:
    jobs = [
        (it["sample_id"], str(mc_dir), it["truth"], it["split"], it["crystal_system"])
        for it in items
    ]
    train_path = args.out / "rr_train.jsonl"
    dev_path = args.out / "rr_dev.jsonl"
    rows = []
    print(f"extracting features: n={len(jobs)} workers={args.workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec is not None:
                rows.append(rec)
            if i % 100 == 0 or i == len(jobs):
                print(f"  extract {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    with open(train_path, "w") as ft, open(dev_path, "w") as fd:
        n_tr = n_dv = 0
        lib_tr = lib_dv = top1_tr = top1_dv = 0
        for r in sorted(rows, key=lambda x: x["sample_id"]):
            line = json.dumps(r) + "\n"
            if r["split"] == "rr_train":
                ft.write(line)
                n_tr += 1
                lib_tr += int(r["lib_strict"])
                top1_tr += int(r["first_mcm20"] == 1)
            else:
                fd.write(line)
                n_dv += 1
                lib_dv += int(r["lib_strict"])
                top1_dv += int(r["first_mcm20"] == 1)

    summary = {
        "n_train": n_tr,
        "n_dev": n_dv,
        "train": {
            "lib_strict": lib_tr / max(n_tr, 1),
            "top1_mcm20": top1_tr / max(n_tr, 1),
            "headroom": (lib_tr - top1_tr) / max(n_tr, 1),
        },
        "dev": {
            "lib_strict": lib_dv / max(n_dv, 1),
            "top1_mcm20": top1_dv / max(n_dv, 1),
            "headroom": (lib_dv - top1_dv) / max(n_dv, 1),
        },
    }
    (args.out / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {train_path} and {dev_path}", flush=True)


def main() -> None:
    args = parse_args()
    args.out = Path(args.out)
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)

    split_path = args.out / "split.json"
    items_path = args.out / "items.jsonl"
    dat_dir = args.out / "dat_seeded"
    pool_path = args.out / f"pool_k{args.k}.json"
    mc_dir = args.out / f"mc_k{args.k}"

    # ---- stage 1: split ----
    if args.skip_split and split_path.exists():
        split = json.loads(split_path.read_text())
        print(f"reuse split {split_path}", flush=True)
    else:
        print("==== stage 1: split ====", flush=True)
        split = make_split(args)
        split_path.write_text(json.dumps(split, indent=2))
        print(f"train={split['n_train']} dev={split['n_dev']} → {split_path}", flush=True)

    # ---- stage 2: items + .dat ----
    if args.skip_items and items_path.exists() and dat_dir.exists():
        items = [json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]
        print(f"reuse items n={len(items)}", flush=True)
    else:
        print("==== stage 2: load peaks + write .dat ====", flush=True)
        items = load_items(split, args)
        with open(items_path, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        write_dats(items, dat_dir, perturb=args.perturb, seed=args.seed)
        with open(items_path, "w") as f:  # rewrite with any perturb metadata
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(
            f"items={len(items)} dats → {dat_dir}  perturb={args.perturb}",
            flush=True,
        )

    # ---- stage 3: seed pool ----
    if args.skip_pool and pool_path.exists():
        pool = json.loads(pool_path.read_text())
        print(f"reuse pool {pool_path}", flush=True)
    else:
        print("==== stage 3: sample seed pool ====", flush=True)
        pool = sample_pool(items, args, pool_path)

    # ---- stage 4: seeded McMaille ----
    if args.skip_reseed and mc_dir.exists() and (mc_dir / "summary.json").exists():
        print(f"reuse reseed {mc_dir}", flush=True)
    else:
        print("==== stage 4: seeded McMaille ====", flush=True)
        run_reseed(pool, dat_dir, mc_dir, args)

    # ---- stage 5: extract features + labels ----
    if args.skip_extract and (args.out / "rr_train.jsonl").exists():
        print(f"reuse extract {args.out}/rr_*.jsonl", flush=True)
    else:
        print("==== stage 5: extract features ====", flush=True)
        extract_all(items, mc_dir, args)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
