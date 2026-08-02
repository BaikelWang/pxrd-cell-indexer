#!/usr/bin/env python3
"""CNRS e2e: native McMaille vs full PXRD-indexer (K=100) on primitive L4.

Protocol
--------
* Data: ``/nanolab/users/wyx/CNRS`` continuous CSV + ``*_sg.cif`` truth
* Peaks: tkh paperlike pick (same as ``eval_cnrs_seedpool.py``), I>=5;
  McMaille .dat keeps the first 20 peaks by 2θ (native Mc convention)
* Wavelength: 1.5406 Å (Cu Kα mapping used by tkh/wyx CNRS)
* Arms:
    1. native McMaille (``mcmaille_anchored``, NGRID=-3)
    2. our full indexer: flow seeds K=100 → symmetrize + LSQ →
       ``mcmaille_seeded`` policy=1, ``MCM_NHKL_CAP=400``
* Metric: primitive L4-strict
    (``find_mapping(0.05, 3°)`` AND ``|det-1|<0.25``)
  Also report L4-loose. Native ranks from .imp (suggested + McM20 Top-20);
  ours ranks from .allcells sorted by McM20.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RUN_LAB = ROOT / "third_party" / "McMaille" / "run_lab"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(RUN_LAB))

from eval_cnrs_seedpool import pick_peaks_paperlike  # noqa: E402
from full_pipeline_mcmaille import (  # noqa: E402
    parse_mcmaille_candidates,
    parse_suggested_best,
    write_mcmaille_dat,
)
from remeasure_l4_prim_vs_conv import l4, parse_allcells, truth_cells  # noqa: E402
from run_mp100_reseed_nn import run_one as reseed_run_one  # noqa: E402
from run_mp100_reseed_nn import write_seed_file, _seed_physically_valid  # noqa: E402

WAVELENGTH = 1.5406
ZEROPOINT = 0.0
MAX_MCM_PEAKS = 20
HKL_FILES = ("cub.hkl", "hex.hkl", "mon.hkl", "ort.hkl", "rho.hkl", "tet.hkl", "tri.hkl")
MCM_ROW = re.compile(
    r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([A-Z])\s*(.*)$"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--ckpt", default="results/flow_seedgen/full6m_equiv_off/best.pt")
    ap.add_argument(
        "--stats",
        default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json",
    )
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-peaks-nn", type=int, default=48)
    ap.add_argument("--intensity-min", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout-native", type=int, default=3600)
    ap.add_argument("--timeout-seeded", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default="results/flow_seedgen/cnrs_e2e_k100",
    )
    ap.add_argument(
        "--skip-native",
        action="store_true",
        help="reuse existing native run dir if present",
    )
    ap.add_argument(
        "--skip-seeded",
        action="store_true",
        help="reuse existing seeded run dir if present",
    )
    ap.add_argument(
        "--skip-sample",
        action="store_true",
        help="reuse existing pool JSON if present",
    )
    return ap.parse_args()


def prepare_samples(args) -> list[dict]:
    cnrs = Path(args.cnrs_dir)
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)

    items = []
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
        if len(tt) < 3:
            print(f"skip {sid}: only {len(tt)} peaks after I>={args.intensity_min}", flush=True)
            continue
        try:
            truth = truth_cells(cif_path)
        except Exception as e:
            print(f"skip cif {sid}: {e}", flush=True)
            continue

        # NN peaks: strongest max_peaks_nn, then by 2θ
        tt_nn, ii_nn = tt.copy(), ii.copy()
        if len(tt_nn) > args.max_peaks_nn:
            top = np.argsort(-ii_nn)[: args.max_peaks_nn]
            tt_nn, ii_nn = tt_nn[top], ii_nn[top]
            order = np.argsort(tt_nn)
            tt_nn, ii_nn = tt_nn[order], ii_nn[order]

        # McMaille peaks: first 20 by 2θ (native convention)
        order = np.argsort(tt)
        tt_mcm = tt[order][:MAX_MCM_PEAKS]
        ii_mcm = np.maximum(ii[order][:MAX_MCM_PEAKS], 1.0)

        items.append(
            {
                "sample_id": sid,
                "cif": str(cif_path),
                "formula": row.get("formula"),
                "prim": truth["prim"],
                "conv": truth["conv"],
                "system": truth["system"],
                "tt_nn": tt_nn.astype(float),
                "ii_nn": ii_nn.astype(float),
                "tt_mcm": tt_mcm.astype(float),
                "ii_mcm": ii_mcm.astype(float),
                "n_peaks_found": int(meta["n_found"]),
            }
        )
    return items


def write_dat_trees(items: list[dict], native_src: Path, seeded_src: Path) -> None:
    """Write identical peak lists; only NGRID differs (-3 native / 5 seeded)."""
    for src, ngrid in ((native_src, -3), (seeded_src, 5)):
        src.mkdir(parents=True, exist_ok=True)
        for it in items:
            sid = it["sample_id"]
            work = src / sid
            work.mkdir(parents=True, exist_ok=True)
            stem = sid.replace("-", "_")
            write_mcmaille_dat(
                work / f"{stem}.dat",
                title=sid,
                wavelength=WAVELENGTH,
                zeropoint=ZEROPOINT,
                ngrid=ngrid,
                peak_tt=it["tt_mcm"],
                peak_i=it["ii_mcm"],
            )


def sample_seed_pool(args, items: list[dict], pool_path: Path) -> dict:
    from pxrd_cell_indexing.data.normalization import GStar6Normalizer
    from pxrd_cell_indexing.geometry import gstar6_to_lattice
    from train_flow_seedgen import SeedGenerator

    device = torch.device(args.device)
    normalizer = GStar6Normalizer.from_json(str(ROOT / args.stats))
    ck = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    model = SeedGenerator(argparse.Namespace(**ck["args"])).to(device)
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
            gen = torch.Generator(device=device).manual_seed(args.seed)
            z = model.sample(
                emb, num_samples=args.k, steps=args.sample_steps, generator=gen
            )[0]
            cells = gstar6_to_lattice(std * z + mean).cpu().numpy()
            cands = [[float(x) for x in row.tolist()] for row in cells]
            per[it["sample_id"]] = {"raw_pred": cands[0], "candidates": cands}
            if i % 20 == 0 or i == len(items):
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
        },
        "per_sample": per,
    }
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(payload))
    return payload


def _native_worker(job) -> dict:
    sid, src_dir, out_dir, timeout, bin_name = job
    work = Path(out_dir) / sid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    stem = sid.replace("-", "_")
    shutil.copy(Path(src_dir) / sid / f"{stem}.dat", work / f"{stem}.dat")
    for hkl in HKL_FILES:
        shutil.copy(RUN_LAB / hkl, work / hkl)
    exe = RUN_LAB / bin_name
    bin_dst = work / exe.name
    shutil.copy(exe, bin_dst)
    bin_dst.chmod(0o755)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "false"
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [f"./{exe.name}", stem],
            cwd=work,
            env=env,
            stdout=open(work / "console.log", "w"),
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = -9
    wall = time.perf_counter() - t0
    return {
        "sample_id": sid,
        "returncode": rc,
        "timed_out": timed_out,
        "wall_s": wall,
        "imp_exists": (work / f"{stem}.imp").exists(),
    }


def run_native(items, src_dir: Path, out_dir: Path, workers: int, timeout: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (it["sample_id"], str(src_dir), str(out_dir), timeout, "mcmaille_anchored")
        for it in items
    ]
    results = []
    print(f"native McMaille: n={len(jobs)} workers={workers} timeout={timeout}s", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_native_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                done = sorted(results, key=lambda r: -r.get("wall_s", 0))
                print(
                    f"  native {i}/{len(jobs)}  elapsed={time.time()-t0:.0f}s  "
                    f"slowest={done[0]['sample_id']}:{done[0]['wall_s']:.0f}s",
                    flush=True,
                )
    results.sort(key=lambda r: r["sample_id"])
    (out_dir / "run_summary.json").write_text(json.dumps(results, indent=2))
    return results


def run_seeded(pool: dict, src_dir: Path, out_dir: Path, args) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sid, rec in sorted(pool["per_sample"].items()):
        seeds = [s for s in rec["candidates"][: args.k] if _seed_physically_valid(s)]
        seed_opts = {
            "symmetrize": True,
            "keep_original": True,
            "ltol": 0.01,
            "atol": 1.0,
            "policy": 1,
            "refine": True,
        }
        jobs.append((sid, src_dir, out_dir, seeds, args.timeout_seeded, seed_opts))

    # Ensure MCM_NHKL_CAP is visible to child Fortran processes via the
    # environment of this process (inherited by ProcessPoolExecutor workers).
    os.environ["MCM_NHKL_CAP"] = "400"
    print(
        f"seeded indexer: n={len(jobs)} workers={args.workers} "
        f"K={args.k} MCM_NHKL_CAP=400 policy=native refine=lsq",
        flush=True,
    )
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(reseed_run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"  seeded {i}/{len(jobs)}  elapsed={time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["sample_id"])
    (out_dir / "summary.json").write_text(
        json.dumps({"n": len(results), "per_sample": results}, indent=2)
    )
    return results


def native_ordered_cells(imp_path: Path, topk: int = 20) -> list[list[float]]:
    if not imp_path.exists():
        return []
    text = imp_path.read_text(errors="replace")
    ordered, seen = [], set()

    def add(p):
        k = tuple(round(x, 4) for x in p)
        if k not in seen:
            seen.add(k)
            ordered.append(p)

    sug = parse_suggested_best(text)
    if sug:
        add([sug["a"], sug["b"], sug["c"], sug["alpha"], sug["beta"], sug["gamma"]])
    for c in parse_mcmaille_candidates(text):
        add([c["a"], c["b"], c["c"], c["alpha"], c["beta"], c["gamma"]])
        if len(ordered) >= topk:
            break
    # fallback regex block (same as remasure)
    if not ordered:
        m = re.search(
            r"FINAL LIST OF CELL PROPOSALS, sorted by McM20\s*:.*?Bravais lattice\s*\n+(.*?)\n\s*=======",
            text,
            flags=re.S | re.I,
        )
        if m:
            for line in m.group(1).splitlines():
                mm = MCM_ROW.match(line.rstrip())
                if mm:
                    g = mm.groups()
                    add([float(g[i]) for i in range(4, 10)])
                if len(ordered) >= topk:
                    break
    return ordered[:topk]


def seeded_ordered_cells(
    run_dir: Path, sid: str, *, rerank: str = "none"
) -> list[list[float]]:
    """Order seeded-McMaille candidates.

    ``rerank``:
      none    — McM20 descending (historical default)
      linear  — V0 equal-weight linear (McM20_pct + nn_dist_pct − 0.25·vol_dev)
    """
    allc = run_dir / sid / f"{sid.replace('-', '_')}.allcells"
    if not allc.exists():
        return []
    if rerank == "linear":
        from pxrd_cell_indexing.rerank import order_allcells

        return order_allcells(allc)
    if rerank != "none":
        raise ValueError(f"unknown rerank={rerank!r}")
    cands = parse_allcells(allc)
    cands.sort(key=lambda c: -c["McM20"])
    return [c["params"] for c in cands]


def score_arm(items, ordered_fn) -> list[dict]:
    rows = []
    for it in items:
        pool = ordered_fn(it["sample_id"])
        truth = it["prim"]
        flags = [l4(p, truth) for p in pool]
        loose = [f[0] for f in flags]
        strict = [f[1] for f in flags]

        def first(arr):
            for i, v in enumerate(arr, 1):
                if v:
                    return i
            return None

        rows.append(
            {
                "sample_id": it["sample_id"],
                "system": it["system"],
                "n_pool": len(pool),
                "prim": {
                    "lib_loose": any(loose),
                    "lib_strict": any(strict),
                    "top1_loose": bool(loose[:1] and loose[0]),
                    "top1_strict": bool(strict[:1] and strict[0]),
                    "top20_loose": any(loose[:20]),
                    "top20_strict": any(strict[:20]),
                    "first_strict": first(strict),
                    "first_loose": first(loose),
                },
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)

    def rate(key):
        return sum(1 for r in rows if r["prim"][key]) / n

    by_sys = {}
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["system"]].append(r)
    for sys, rs in sorted(buckets.items()):
        m = max(len(rs), 1)
        by_sys[sys] = {
            "n": len(rs),
            "top1_strict": sum(1 for r in rs if r["prim"]["top1_strict"]) / m,
            "top20_strict": sum(1 for r in rs if r["prim"]["top20_strict"]) / m,
            "lib_strict": sum(1 for r in rs if r["prim"]["lib_strict"]) / m,
        }
    return {
        "n": len(rows),
        "mean_pool": sum(r["n_pool"] for r in rows) / n,
        "prim": {
            "top1_loose": rate("top1_loose"),
            "top1_strict": rate("top1_strict"),
            "top20_loose": rate("top20_loose"),
            "top20_strict": rate("top20_strict"),
            "lib_loose": rate("lib_loose"),
            "lib_strict": rate("lib_strict"),
        },
        "by_system": by_sys,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    native_src = out / "dat_native"
    seeded_src = out / "dat_seeded"
    native_run = out / "native_mcmaille"
    seeded_run = out / "indexer_k100"
    pool_path = out / f"pool_k{args.k}.json"
    report_path = out / "l4_prim_compare.json"

    print("==== prepare CNRS samples ====", flush=True)
    items = prepare_samples(args)
    print(f"usable samples: {len(items)}", flush=True)
    write_dat_trees(items, native_src, seeded_src)
    meta = {
        "cnrs_dir": args.cnrs_dir,
        "ckpt": args.ckpt,
        "k": args.k,
        "wavelength": WAVELENGTH,
        "max_mcm_peaks": MAX_MCM_PEAKS,
        "max_peaks_nn": args.max_peaks_nn,
        "intensity_min": args.intensity_min,
        "peak_protocol": "tkh paperlike + Imin + first20 by 2θ for McMaille",
        "n_samples": len(items),
        "sample_ids": [it["sample_id"] for it in items],
    }
    (out / "protocol.json").write_text(json.dumps(meta, indent=2))

    if args.skip_sample and pool_path.exists():
        print(f"reuse pool {pool_path}", flush=True)
        pool = json.loads(pool_path.read_text())
    else:
        pool = sample_seed_pool(args, items, pool_path)

    if args.skip_native and native_run.exists() and any(native_run.iterdir()):
        print(f"reuse native run {native_run}", flush=True)
    else:
        run_native(items, native_src, native_run, args.workers, args.timeout_native)

    if args.skip_seeded and seeded_run.exists() and any(seeded_run.iterdir()):
        print(f"reuse seeded run {seeded_run}", flush=True)
    else:
        run_seeded(pool, seeded_src, seeded_run, args)

    print("==== score primitive L4 ====", flush=True)

    def native_fn(sid):
        return native_ordered_cells(native_run / sid / f"{sid.replace('-', '_')}.imp")

    def seeded_fn(sid):
        return seeded_ordered_cells(seeded_run, sid)

    native_rows = score_arm(items, native_fn)
    seeded_rows = score_arm(items, seeded_fn)
    report = {
        "protocol": meta,
        "native_mcmaille": {
            "run_dir": str(native_run),
            **summarize(native_rows),
            "per_sample": native_rows,
        },
        "indexer_k100": {
            "run_dir": str(seeded_run),
            "pool_json": str(pool_path),
            **summarize(seeded_rows),
            "per_sample": seeded_rows,
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    def show(label, block):
        p = block["prim"]
        print(
            f"{label:16s} n={block['n']}  "
            f"L4-strict Top-1={p['top1_strict']:.1%}  "
            f"Top-20={p['top20_strict']:.1%}  "
            f"lib={p['lib_strict']:.1%}  "
            f"(loose @1={p['top1_loose']:.1%} @20={p['top20_loose']:.1%})",
            flush=True,
        )

    print("==== CNRS primitive L4 (strict) ====", flush=True)
    show("native McMaille", report["native_mcmaille"])
    show("indexer K=100", report["indexer_k100"])
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
