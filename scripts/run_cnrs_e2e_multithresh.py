#!/usr/bin/env python3
"""CNRS e2e with multi-threshold seed-pool union.

Same pipeline as ``run_cnrs_e2e_compare.py``, except the flow model is queried
once per intensity threshold and the candidate pools are unioned before
symmetrize / LSQ / seeded McMaille. McMaille peak lists stay at the native
convention (I>=5, first 20 by 2θ) so the only changed variable is the seed
source. Native McMaille can be reused via ``--skip-native``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cnrs_seedpool import pick_peaks_paperlike  # noqa: E402
from remeasure_l4_prim_vs_conv import truth_cells  # noqa: E402
from run_cnrs_e2e_compare import (  # noqa: E402
    MAX_MCM_PEAKS,
    WAVELENGTH,
    native_ordered_cells,
    run_native,
    run_seeded,
    sample_seed_pool,
    score_arm,
    seeded_ordered_cells,
    summarize,
    write_dat_trees,
)
from run_mp100_reseed_nn import _seed_physically_valid  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--ckpt", default="results/flow_seedgen/pxrd_indexer_full6m_v2/best.pt")
    ap.add_argument(
        "--stats",
        default="data/processed/lattice_gstar6_stats_full_niggli_seed42.json",
    )
    ap.add_argument("--imins", default="5.0,1.0", help="comma-separated intensity thresholds")
    ap.add_argument("--k", type=int, default=100, help="seeds sampled per threshold")
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-peaks-nn", type=int, default=48)
    ap.add_argument("--mcm-intensity-min", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout-native", type=int, default=3600)
    ap.add_argument("--timeout-seeded", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default="results/flow_seedgen/cnrs_e2e_multithresh_5_1",
    )
    ap.add_argument(
        "--reuse-native-from",
        default="results/flow_seedgen/cnrs_e2e_k100_v2",
        help="copy/hardlink native_mcmaille from this prior run (empty = run native)",
    )
    ap.add_argument("--skip-sample", action="store_true")
    ap.add_argument("--skip-seeded", action="store_true")
    ap.add_argument(
        "--rerank",
        choices=["none", "linear"],
        default="none",
        help="none=McM20; linear=V0 equal-weight reranker",
    )
    return ap.parse_args()


def prepare_base_items(args) -> list[dict]:
    """Truth + McMaille peaks (fixed I>=5 protocol) + raw paperlike peaks."""
    cnrs = Path(args.cnrs_dir)
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)

    items = []
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        csv_path, cif_path = cnrs / f"{sid}.csv", cnrs / f"{sid}_sg.cif"
        if not csv_path.exists() or not cif_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tt_all, ii_all, meta = pick_peaks_paperlike(
            df["two_theta_deg"].to_numpy(), df["intensity"].to_numpy()
        )
        keep_mcm = ii_all >= args.mcm_intensity_min
        tt_m, ii_m = tt_all[keep_mcm], ii_all[keep_mcm]
        if len(tt_m) < 3:
            print(f"skip {sid}: <3 peaks at I>={args.mcm_intensity_min}", flush=True)
            continue
        try:
            truth = truth_cells(cif_path)
        except Exception as e:
            print(f"skip cif {sid}: {e}", flush=True)
            continue

        order = np.argsort(tt_m)
        tt_mcm = tt_m[order][:MAX_MCM_PEAKS]
        ii_mcm = np.maximum(ii_m[order][:MAX_MCM_PEAKS], 1.0)

        items.append(
            {
                "sample_id": sid,
                "cif": str(cif_path),
                "formula": row.get("formula"),
                "prim": truth["prim"],
                "conv": truth["conv"],
                "system": truth["system"],
                "tt_all": tt_all.astype(float),
                "ii_all": ii_all.astype(float),
                "tt_mcm": tt_mcm.astype(float),
                "ii_mcm": ii_mcm.astype(float),
                # placeholders required by write_dat_trees / sample_seed_pool API
                "tt_nn": tt_mcm.astype(float),
                "ii_nn": ii_mcm.astype(float),
                "n_peaks_found": int(meta["n_found"]),
            }
        )
    return items


def nn_view(item: dict, imin: float, max_peaks: int) -> dict:
    keep = item["ii_all"] >= imin
    tt, ii = item["tt_all"][keep], item["ii_all"][keep]
    if len(tt) > max_peaks:
        top = np.argsort(-ii)[:max_peaks]
        tt, ii = tt[top], ii[top]
        order = np.argsort(tt)
        tt, ii = tt[order], ii[order]
    out = dict(item)
    out["tt_nn"] = tt.astype(float)
    out["ii_nn"] = ii.astype(float)
    return out


def dedupe_cells(cells: list[list[float]], atol: float = 1e-3) -> list[list[float]]:
    """O(n) dedupe via rounded keys (allclose was O(n²) and stalls at K≈1000×2)."""
    # Quantize so nearby cells collide; scale matches historical atol≈1e-3 Å/°.
    q = max(atol, 1e-4)
    seen: set[tuple[int, ...]] = set()
    uniq: list[list[float]] = []
    for c in cells:
        if not _seed_physically_valid(c):
            continue
        key = tuple(int(round(float(x) / q)) for x in c[:6])
        if key in seen:
            continue
        seen.add(key)
        uniq.append([float(x) for x in c[:6]])
    return uniq


def sample_union_pool(args, items: list[dict], pool_path: Path) -> dict:
    imins = [float(x) for x in args.imins.split(",")]
    per: dict[str, dict] = {
        it["sample_id"]: {
            "raw_pred": None,
            "candidates": [],
            "by_threshold": {},
            "n_peaks_by_threshold": {},
        }
        for it in items
    }

    for imin in imins:
        print(f"==== sample seeds at I>={imin} K={args.k} ====", flush=True)
        views = []
        for it in items:
            v = nn_view(it, imin, args.max_peaks_nn)
            if len(v["tt_nn"]) < 3:
                print(f"  skip {it['sample_id']} at I>={imin}: <3 peaks", flush=True)
                continue
            views.append(v)
            per[it["sample_id"]]["n_peaks_by_threshold"][str(imin)] = len(v["tt_nn"])

        # sample_seed_pool writes its own file; use a temp path per threshold
        tmp = pool_path.parent / f"_tmp_pool_i{imin}.json"
        sub = sample_seed_pool(args, views, tmp)
        for sid, rec in sub["per_sample"].items():
            cands = rec["candidates"]
            per[sid]["by_threshold"][str(imin)] = cands
            if per[sid]["raw_pred"] is None and cands:
                per[sid]["raw_pred"] = cands[0]
            per[sid]["candidates"].extend(cands)

    for sid, rec in per.items():
        before = len(rec["candidates"])
        rec["candidates"] = dedupe_cells(rec["candidates"])
        rec["n_raw"] = before
        rec["n_union"] = len(rec["candidates"])
        if rec["raw_pred"] is None and rec["candidates"]:
            rec["raw_pred"] = rec["candidates"][0]

    payload = {
        "summary": {
            "ckpt": args.ckpt,
            "imins": imins,
            "top_k_per_threshold": args.k,
            "sample_steps": args.sample_steps,
            "seed": args.seed,
            "n_samples": len(per),
            "wavelength": WAVELENGTH,
            "mean_union": float(np.mean([r["n_union"] for r in per.values()])) if per else 0.0,
        },
        "per_sample": per,
    }
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(payload))
    print(
        f"union pool: mean |candidates|={payload['summary']['mean_union']:.1f} "
        f"wrote {pool_path}",
        flush=True,
    )
    return payload


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    native_src = out / "dat_native"
    seeded_src = out / "dat_seeded"
    native_run = out / "native_mcmaille"
    seeded_run = out / "indexer_union"
    pool_path = out / "pool_union.json"
    report_path = out / "l4_prim_compare.json"

    print("==== prepare CNRS samples ====", flush=True)
    items = prepare_base_items(args)
    print(f"usable samples: {len(items)}", flush=True)
    write_dat_trees(items, native_src, seeded_src)

    meta = {
        "cnrs_dir": args.cnrs_dir,
        "ckpt": args.ckpt,
        "imins": [float(x) for x in args.imins.split(",")],
        "k_per_threshold": args.k,
        "mcm_intensity_min": args.mcm_intensity_min,
        "max_peaks_nn": args.max_peaks_nn,
        "wavelength": WAVELENGTH,
        "peak_protocol": (
            f"NN: paperlike pick at each I in {args.imins}, K={args.k} each, "
            f"union+dedupe; McMaille .dat: I>={args.mcm_intensity_min} first20 by 2θ"
        ),
        "n_samples": len(items),
        "sample_ids": [it["sample_id"] for it in items],
    }
    (out / "protocol.json").write_text(json.dumps(meta, indent=2))

    if args.skip_sample and pool_path.exists():
        print(f"reuse pool {pool_path}", flush=True)
        pool = json.loads(pool_path.read_text())
    else:
        t0 = time.time()
        pool = sample_union_pool(args, items, pool_path)
        print(f"sampling done in {time.time()-t0:.0f}s", flush=True)

    # native arm
    reuse = Path(args.reuse_native_from) if args.reuse_native_from else None
    if reuse and not reuse.is_absolute():
        reuse = ROOT / reuse
    if reuse and (reuse / "native_mcmaille").exists():
        if native_run.exists():
            shutil.rmtree(native_run)
        print(f"hardlink-copy native from {reuse / 'native_mcmaille'}", flush=True)
        shutil.copytree(reuse / "native_mcmaille", native_run, copy_function=os_link)
    elif native_run.exists() and any(native_run.iterdir()):
        print(f"reuse existing native {native_run}", flush=True)
    else:
        run_native(items, native_src, native_run, args.workers, args.timeout_native)

    if args.skip_seeded and seeded_run.exists() and any(seeded_run.iterdir()):
        print(f"reuse seeded {seeded_run}", flush=True)
    else:
        # run_seeded expects pool["per_sample"][sid]["candidates"] and uses args.k
        # as a slice cap — raise k so the full union is injected.
        args.k = max(
            args.k,
            max((len(r["candidates"]) for r in pool["per_sample"].values()), default=args.k),
        )
        print(f"seeded McMaille with union pool (inject up to K={args.k})", flush=True)
        run_seeded(pool, seeded_src, seeded_run, args)

    print("==== score primitive L4 ====", flush=True)

    def native_fn(sid):
        return native_ordered_cells(native_run / sid / f"{sid.replace('-', '_')}.imp")

    def seeded_fn(sid):
        return seeded_ordered_cells(seeded_run, sid, rerank=args.rerank)

    # also score the raw union pool (pre-McMaille) for attribution
    raw_rows = []
    for it in items:
        cands = pool["per_sample"][it["sample_id"]]["candidates"]
        from remeasure_l4_prim_vs_conv import l4

        strict = [l4(c, it["prim"])[1] for c in cands]
        loose = [l4(c, it["prim"])[0] for c in cands]
        raw_rows.append(
            {
                "sample_id": it["sample_id"],
                "system": it["system"],
                "n_pool": len(cands),
                "prim": {
                    "lib_loose": any(loose),
                    "lib_strict": any(strict),
                    "top1_loose": bool(loose[:1] and loose[0]),
                    "top1_strict": bool(strict[:1] and strict[0]),
                    "top20_loose": any(loose[:20]),
                    "top20_strict": any(strict[:20]),
                },
            }
        )

    native_rows = score_arm(items, native_fn)
    seeded_rows = score_arm(items, seeded_fn)
    report = {
        "protocol": meta,
        "raw_union_pool": {"pool_json": str(pool_path), **summarize(raw_rows), "per_sample": raw_rows},
        "native_mcmaille": {
            "run_dir": str(native_run),
            **summarize(native_rows),
            "per_sample": native_rows,
        },
        "indexer_union": {
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
            f"{label:18s} n={block['n']}  "
            f"L4-strict Top-1={p['top1_strict']:.1%}  "
            f"Top-20={p['top20_strict']:.1%}  "
            f"lib={p['lib_strict']:.1%}  "
            f"(loose @1={p['top1_loose']:.1%} @20={p['top20_loose']:.1%})",
            flush=True,
        )

    print("==== CNRS primitive L4 (strict) — multi-threshold union ====", flush=True)
    show("raw union pool", report["raw_union_pool"])
    show("native McMaille", report["native_mcmaille"])
    show("indexer union", report["indexer_union"])
    print(f"wrote {report_path}", flush=True)


def os_link(src, dst):
    import os

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
