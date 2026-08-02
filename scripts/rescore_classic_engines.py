#!/usr/bin/env python3
"""Re-score existing classic-engine outputs after a parser fix (no re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from remeasure_l4_prim_vs_conv import l4, truth_cells  # noqa: E402
from run_cnrs_classic_engines import parse_cells, summarize  # noqa: E402

CNRS = Path("/nanolab/users/wyx/CNRS")
BIN = {
    "dicvol06": ROOT / "third_party/dicvol06/dicvol91",
    "treor90": ROOT / "third_party/treor90/treor90",
    "ito": ROOT / "third_party/ito/ito13",
}


def load_truth_cache(bench: Path) -> dict:
    cache_path = bench / "truth_prim_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    samples = json.loads((bench / "classic_samples.json").read_text())["sample_ids"]
    cache = {}
    for sid in samples:
        t = truth_cells(CNRS / f"{sid}_sg.cif")
        cache[sid] = {"system": t["system"], "prim": t["prim"]}
    cache_path.write_text(json.dumps(cache))
    return cache


def gather_text(work: Path) -> str:
    chunks = []
    # Prefer condensed .con; never ingest multi-GB runaway dumps.
    for name in ("input.con", "input.imp", "input.out", "input.lst"):
        p = work / name
        if not p.exists():
            continue
        size = p.stat().st_size
        if size > 5_000_000:
            continue
        chunks.append(p.read_bytes()[:2_000_000].decode("utf-8", errors="replace"))
    return "\n".join(chunks)


def score_dir(work: Path, eng: str, truth_cache: dict) -> dict:
    sid = work.name
    if sid not in truth_cache:
        raw = truth_cells(CNRS / f"{sid}_sg.cif")
        truth_cache[sid] = {"system": raw["system"], "prim": raw["prim"]}
    t = truth_cache[sid]
    cells = parse_cells(gather_text(work))
    flags = [l4(c, t["prim"]) for c in cells]
    loose = [f[0] for f in flags]
    strict = [f[1] for f in flags]

    def first(arr):
        for i, v in enumerate(arr, 1):
            if v:
                return i
        return None

    old = {}
    if (work / "status.json").exists():
        old = json.loads((work / "status.json").read_text())
    row = {
        "sample_id": sid,
        "system": t["system"],
        "engine": eng,
        "status": "ok" if cells else "no_solution",
        "error": old.get("error", ""),
        "elapsed_s": old.get("elapsed_s"),
        "n_pool": len(cells),
        "prim": {
            "lib_loose": any(loose),
            "lib_strict": any(strict),
            "top1_loose": bool(loose[:1] and loose[0]),
            "top1_strict": bool(strict[:1] and strict[0]),
            "top20_loose": any(loose[:20]),
            "top20_strict": any(strict[:20]),
            "first_strict": first(strict),
            "first_loose": first(loose),
        }
        if cells
        else {
            "lib_loose": False,
            "lib_strict": False,
            "top1_loose": False,
            "top1_strict": False,
            "top20_loose": False,
            "top20_strict": False,
            "first_strict": None,
            "first_loose": None,
        },
    }
    (work / "candidates.json").write_text(
        json.dumps({"cells": cells[:50], "n_cells": len(cells)}, indent=2)
    )
    (work / "status.json").write_text(json.dumps(row, indent=2))
    return row


def rescore(eng: str, bench: Path) -> dict:
    base = bench / eng
    samples = json.loads((bench / "classic_samples.json").read_text())["sample_ids"]
    truth_cache = load_truth_cache(bench)
    rows = []
    for sid in samples:
        work = base / sid
        if not work.is_dir() or not (work / "input.dat").exists():
            t = truth_cache[sid]
            rows.append(
                {
                    "sample_id": sid,
                    "system": t["system"],
                    "engine": eng,
                    "status": "missing",
                    "error": "no run artifact",
                    "elapsed_s": None,
                    "n_pool": 0,
                    "prim": {
                        "lib_loose": False,
                        "lib_strict": False,
                        "top1_loose": False,
                        "top1_strict": False,
                        "top20_loose": False,
                        "top20_strict": False,
                        "first_strict": None,
                        "first_loose": None,
                    },
                }
            )
            continue
        rows.append(score_dir(work, eng, truth_cache))
        if len(rows) % 20 == 0:
            print(f"  {eng} rescored {len(rows)}/{len(samples)}", flush=True)
    summary = {
        "engine": eng,
        "status": "ok",
        "binary": str(BIN[eng].resolve()),
        "binary_how": "rescored native",
        "use_wine": False,
        **summarize(rows, len(samples)),
        "per_sample": rows,
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    p = summary["prim"]
    print(
        f"{eng}: Top-1={p['top1_strict']:.1%} Top-20={p['top20_strict']:.1%} "
        f"lib={p['lib_strict']:.1%} n_ok={sum(1 for r in rows if r['status']=='ok')}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="results/cnrs_benchmark")
    ap.add_argument("--engines", nargs="+", default=["dicvol06", "treor90"])
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    if not bench.is_absolute():
        bench = ROOT / bench
    for eng in args.engines:
        rescore(eng, bench)


if __name__ == "__main__":
    main()
