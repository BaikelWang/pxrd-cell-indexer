#!/usr/bin/env python3
"""Batch-run classic powder indexers (DICVOL91 / TREOR90 / ITO13) on MP100.

Peak lists are taken from the same McMaille ``.dat`` files used by JADE/Mc
benchmarks (``mp100_seeded_phase5``), first 20 peaks by 2θ, λ from the .dat
header (typically 1.54184 Å). Scoring uses primitive L4-strict — same ruler as
the CNRS competitor board.

Reuses writers / parsers / runner from ``run_cnrs_classic_engines.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_cnrs_classic_engines as classic  # noqa: E402
from remeasure_l4_prim_vs_conv import truth_cells  # noqa: E402

CIF_DIR = ROOT / "data" / "MP-100samples-benchmark"
DEFAULT_SRC = ROOT / "third_party" / "McMaille" / "run_lab" / "mp100_seeded_phase5"
ENGINES = ("dicvol06", "treor90", "ito")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-run", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--cif-dir", type=Path, default=CIF_DIR)
    ap.add_argument("--out-dir", default="results/mp100_benchmark")
    ap.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    ap.add_argument("--max-peaks", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout-dicvol06", type=int, default=300)
    ap.add_argument("--timeout-treor90", type=int, default=180)
    ap.add_argument("--timeout-ito", type=int, default=180)
    ap.add_argument("--max-output-mb", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def read_dat_peaks(dat: Path) -> tuple[float, list[float], list[float]]:
    """Parse McMaille .dat → (wavelength, two_theta, intensity)."""
    lines = dat.read_text(errors="replace").splitlines()
    wave = 1.54184
    peaks: list[tuple[float, float]] = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 3:
            try:
                w = float(parts[0])
                zp = float(parts[1])
                # wavelength / zeropoint / NGRID header
                if 0.5 < w < 3.0 and abs(zp) < 1.0:
                    wave = w
                    continue
            except ValueError:
                pass
        if len(parts) >= 2:
            try:
                tt = float(parts[0])
                ii = float(parts[1])
            except ValueError:
                continue
            if 0.0 < tt < 180.0 and ii >= 0.0:
                peaks.append((tt, ii))
    peaks.sort(key=lambda x: x[0])
    if not peaks:
        raise ValueError(f"no peaks in {dat}")
    tt = [p[0] for p in peaks]
    ii = [max(p[1], 1.0) for p in peaks]
    return wave, tt, ii


def prepare_samples(
    src_run: Path, cif_dir: Path, max_peaks: int, limit: int
) -> tuple[list[dict], float]:
    sids = sorted(p.name for p in src_run.iterdir() if p.name.startswith("mp-"))
    if limit:
        sids = sids[:limit]
    items = []
    wave_ref = None
    for sid in sids:
        stem = sid.replace("-", "_")
        dat = src_run / sid / f"{stem}.dat"
        cif = cif_dir / f"{sid}.cif"
        if not dat.exists() or not cif.exists():
            continue
        wave, tt, ii = read_dat_peaks(dat)
        wave_ref = wave if wave_ref is None else wave_ref
        tt20 = tt[:max_peaks]
        ii20 = ii[:max_peaks]
        if len(tt20) < 3:
            continue
        truth = truth_cells(cif)
        items.append(
            {
                "sample_id": sid,
                "system": truth["system"],
                "truth_prim": truth["prim"],
                "two_theta": tt20,
                "intensity": ii20,
                "wavelength": wave,
            }
        )
    return items, float(wave_ref or 1.54184)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    items, wave = prepare_samples(args.src_run, args.cif_dir, args.max_peaks, args.limit)
    # Override module wavelength used by writers / peaks.json
    classic.WAVELENGTH = wave
    classic.MAX_PEAKS = args.max_peaks

    print(f"usable samples: {len(items)}  λ={wave}  max_peaks={args.max_peaks}", flush=True)
    (out / "classic_samples.json").write_text(
        json.dumps(
            {
                "n": len(items),
                "sample_ids": [it["sample_id"] for it in items],
                "wavelength": wave,
                "max_peaks": args.max_peaks,
                "src_run": str(args.src_run),
                "peak_protocol": "McMaille .dat first N by 2θ (same peak table as JADE/Mc)",
            },
            indent=2,
        )
    )

    timeouts = {
        "dicvol06": args.timeout_dicvol06,
        "treor90": args.timeout_treor90,
        "ito": args.timeout_ito,
    }

    all_summaries = {}
    for engine in args.engines:
        out_eng = out / engine
        binary, how = classic.find_binary(engine)
        print(f"==== {engine}: {how} ====", flush=True)
        if binary is None:
            all_summaries[engine] = classic.write_unavailable(
                engine, out_eng, how, len(items)
            )
            continue

        use_wine = classic.wine_needed(binary)
        payloads = []
        for it in items:
            # per-sample wavelength: temporarily stash; run_one uses classic.WAVELENGTH
            payloads.append(
                {
                    **it,
                    "engine": engine,
                    "binary": str(binary.resolve()),
                    "use_wine": use_wine,
                    "timeout": timeouts[engine],
                    "max_output_mb": args.max_output_mb,
                    "work_dir": str(out_eng / it["sample_id"]),
                }
            )

        # Writers use module WAVELENGTH; if any sample differs, write via custom path.
        # MP100 .dat files share one λ — set once.
        classic.WAVELENGTH = wave

        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(classic.run_one, p): p["sample_id"] for p in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                rows.append(row)
                if i % 10 == 0 or i == len(futs):
                    print(
                        f"  {engine} {i}/{len(futs)} last={row['sample_id']} {row['status']}",
                        flush=True,
                    )
        rows.sort(key=lambda r: r["sample_id"])
        summary = {
            "engine": engine,
            "status": "ok",
            "binary": str(binary),
            "binary_how": how,
            "use_wine": use_wine,
            "wavelength": wave,
            **classic.summarize(rows, len(items)),
            "per_sample": rows,
        }
        (out_eng / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[engine] = {
            k: summary[k]
            for k in (
                "engine",
                "status",
                "binary",
                "binary_how",
                "n",
                "prim",
                "by_system",
                "n_timeout",
                "n_fail",
            )
            if k in summary
        }
        p = summary["prim"]
        print(
            f"  {engine} L4-strict Top-1={p['top1_strict']:.1%} "
            f"Top-20={p['top20_strict']:.1%} lib={p['lib_strict']:.1%}",
            flush=True,
        )

    (out / "classic_engines_overview.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"wrote {out / 'classic_engines_overview.json'}", flush=True)


if __name__ == "__main__":
    main()
