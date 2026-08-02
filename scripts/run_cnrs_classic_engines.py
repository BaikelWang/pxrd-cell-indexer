#!/usr/bin/env python3
"""Batch-run classic powder indexers (DICVOL91 / TREOR90 / ITO13) on CNRS.

Reuses paperlike peak picking from ``run_cnrs_e2e_compare.py`` / ``eval_cnrs_seedpool``.
Engines without a detectable binary are recorded as ``unavailable`` (not dropped).

Native Linux binaries (gfortran -std=legacy) live under ``third_party/<engine>/``:
  dicvol06/dicvol91, treor90/treor90, ito/ito13
(DICVOL06 Windows binary was unreachable; DICVOL91 is the closest public drop-in.)

Output layout::
  results/cnrs_benchmark/<engine>/<sid>/
    peaks.json, input.dat, stdout.txt, candidates.json, status.json
  results/cnrs_benchmark/<engine>/summary.json
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cnrs_seedpool import pick_peaks_paperlike  # noqa: E402
from remeasure_l4_prim_vs_conv import l4, truth_cells  # noqa: E402

WAVELENGTH = 1.5406
MAX_PEAKS = 20
ENGINES = ("dicvol06", "treor90", "ito")

# Candidate lattice line patterns (engine-dependent; keep permissive).
CELL_PATTERNS = [
    # a b c alpha beta gamma [FOM...]
    re.compile(
        r"(?P<a>\d+\.\d+)\s+(?P<b>\d+\.\d+)\s+(?P<c>\d+\.\d+)\s+"
        r"(?P<al>\d+\.\d+)\s+(?P<be>\d+\.\d+)\s+(?P<ga>\d+\.\d+)"
    ),
]
# Named forms used by DICVOL / TREOR condensed output.
# Keep patterns line-local (no re.S + .*?) — large .imp files otherwise hang.
NAMED_CELL_PATTERNS = [
    re.compile(
        r"A\s*=\s*(?P<a>\d+\.\d+)\s+B\s*=\s*(?P<b>\d+\.\d+)\s+C\s*=\s*(?P<c>\d+\.\d+)"
        r"(?:\s+ALFA?\s*=\s*(?P<al>\d+\.\d+)\s+BETA\s*=\s*(?P<be>\d+\.\d+)\s+"
        r"GAMMA\s*=\s*(?P<ga>\d+\.\d+))?",
        re.I,
    ),
    re.compile(
        r"DIRECT PARAMETERS\s*:\s*A=\s*(?P<a>\d+\.\d+)\s+B=\s*(?P<b>\d+\.\d+)\s+"
        r"C=\s*(?P<c>\d+\.\d+)(?:\s+ALPHA=\s*(?P<al>\d+\.\d+)\s+BETA=\s*(?P<be>\d+\.\d+)\s+"
        r"GAMMA=\s*(?P<ga>\d+\.\d+))?",
        re.I,
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument(
        "--out-dir",
        default="results/cnrs_benchmark",
        help="parent benchmark directory",
    )
    ap.add_argument(
        "--engines",
        nargs="+",
        default=list(ENGINES),
        choices=list(ENGINES),
    )
    ap.add_argument("--intensity-min", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout-dicvol06", type=int, default=300)
    ap.add_argument("--timeout-treor90", type=int, default=180)
    ap.add_argument("--timeout-ito", type=int, default=180)
    ap.add_argument(
        "--max-output-mb",
        type=float,
        default=20.0,
        help="kill engine if any .imp/.con grows beyond this (runaway dumps)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--force-unavailable",
        action="store_true",
        help="skip binary detection and write unavailable for all selected engines",
    )
    return ap.parse_args()


def find_binary(engine: str) -> tuple[Path | None, str]:
    """Return (path, how) or (None, reason). Prefer native Linux, then Wine .exe."""
    base = ROOT / "third_party" / engine
    names = {
        "dicvol06": [
            "dicvol91",  # native gfortran build (preferred)
            "dicvol06",
            "DICVOL06",
            "DICVOL91.exe",
            "dicvol91.exe",
            "dicvol06.exe",
            "DICVOL06.EXE",
            "dicvol04.exe",
        ],
        "treor90": [
            "treor90",  # native gfortran build (preferred)
            "TREOR90",
            "TREOR90.exe",
            "treor90.exe",
            "treor.exe",
        ],
        "ito": [
            "ito13",  # native gfortran build (preferred)
            "ito12",
            "ITO12",
            "ito",
            "ITO",
            "ito13.exe",
            "ito12.exe",
            "ITO12.EXE",
            "ito.exe",
            "ITO.EXE",
        ],
    }[engine]
    env_key = f"{engine.upper()}_BIN"
    if os.environ.get(env_key):
        p = Path(os.environ[env_key])
        if p.exists():
            return p, f"env:{env_key}"
    if not base.is_dir():
        return None, f"missing dir {base}"
    for name in names:
        p = base / name
        if p.is_file():
            return p, f"third_party/{engine}/{name}"
    # any executable under the tree
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith((".md", ".txt", ".dat", ".pdf", ".html")):
            continue
        if "dicvol" in low or "treor" in low or low.startswith("ito"):
            return p, f"discovered:{p.relative_to(ROOT)}"
    return None, f"no binary under {base}"


def wine_needed(path: Path) -> bool:
    return path.suffix.lower() == ".exe" or "windows" in path.name.lower()


def prepare_samples(cnrs_dir: Path, intensity_min: float, limit: int) -> list[dict]:
    manifest = pd.read_csv(cnrs_dir / "cnrs_manifest.csv")
    if limit:
        manifest = manifest.head(limit)
    items = []
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        csv_path = cnrs_dir / f"{sid}.csv"
        cif_path = cnrs_dir / f"{sid}_sg.cif"
        if not csv_path.exists() or not cif_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tt, ii, _ = pick_peaks_paperlike(
            df["two_theta_deg"].to_numpy(), df["intensity"].to_numpy()
        )
        keep = ii >= intensity_min
        tt, ii = tt[keep], ii[keep]
        if len(tt) < 3:
            continue
        try:
            truth = truth_cells(cif_path)
        except Exception:
            continue
        order = np.argsort(tt)
        tt20 = tt[order][:MAX_PEAKS]
        ii20 = np.maximum(ii[order][:MAX_PEAKS], 1.0)
        items.append(
            {
                "sample_id": sid,
                "system": truth["system"],
                "truth_prim": truth["prim"],
                "two_theta": tt20.tolist(),
                "intensity": ii20.tolist(),
            }
        )
    return items


def write_dicvol_dat(path: Path, sid: str, two_theta: list[float], wave: float) -> None:
    """DICVOL91 cards 1–5 + peak list (2θ). Stem-only launch appends ``.dat``."""
    n = len(two_theta)
    # Card2: N ITYPE=2(2θ) JC JT JH JO JM JTR — search all systems
    lines = [
        f"{sid}\n",
        f"{n} 2 1 1 1 1 1 1\n",
        "0. 0. 0. 0. 0. 0. 0.\n",
        f"{wave} 0. 0. 0.\n",
        "0. 0. 0.\n",
    ]
    for tt in two_theta:
        lines.append(f"{tt:.3f}\n")
    path.write_text("".join(lines))


def write_treor_dat(path: Path, sid: str, two_theta: list[float], wave: float) -> None:
    """Match the cristal.org TREOR90 sample layout (keywords after blank line)."""
    lines = [f" {sid}\n"]
    for tt in two_theta:
        lines.append(f"{tt:16.5f}\n")
    lines.append("\n")
    lines.append(f" WAVE={wave},\n")
    lines.append(" CHOICE=3,\n")
    lines.append(" MONO=130.0,\n")
    lines.append(" VOL=2000,\n")
    lines.append(" END*\n")
    lines.append("  0.00\n")
    path.write_text("".join(lines))


def write_ito_dat(path: Path, sid: str, two_theta: list[float], wave: float) -> None:
    """ITO13: title + fixed-column params + one 2θ per line (ascending)."""
    # Col1 MAN=9 suppresses the long instruction dump; wavelength in cols 21-30.
    param = f"9009 0 0 0          {wave:10.5f}"
    lines = [f"{sid}\n", f"{param}\n", " 0.000\n"]
    for tt in two_theta:
        lines.append(f"{tt:10.5f}\n")
    lines.append("0.0\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def _sane_cell(cell: list[float]) -> bool:
    if not (1.0 < cell[0] < 200 and 1.0 < cell[1] < 200 and 1.0 < cell[2] < 200):
        return False
    return all(20.0 < x < 180.0 for x in cell[3:])


def parse_cells(text: str) -> list[list[float]]:
    cells: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()

    def add(a, b, c, al, be, ga):
        cell = [float(a), float(b), float(c), float(al), float(be), float(ga)]
        if not _sane_cell(cell):
            return
        key = tuple(round(x, 4) for x in cell)
        if key in seen:
            return
        seen.add(key)
        cells.append(cell)

    # 1) Named A=/B=/C=/ALFA= blocks (may span 1–2 lines)
    for pat in NAMED_CELL_PATTERNS:
        for m in pat.finditer(text):
            al = m.groupdict().get("al") or "90.0"
            be = m.groupdict().get("be") or "90.0"
            ga = m.groupdict().get("ga") or "90.0"
            # Orthorhombic DICVOL often omits angles → default 90
            add(m.group("a"), m.group("b"), m.group("c"), al, be, ga)

    # 2) TREOR compact: A=.. B=.. C=.. \n ALFA=.. BETA=.. GAMMA=..
    for m in re.finditer(
        r"A=\s*(\d+\.\d+)\s+B=\s*(\d+\.\d+)\s+C=\s*(\d+\.\d+)\s*\n"
        r"\s*ALFA=\s*(\d+\.\d+)\s+BETA=\s*(\d+\.\d+)\s+GAMMA=\s*(\d+\.\d+)",
        text,
        re.I,
    ):
        add(*m.groups())

    # 2b) TREOR refined cycle block (one parameter per line):
    #   A = 14.055882  0.003397 A   ALFA = 90.000000  0.000000 DEG
    #   B = 14.055882  ...          BETA = ...
    #   C =  6.007120  ...         GAMMA = ...
    for m in re.finditer(
        r"A\s*=\s*(?P<a>\d+\.\d+)[^\n]*ALFA\s*=\s*(?P<al>\d+\.\d+)[^\n]*\n"
        r"[^\n]*B\s*=\s*(?P<b>\d+\.\d+)[^\n]*BETA\s*=\s*(?P<be>\d+\.\d+)[^\n]*\n"
        r"[^\n]*C\s*=\s*(?P<c>\d+\.\d+)[^\n]*GAMMA\s*=\s*(?P<ga>\d+\.\d+)",
        text,
        re.I,
    ):
        add(
            m.group("a"),
            m.group("b"),
            m.group("c"),
            m.group("al"),
            m.group("be"),
            m.group("ga"),
        )

    # 3) ITO "DIRECT CONSTANTS" table: a b c al be ga [volume]
    for block in re.finditer(
        r"DIRECT CONSTANTS OF THESE LATTICES.*?GAMMA\s+VOLUME\s*\n(.*?)(?:\n\s*\n|\Z)",
        text,
        re.I | re.S,
    ):
        for line in block.group(1).splitlines():
            m = re.match(
                r"\s*(?P<a>\d+\.\d+)\s+(?P<b>\d+\.\d+)\s+(?P<c>\d+\.\d+)\s+"
                r"(?P<al>\d+\.\d+)\s+(?P<be>\d+\.\d+)\s+(?P<ga>\d+\.\d+)"
                r"(?:\s+(?P<vol>\d+\.\d+))?",
                line,
            )
            if not m:
                continue
            add(
                m.group("a"),
                m.group("b"),
                m.group("c"),
                m.group("al"),
                m.group("be"),
                m.group("ga"),
            )

    # 4) Remaining dense rows — only if a 7th volume column is present (avoids Q-tables)
    for line in text.splitlines():
        up = line.upper()
        if "Q(" in up or "INDEXED" in up or "SST-" in up:
            continue
        m = re.match(
            r"\s*(?P<a>\d+\.\d+)\s+(?P<b>\d+\.\d+)\s+(?P<c>\d+\.\d+)\s+"
            r"(?P<al>\d+\.\d+)\s+(?P<be>\d+\.\d+)\s+(?P<ga>\d+\.\d+)\s+"
            r"(?P<vol>\d+\.\d+)\s*$",
            line,
        )
        if not m:
            continue
        vol = float(m.group("vol"))
        if not (10.0 < vol < 20000.0):
            continue
        if max(float(m.group("a")), float(m.group("b")), float(m.group("c"))) > 80:
            continue
        add(
            m.group("a"),
            m.group("b"),
            m.group("c"),
            m.group("al"),
            m.group("be"),
            m.group("ga"),
        )
    return cells


def run_one(payload: dict) -> dict:
    sid = payload["sample_id"]
    engine = payload["engine"]
    work = Path(payload["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    (work / "peaks.json").write_text(
        json.dumps(
            {
                "sample_id": sid,
                "wavelength": WAVELENGTH,
                "two_theta": payload["two_theta"],
                "intensity": payload["intensity"],
            },
            indent=2,
        )
    )
    # Engines prompt for a stem and open ``<stem>.dat`` themselves.
    stem = "input"
    dat = work / f"{stem}.dat"
    if engine == "dicvol06":
        write_dicvol_dat(dat, sid, payload["two_theta"], WAVELENGTH)
    elif engine == "treor90":
        write_treor_dat(dat, sid, payload["two_theta"], WAVELENGTH)
    else:
        write_ito_dat(dat, sid, payload["two_theta"], WAVELENGTH)

    binary = Path(payload["binary"])
    use_wine = bool(payload["use_wine"])
    timeout = int(payload["timeout"])
    stdout_path = work / "stdout.txt"
    t0 = time.time()
    status = "ok"
    err = ""
    # TREOR asks for "any number" before exit; ITO may SEGV after writing .imp.
    stdin_text = f"{stem}\n0\n"
    max_bytes = int(float(payload.get("max_output_mb", 20.0)) * 1024 * 1024)
    try:
        cmd = [str(binary)]
        if use_wine:
            wine = shutil.which("wine64") or shutil.which("wine")
            if not wine:
                raise RuntimeError("wine not found")
            cmd = [wine, str(binary)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(work),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        try:
            proc.stdin.write(stdin_text)
        finally:
            proc.stdin.close()
        t_deadline = time.time() + timeout
        killed_for = ""
        while proc.poll() is None:
            if time.time() > t_deadline:
                proc.kill()
                killed_for = f"timeout>{timeout}s"
                break
            for p in work.iterdir():
                if (
                    p.suffix.lower() in {".imp", ".con", ".out", ".lst"}
                    and p.stat().st_size > max_bytes
                ):
                    proc.kill()
                    killed_for = f"output>{max_bytes // (1024 * 1024)}MB:{p.name}"
                    break
            if killed_for:
                break
            time.sleep(0.2)
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.wait(timeout=5)
        out = (stdout or "") + "\n" + (stderr or "")
        extras = []
        for p in sorted(work.iterdir()):
            if p.suffix.lower() not in {".imp", ".out", ".lst", ".prf", ".new", ".con"}:
                continue
            # Cap read size — runaway dumps must not enter the regexer.
            raw = p.read_bytes()[: min(p.stat().st_size, 2_000_000)]
            extras.append(raw.decode("utf-8", errors="replace"))
        out = out + "\n" + "\n".join(extras)
        stdout_path.write_text(out[:2_000_000])
        if killed_for.startswith("timeout"):
            status = "timeout"
            err = killed_for
        elif killed_for.startswith("output"):
            status = "ok" if extras else "error"
            err = killed_for
        elif proc.returncode not in (0, None) and not extras:
            status = "nonzero_exit"
            err = f"rc={proc.returncode}"
        elif proc.returncode not in (0, None) and extras:
            status = "ok"
            err = f"rc={proc.returncode} (outputs present)"
    except Exception as e:
        status = "error"
        err = str(e)
        stdout_path.write_text(err)
        out = err

    cells = parse_cells(out) if status in ("ok", "nonzero_exit", "timeout", "error") else []
    if status in ("timeout", "error") and cells:
        status = "ok"
        err = (err + "; recovered cells from partial output").strip("; ")
    if status == "ok" and not cells:
        status = "no_solution"
    if status == "nonzero_exit" and cells:
        status = "ok"

    # score vs primitive
    truth = payload["truth_prim"]
    flags = [l4(c, truth) for c in cells]
    loose = [f[0] for f in flags]
    strict = [f[1] for f in flags]

    def first(arr):
        for i, v in enumerate(arr, 1):
            if v:
                return i
        return None

    candidates = {
        "cells": cells[:50],
        "n_cells": len(cells),
    }
    (work / "candidates.json").write_text(json.dumps(candidates, indent=2))
    row = {
        "sample_id": sid,
        "system": payload["system"],
        "engine": engine,
        "status": status,
        "error": err,
        "elapsed_s": time.time() - t0,
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
        },
    }
    # misses stay False for all match keys when no cells
    if not cells:
        row["prim"] = {
            "lib_loose": False,
            "lib_strict": False,
            "top1_loose": False,
            "top1_strict": False,
            "top20_loose": False,
            "top20_strict": False,
            "first_strict": None,
            "first_loose": None,
        }
    (work / "status.json").write_text(json.dumps(row, indent=2))
    return row


def summarize(rows: list[dict], n_denom: int) -> dict:
    # denominator fixed to full sample list (misses included)
    n = max(n_denom, 1)

    def rate(key):
        return sum(1 for r in rows if r["prim"][key]) / n

    from collections import defaultdict

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["system"]].append(r)
    by_sys = {}
    for sys, rs in sorted(buckets.items()):
        m = max(len(rs), 1)
        by_sys[sys] = {
            "n": len(rs),
            "top1_strict": sum(1 for r in rs if r["prim"]["top1_strict"]) / m,
            "top20_strict": sum(1 for r in rs if r["prim"]["top20_strict"]) / m,
            "lib_strict": sum(1 for r in rs if r["prim"]["lib_strict"]) / m,
        }
    n_timeout = sum(1 for r in rows if r["status"] == "timeout")
    n_nosol = sum(1 for r in rows if r["status"] in ("no_solution", "error", "nonzero_exit"))
    return {
        "n": n_denom,
        "n_rows": len(rows),
        "mean_pool": sum(r["n_pool"] for r in rows) / max(len(rows), 1),
        "n_timeout": n_timeout,
        "n_fail": n_nosol,
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


def write_unavailable(engine: str, out_eng: Path, reason: str, n: int) -> dict:
    out_eng.mkdir(parents=True, exist_ok=True)
    summary = {
        "engine": engine,
        "status": "unavailable",
        "reason": reason,
        "n": n,
        "prim": None,
        "by_system": {},
        "per_sample": [],
    }
    (out_eng / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    items = prepare_samples(Path(args.cnrs_dir), args.intensity_min, args.limit)
    print(f"usable samples: {len(items)}", flush=True)
    (out / "classic_samples.json").write_text(
        json.dumps(
            {
                "n": len(items),
                "sample_ids": [it["sample_id"] for it in items],
                "wavelength": WAVELENGTH,
                "max_peaks": MAX_PEAKS,
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
        if args.force_unavailable:
            binary, how = None, "forced unavailable"
        else:
            binary, how = find_binary(engine)
        print(f"==== {engine}: {how} ====", flush=True)
        if binary is None:
            all_summaries[engine] = write_unavailable(engine, out_eng, how, len(items))
            continue

        use_wine = wine_needed(binary)
        payloads = []
        for it in items:
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

        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, p): p["sample_id"] for p in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                rows.append(row)
                if i % 10 == 0 or i == len(futs):
                    print(f"  {engine} {i}/{len(futs)} last={row['sample_id']} {row['status']}", flush=True)
        rows.sort(key=lambda r: r["sample_id"])
        summary = {
            "engine": engine,
            "status": "ok",
            "binary": str(binary),
            "binary_how": how,
            "use_wine": use_wine,
            **summarize(rows, len(items)),
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

    (out / "classic_engines_overview.json").write_text(
        json.dumps(all_summaries, indent=2)
    )
    print(f"wrote {out / 'classic_engines_overview.json'}", flush=True)


if __name__ == "__main__":
    main()
