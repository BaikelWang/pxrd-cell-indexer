#!/usr/bin/env python3
"""L4 match rate on MP100 against PRIMITIVE vs CONVENTIONAL truth.

Engines: JADE9 (Top-1), native McMaille (Top-1/Top-20), Ours (phase4 pool).

L4 loose  = find_mapping(ltol=0.05, atol=3deg)
L4 strict = loose AND |det(scale)-1| < 0.25

Motivation: RealPXRD-solver emits primitive cells, so conventional truth makes
genuine primitive solutions look like |det|=2/4 "subcells".

Self-contained: no project modules required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
CIF_DIR = PROJECT / "data/MP-100samples-benchmark"
PHASE4 = PROJECT / "third_party/McMaille/run_lab/mp100_seeded_phase4"
MCM_ORIG = PROJECT / "third_party/McMaille/run_lab/mp100_compare/original"
JADE_JSON = PROJECT / "results/jade9_mp100_top1_remeasure.json"
MCM_JSON = PROJECT / "results/native_mcmaille_mp100_top1_top20_remeasure.json"

LTOL, ATOL, DET_TOL = 0.05, 3.0, 0.25
TOPK = 20


# ---------------------------------------------------------------- truth cells
def truth_cells(cif: Path) -> dict:
    """Conventional and primitive standard lattices from a CIF."""
    s = Structure.from_file(cif)
    ana = SpacegroupAnalyzer(s, symprec=0.01)
    conv = ana.get_conventional_standard_structure().lattice
    prim = ana.get_primitive_standard_structure().lattice
    p6 = [prim.a, prim.b, prim.c, prim.alpha, prim.beta, prim.gamma]
    c6 = [conv.a, conv.b, conv.c, conv.alpha, conv.beta, conv.gamma]
    return {
        "conv": c6,
        "prim": p6,
        "conv_vol": float(conv.volume),
        "prim_vol": float(prim.volume),
        "z_ratio": float(conv.volume / prim.volume) if prim.volume > 0 else None,
        "system": ana.get_crystal_system(),
    }


# ------------------------------------------------------------------- matching
def l4(pred, truth) -> tuple[bool, bool, float | None]:
    """Return (loose, strict, det)."""
    if pred is None:
        return False, False, None
    try:
        r = Lattice.from_parameters(*pred).find_mapping(
            Lattice.from_parameters(*truth), ltol=LTOL, atol=ATOL
        )
        if r is None:
            return False, False, None
        det = abs(float(np.linalg.det(r[2])))
        return True, abs(det - 1.0) < DET_TOL, det
    except Exception:
        return False, False, None


def first_hit(ordered, truth, strict: bool) -> int | None:
    for i, p in enumerate(ordered, 1):
        lo, st, _ = l4(p, truth)
        if (st if strict else lo):
            return i
    return None


# --------------------------------------------------------------- data parsing
ALLCELLS_ROW = re.compile(
    r"^\s*\d+\s+\d+\s+\d+\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([A-Z])"
)


def parse_allcells(path: Path) -> list[dict]:
    """Parse phase4 .allcells: n_indexed McM20 volume Rp a b c al be ga bravais."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = ALLCELLS_ROW.match(line)
        if not m:
            continue
        g = m.groups()
        out.append(
            {
                "n_indexed": int(g[0]),
                "McM20": float(g[1]),
                "volume": float(g[2]),
                "Rp": float(g[3]),
                "params": [float(g[i]) for i in range(4, 10)],
                "bravais": g[10],
            }
        )
    return out


MCM_ROW = re.compile(
    r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([A-Z])\s*(.*)$"
)


def mcm_ordered(sid: str) -> list[list[float]]:
    """Native McMaille output: suggested-best first, then McM20 list."""
    imp = MCM_ORIG / sid / f"{sid.replace('-', '_')}.imp"
    if not imp.exists():
        return []
    text = imp.read_text(errors="replace")
    ordered, seen = [], set()

    def add(p):
        k = tuple(round(x, 4) for x in p)
        if k not in seen:
            seen.add(k)
            ordered.append(p)

    m = re.search(
        r"It is suggested that the correct cell could be\s*:.*?Bravais lattice\s*\n+"
        r"\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([A-Z])",
        text,
        flags=re.S | re.I,
    )
    if m:
        g = m.groups()
        add([float(g[i]) for i in range(3, 9)])

    blk = re.search(
        r"FINAL LIST OF CELL PROPOSALS, sorted by McM20\s*:.*?Bravais lattice\s*\n+(.*?)\n\s*=======",
        text,
        flags=re.S | re.I,
    )
    if blk:
        for line in blk.group(1).splitlines():
            mm = MCM_ROW.match(line.rstrip())
            if mm:
                g = mm.groups()
                add([float(g[i]) for i in range(4, 10)])
            if len(ordered) >= TOPK:
                break
    return ordered[:TOPK]


# ---------------------------------------------------------------- per sample
def _worker(sid: str) -> dict:
    cif = CIF_DIR / f"{sid}.cif"
    if not cif.exists():
        return {"sample_id": sid, "status": "missing_cif"}
    t = truth_cells(cif)
    row = {
        "sample_id": sid,
        "status": "ok",
        "system": t["system"],
        "conv_vol": t["conv_vol"],
        "prim_vol": t["prim_vol"],
        "z_ratio": t["z_ratio"],
    }

    # --- Ours: phase4 pool ---
    allc = PHASE4 / sid / f"{sid.replace('-', '_')}.allcells"
    pool = []
    if allc.exists():
        cands = parse_allcells(allc)
        # rank by McM20 desc (native FoM ordering already in file order too)
        cands.sort(key=lambda c: -c["McM20"])
        pool = [c["params"] for c in cands]
    row["n_pool"] = len(pool)

    for tag in ("conv", "prim"):
        truth = t[tag]
        # library reachability
        lib_loose = any(l4(p, truth)[0] for p in pool)
        lib_strict = any(l4(p, truth)[1] for p in pool)
        r1_loose = first_hit(pool[:1], truth, False)
        r1_strict = first_hit(pool[:1], truth, True)
        r20_loose = first_hit(pool[:TOPK], truth, False)
        r20_strict = first_hit(pool[:TOPK], truth, True)
        row[f"ours_{tag}"] = {
            "lib_loose": lib_loose,
            "lib_strict": lib_strict,
            "top1_loose": r1_loose is not None,
            "top1_strict": r1_strict is not None,
            "top20_loose": r20_loose is not None,
            "top20_strict": r20_strict is not None,
        }

    # --- McMaille ---
    mo = mcm_ordered(sid)
    row["n_mcm"] = len(mo)
    for tag in ("conv", "prim"):
        truth = t[tag]
        r1l, r1s, det = l4(mo[0], truth) if mo else (False, False, None)
        row[f"mcm_{tag}"] = {
            "top1_loose": r1l,
            "top1_strict": r1s,
            "top1_det": det,
            "top20_loose": first_hit(mo, truth, False) is not None,
            "top20_strict": first_hit(mo, truth, True) is not None,
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--output", type=Path, default=PROJECT / "results/l4_prim_vs_conv.json"
    )
    args = ap.parse_args()

    sids = sorted(
        p.name for p in PHASE4.iterdir() if p.is_dir() and p.name.startswith("mp-")
    )
    if args.limit:
        sids = sids[: args.limit]
    print(f"Measuring {len(sids)} samples (conv vs prim truth)...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, s) for s in sids]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(sids)}", flush=True)
    rows.sort(key=lambda r: r["sample_id"])

    # --- JADE from stored preds ---
    jade = json.loads(JADE_JSON.read_text())
    jade_by = {r["sample_id"]: r for r in jade["per_sample"]}
    jade_rows = []
    for r in rows:
        sid = r["sample_id"]
        jr = jade_by.get(sid, {})
        pred = jr.get("pred") if jr.get("status") == "parsed" else None
        cif = CIF_DIR / f"{sid}.cif"
        t = truth_cells(cif)
        e = {"sample_id": sid, "pred": pred}
        for tag in ("conv", "prim"):
            lo, st, det = l4(pred, t[tag])
            e[f"jade_{tag}"] = {"top1_loose": lo, "top1_strict": st, "top1_det": det}
        jade_rows.append(e)

    n = 100

    def rate(rs, key, field):
        return sum(1 for r in rs if (r.get(key) or {}).get(field)) / n

    summary = {
        "protocol": {
            "L4_loose": f"find_mapping ltol={LTOL} atol={ATOL}",
            "L4_strict": f"loose AND |det(scale)-1|<{DET_TOL}",
            "denominator": n,
            "truth_conv": "SpacegroupAnalyzer.get_conventional_standard_structure",
            "truth_prim": "SpacegroupAnalyzer.get_primitive_standard_structure",
            "ours": "phase4 .allcells pool, ranked by McM20 desc",
            "mcmaille": "original .imp suggested + McM20 top20",
            "jade": "jade-index .hkl Top-1",
        },
        "median_z_ratio": float(
            np.median([r["z_ratio"] for r in rows if r.get("z_ratio")])
        ),
        "z_ratio_hist": {},
    }
    zs = [round(r["z_ratio"], 2) for r in rows if r.get("z_ratio")]
    for z in sorted(set(zs)):
        summary["z_ratio_hist"][str(z)] = zs.count(z)

    for tag in ("conv", "prim"):
        summary[f"truth_{tag}"] = {
            "JADE9": {
                "top1_loose": rate(jade_rows, f"jade_{tag}", "top1_loose"),
                "top1_strict": rate(jade_rows, f"jade_{tag}", "top1_strict"),
            },
            "McMaille_original": {
                "top1_loose": rate(rows, f"mcm_{tag}", "top1_loose"),
                "top1_strict": rate(rows, f"mcm_{tag}", "top1_strict"),
                "top20_loose": rate(rows, f"mcm_{tag}", "top20_loose"),
                "top20_strict": rate(rows, f"mcm_{tag}", "top20_strict"),
            },
            "Ours_phase4_pool": {
                "top1_loose": rate(rows, f"ours_{tag}", "top1_loose"),
                "top1_strict": rate(rows, f"ours_{tag}", "top1_strict"),
                "top20_loose": rate(rows, f"ours_{tag}", "top20_loose"),
                "top20_strict": rate(rows, f"ours_{tag}", "top20_strict"),
                "lib_loose": rate(rows, f"ours_{tag}", "lib_loose"),
                "lib_strict": rate(rows, f"ours_{tag}", "lib_strict"),
            },
        }

    out = {"summary": summary, "per_sample": rows, "jade_per_sample": jade_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n======== SUMMARY ========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
