#!/usr/bin/env python3
"""Evaluate OpenAlphaDiffract (HF public weights) on CNRS with primitive L4-strict.

Input contract (official OpenAlphaDiffract UI / simulator.yaml):
  * 8192-point continuous PXRD intensity on a fixed 2θ grid
  * Training-time grid: **5–20°** at λ=0.6199 Å (20 keV) — NOT 5–120°
  * Intensity floored at 0, then linearly scaled to [0, 100]

CNRS CSVs are Cu Kα (λ=1.5406 Å). We convert 2θ via the same Bragg helper as
their UI (``XRDContext.jsx`` / ``xrd-processing.js``), crop to 5–20°, resample
to 8192, then normalize. L4 ruler vs primitive CIF is unchanged.

Top-1 only (single LP head prediction).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenAlphaDiffract"))

from model import AlphaDiffract  # noqa: E402
from remeasure_l4_prim_vs_conv import l4, truth_cells  # noqa: E402

CUKA_A = 1.5406
# Official UI / simulator constant (≈ 20 keV)
LAMBDA_20KEV = 0.6199
N_BINS = 8192
TT_MIN, TT_MAX = 5.0, 20.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument(
        "--model-dir",
        default="third_party/OpenAlphaDiffract",
    )
    ap.add_argument(
        "--out-dir",
        default="results/cnrs_benchmark/openalphadiffract",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cuka-wavelength", type=float, default=CUKA_A)
    ap.add_argument(
        "--bragg-mode",
        choices=("ui", "physical"),
        default="ui",
        help="ui=match OpenAlpha XRDContext (sin of 2θ as shipped); "
        "physical=correct Bragg with θ=2θ/2",
    )
    return ap.parse_args()


def convert_wavelength_ui(two_theta_deg: float, source_wl: float, target_wl: float) -> float | None:
    """Match OpenAlphaDiffract ``convertWavelength`` (operates on the passed angle)."""
    if abs(source_wl - target_wl) < 1e-4:
        return float(two_theta_deg)
    theta_rad = np.deg2rad(two_theta_deg)
    sin_theta2 = (target_wl / source_wl) * np.sin(theta_rad)
    if abs(sin_theta2) > 1.0:
        return None
    return float(np.rad2deg(np.arcsin(sin_theta2)))


def convert_wavelength_physical(
    two_theta_deg: float, source_wl: float, target_wl: float
) -> float | None:
    """Physically correct Bragg: sin(θ₂)=(λ₂/λ₁)sin(θ₁) with θ=2θ/2."""
    if abs(source_wl - target_wl) < 1e-4:
        return float(two_theta_deg)
    th = np.deg2rad(two_theta_deg / 2.0)
    sin_th2 = (target_wl / source_wl) * np.sin(th)
    if abs(sin_th2) > 1.0:
        return None
    return float(2.0 * np.rad2deg(np.arcsin(sin_th2)))


def cuka_to_20kev_pattern(
    tt_cuka: np.ndarray,
    intensity: np.ndarray,
    wave_cuka: float,
    wave_20: float = LAMBDA_20KEV,
    n_bins: int = N_BINS,
    tt_min: float = TT_MIN,
    tt_max: float = TT_MAX,
    bragg_mode: str = "ui",
) -> np.ndarray:
    """Convert CuKα spectrum onto the OpenAlpha 5–20° / 8192 grid."""
    tt = np.asarray(tt_cuka, dtype=np.float64)
    ii = np.asarray(intensity, dtype=np.float64)
    conv_fn = convert_wavelength_ui if bragg_mode == "ui" else convert_wavelength_physical
    tt20_list = []
    ii_list = []
    for x, y in zip(tt, ii):
        x2 = conv_fn(float(x), wave_cuka, wave_20)
        if x2 is None:
            continue
        if tt_min <= x2 <= tt_max:
            tt20_list.append(x2)
            ii_list.append(float(y))
    grid = np.linspace(tt_min, tt_max, n_bins, dtype=np.float64)
    if len(tt20_list) < 2:
        return np.zeros(n_bins, dtype=np.float32)
    tt_v = np.asarray(tt20_list, dtype=np.float64)
    ii_v = np.asarray(ii_list, dtype=np.float64)
    order = np.argsort(tt_v)
    tt_v, ii_v = tt_v[order], ii_v[order]
    uniq_tt, inv = np.unique(np.round(tt_v, 6), return_inverse=True)
    sums = np.zeros(len(uniq_tt), dtype=np.float64)
    counts = np.zeros(len(uniq_tt), dtype=np.float64)
    np.add.at(sums, inv, ii_v)
    np.add.at(counts, inv, 1.0)
    uniq_i = sums / np.maximum(counts, 1.0)
    y = np.interp(grid, uniq_tt, uniq_i, left=0.0, right=0.0)
    y = np.maximum(y, 0.0)
    ymin = float(y.min())
    ymax = float(y.max())
    # UI: floor already applied; scale min→0 max→100 (constant → zeros)
    if ymax - ymin < 1e-12:
        return np.zeros(n_bins, dtype=np.float32)
    y = (y - ymin) / (ymax - ymin) * 100.0
    return y.astype(np.float32)


def prepare_items(
    cnrs: Path, limit: int, wave_cuka: float, bragg_mode: str
) -> list[dict]:
    manifest = pd.read_csv(cnrs / "cnrs_manifest.csv")
    if limit:
        manifest = manifest.head(limit)
    items = []
    for _, row in manifest.iterrows():
        sid = f"{int(row['sample_id']):06d}"
        csv_path = cnrs / f"{sid}.csv"
        cif_path = cnrs / f"{sid}_sg.cif"
        if not csv_path.exists() or not cif_path.exists():
            print(f"skip missing {sid}", flush=True)
            continue
        df = pd.read_csv(csv_path)
        try:
            truth = truth_cells(cif_path)
        except Exception as e:
            print(f"skip cif {sid}: {e}", flush=True)
            continue
        pattern = cuka_to_20kev_pattern(
            df["two_theta_deg"].to_numpy(),
            df["intensity"].to_numpy(),
            wave_cuka,
            bragg_mode=bragg_mode,
        )
        if float(pattern.max()) <= 0:
            print(f"skip {sid}: empty converted pattern", flush=True)
            continue
        items.append(
            {
                "sample_id": sid,
                "system": truth["system"],
                "truth_prim": truth["prim"],
                "pattern": pattern,
            }
        )
    return items


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)

    def rate(key):
        return sum(1 for r in rows if r["prim"][key]) / n

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
    return {
        "n": len(rows),
        "mean_pool": 1.0,
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
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"loading OpenAlphaDiffract from {model_dir} on {device}", flush=True)
    model = AlphaDiffract.from_pretrained(str(model_dir), device=device)
    model.eval()

    items = prepare_items(
        Path(args.cnrs_dir), args.limit, args.cuka_wavelength, args.bragg_mode
    )
    print(
        f"usable samples: {len(items)}  grid={TT_MIN}-{TT_MAX}°  "
        f"λ={LAMBDA_20KEV}  bragg={args.bragg_mode}",
        flush=True,
    )

    rows = []
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            x = torch.from_numpy(np.stack([b["pattern"] for b in batch], axis=0)).to(
                device
            )
            out_m = model(x)
            lp = out_m["lp"].detach().cpu().numpy()
            cs = torch.softmax(out_m["cs_logits"], dim=-1).argmax(dim=-1).cpu().numpy()
            sg = torch.softmax(out_m["sg_logits"], dim=-1).argmax(dim=-1).cpu().numpy()
            for i, it in enumerate(batch):
                cell = [float(v) for v in lp[i].tolist()]
                loose, strict, det = l4(cell, it["truth_prim"])
                row = {
                    "sample_id": it["sample_id"],
                    "system": it["system"],
                    "n_pool": 1,
                    "status": "ok",
                    "pred_cell": cell,
                    "pred_cs": AlphaDiffract.CRYSTAL_SYSTEMS[int(cs[i])],
                    "pred_sg": int(sg[i]) + 1,
                    "prim": {
                        "lib_loose": bool(loose),
                        "lib_strict": bool(strict),
                        "top1_loose": bool(loose),
                        "top1_strict": bool(strict),
                        # single prediction: top20/lib == top1
                        "top20_loose": bool(loose),
                        "top20_strict": bool(strict),
                        "first_strict": 1 if strict else None,
                        "first_loose": 1 if loose else None,
                        "det": det,
                    },
                }
                rows.append(row)
                preds.append(
                    {
                        "sample_id": it["sample_id"],
                        "cell": cell,
                        "cs": row["pred_cs"],
                        "sg": row["pred_sg"],
                    }
                )
            print(
                f"  inferred {min(start + args.batch_size, len(items))}/{len(items)}",
                flush=True,
            )

    summary = {
        "engine": "openalphadiffract",
        "status": "ok",
        "model_dir": str(model_dir),
        "device": device,
        "elapsed_s": time.time() - t0,
        "input": {
            "n_bins": N_BINS,
            "tt_min_deg": TT_MIN,
            "tt_max_deg": TT_MAX,
            "model_wavelength_A": LAMBDA_20KEV,
            "model_energy_keV": 20.0,
            "source_wavelength_A": args.cuka_wavelength,
            "bragg_mode": args.bragg_mode,
            "conversion": (
                "OpenAlpha UI pipeline: wavelength convert → crop 5–20° → "
                "interp 8192 → floor≥0 → min-max [0,100]"
            ),
        },
        **summarize(rows),
        "per_sample": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "predictions.json").write_text(json.dumps(preds, indent=2))
    # per-sample dirs (lightweight)
    for r, it in zip(rows, items):
        d = out / r["sample_id"]
        d.mkdir(exist_ok=True)
        (d / "status.json").write_text(json.dumps(r, indent=2))
        np.save(d / "pattern_8192.npy", it["pattern"])

    p = summary["prim"]
    print(
        f"OpenAlphaDiffract n={summary['n']}  "
        f"L4-strict Top-1={p['top1_strict']:.1%}  "
        f"(loose={p['top1_loose']:.1%})",
        flush=True,
    )
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
