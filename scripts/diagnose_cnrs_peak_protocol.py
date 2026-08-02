#!/usr/bin/env python3
"""Compare the CNRS eval peak protocol against the training peak protocol.

The model is trained on pymatgen stick patterns filtered by ``I > 5`` (max
normalised to 100). CNRS evaluation instead peak-picks a *continuous* pattern
and applies the same ``I >= 5``. If the picker recovers far fewer peaks than the
sticks, the model sees an input distribution it was never trained on, and the
loss is worst exactly where patterns are peak-dense (low symmetry).

Reports, per sample and per crystal system, the stick count, the picked count,
and the intensity threshold the picker would need to match the stick count.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pymatgen.analysis.diffraction.xrd import XRDCalculator  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from eval_cnrs_seedpool import pick_peaks_paperlike  # noqa: E402

TRAIN_IMIN = 5.0
TT_CAP = 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnrs-dir", default="/nanolab/users/wyx/CNRS")
    ap.add_argument("--attribution", default="results/flow_seedgen/cnrs_e2e_k100/ortho_attribution.json")
    ap.add_argument("--out", default="results/flow_seedgen/cnrs_e2e_k100/peak_protocol_gap.json")
    args = ap.parse_args()

    att = json.loads((ROOT / args.attribution).read_text())
    hit = {r["sample_id"]: r for r in att["per_sample"]}

    cnrs = Path(args.cnrs_dir)
    rows = []
    for sid, rec in sorted(hit.items()):
        cif = cnrs / f"{sid}_sg.cif"
        csv = cnrs / f"{sid}.csv"
        if not cif.exists() or not csv.exists():
            continue
        df = pd.read_csv(csv)
        tt_raw = df["two_theta_deg"].to_numpy()
        lo, hi = float(tt_raw.min()), min(float(tt_raw.max()), TT_CAP)

        conv = SpacegroupAnalyzer(
            Structure.from_file(cif), symprec=0.01
        ).get_conventional_standard_structure()
        pat = XRDCalculator(wavelength="CuKa").get_pattern(conv, two_theta_range=(lo, hi))
        sy = np.asarray(pat.y)
        n_stick = int((sy > TRAIN_IMIN).sum())

        tt, ii, _ = pick_peaks_paperlike(tt_raw, df["intensity"].to_numpy())
        keep = (tt <= hi)
        tt, ii = tt[keep], ii[keep]
        n_pick5 = int((ii >= TRAIN_IMIN).sum())

        # Threshold that would make the picker return as many peaks as training.
        srt = np.sort(ii)[::-1]
        thr_match = float(srt[min(n_stick, len(srt)) - 1]) if len(srt) else np.nan

        rows.append(
            {
                "sample_id": sid,
                "system": rec["system"],
                "strict": bool(rec["L0_raw"]["strict"]),
                "loose": bool(rec["L0_raw"]["loose"]),
                "n_stick_train": n_stick,
                "n_picked_all": int(len(ii)),
                "n_picked_i5": n_pick5,
                "ratio": n_pick5 / max(n_stick, 1),
                "thr_to_match": thr_match,
                "first_tt_stick": float(np.asarray(pat.x)[sy > TRAIN_IMIN][0]) if n_stick else None,
                "first_tt_pick5": float(tt[ii >= TRAIN_IMIN][0]) if n_pick5 else None,
            }
        )

    d = pd.DataFrame(rows)
    g = d.groupby("system").agg(
        n=("sample_id", "size"),
        strict=("strict", "mean"),
        train_peaks=("n_stick_train", "mean"),
        eval_peaks=("n_picked_i5", "mean"),
        ratio=("ratio", "mean"),
        thr_to_match=("thr_to_match", "median"),
    )
    print("=== peaks the model was trained to see vs peaks it gets on CNRS ===")
    print(g.round(3).to_string())
    print(
        f"\nALL n={len(d)} strict={d['strict'].mean():.3f} "
        f"train_peaks={d['n_stick_train'].mean():.1f} eval_peaks={d['n_picked_i5'].mean():.1f} "
        f"ratio={d['ratio'].mean():.3f}"
    )

    print("\n=== hit rate vs how much of the pattern survives ===")
    d["rb"] = pd.cut(d["ratio"], [0, 0.35, 0.55, 0.8, 1.2, 99])
    print(
        d.groupby("rb", observed=True)
        .agg(n=("strict", "size"), strict=("strict", "mean"), loose=("loose", "mean"))
        .round(3)
        .to_string()
    )

    (ROOT / args.out).write_text(
        json.dumps({"by_system": g.round(4).to_dict("index"), "per_sample": rows}, indent=2)
    )
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
