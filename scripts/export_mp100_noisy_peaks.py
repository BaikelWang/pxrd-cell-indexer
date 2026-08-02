#!/usr/bin/env python3
"""Export MP100 peaks with RealPXRD-style online noise (frozen, reproducible).

Writes:
  - peaks.jsonl — one row per sample (two_theta/intensity + truth metadata)
  - xy/<sample_id>.xy — two-column peak lists for external tools (McMaille etc.)
  - meta.json — augment protocol

Ideal (no-noise) MP100 is unchanged; this is an optional noisy twin.

Usage:
    python scripts/export_mp100_noisy_peaks.py --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxrd_cell_indexing.data.dataset import SpectrumAugmentConfig
from pxrd_cell_indexing.data.mp100 import DEFAULT_MP100_AUGMENT_SEED, load_mp100_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MP100_DIR = PROJECT_ROOT / "data" / "MP-100samples-benchmark"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "mp100_realpxrd_augment_seed42"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mp100-dir", type=Path, default=DEFAULT_MP100_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--convention", choices=("primitive", "reduced", "niggli"), default="niggli")
    p.add_argument("--seed", type=int, default=DEFAULT_MP100_AUGMENT_SEED)
    p.add_argument("--noise-level", type=float, default=0.05)
    p.add_argument("--shift-range", type=float, default=0.1)
    p.add_argument("--scale-min", type=float, default=0.8)
    p.add_argument("--scale-max", type=float, default=1.2)
    args = p.parse_args()

    cfg = SpectrumAugmentConfig(
        noise_level=args.noise_level,
        shift_range=args.shift_range,
        scale_range=(args.scale_min, args.scale_max),
    )
    samples = load_mp100_dataset(
        args.mp100_dir,
        convention=args.convention,
        xrd_augment=True,
        augment_seed=args.seed,
        augment_config=cfg,
    )

    out = args.output_dir
    xy_dir = out / "xy"
    out.mkdir(parents=True, exist_ok=True)
    xy_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "peaks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for s in samples:
            row = {
                "sample_id": s.sample_id,
                "cif_path": str(s.cif_path),
                "crystal_system": s.crystal_system,
                "convention": s.convention,
                "peak_num": s.peak_num,
                "two_theta": [float(x) for x in s.two_theta.tolist()],
                "intensity": [float(x) for x in s.intensity.tolist()],
                "truth_lattice": [float(x) for x in s.truth_lattice.tolist()],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            xy_path = xy_dir / f"{s.sample_id}.xy"
            with xy_path.open("w", encoding="utf-8") as xy:
                for tt, ii in zip(s.two_theta.tolist(), s.intensity.tolist(), strict=True):
                    xy.write(f"{tt:.8f} {ii:.8f}\n")

    meta = {
        "mp100_dir": str(args.mp100_dir.resolve()),
        "n_samples": len(samples),
        "convention": args.convention,
        "xrd_augment": True,
        "augment_seed": args.seed,
        "augment": {
            "noise_level": args.noise_level,
            "shift_range": args.shift_range,
            "scale_range": [args.scale_min, args.scale_max],
            "protocol": "RealPXRD augment_spectrum (online; not baked into CIF)",
        },
        "peaks_jsonl": str(jsonl_path),
        "xy_dir": str(xy_dir),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
