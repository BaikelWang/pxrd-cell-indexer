#!/usr/bin/env python3
"""A2: sweep Top-K pool scale sets + volume guards on MP100 (strict elementwise)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Matrix from 完胜引擎攻关方案 v1 §7 A2.
SWEEP_RUNS: list[dict[str, object]] = [
    {"id": "A2a", "top_k": 20, "scale_set": "default", "pool_vol": -1.0},
    {"id": "A2b", "top_k": 20, "scale_set": "extended", "pool_vol": -1.0},
    {"id": "A2c", "top_k": 20, "scale_set": "extended", "pool_vol": math.log(2.0)},
    {"id": "A2d", "top_k": 40, "scale_set": "extended", "pool_vol": math.log(2.0)},
    {"id": "A2e", "top_k": 40, "scale_set": "extended", "pool_vol": math.log(1.5)},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "results/experiments/scale_100k_no_cs_matrix6_seed42/checkpoints/best.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/scale_100k_no_cs_matrix6.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ltol", type=float, default=0.05)
    parser.add_argument("--atol-deg", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/beat_engine",
    )
    parser.add_argument(
        "--also-valid",
        action="store_true",
        help="Also run valid1400 for each setting (slower)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for run in SWEEP_RUNS:
        run_id = str(run["id"])
        out_path = args.output_dir / f"{run_id.lower()}_mp100_strict.json"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/eval_mp100.py"),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--device",
            args.device,
            "--ltol",
            str(args.ltol),
            "--atol-deg",
            str(args.atol_deg),
            "--top-k",
            str(run["top_k"]),
            "--scale-set",
            str(run["scale_set"]),
            "--pool-max-log-volume-ratio",
            str(run["pool_vol"]),
            "--output-path",
            str(out_path),
        ]
        print(">>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        row = {
            "id": run_id,
            "top_k": run["top_k"],
            "scale_set": run["scale_set"],
            "pool_max_log_volume_ratio": run["pool_vol"],
            "topk_lattice_match_rate": metrics["topk_lattice_match_rate"],
            "topk_elementwise_rate": metrics["topk_elementwise_rate"],
            "topk_volume_guarded_rate": metrics["topk_volume_guarded_rate"],
            "topk_mapping_vs_elementwise_gap_rate": metrics[
                "topk_mapping_vs_elementwise_gap_rate"
            ],
            "fom_top1_lattice_match_rate": metrics["fom_top1_lattice_match_rate"],
            "fom_top1_elementwise_rate": metrics["fom_top1_elementwise_rate"],
            "raw_top1_elementwise_rate": metrics["raw_top1_elementwise_rate"],
            "output": str(out_path),
        }
        summary_rows.append(row)
        print(
            f"[{run_id}] topk_elem={row['topk_elementwise_rate']:.1%} "
            f"topk_map={row['topk_lattice_match_rate']:.1%} "
            f"gap={row['topk_mapping_vs_elementwise_gap_rate']:.1%} "
            f"fom_elem={row['fom_top1_elementwise_rate']:.1%}",
            flush=True,
        )

        if args.also_valid:
            valid_out = args.output_dir / f"{run_id.lower()}_valid_strict.json"
            vcmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/eval_valid.py"),
                "--config",
                str(args.config),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                args.device,
                "--ltol",
                str(args.ltol),
                "--atol-deg",
                str(args.atol_deg),
                "--top-k",
                str(run["top_k"]),
                "--scale-set",
                str(run["scale_set"]),
                "--pool-max-log-volume-ratio",
                str(run["pool_vol"]),
                "--output-path",
                str(valid_out),
            ]
            print(">>>", " ".join(vcmd), flush=True)
            subprocess.run(vcmd, check=True, cwd=PROJECT_ROOT)

    summary_path = args.output_dir / "a2_pool_sweep_summary.json"
    summary_path.write_text(
        json.dumps({"ltol": args.ltol, "atol_deg": args.atol_deg, "runs": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
