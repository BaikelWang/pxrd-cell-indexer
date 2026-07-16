#!/usr/bin/env python3
"""Summarize R3 runs with non-cubic strict elem + pull90 via diagnose."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NON_CUBIC = {"tetragonal", "orthorhombic", "hexagonal", "trigonal", "monoclinic", "triclinic"}


def summarize_metrics(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    best = max(metrics, key=lambda e: e["valid"].get("strict_raw_top1_elementwise_rate", -1.0))
    v = best["valid"]
    return {
        "run": run_dir.name,
        "best_epoch": best["epoch"],
        "train_loss": best.get("train", {}).get("loss"),
        "strict_elem": v.get("strict_raw_top1_elementwise_rate"),
        "angle_mae": v.get("angle_mae"),
        "status": "done" if (run_dir / "summary.json").exists() else "running",
    }


def load_diag(path: Path) -> dict:
    d = json.loads(path.read_text())
    o = d["valid"]["overall"]
    by = d["valid"]["by_crystal_system"]
    non = [by[cs]["elem_ok_rate"] for cs in NON_CUBIC if cs in by]
    return {
        "elem": o["elem_ok_rate"],
        "angle_mae": o["angle_mae"],
        "pull90": o["pulled_to_90_rate"],
        "non_cubic_elem": float(sum(non) / len(non)) if non else None,
        "by_cs_elem": {cs: by[cs]["elem_ok_rate"] for cs in by},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Matching config yaml/json; default = results/experiments/<run>/config.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip-diagnose", action="store_true")
    p.add_argument("--diag-subdir", type=str, default="r3")
    args = p.parse_args()
    if args.configs is None:
        args.configs = [str(PROJECT_ROOT / "results/experiments" / r / "config.json") for r in args.runs]
    assert len(args.runs) == len(args.configs)

    rows = []
    for run, cfg in zip(args.runs, args.configs):
        run_dir = PROJECT_ROOT / "results/experiments" / run
        if not (run_dir / "metrics.json").exists():
            rows.append({"run": run, "status": "missing"})
            continue
        row = summarize_metrics(run_dir)
        ckpt = run_dir / "checkpoints" / "best.pt"
        diag_tag = f"r3_{run}"
        diag_dir = PROJECT_ROOT / "results/beat_engine/raw_diag" / args.diag_subdir
        diag_path = diag_dir / f"{diag_tag}_raw_diagnosis.json"
        if not args.skip_diagnose and ckpt.exists():
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/diagnose_raw_errors.py"),
                "--config",
                str(cfg),
                "--checkpoint",
                str(ckpt),
                "--output-dir",
                str(diag_dir),
                "--tag",
                diag_tag,
                "--skip-mp100",
            ]
            env = {**os.environ, "PYTHONPATH": "src"}
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        if diag_path.exists():
            row["diag"] = load_diag(diag_path)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
