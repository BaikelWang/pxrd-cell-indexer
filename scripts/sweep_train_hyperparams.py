#!/usr/bin/env python3
"""Hyperparameter sweep launcher for loss/training configs on 100k."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg = _load_yaml(args.base_config if args.base_config.is_absolute() else PROJECT_ROOT / args.base_config)
    grid: list[dict[str, Any]] = []

    for loss_mode in args.loss_modes:
        for head_lr in args.head_lrs:
            for batch_size in args.batch_sizes:
                grid.append(
                    {
                        "loss_mode": loss_mode,
                        "head_lr": head_lr,
                        "batch_size": batch_size,
                    }
                )

    results: list[dict[str, Any]] = []
    sweep_dir = PROJECT_ROOT / "configs" / "sweeps"
    for idx, combo in enumerate(grid):
        cfg = copy.deepcopy(base_cfg)
        exp_name = (
            f"sweep_{combo['loss_mode']}_hlr{combo['head_lr']}_bs{combo['batch_size']}_seed{cfg.get('seed', 42)}"
        )
        cfg["experiment_name"] = exp_name
        cfg.setdefault("loss", {})
        cfg["loss"]["mode"] = combo["loss_mode"]
        cfg.setdefault("optim", {})
        cfg["optim"]["head_lr"] = combo["head_lr"]
        cfg.setdefault("data", {})
        cfg["data"]["batch_size"] = combo["batch_size"]
        cfg["optim"]["profile_timing"] = idx == 0

        cfg_path = sweep_dir / f"{exp_name}.yaml"
        _write_yaml(cfg_path, cfg)

        if args.dry_run:
            results.append({"config": str(cfg_path), "status": "dry_run", **combo})
            continue

        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        summary_path = PROJECT_ROOT / "results" / "experiments" / exp_name / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
        valid_raw = 0.0
        history = summary.get("history", [])
        if history:
            valid_raw = history[-1].get("valid", {}).get("raw_top1_lattice_match_rate", 0.0)
            if not valid_raw:
                valid_raw = history[-1].get("valid", {}).get("top1_lattice_match_rate", 0.0)
        results.append(
            {
                "config": str(cfg_path),
                "status": "ok" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "best_valid_metric": summary.get("best_valid_metric"),
                "valid_raw_top1_last_epoch": valid_raw,
                **combo,
            }
        )

    results.sort(
        key=lambda item: (
            item.get("best_valid_metric") or 0.0,
            item.get("valid_raw_top1_last_epoch") or 0.0,
        ),
        reverse=True,
    )
    output = {"best": results[0] if results else None, "grid": results}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output.get("best"), indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep loss modes and training hyperparameters")
    parser.add_argument(
        "--base-config",
        type=Path,
        default="configs/scale_100k_no_cs_matrix6.yaml",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "train_sweep_100k.json",
    )
    parser.add_argument(
        "--loss-modes",
        nargs="+",
        default=["baseline", "length_angle", "cs_mask", "cs_reweight", "combined"],
    )
    parser.add_argument("--head-lrs", nargs="+", type=float, default=[0.0005, 0.001, 0.002])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
