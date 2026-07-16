#!/usr/bin/env python3
"""Summarize R1 experiment runs: strict elem, angle, per-CS, non-cubic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS

NON_CUBIC = [c for c in CRYSTAL_SYSTEMS if c != "cubic"]


def summarize_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"run": str(run_dir), "status": "missing"}
    metrics = json.loads(metrics_path.read_text())
    best = max(metrics, key=lambda e: e["valid"].get("strict_raw_top1_elementwise_rate", -1.0))
    v = best["valid"]
    t = best.get("train", {})
    per_cs_path = run_dir / f"valid_per_cs_epoch{best['epoch']}.json"
    per_cs = json.loads(per_cs_path.read_text()) if per_cs_path.exists() else {}
    # Prefer elementwise if present; else proxy.
    cs_elem = {}
    for cs in CRYSTAL_SYSTEMS:
        row = per_cs.get(cs, {})
        if not row:
            continue
        cs_elem[cs] = {
            "count": row.get("count"),
            "lattice_mae": row.get("lattice_mae"),
            "proxy": row.get("top1_lattice_match_proxy"),
        }
    non_cubic_proxy_vals = [
        cs_elem[cs]["proxy"]
        for cs in NON_CUBIC
        if cs in cs_elem and cs_elem[cs].get("proxy") is not None
    ]
    cubic_proxy = cs_elem.get("cubic", {}).get("proxy")
    return {
        "run": run_dir.name,
        "status": "done" if (run_dir / "summary.json").exists() else "running",
        "n_epochs": len(metrics),
        "best_epoch": best["epoch"],
        "train_loss": t.get("loss"),
        "valid_loss": v.get("loss"),
        "strict_elem": v.get("strict_raw_top1_elementwise_rate"),
        "strict_map": v.get("strict_raw_top1_lattice_match_rate"),
        "angle_mae": v.get("angle_mae"),
        "length_mape": v.get("length_mape"),
        "loose_raw": v.get("raw_top1_lattice_match_rate"),
        "cubic_proxy": cubic_proxy,
        "non_cubic_proxy_mean": (
            float(sum(non_cubic_proxy_vals) / len(non_cubic_proxy_vals))
            if non_cubic_proxy_vals
            else None
        ),
        "per_cs_proxy": cs_elem,
        "last_train_loss": metrics[-1].get("train", {}).get("loss"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    root = Path("results/experiments")
    rows = [summarize_run(root / name if not name.startswith("/") else Path(name)) for name in args.runs]
    text = json.dumps(rows, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
