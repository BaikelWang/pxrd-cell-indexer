#!/usr/bin/env python3
"""Deprecated: use scripts/eval_valid.py which reports raw + rerank + FOM together."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper around eval_valid.py (dual-metric eval)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_valid.py"),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--output-path",
        str(args.output_path),
        "--device",
        args.device,
        "--rerank",
        "none",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
