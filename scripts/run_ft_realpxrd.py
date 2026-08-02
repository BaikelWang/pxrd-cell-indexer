#!/usr/bin/env python3
"""Launcher: python scripts/run_ft_realpxrd.py train --arm B ..."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval"):
        print("Usage: run_ft_realpxrd.py train|eval --arm A|B|C ...")
        sys.exit(2)
    cmd = sys.argv.pop(1)
    if cmd == "train":
        from ft_realpxrd.train import main as train_main

        train_main()
    else:
        from ft_realpxrd.eval_mp100 import main as eval_main

        eval_main()


if __name__ == "__main__":
    main()
