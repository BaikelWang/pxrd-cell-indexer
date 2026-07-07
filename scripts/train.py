#!/usr/bin/env python3
"""Training entry point for indexing smoke experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxrd_cell_indexing.training.config import TrainConfig
from pxrd_cell_indexing.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = TrainConfig.from_yaml(config_path).resolve_paths(PROJECT_ROOT)
    trainer = Trainer(config)
    result = trainer.train()
    summary_path = config.run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PXRD cell indexing model")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
