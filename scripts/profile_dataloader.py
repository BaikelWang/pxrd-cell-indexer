#!/usr/bin/env python3
"""Profile train dataloader vs GPU compute on a short run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pxrd_cell_indexing.data.dataset import PXRDDatasetConfig, PeakFilterConfig, build_dataloader
from pxrd_cell_indexing.data.normalization import build_lattice_normalizer, head_output_dim
from pxrd_cell_indexing.model.heads import HeadConfig, build_indexing_model
from pxrd_cell_indexing.training.config import TrainConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> dict:
    config = TrainConfig.from_yaml(args.config).resolve_paths(PROJECT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = build_lattice_normalizer(config.data)
    model = build_indexing_model(
        checkpoint_path=config.model.encoder_checkpoint,
        head_config=HeadConfig(
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            output_dim=head_output_dim(config.data.representation),
        ),
        freeze_encoder=config.model.freeze_encoder,
        normalize_embedding=config.model.normalize_embedding,
    ).to(device)
    model.eval()

    dataset_cfg = PXRDDatasetConfig(
        lmdb_path=Path(config.data.train_lmdb),
        split="train",
        sample_list_path=Path(config.data.train_jsonl),
        peak_filter=PeakFilterConfig(),
        xrd_augment=config.data.train_augment,
        strict=False,
        seed_base=config.seed,
    )
    loader = build_dataloader(
        dataset_cfg,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
    )

    data_wait_s = 0.0
    compute_s = 0.0
    steps = 0
    iter_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            if steps >= args.max_steps:
                break
            data_wait_s += time.perf_counter() - iter_start
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            batch["lattice_norm"] = normalizer.normalize(batch["lattice"])
            compute_start = time.perf_counter()
            _ = model(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_s += time.perf_counter() - compute_start
            steps += 1
            iter_start = time.perf_counter()

    result = {
        "config": str(args.config),
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "prefetch_factor": config.data.prefetch_factor,
        "persistent_workers": config.data.persistent_workers,
        "steps": steps,
        "data_wait_ms_per_step": 1000.0 * data_wait_s / max(steps, 1),
        "compute_ms_per_step": 1000.0 * compute_s / max(steps, 1),
        "bottleneck": "io" if data_wait_s > compute_s else "compute",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile dataloader vs compute")
    parser.add_argument("--config", type=Path, default="configs/scale_100k_no_cs_matrix6.yaml")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "profile_dataloader_matrix6.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
