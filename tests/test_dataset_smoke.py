"""Smoke tests for M1.4 dataset and dataloader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pxrd_cell_indexing.data.dataset import (
    PXRDDataset,
    PXRDDatasetConfig,
    SpectrumAugmentConfig,
    augment_spectrum,
    build_dataloader,
    collate_peak_batch,
)
from pxrd_cell_indexing.model.encoder.loader import (
    REALPXRD_ENCODER_CHECKPOINT,
    load_xrd_encoder_from_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_LMDB = (
    PROJECT_ROOT / ".." / ".." / "alex_aflow_oqmd_mp" / "datasets" / "pxrd_241113_train.lmdb"
).resolve()
VALID_LMDB = (
    PROJECT_ROOT / ".." / ".." / "alex_aflow_oqmd_mp" / "datasets" / "pxrd_241113_valid.lmdb"
).resolve()
TRAIN_JSONL = DATA_DIR / "train10k_seed42.jsonl"
VALID_JSONL = DATA_DIR / "valid1400_seed42.jsonl"


def _train_config(**overrides: object) -> PXRDDatasetConfig:
    base = {
        "lmdb_path": TRAIN_LMDB,
        "split": "train",
        "sample_list_path": TRAIN_JSONL,
        "xrd_augment": False,
        "strict": True,
    }
    base.update(overrides)
    return PXRDDatasetConfig(**base)  # type: ignore[arg-type]


def _valid_config(**overrides: object) -> PXRDDatasetConfig:
    base = {
        "lmdb_path": VALID_LMDB,
        "split": "valid",
        "sample_list_path": VALID_JSONL,
        "xrd_augment": False,
        "strict": True,
    }
    base.update(overrides)
    return PXRDDatasetConfig(**base)  # type: ignore[arg-type]


@pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="train10k jsonl missing")
def test_dataset_length_train() -> None:
    dataset = PXRDDataset(_train_config())
    assert len(dataset) == 10_000


@pytest.mark.skipif(not VALID_JSONL.exists(), reason="valid1400 jsonl missing")
def test_dataset_length_valid() -> None:
    dataset = PXRDDataset(_valid_config())
    assert len(dataset) == 1_400


@pytest.mark.skipif(not TRAIN_JSONL.exists() or not TRAIN_LMDB.exists(), reason="data missing")
def test_single_sample_matches_metadata() -> None:
    dataset = PXRDDataset(_train_config())
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=20, replace=False)
    for idx in indices:
        sample = dataset[int(idx)]
        record = dataset.records[int(idx)]
        assert sample["peak_num"] == record["peak_num_filtered"]
        assert sample["peak_num"] == sample["two_theta"].shape[0]
        assert sample["sample_id"] == record["lmdb_key"]


@pytest.mark.skipif(not TRAIN_JSONL.exists() or not TRAIN_LMDB.exists(), reason="data missing")
def test_collate_shapes() -> None:
    dataset = PXRDDataset(_train_config())
    samples = [dataset[i] for i in range(8)]
    batch = collate_peak_batch(samples)
    assert batch["pxrd_x"].shape == (sum(s["peak_num"] for s in samples), 1)
    assert batch["pxrd_y"].shape == batch["pxrd_x"].shape
    assert batch["peak_num"].dtype == torch.long
    assert batch["peak_num"].shape == (8,)
    assert batch["lattice"].shape == (8, 6)
    assert batch["crystal_system_idx"].shape == (8,)


@pytest.mark.skipif(
    not TRAIN_JSONL.exists()
    or not TRAIN_LMDB.exists()
    or not REALPXRD_ENCODER_CHECKPOINT.exists(),
    reason="data or checkpoint missing",
)
def test_encoder_integration() -> None:
    dataset = PXRDDataset(_train_config())
    batch = collate_peak_batch([dataset[i] for i in range(4)])
    encoder, _ = load_xrd_encoder_from_checkpoint(REALPXRD_ENCODER_CHECKPOINT)
    encoder.eval()
    with torch.no_grad():
        output = encoder(batch["pxrd_x"], batch["pxrd_y"], batch["peak_num"])
    assert output.shape == (4, 512)
    assert torch.isfinite(output).all()


def test_augment_spectrum_quirk() -> None:
    rng = np.random.default_rng(123)
    two_theta = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    intensity = np.array([12.0, 45.0, 80.0], dtype=np.float32)
    pre_filter = intensity.copy()
    aug_x, aug_y = augment_spectrum(
        two_theta,
        intensity,
        config=SpectrumAugmentConfig(),
        rng=rng,
        pre_filter_intensity=pre_filter,
    )
    assert aug_x.shape[0] == two_theta.shape[0]
    assert aug_y.shape[0] == intensity.shape[0]
    assert np.isclose(aug_y.max(), 100.0)
    assert np.all(aug_y >= 0)


@pytest.mark.skipif(not TRAIN_JSONL.exists() or not TRAIN_LMDB.exists(), reason="data missing")
def test_build_dataloader_iterates() -> None:
    loader = build_dataloader(_train_config(), batch_size=8, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert batch["peak_num"].shape[0] == 8
