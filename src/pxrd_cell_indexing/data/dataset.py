"""Dataset and dataloader for RealPXRD-compatible baseline."""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import lmdb
import numpy as np
import torch
from numpy.random import Generator
from torch.utils.data import DataLoader, Dataset, get_worker_info

from pxrd_cell_indexing.types import CRYSTAL_SYSTEM_TO_IDX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeakFilterConfig:
    """Peak filtering defaults for the first RealPXRD-compatible baseline."""

    intensity_min: float = 5.0
    max_peaks: int | None = None
    input_axis: str = "two_theta"


@dataclass(frozen=True)
class SpectrumAugmentConfig:
    """Training-time spectrum augmentation (values match RealPXRD, D21).

    Applied on the training split only (valid/test never augmented), after the
    ``y > 5`` filter, replicating ``app/data/dataset.py::augment_spectrum``.
    """

    noise_level: float = 0.05
    shift_range: float = 0.1
    scale_range: tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class PXRDDatasetConfig:
    """High-level dataset configuration.

    ``xrd_augment`` follows RealPXRD's train-only convention (D21): keep it False
    during M1.4 dataloader debugging for determinism, then enable for the training
    split during smoke training while leaving valid/test augmentation off.

    Reproducible augmentation requires the same ``seed_base`` and ``num_workers``.
    """

    lmdb_path: Path
    split: str = "train"
    sample_list_path: Path | None = None
    peak_filter: PeakFilterConfig = PeakFilterConfig()
    xrd_augment: bool = False
    augment: SpectrumAugmentConfig = SpectrumAugmentConfig()
    strict: bool = False
    seed_base: int = 42


class SampleMetadata(TypedDict):
    lmdb_key: str
    atom_num: int
    peak_num_raw: int
    peak_num_filtered: int
    two_theta_min: float
    two_theta_max: float
    crystal_system: str
    lattice_a: float
    lattice_b: float
    lattice_c: float
    lattice_alpha: float
    lattice_beta: float
    lattice_gamma: float


class DatasetSample(TypedDict):
    two_theta: np.ndarray
    intensity: np.ndarray
    peak_num: int
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    sample_id: str


def load_sample_list(path: Path) -> list[SampleMetadata]:
    """Load jsonl metadata records produced by investigate_*_sample.py."""
    records: list[SampleMetadata] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("crystal_system") is None:
                continue
            records.append(
                SampleMetadata(
                    lmdb_key=str(raw["lmdb_key"]),
                    atom_num=int(raw["atom_num"]),
                    peak_num_raw=int(raw["peak_num_raw"]),
                    peak_num_filtered=int(raw["peak_num_filtered"]),
                    two_theta_min=float(raw["two_theta_min"]),
                    two_theta_max=float(raw["two_theta_max"]),
                    crystal_system=str(raw["crystal_system"]),
                    lattice_a=float(raw["lattice_a"]),
                    lattice_b=float(raw["lattice_b"]),
                    lattice_c=float(raw["lattice_c"]),
                    lattice_alpha=float(raw["lattice_alpha"]),
                    lattice_beta=float(raw["lattice_beta"]),
                    lattice_gamma=float(raw["lattice_gamma"]),
                )
            )
    return records


def filter_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    intensity_min: float,
    max_peaks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply y>5 filtering and optional max_peaks truncation."""
    mask = intensity > intensity_min
    two_theta = two_theta[mask]
    intensity = intensity[mask]
    if max_peaks is not None and two_theta.shape[0] > max_peaks:
        two_theta = two_theta[:max_peaks]
        intensity = intensity[:max_peaks]
    return two_theta.astype(np.float32), intensity.astype(np.float32)


def augment_spectrum(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    config: SpectrumAugmentConfig = SpectrumAugmentConfig(),
    rng: Generator | None = None,
    *,
    pre_filter_intensity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Port RealPXRD ``app/data/dataset.py::augment_spectrum`` with D23 quirk.

    D23: upstream re-applies ``pxrd_y > 5`` using the *pre-augmentation* filtered
    intensity array (always all-True), not the augmented intensities. We replicate
    that no-op second filter for checkpoint-compatible training.
    """
    if rng is None:
        rng = np.random.default_rng()

    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if pre_filter_intensity is None:
        pre_filter_intensity = intensity.copy()

    noise = rng.normal(0, config.noise_level * np.max(intensity), size=intensity.shape)
    intensity_aug = intensity + noise

    shift = rng.uniform(-config.shift_range, config.shift_range, size=two_theta.shape)
    two_theta_aug = two_theta + shift

    scale_factor = rng.uniform(*config.scale_range)
    intensity_aug = intensity_aug * scale_factor
    intensity_aug = np.where(intensity_aug < 0, intensity, intensity_aug)

    max_intensity = np.max(intensity_aug)
    if max_intensity > 0:
        intensity_aug = intensity_aug / max_intensity * 100.0

    # D23: upstream bug/no-op — mask uses pre-augment filtered intensities.
    mask_aug = pre_filter_intensity > 5
    two_theta_final = two_theta_aug[mask_aug]
    intensity_final = intensity_aug[mask_aug]

    return two_theta_final.astype(np.float32), intensity_final.astype(np.float32)


class PXRDDataset(Dataset[DatasetSample]):
    """LMDB-backed dataset driven by a precomputed jsonl sample list."""

    def __init__(self, config: PXRDDatasetConfig) -> None:
        if config.sample_list_path is None:
            raise ValueError("sample_list_path is required")
        self.config = config
        self.records = load_sample_list(config.sample_list_path)
        self._env: lmdb.Environment | None = None
        self._rng: Generator | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.config.lmdb_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        return self._env

    def _ensure_rng(self) -> Generator:
        if self._rng is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self._rng = np.random.default_rng(self.config.seed_base + worker_id)
        return self._rng

    def _load_lmdb_entry(self, lmdb_key: str) -> dict[str, Any]:
        env = self._ensure_env()
        raw = env.begin().get(lmdb_key.encode("ascii"))
        if raw is None:
            raise KeyError(f"LMDB key not found: {lmdb_key}")
        return pickle.loads(gzip.decompress(raw))

    def __getitem__(self, index: int) -> DatasetSample:
        record = self.records[index]
        data = self._load_lmdb_entry(record["lmdb_key"])

        pxrd_x = np.asarray(data["pxrd_x"], dtype=np.float32)
        pxrd_y = np.asarray(data["pxrd_y"], dtype=np.float32)

        two_theta, intensity = filter_peaks(
            pxrd_x,
            pxrd_y,
            intensity_min=self.config.peak_filter.intensity_min,
            max_peaks=self.config.peak_filter.max_peaks,
        )

        if not self.config.xrd_augment:
            if two_theta.shape[0] != record["peak_num_filtered"]:
                message = (
                    f"peak count mismatch for {record['lmdb_key']}: "
                    f"got {two_theta.shape[0]}, expected {record['peak_num_filtered']}"
                )
                if self.config.strict:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
        else:
            pre_filter = intensity.copy()
            two_theta, intensity = augment_spectrum(
                two_theta,
                intensity,
                config=self.config.augment,
                rng=self._ensure_rng(),
                pre_filter_intensity=pre_filter,
            )

        peak_num = int(two_theta.shape[0])
        return DatasetSample(
            two_theta=two_theta,
            intensity=intensity,
            peak_num=peak_num,
            crystal_system=record["crystal_system"],
            a=record["lattice_a"],
            b=record["lattice_b"],
            c=record["lattice_c"],
            alpha=record["lattice_alpha"],
            beta=record["lattice_beta"],
            gamma=record["lattice_gamma"],
            sample_id=record["lmdb_key"],
        )


def collate_peak_batch(samples: list[DatasetSample]) -> dict[str, Any]:
    """Flatten variable-length peaks for BertModel batch_input."""
    pxrd_x_parts = [
        torch.from_numpy(sample["two_theta"]).view(-1, 1) for sample in samples
    ]
    pxrd_y_parts = [
        torch.from_numpy(sample["intensity"]).view(-1, 1) for sample in samples
    ]
    pxrd_x = torch.cat(pxrd_x_parts, dim=0)
    pxrd_y = torch.cat(pxrd_y_parts, dim=0)
    peak_num = torch.tensor([sample["peak_num"] for sample in samples], dtype=torch.long)
    crystal_system_idx = torch.tensor(
        [CRYSTAL_SYSTEM_TO_IDX[sample["crystal_system"]] for sample in samples],
        dtype=torch.long,
    )
    lattice = torch.tensor(
        [
            [
                sample["a"],
                sample["b"],
                sample["c"],
                sample["alpha"],
                sample["beta"],
                sample["gamma"],
            ]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    sample_ids = [sample["sample_id"] for sample in samples]
    return {
        "pxrd_x": pxrd_x,
        "pxrd_y": pxrd_y,
        "peak_num": peak_num,
        "crystal_system_idx": crystal_system_idx,
        "lattice": lattice,
        "sample_id": sample_ids,
    }


def build_train_dataset(config: PXRDDatasetConfig) -> PXRDDataset:
    """Construct a dataset from configuration."""
    return PXRDDataset(config)


def build_dataloader(
    config: PXRDDatasetConfig,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader[DatasetSample]:
    """Build a DataLoader with RealPXRD-compatible collate defaults."""
    dataset = build_train_dataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_peak_batch,
    )
