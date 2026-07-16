"""Tests for matrix6 lattice normalization (Decision B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.normalization import (
    MatrixLatticeNormalizer,
    _lattice6_to_matrix6_numpy,
    _matrix6_to_lattice6_numpy,
    build_lattice_normalizer,
    compute_matrix6_stats_from_records,
)
from pxrd_cell_indexing.training.config import DataConfig


def test_matrix6_component_round_trip_identity() -> None:
    components = np.array([[4.0, 2.0, -1.5, 3.2, 1.1, 6.5]], dtype=np.float64)
    restored = _matrix6_to_lattice6_numpy(components)
    back = _lattice6_to_matrix6_numpy(restored)
    assert np.allclose(back, components, rtol=1e-5, atol=1e-4)


def test_matrix6_normalizer_round_trip_torch() -> None:
    normalizer = MatrixLatticeNormalizer(
        component_mean=(0.0, 2.0, 0.0, 1.0, 0.5, 3.0),
        component_std=(1.0, 0.5, 0.8, 0.7, 0.6, 1.2),
    )
    lattice = torch.tensor([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=torch.float32)
    restored = normalizer.denormalize(normalizer.normalize(lattice))
    assert torch.allclose(restored, lattice, rtol=1e-4, atol=1e-3)


def test_matrix6_normalizer_round_trip_numpy() -> None:
    normalizer = MatrixLatticeNormalizer(
        component_mean=(0.0, 2.0, 0.0, 1.0, 0.5, 3.0),
        component_std=(1.0, 0.5, 0.8, 0.7, 0.6, 1.2),
    )
    lattice = np.array([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=np.float32)
    restored = normalizer.denormalize_numpy(normalizer.normalize_numpy(lattice))
    assert np.allclose(restored, lattice, rtol=1e-4, atol=1e-3)


def test_matrix6_stats_from_records() -> None:
    records = [
        {
            "lattice_a": 5.0,
            "lattice_b": 5.0,
            "lattice_c": 6.0,
            "lattice_alpha": 90.0,
            "lattice_beta": 90.0,
            "lattice_gamma": 90.0,
        },
        {
            "lattice_a": 3.0,
            "lattice_b": 3.0,
            "lattice_c": 3.0,
            "lattice_alpha": 60.0,
            "lattice_beta": 60.0,
            "lattice_gamma": 60.0,
        },
    ]
    stats = compute_matrix6_stats_from_records(records)
    normalizer = MatrixLatticeNormalizer.from_stats(stats)
    assert all(value > 0 for value in normalizer.component_std)


def test_matrix6_round_trip_on_train100k_sample() -> None:
    jsonl_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "train100k_seed42.jsonl"
    )
    if not jsonl_path.exists():
        return

    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 32:
                break
            records.append(json.loads(line))

    stats = compute_matrix6_stats_from_records(records)
    normalizer = MatrixLatticeNormalizer.from_stats(stats)
    for record in records:
        lattice = np.array(
            [
                [
                    record["lattice_a"],
                    record["lattice_b"],
                    record["lattice_c"],
                    record["lattice_alpha"],
                    record["lattice_beta"],
                    record["lattice_gamma"],
                ]
            ],
            dtype=np.float32,
        )
        restored = normalizer.denormalize_numpy(normalizer.normalize_numpy(lattice))
        assert np.allclose(restored, lattice, rtol=1e-4, atol=1e-2)


def test_build_lattice_normalizer_factory() -> None:
    data_config = DataConfig(
        train_lmdb="train.lmdb",
        valid_lmdb="valid.lmdb",
        train_jsonl="train.jsonl",
        valid_jsonl="valid.jsonl",
        lattice_stats="data/processed/lattice_stats_100k_seed42.json",
        representation="angles",
    )
    normalizer = build_lattice_normalizer(data_config)
    assert normalizer.__class__.__name__ == "LatticeNormalizer"
