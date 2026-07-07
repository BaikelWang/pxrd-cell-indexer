"""Tests for lattice normalization."""

from __future__ import annotations

import numpy as np
import torch

from pxrd_cell_indexing.data.normalization import (
    LatticeNormalizer,
    compute_lattice_stats_from_records,
)


def test_round_trip_numpy() -> None:
    normalizer = LatticeNormalizer(
        length_log_mean=1.7,
        length_log_std=0.4,
        angle_mean=88.0,
        angle_std=17.0,
    )
    lattice = np.array([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=np.float32)
    restored = normalizer.denormalize_numpy(normalizer.normalize_numpy(lattice))
    assert np.allclose(restored, lattice, rtol=1e-5, atol=1e-4)


def test_round_trip_torch() -> None:
    normalizer = LatticeNormalizer(
        length_log_mean=1.7,
        length_log_std=0.4,
        angle_mean=88.0,
        angle_std=17.0,
    )
    lattice = torch.tensor([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=torch.float32)
    restored = normalizer.denormalize(normalizer.normalize(lattice))
    assert torch.allclose(restored, lattice, rtol=1e-5, atol=1e-4)


def test_compute_stats_shape() -> None:
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
    stats = compute_lattice_stats_from_records(records)
    normalizer = LatticeNormalizer.from_stats(stats)
    assert normalizer.length_log_std > 0
    assert normalizer.angle_std > 0
