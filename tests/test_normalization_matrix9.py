"""Tests for matrix9 (full unconstrained 3x3 matrix) lattice normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.normalization import (
    Matrix9Normalizer,
    build_lattice_normalizer,
    compute_matrix9_stats_from_records,
    head_output_dim,
)
from pxrd_cell_indexing.training.config import DataConfig


def _default_normalizer() -> Matrix9Normalizer:
    return Matrix9Normalizer(
        component_mean=(0.0, 0.0, 2.0, 0.0, 1.0, 0.5, 0.0, 0.0, 3.0),
        component_std=(1.0, 1e-8, 0.5, 0.8, 0.7, 0.6, 1e-8, 1e-8, 1.2),
    )


def test_matrix9_normalizer_round_trip_torch() -> None:
    normalizer = _default_normalizer()
    lattice = torch.tensor([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=torch.float32)
    restored = normalizer.denormalize(normalizer.normalize(lattice))
    assert torch.allclose(restored, lattice, rtol=1e-4, atol=1e-3)


def test_matrix9_normalizer_round_trip_numpy() -> None:
    normalizer = _default_normalizer()
    lattice = np.array([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=np.float32)
    restored = normalizer.denormalize_numpy(normalizer.normalize_numpy(lattice))
    assert np.allclose(restored, lattice, rtol=1e-4, atol=1e-3)


def test_matrix9_decode_robust_to_nonzero_structural_positions() -> None:
    """Head noise at structurally-zero positions [0,1],[2,0],[2,1] must not break decode."""
    normalizer = _default_normalizer()
    lattice = torch.tensor([[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=torch.float32)
    norm = normalizer.normalize(lattice)
    noisy = norm.clone()
    # Inject noise at the structurally-zero component indices (1, 6, 7 in flattened order).
    noisy[:, 1] += 0.3
    noisy[:, 6] += 0.3
    noisy[:, 7] -= 0.2
    restored = normalizer.denormalize(noisy)
    assert torch.isfinite(restored).all()
    assert restored.shape == lattice.shape
    # Should stay close to truth despite noise (lengths/angles derived from norms/dot products).
    assert torch.allclose(restored[:, :3], lattice[:, :3], atol=0.5)


def test_matrix9_stats_from_records() -> None:
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
    stats = compute_matrix9_stats_from_records(records)
    assert len(stats.component_mean) == 9
    assert len(stats.component_std) == 9
    normalizer = Matrix9Normalizer.from_stats(stats)
    # Structurally-zero positions (indices 1, 6, 7) have zero true variance;
    # from_stats must clamp std away from zero to avoid division blow-up.
    assert all(value > 0 for value in normalizer.component_std)


def test_matrix9_round_trip_on_train100k_sample() -> None:
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

    stats = compute_matrix9_stats_from_records(records)
    normalizer = Matrix9Normalizer.from_stats(stats)
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


def test_head_output_dim() -> None:
    assert head_output_dim("angles") == 6
    assert head_output_dim("matrix6") == 6
    assert head_output_dim("matrix9") == 9


def test_build_lattice_normalizer_matrix9_factory(tmp_path: Path) -> None:
    stats_path = tmp_path / "matrix9_stats.json"
    normalizer = _default_normalizer()
    normalizer.save(stats_path)

    data_config = DataConfig(
        train_lmdb="train.lmdb",
        valid_lmdb="valid.lmdb",
        train_jsonl="train.jsonl",
        valid_jsonl="valid.jsonl",
        lattice_stats=str(stats_path),
        representation="matrix9",
    )
    built = build_lattice_normalizer(data_config)
    assert isinstance(built, Matrix9Normalizer)
