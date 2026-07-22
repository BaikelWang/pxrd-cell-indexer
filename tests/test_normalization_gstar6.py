"""Tests for A3 gstar6 reciprocal-metric Cholesky normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pxrd_cell_indexing.data.normalization import (
    GStar6Normalizer,
    build_lattice_normalizer,
    compute_gstar6_stats_from_records,
    head_output_dim,
)
from pxrd_cell_indexing.geometry import (
    gstar6_min_eig,
    gstar6_to_lattice,
    lattice_params_to_matrix,
    lattice_to_gstar6,
    reciprocal_metric_from_gstar6,
)
from pxrd_cell_indexing.training.config import DataConfig


def _sample_valid_cells(n: int = 32, seed: int = 0) -> torch.Tensor:
    """Sample geometrically valid cells (positive volume, well-conditioned G)."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        lengths = rng.uniform(2.5, 12.0, size=(3,))
        angles = rng.uniform(60.0, 120.0, size=(3,))
        cell = np.concatenate([lengths, angles]).astype(np.float64)
        tensor = torch.tensor(cell, dtype=torch.float64)
        matrix = lattice_params_to_matrix(tensor)
        vol = float(torch.linalg.det(matrix).abs())
        g = matrix @ matrix.T
        eig = torch.linalg.eigvalsh(g)
        if vol < 1.0 or float(eig.min()) < 1e-3:
            continue
        # Round-trip sanity in float64 before accepting the sample.
        try:
            restored = gstar6_to_lattice(lattice_to_gstar6(tensor.unsqueeze(0)))
            if not torch.allclose(restored, tensor.unsqueeze(0), rtol=1e-6, atol=1e-5):
                continue
        except Exception:
            continue
        out.append(cell)
    if len(out) < n:
        raise RuntimeError(f"only sampled {len(out)}/{n} valid cells")
    return torch.tensor(np.stack(out, axis=0), dtype=torch.float64)


def test_gstar6_round_trip_float64_near_machine_precision() -> None:
    cells = _sample_valid_cells(64, seed=1)
    packed = lattice_to_gstar6(cells)
    restored = gstar6_to_lattice(packed)
    assert torch.allclose(restored, cells, rtol=1e-6, atol=1e-5)


def test_gstar6_round_trip_float32_far_below_strict() -> None:
    cells = _sample_valid_cells(64, seed=2).to(dtype=torch.float32)
    packed = lattice_to_gstar6(cells)
    restored = gstar6_to_lattice(packed)
    # Strict gate is 5% / 3°; float32 decode must be far tighter.
    length_rel = (restored[:, :3] - cells[:, :3]).abs() / cells[:, :3].clamp_min(1e-6)
    angle_abs = (restored[:, 3:] - cells[:, 3:]).abs()
    assert float(length_rel.max()) < 1e-4
    assert float(angle_abs.max()) < 1e-3


def test_gstar6_reconstructed_metric_is_spd() -> None:
    cells = _sample_valid_cells(48, seed=3)
    packed = lattice_to_gstar6(cells)
    g_star = reciprocal_metric_from_gstar6(packed)
    eig = torch.linalg.eigvalsh(0.5 * (g_star + g_star.transpose(-1, -2)))
    assert torch.all(eig > 0)
    assert torch.all(gstar6_min_eig(packed) > 0)


def test_gstar6_normalizer_round_trip_torch_numpy() -> None:
    records = [
        {
            "lattice_a": 5.0,
            "lattice_b": 5.1,
            "lattice_c": 6.2,
            "lattice_alpha": 90.0,
            "lattice_beta": 90.0,
            "lattice_gamma": 120.0,
        },
        {
            "lattice_a": 3.2,
            "lattice_b": 4.1,
            "lattice_c": 5.5,
            "lattice_alpha": 80.0,
            "lattice_beta": 85.0,
            "lattice_gamma": 95.0,
        },
    ]
    stats = compute_gstar6_stats_from_records(records)
    normalizer = GStar6Normalizer.from_stats(stats)
    lattice_t = torch.tensor(
        [[5.0, 5.1, 6.2, 90.0, 90.0, 120.0]], dtype=torch.float32
    )
    restored_t = normalizer.denormalize(normalizer.normalize(lattice_t))
    assert torch.allclose(restored_t, lattice_t, rtol=1e-4, atol=1e-3)

    lattice_n = lattice_t.numpy()
    restored_n = normalizer.denormalize_numpy(normalizer.normalize_numpy(lattice_n))
    assert np.allclose(restored_n, lattice_n, rtol=1e-4, atol=1e-3)


def test_gstar6_finite_gradient_through_denormalize() -> None:
    normalizer = GStar6Normalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    # Start near a valid encoding of a cubic-ish cell.
    cell = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    packed = lattice_to_gstar6(cell).detach().requires_grad_(True)
    norm = (packed - normalizer._mean_tensor(device=packed.device, dtype=packed.dtype)) / (
        normalizer._std_tensor(device=packed.device, dtype=packed.dtype)
    )
    decoded = normalizer.denormalize(norm)
    loss = decoded.pow(2).mean()
    loss.backward()
    assert packed.grad is not None
    assert torch.isfinite(packed.grad).all()


def test_gstar6_batch_shapes_and_head_dim() -> None:
    cells = _sample_valid_cells(8, seed=4).to(dtype=torch.float32)
    packed = lattice_to_gstar6(cells)
    assert packed.shape == (8, 6)
    restored = gstar6_to_lattice(packed)
    assert restored.shape == (8, 6)
    assert head_output_dim("gstar6") == 6
    assert head_output_dim("matrix6") == 6


def test_gstar6_extreme_but_valid_cell_stays_finite() -> None:
    # Moderately anisotropic / skewed but still a valid open cell.
    cell = torch.tensor([[2.5, 8.0, 11.0, 70.0, 100.0, 115.0]], dtype=torch.float64)
    packed = lattice_to_gstar6(cell)
    restored = gstar6_to_lattice(packed)
    assert torch.isfinite(packed).all()
    assert torch.isfinite(restored).all()
    assert torch.allclose(restored, cell, rtol=1e-5, atol=1e-4)


def test_gstar6_round_trip_on_train100k_sample() -> None:
    jsonl_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "train100k_niggli_seed42.jsonl"
    )
    if not jsonl_path.exists():
        return
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 64:
                break
            records.append(json.loads(line))
    stats = compute_gstar6_stats_from_records(records)
    normalizer = GStar6Normalizer.from_stats(stats)
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


def test_build_lattice_normalizer_gstar6_factory(tmp_path: Path) -> None:
    stats_path = tmp_path / "gstar6_stats.json"
    payload = {
        "component_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "component_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "representation": "gstar6",
        "convention": "niggli",
        "pack_order": "logL11,logL22,logL33,L21,L31,L32",
    }
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    data_config = DataConfig(
        train_lmdb="train.lmdb",
        valid_lmdb="valid.lmdb",
        train_jsonl="train.jsonl",
        valid_jsonl="valid.jsonl",
        lattice_stats=str(stats_path),
        representation="gstar6",
    )
    built = build_lattice_normalizer(data_config)
    assert built.__class__.__name__ == "GStar6Normalizer"
