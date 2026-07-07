"""Lattice parameter normalization for regression targets (D24)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LatticeStats:
    """Statistics computed from training split only."""

    length_log_mean: float
    length_log_std: float
    angle_mean: float
    angle_std: float
    source: str = "train10k_seed42.jsonl"


@dataclass
class LatticeNormalizer:
    """Normalize primitive lattice params: log+z-score lengths, z-score angles."""

    length_log_mean: float
    length_log_std: float
    angle_mean: float
    angle_std: float

    @classmethod
    def from_stats(cls, stats: LatticeStats) -> LatticeNormalizer:
        return cls(
            length_log_mean=stats.length_log_mean,
            length_log_std=max(stats.length_log_std, 1e-8),
            angle_mean=stats.angle_mean,
            angle_std=max(stats.angle_std, 1e-8),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> LatticeNormalizer:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        stats = LatticeStats(
            length_log_mean=float(raw["length_log_mean"]),
            length_log_std=float(raw["length_log_std"]),
            angle_mean=float(raw["angle_mean"]),
            angle_std=float(raw["angle_std"]),
            source=str(raw.get("source", "")),
        )
        return cls.from_stats(stats)

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            LatticeStats(
                length_log_mean=self.length_log_mean,
                length_log_std=self.length_log_std,
                angle_mean=self.angle_mean,
                angle_std=self.angle_std,
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def normalize_numpy(self, lattice: np.ndarray) -> np.ndarray:
        """Transform [..., 6] lattice (a,b,c,alpha,beta,gamma) to normalized space."""
        arr = np.asarray(lattice, dtype=np.float64)
        out = arr.copy()
        lengths = arr[..., :3]
        angles = arr[..., 3:]
        out[..., :3] = (np.log(lengths) - self.length_log_mean) / self.length_log_std
        out[..., 3:] = (angles - self.angle_mean) / self.angle_std
        return out.astype(np.float32)

    def denormalize_numpy(self, lattice_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice_norm, dtype=np.float64)
        out = arr.copy()
        out[..., :3] = np.exp(arr[..., :3] * self.length_log_std + self.length_log_mean)
        out[..., 3:] = arr[..., 3:] * self.angle_std + self.angle_mean
        return out.astype(np.float32)

    def normalize(self, lattice: torch.Tensor) -> torch.Tensor:
        lengths = lattice[..., :3]
        angles = lattice[..., 3:]
        norm_lengths = (torch.log(lengths) - self.length_log_mean) / self.length_log_std
        norm_angles = (angles - self.angle_mean) / self.angle_std
        return torch.cat([norm_lengths, norm_angles], dim=-1)

    def denormalize(self, lattice_norm: torch.Tensor) -> torch.Tensor:
        lengths = torch.exp(lattice_norm[..., :3] * self.length_log_std + self.length_log_mean)
        angles = lattice_norm[..., 3:] * self.angle_std + self.angle_mean
        return torch.cat([lengths, angles], dim=-1)


def compute_lattice_stats_from_records(records: list[dict[str, Any]]) -> LatticeStats:
    """Compute shared length/angle stats from jsonl-like records."""
    lengths: list[float] = []
    angles: list[float] = []
    for record in records:
        lengths.extend(
            [record["lattice_a"], record["lattice_b"], record["lattice_c"]]
        )
        angles.extend(
            [
                record["lattice_alpha"],
                record["lattice_beta"],
                record["lattice_gamma"],
            ]
        )
    length_arr = np.asarray(lengths, dtype=np.float64)
    angle_arr = np.asarray(angles, dtype=np.float64)
    log_lengths = np.log(length_arr)
    return LatticeStats(
        length_log_mean=float(log_lengths.mean()),
        length_log_std=float(log_lengths.std()),
        angle_mean=float(angle_arr.mean()),
        angle_std=float(angle_arr.std()),
    )
