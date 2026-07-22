"""Lattice parameter normalization for regression targets (D24, Decision B matrix6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from pxrd_cell_indexing.geometry import (
    gstar6_to_lattice,
    lattice_lengths_angles,
    lattice_params_to_matrix,
    lattice_to_gstar6,
)

if TYPE_CHECKING:
    from pxrd_cell_indexing.training.config import DataConfig

LatticeRepresentation = Literal["angles", "matrix6", "matrix9", "gstar6"]

# Under our fixed canonical convention (lattice_params_to_matrix), matrix
# positions [0,1], [2,0], [2,1] (a_y, c_x, c_y) are structurally always zero.
# matrix9 regresses all 9 elements unconstrained (RealPXRD-style): these three
# carry zero true variance, but the head is free to output nonzero noise there
# and decode remains robust because lattice_lengths_angles derives
# lengths/angles from vector norms/dot products, not assumed exact zeros.


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


@dataclass(frozen=True)
class MatrixLatticeStats:
    """Per-component z-score stats for canonical 3x3 free components."""

    component_mean: tuple[float, float, float, float, float, float]
    component_std: tuple[float, float, float, float, float, float]
    source: str = "train100k_seed42.jsonl"


def _record_to_lattice6(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            record["lattice_a"],
            record["lattice_b"],
            record["lattice_c"],
            record["lattice_alpha"],
            record["lattice_beta"],
            record["lattice_gamma"],
        ],
        dtype=np.float64,
    )


def _lattice6_to_matrix6_numpy(lattice: np.ndarray) -> np.ndarray:
    """Convert (a,b,c,alpha,beta,gamma) to canonical free components."""
    arr = np.asarray(lattice, dtype=np.float64).reshape(-1, 6)
    matrices = lattice_params_to_matrix(torch.tensor(arr, dtype=torch.float64)).numpy()
    return np.stack(
        [
            matrices[..., 0, 0],
            matrices[..., 0, 2],
            matrices[..., 1, 0],
            matrices[..., 1, 1],
            matrices[..., 1, 2],
            matrices[..., 2, 2],
        ],
        axis=-1,
    )


def _matrix6_to_lattice6_numpy(components: np.ndarray) -> np.ndarray:
    """Reconstruct (a,b,c,alpha,beta,gamma) from canonical free components."""
    arr = np.asarray(components, dtype=np.float64).reshape(-1, 6)
    matrices = np.zeros((arr.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = arr[:, 0]
    matrices[:, 0, 2] = arr[:, 1]
    matrices[:, 1, 0] = arr[:, 2]
    matrices[:, 1, 1] = arr[:, 3]
    matrices[:, 1, 2] = arr[:, 4]
    matrices[:, 2, 2] = arr[:, 5]
    tensor = torch.tensor(matrices, dtype=torch.float64)
    lengths, angles = lattice_lengths_angles(tensor)
    return np.concatenate([lengths.numpy(), angles.numpy()], axis=-1).astype(np.float32)


def _lattice6_to_matrix6_torch(lattice: torch.Tensor) -> torch.Tensor:
    matrix = lattice_params_to_matrix(lattice)
    return torch.stack(
        [
            matrix[..., 0, 0],
            matrix[..., 0, 2],
            matrix[..., 1, 0],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 2],
        ],
        dim=-1,
    )


def _matrix6_to_lattice6_torch(components: torch.Tensor) -> torch.Tensor:
    arr = components.reshape(-1, 6)
    batch = arr.shape[0]
    matrix = torch.zeros(batch, 3, 3, device=arr.device, dtype=arr.dtype)
    matrix[:, 0, 0] = arr[:, 0]
    matrix[:, 0, 2] = arr[:, 1]
    matrix[:, 1, 0] = arr[:, 2]
    matrix[:, 1, 1] = arr[:, 3]
    matrix[:, 1, 2] = arr[:, 4]
    matrix[:, 2, 2] = arr[:, 5]
    lengths, angles = lattice_lengths_angles(matrix)
    return torch.cat([lengths, angles], dim=-1)


@dataclass
class MatrixLatticeNormalizer:
    """Normalize lattice via canonical 3x3 free components (Decision B)."""

    component_mean: tuple[float, float, float, float, float, float]
    component_std: tuple[float, float, float, float, float, float]

    @classmethod
    def from_stats(cls, stats: MatrixLatticeStats) -> MatrixLatticeNormalizer:
        std = tuple(max(value, 1e-8) for value in stats.component_std)
        return cls(component_mean=stats.component_mean, component_std=std)

    @classmethod
    def from_json(cls, path: str | Path) -> MatrixLatticeNormalizer:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        stats = MatrixLatticeStats(
            component_mean=tuple(float(value) for value in raw["component_mean"]),
            component_std=tuple(float(value) for value in raw["component_std"]),
            source=str(raw.get("source", "")),
        )
        return cls.from_stats(stats)

    def _mean_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_mean, device=device, dtype=dtype)

    def _std_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_std, device=device, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            MatrixLatticeStats(
                component_mean=self.component_mean,
                component_std=self.component_std,
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def normalize_numpy(self, lattice: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice, dtype=np.float64)
        components = _lattice6_to_matrix6_numpy(arr)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        return ((components - mean) / std).astype(np.float32)

    def denormalize_numpy(self, lattice_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice_norm, dtype=np.float64)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        components = arr * std + mean
        return _matrix6_to_lattice6_numpy(components)

    def normalize(self, lattice: torch.Tensor) -> torch.Tensor:
        components = _lattice6_to_matrix6_torch(lattice)
        mean = self._mean_tensor(device=components.device, dtype=components.dtype)
        std = self._std_tensor(device=components.device, dtype=components.dtype)
        return (components - mean) / std

    def denormalize(self, lattice_norm: torch.Tensor) -> torch.Tensor:
        mean = self._mean_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        std = self._std_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        components = lattice_norm * std + mean
        return _matrix6_to_lattice6_torch(components)


def compute_matrix6_stats_from_records(records: list[dict[str, Any]]) -> MatrixLatticeStats:
    """Compute per-component z-score stats from jsonl-like records."""
    if not records:
        raise ValueError("records must not be empty")
    lattice_batch = np.stack([_record_to_lattice6(record) for record in records], axis=0)
    components = _lattice6_to_matrix6_numpy(lattice_batch)
    mean = components.mean(axis=0)
    std = components.std(axis=0)
    return MatrixLatticeStats(
        component_mean=tuple(float(value) for value in mean.reshape(-1)),
        component_std=tuple(float(value) for value in std.reshape(-1)),
    )


def _lattice6_to_matrix9_numpy(lattice: np.ndarray) -> np.ndarray:
    arr = np.asarray(lattice, dtype=np.float64).reshape(-1, 6)
    matrices = lattice_params_to_matrix(torch.tensor(arr, dtype=torch.float64)).numpy()
    return matrices.reshape(-1, 9)


def _lattice6_to_matrix9_torch(lattice: torch.Tensor) -> torch.Tensor:
    matrix = lattice_params_to_matrix(lattice)
    return matrix.reshape(*matrix.shape[:-2], 9)


@dataclass(frozen=True)
class Matrix9Stats:
    """Per-element z-score stats for the full unconstrained 3x3 matrix (Decision B, true matrix9)."""

    component_mean: tuple[float, ...]
    component_std: tuple[float, ...]
    source: str = "train100k_seed42.jsonl"


@dataclass
class Matrix9Normalizer:
    """Normalize/denormalize via full unconstrained 9-element 3x3 matrix regression.

    Unlike ``MatrixLatticeNormalizer`` (6 canonical free components), this
    exposes all 9 matrix elements to the head, mirroring RealPXRD's
    ``lattice_out = nn.Linear(hidden_dim, 9)`` design: the three structurally
    zero positions under our fixed convention are still regressed (their
    target is always 0), so the head has genuine spare capacity/no hard
    constraint, and decode is robust to any noise the head puts there because
    ``lattice_lengths_angles`` derives lengths/angles from vector norms and
    dot products (rotation invariant), not from assuming exact zeros.
    """

    component_mean: tuple[float, ...]
    component_std: tuple[float, ...]

    @classmethod
    def from_stats(cls, stats: Matrix9Stats) -> Matrix9Normalizer:
        std = tuple(max(value, 1e-8) for value in stats.component_std)
        return cls(component_mean=stats.component_mean, component_std=std)

    @classmethod
    def from_json(cls, path: str | Path) -> Matrix9Normalizer:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        stats = Matrix9Stats(
            component_mean=tuple(float(value) for value in raw["component_mean"]),
            component_std=tuple(float(value) for value in raw["component_std"]),
            source=str(raw.get("source", "")),
        )
        return cls.from_stats(stats)

    def _mean_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_mean, device=device, dtype=dtype)

    def _std_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_std, device=device, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            Matrix9Stats(component_mean=self.component_mean, component_std=self.component_std)
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def normalize_numpy(self, lattice: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice, dtype=np.float64)
        components = _lattice6_to_matrix9_numpy(arr)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        return ((components - mean) / std).astype(np.float32)

    def denormalize_numpy(self, lattice_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice_norm, dtype=np.float64)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        components = (arr * std + mean).reshape(-1, 3, 3)
        tensor = torch.tensor(components, dtype=torch.float64)
        lengths, angles = lattice_lengths_angles(tensor)
        return np.concatenate([lengths.numpy(), angles.numpy()], axis=-1).astype(np.float32)

    def normalize(self, lattice: torch.Tensor) -> torch.Tensor:
        components = _lattice6_to_matrix9_torch(lattice)
        mean = self._mean_tensor(device=components.device, dtype=components.dtype)
        std = self._std_tensor(device=components.device, dtype=components.dtype)
        return (components - mean) / std

    def denormalize(self, lattice_norm: torch.Tensor) -> torch.Tensor:
        mean = self._mean_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        std = self._std_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        components = (lattice_norm * std + mean).reshape(-1, 3, 3)
        lengths, angles = lattice_lengths_angles(components)
        return torch.cat([lengths, angles], dim=-1)


def compute_matrix9_stats_from_records(records: list[dict[str, Any]]) -> Matrix9Stats:
    """Compute per-element z-score stats for the full 3x3 matrix from jsonl records."""
    if not records:
        raise ValueError("records must not be empty")
    lattice_batch = np.stack([_record_to_lattice6(record) for record in records], axis=0)
    components = _lattice6_to_matrix9_numpy(lattice_batch)
    mean = components.mean(axis=0)
    std = components.std(axis=0)
    return Matrix9Stats(
        component_mean=tuple(float(value) for value in mean.reshape(-1)),
        component_std=tuple(float(value) for value in std.reshape(-1)),
    )


@dataclass(frozen=True)
class GStar6Stats:
    """Per-component z-score stats for reciprocal-metric Cholesky (A3 gstar6).

    Component order: ``[log L11, log L22, log L33, L21, L31, L32]``.
    """

    component_mean: tuple[float, float, float, float, float, float]
    component_std: tuple[float, float, float, float, float, float]
    source: str = "train100k_niggli_seed42.jsonl"
    representation: str = "gstar6"
    convention: str = "niggli"
    pack_order: str = "logL11,logL22,logL33,L21,L31,L32"


def _lattice6_to_gstar6_numpy(lattice: np.ndarray) -> np.ndarray:
    arr = np.asarray(lattice, dtype=np.float64).reshape(-1, 6)
    tensor = torch.tensor(arr, dtype=torch.float64)
    return lattice_to_gstar6(tensor).numpy()


def _gstar6_to_lattice6_numpy(components: np.ndarray) -> np.ndarray:
    arr = np.asarray(components, dtype=np.float64).reshape(-1, 6)
    tensor = torch.tensor(arr, dtype=torch.float64)
    return gstar6_to_lattice(tensor).numpy().astype(np.float32)


def _lattice6_to_gstar6_torch(lattice: torch.Tensor) -> torch.Tensor:
    return lattice_to_gstar6(lattice)


def _gstar6_to_lattice6_torch(components: torch.Tensor) -> torch.Tensor:
    return gstar6_to_lattice(components)


@dataclass
class GStar6Normalizer:
    """Normalize lattice via packed reciprocal-metric Cholesky (A3).

    ``normalize`` maps physical lattice6 → z-scored gstar6.
    ``denormalize`` maps z-scored gstar6 → physical lattice6 for eval/loss.
    """

    component_mean: tuple[float, float, float, float, float, float]
    component_std: tuple[float, float, float, float, float, float]

    @classmethod
    def from_stats(cls, stats: GStar6Stats) -> GStar6Normalizer:
        std = tuple(max(value, 1e-8) for value in stats.component_std)
        return cls(component_mean=stats.component_mean, component_std=std)

    @classmethod
    def from_json(cls, path: str | Path) -> GStar6Normalizer:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        stats = GStar6Stats(
            component_mean=tuple(float(value) for value in raw["component_mean"]),
            component_std=tuple(float(value) for value in raw["component_std"]),
            source=str(raw.get("source", "")),
            representation=str(raw.get("representation", "gstar6")),
            convention=str(raw.get("convention", "niggli")),
            pack_order=str(
                raw.get("pack_order", "logL11,logL22,logL33,L21,L31,L32")
            ),
        )
        return cls.from_stats(stats)

    def _mean_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_mean, device=device, dtype=dtype)

    def _std_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.component_std, device=device, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            GStar6Stats(
                component_mean=self.component_mean,
                component_std=self.component_std,
            )
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def normalize_numpy(self, lattice: np.ndarray) -> np.ndarray:
        components = _lattice6_to_gstar6_numpy(lattice)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        return ((components - mean) / std).astype(np.float32)

    def denormalize_numpy(self, lattice_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(lattice_norm, dtype=np.float64)
        mean = np.asarray(self.component_mean, dtype=np.float64)
        std = np.asarray(self.component_std, dtype=np.float64)
        components = arr * std + mean
        return _gstar6_to_lattice6_numpy(components)

    def normalize(self, lattice: torch.Tensor) -> torch.Tensor:
        components = _lattice6_to_gstar6_torch(lattice)
        mean = self._mean_tensor(device=components.device, dtype=components.dtype)
        std = self._std_tensor(device=components.device, dtype=components.dtype)
        return (components - mean) / std

    def denormalize(self, lattice_norm: torch.Tensor) -> torch.Tensor:
        mean = self._mean_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        std = self._std_tensor(device=lattice_norm.device, dtype=lattice_norm.dtype)
        components = lattice_norm * std + mean
        return _gstar6_to_lattice6_torch(components)


def compute_gstar6_stats_from_records(records: list[dict[str, Any]]) -> GStar6Stats:
    """Compute per-component z-score stats for gstar6 from jsonl-like records."""
    if not records:
        raise ValueError("records must not be empty")
    lattice_batch = np.stack([_record_to_lattice6(record) for record in records], axis=0)
    components = _lattice6_to_gstar6_numpy(lattice_batch)
    mean = components.mean(axis=0)
    std = components.std(axis=0)
    return GStar6Stats(
        component_mean=tuple(float(value) for value in mean.reshape(-1)),
        component_std=tuple(float(value) for value in std.reshape(-1)),
    )


def head_output_dim(representation: LatticeRepresentation | str) -> int:
    """Regression head width implied by the lattice representation."""
    if representation in ("angles", "matrix6", "gstar6"):
        return 6
    if representation == "matrix9":
        return 9
    raise ValueError(f"Unsupported lattice representation: {representation!r}")


def build_lattice_normalizer(
    data_config: DataConfig,
) -> LatticeNormalizer | MatrixLatticeNormalizer | Matrix9Normalizer | GStar6Normalizer:
    """Construct the lattice normalizer implied by ``data_config.representation``."""
    representation = getattr(data_config, "representation", "angles")
    if representation == "matrix6":
        return MatrixLatticeNormalizer.from_json(data_config.lattice_stats)
    if representation == "matrix9":
        return Matrix9Normalizer.from_json(data_config.lattice_stats)
    if representation == "gstar6":
        return GStar6Normalizer.from_json(data_config.lattice_stats)
    if representation == "angles":
        return LatticeNormalizer.from_json(data_config.lattice_stats)
    raise ValueError(f"Unsupported lattice representation: {representation!r}")
