"""Evaluation metrics for indexing smoke training and formal lattice match."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pymatgen.core.lattice import Lattice

from pxrd_cell_indexing.data.normalization import LatticeNormalizer
from pxrd_cell_indexing.geometry import lattice_lengths_angles, lattice_params_to_matrix
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, LatticeCandidate


@dataclass(frozen=True)
class MetricBundle:
    crystal_system_accuracy: float
    lattice_mae: float
    length_mape: float
    top1_lattice_match_proxy: float
    per_crystal_system: dict[str, dict[str, float]]


DEFAULT_LTOL = 0.3
DEFAULT_ATOL_DEG = 10.0


def crystal_system_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def lattice_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.abs(pred - target).mean().item())


def length_mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_lengths = pred[..., :3]
    target_lengths = target[..., :3]
    denom = torch.clamp(target_lengths.abs(), min=1e-6)
    return float((torch.abs(pred_lengths - target_lengths) / denom).mean().item())


def lattice_match_proxy(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ltol: float = 0.3,
    atol_deg: float = 10.0,
) -> torch.Tensor:
    """Return per-sample boolean match using length/angle tolerances."""
    pred = pred.reshape(-1, 6)
    target = target.reshape(-1, 6)
    pred_matrix = lattice_params_to_matrix(pred)
    target_matrix = lattice_params_to_matrix(target)
    pred_lengths, pred_angles = lattice_lengths_angles(pred_matrix)
    target_lengths, target_angles = lattice_lengths_angles(target_matrix)

    length_ok = torch.all(
        torch.abs(pred_lengths - target_lengths)
        <= ltol * torch.clamp(target_lengths.abs(), min=1e-6),
        dim=-1,
    )
    angle_ok = torch.all(torch.abs(pred_angles - target_angles) <= atol_deg, dim=-1)
    if length_ok.ndim == 0:
        return length_ok & angle_ok
    return (length_ok & angle_ok).reshape(-1)


def top1_lattice_match_proxy(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    matches = lattice_match_proxy(pred, target, ltol=ltol, atol_deg=atol_deg)
    return float(matches.float().mean().item())


def lattice_params_to_pmg_lattice(params: Sequence[float] | np.ndarray) -> Lattice:
    """Convert six lattice parameters to a pymatgen ``Lattice``."""
    values = np.asarray(params, dtype=np.float64).reshape(6)
    return Lattice.from_parameters(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def lattice_match_pymatgen(
    pred: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> bool:
    """True if ``pred`` lattice is equivalent to ``target`` under pymatgen mapping."""
    pred_lat = lattice_params_to_pmg_lattice(pred)
    target_lat = lattice_params_to_pmg_lattice(target)
    return pred_lat.find_mapping(target_lat, ltol=ltol, atol=atol_deg) is not None


def top1_lattice_match_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Top-1 lattice match rate using pymatgen ``Lattice.find_mapping``."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    matches = [
        lattice_match_pymatgen(pred_array[idx], target_array[idx], ltol=ltol, atol_deg=atol_deg)
        for idx in range(pred_array.shape[0])
    ]
    return float(np.mean(matches)) if matches else 0.0


def top1_joint_match_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    pred_cs_idx: Sequence[int] | torch.Tensor | np.ndarray,
    target_cs_idx: Sequence[int] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Top-1 lattice match AND correct crystal system (stricter than lattice-only)."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    pred_cs = _as_1d_int_array(pred_cs_idx)
    target_cs = _as_1d_int_array(target_cs_idx)
    if pred_array.shape[0] != target_array.shape[0] or pred_array.shape[0] != pred_cs.shape[0]:
        raise ValueError("preds, targets, and crystal-system indices must have the same length")
    matches = [
        pred_cs[idx] == target_cs[idx]
        and lattice_match_pymatgen(
            pred_array[idx], target_array[idx], ltol=ltol, atol_deg=atol_deg
        )
        for idx in range(pred_array.shape[0])
    ]
    return float(np.mean(matches)) if matches else 0.0


def topk_lattice_match_rate(
    candidate_lists: Sequence[Sequence[LatticeCandidate]],
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Oracle Top-K recall: any candidate matching truth counts as a hit."""
    target_array = _as_2d_array(targets)
    hits: list[bool] = []
    for idx, candidates in enumerate(candidate_lists):
        truth = target_array[idx]
        hit = any(
            lattice_match_pymatgen(
                [
                    candidate.a,
                    candidate.b,
                    candidate.c,
                    candidate.alpha,
                    candidate.beta,
                    candidate.gamma,
                ],
                truth,
                ltol=ltol,
                atol_deg=atol_deg,
            )
            for candidate in candidates
        )
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def _as_2d_array(values: Sequence[Sequence[float]] | torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().reshape(-1, 6)
    return np.asarray(values, dtype=np.float64).reshape(-1, 6)


def _as_1d_int_array(values: Sequence[int] | torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().reshape(-1).astype(np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def evaluate_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    normalizer: LatticeNormalizer,
) -> dict[str, float]:
    pred_norm = outputs["lattice_norm"]
    pred = normalizer.denormalize(pred_norm)
    target = batch["lattice"]
    return {
        "crystal_system_accuracy": crystal_system_accuracy(
            outputs["crystal_system_logits"], batch["crystal_system_idx"]
        ),
        "lattice_mae": lattice_mae(pred, target),
        "length_mape": length_mape(pred, target),
        "top1_lattice_match_proxy": top1_lattice_match_proxy(pred, target),
    }


def evaluate_by_crystal_system(
    pred: torch.Tensor,
    target: torch.Tensor,
    crystal_system_idx: torch.Tensor,
) -> dict[str, dict[str, float]]:
    per_cs: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(CRYSTAL_SYSTEMS):
        mask = crystal_system_idx == idx
        if not mask.any():
            continue
        per_cs[name] = {
            "lattice_mae": lattice_mae(pred[mask], target[mask]),
            "top1_lattice_match_proxy": top1_lattice_match_proxy(
                pred[mask], target[mask]
            ),
            "count": float(mask.sum().item()),
        }
    return per_cs
