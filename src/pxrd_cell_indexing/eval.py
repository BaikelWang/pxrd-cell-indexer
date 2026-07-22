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
from pxrd_cell_indexing.model.bravais import best_hypothesis
from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, CRYSTAL_SYSTEM_TO_IDX, LatticeCandidate


@dataclass(frozen=True)
class MetricBundle:
    crystal_system_accuracy: float
    lattice_mae: float
    length_mape: float
    top1_lattice_match_proxy: float
    per_crystal_system: dict[str, dict[str, float]]


DEFAULT_LTOL = 0.3
DEFAULT_ATOL_DEG = 10.0
# |log(V_pred / V_truth)| bound; log(2) ≈ 0.693 rejects >2× volume mismatch.
DEFAULT_VOLUME_LOG_RATIO_MAX = float(np.log(2.0))


def infer_crystal_system_idx_from_lattice(
    pred_lattice: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    identity_penalty_score: float = 1.0,
) -> np.ndarray:
    """Post-hoc crystal system from best Bravais snap hypothesis (-1 if identity)."""
    pred_array = _as_2d_array(pred_lattice)
    indices: list[int] = []
    for row in pred_array:
        hypothesis = best_hypothesis(row, identity_penalty_score=identity_penalty_score)
        if hypothesis.bravais_key == "identity" or hypothesis.crystal_system_label is None:
            indices.append(-1)
            continue
        indices.append(CRYSTAL_SYSTEM_TO_IDX[hypothesis.crystal_system_label])
    return np.asarray(indices, dtype=np.int64)


def crystal_system_accuracy_from_lattice(
    pred_lattice: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> float:
    """Diagnostic crystal-system accuracy using post-hoc lattice inference."""
    pred_cs = infer_crystal_system_idx_from_lattice(pred_lattice)
    target_cs = _as_1d_int_array(targets)
    valid = pred_cs >= 0
    if not valid.any():
        return 0.0
    return float((pred_cs[valid] == target_cs[valid]).mean())


def crystal_system_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Classifier CS accuracy from ``crystal_system_logits`` argmax."""
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


PARAM_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
PEAK_COUNT_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("le10", 0, 10),
    ("11_20", 11, 20),
    ("21_40", 21, 40),
    ("gt40", 41, None),
)


def elementwise_param_ok_mask(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> np.ndarray:
    """Boolean mask ``[N, 6]`` for independent length/angle tolerances."""
    pred_t = torch.as_tensor(_as_2d_array(preds), dtype=torch.float64)
    target_t = torch.as_tensor(_as_2d_array(targets), dtype=torch.float64)
    pred_matrix = lattice_params_to_matrix(pred_t)
    target_matrix = lattice_params_to_matrix(target_t)
    pred_lengths, pred_angles = lattice_lengths_angles(pred_matrix)
    target_lengths, target_angles = lattice_lengths_angles(target_matrix)
    length_ok = torch.abs(pred_lengths - target_lengths) <= ltol * torch.clamp(
        target_lengths.abs(), min=1e-6
    )
    angle_ok = torch.abs(pred_angles - target_angles) <= atol_deg
    return torch.cat([length_ok, angle_ok], dim=-1).detach().cpu().numpy().astype(bool)


def six_param_pass_rates(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> dict[str, float]:
    """Independent pass rate for each of a,b,c,α,β,γ under strict tolerances."""
    ok = elementwise_param_ok_mask(preds, targets, ltol=ltol, atol_deg=atol_deg)
    if ok.size == 0:
        return {name: 0.0 for name in PARAM_NAMES}
    return {name: float(ok[:, i].mean()) for i, name in enumerate(PARAM_NAMES)}


def stratify_elementwise_by_peak_count(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    peak_counts: Sequence[int] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> dict[str, dict[str, float]]:
    """Strict elementwise rate stratified by peak-count bins (A0)."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    peaks = _as_1d_int_array(peak_counts)
    out: dict[str, dict[str, float]] = {}
    for name, lo, hi in PEAK_COUNT_BINS:
        if hi is None:
            mask = peaks >= lo
        else:
            mask = (peaks >= lo) & (peaks <= hi)
        count = int(mask.sum())
        if count == 0:
            out[name] = {"count": 0.0, "strict_elementwise_rate": 0.0}
            continue
        rate = top1_elementwise_match_rate(
            pred_array[mask], target_array[mask], ltol=ltol, atol_deg=atol_deg
        )
        out[name] = {"count": float(count), "strict_elementwise_rate": float(rate)}
    return out


def stratify_elementwise_by_crystal_system(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    crystal_system_idx: Sequence[int] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> dict[str, dict[str, float]]:
    """Strict elementwise rate per crystal system (A0)."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    cs = _as_1d_int_array(crystal_system_idx)
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(CRYSTAL_SYSTEMS):
        mask = cs == idx
        count = int(mask.sum())
        if count == 0:
            continue
        out[name] = {
            "count": float(count),
            "strict_elementwise_rate": float(
                top1_elementwise_match_rate(
                    pred_array[mask], target_array[mask], ltol=ltol, atol_deg=atol_deg
                )
            ),
        }
    return out


def cs_correct_subset_elementwise_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    classifier_cs_idx: Sequence[int] | torch.Tensor | np.ndarray,
    target_cs_idx: Sequence[int] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> dict[str, float]:
    """Strict elementwise on the subset where classifier CS matches GT."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    pred_cs = _as_1d_int_array(classifier_cs_idx)
    target_cs = _as_1d_int_array(target_cs_idx)
    mask = pred_cs == target_cs
    count = int(mask.sum())
    if count == 0:
        return {"count": 0.0, "strict_elementwise_rate": 0.0}
    return {
        "count": float(count),
        "strict_elementwise_rate": float(
            top1_elementwise_match_rate(
                pred_array[mask], target_array[mask], ltol=ltol, atol_deg=atol_deg
            )
        ),
    }


def build_a0_metrics_block(
    *,
    preds_predicted: Sequence[Sequence[float]] | np.ndarray,
    preds_oracle: Sequence[Sequence[float]] | np.ndarray | None,
    targets: Sequence[Sequence[float]] | np.ndarray,
    peak_counts: Sequence[int] | np.ndarray,
    target_cs_idx: Sequence[int] | np.ndarray,
    classifier_cs_idx: Sequence[int] | np.ndarray | None,
    lattice_inferred_cs_idx: Sequence[int] | np.ndarray | None = None,
    ltol: float = 0.05,
    atol_deg: float = 3.0,
) -> dict[str, Any]:
    """Unified A0 metrics schema used by trainer dumps and protocol eval."""
    pred_arr = _as_2d_array(preds_predicted)
    tgt_arr = _as_2d_array(targets)
    target_cs = _as_1d_int_array(target_cs_idx)
    if lattice_inferred_cs_idx is None:
        lattice_cs = infer_crystal_system_idx_from_lattice(pred_arr)
    else:
        lattice_cs = _as_1d_int_array(lattice_inferred_cs_idx)

    pred_elem = top1_elementwise_match_rate(pred_arr, tgt_arr, ltol=ltol, atol_deg=atol_deg)
    block: dict[str, Any] = {
        "strict_ltol": ltol,
        "strict_atol_deg": atol_deg,
        "strict_raw_top1_elementwise_rate": float(pred_elem),
        "angle_mae": angle_mae(
            torch.as_tensor(pred_arr, dtype=torch.float32),
            torch.as_tensor(tgt_arr, dtype=torch.float32),
        ),
        "length_mae": length_mae(
            torch.as_tensor(pred_arr, dtype=torch.float32),
            torch.as_tensor(tgt_arr, dtype=torch.float32),
        ),
        "length_mape": length_mape(
            torch.as_tensor(pred_arr, dtype=torch.float32),
            torch.as_tensor(tgt_arr, dtype=torch.float32),
        ),
        "lattice_inferred_cs_accuracy": float((lattice_cs == target_cs).mean())
        if len(target_cs)
        else 0.0,
        "six_param_pass_rates": six_param_pass_rates(
            pred_arr, tgt_arr, ltol=ltol, atol_deg=atol_deg
        ),
        "per_crystal_system_strict_elementwise": stratify_elementwise_by_crystal_system(
            pred_arr, tgt_arr, target_cs, ltol=ltol, atol_deg=atol_deg
        ),
        "by_peak_count_strict_elementwise": stratify_elementwise_by_peak_count(
            pred_arr, tgt_arr, peak_counts, ltol=ltol, atol_deg=atol_deg
        ),
    }

    if classifier_cs_idx is not None:
        clf = _as_1d_int_array(classifier_cs_idx)
        block["classifier_cs_accuracy"] = float((clf == target_cs).mean()) if len(target_cs) else 0.0
        subset = cs_correct_subset_elementwise_rate(
            pred_arr, tgt_arr, clf, target_cs, ltol=ltol, atol_deg=atol_deg
        )
        block["cs_correct_subset_lattice_elementwise"] = subset["strict_elementwise_rate"]
        block["cs_correct_subset_count"] = subset["count"]
    else:
        block["classifier_cs_accuracy"] = None
        block["cs_correct_subset_lattice_elementwise"] = None
        block["cs_correct_subset_count"] = None

    if preds_oracle is not None:
        oracle_arr = _as_2d_array(preds_oracle)
        oracle_elem = top1_elementwise_match_rate(
            oracle_arr, tgt_arr, ltol=ltol, atol_deg=atol_deg
        )
        block["oracle_cs_strict_elementwise_rate"] = float(oracle_elem)
        block["predicted_cs_strict_elementwise_rate"] = float(pred_elem)
        block["oracle_predicted_route_gap_pp"] = float((oracle_elem - pred_elem) * 100.0)
    else:
        block["oracle_cs_strict_elementwise_rate"] = None
        block["predicted_cs_strict_elementwise_rate"] = float(pred_elem)
        block["oracle_predicted_route_gap_pp"] = None

    # Backward-compatible alias: historical key meant lattice-inferred CS.
    block["crystal_system_accuracy"] = block["lattice_inferred_cs_accuracy"]
    return block


def lattice_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.abs(pred - target).mean().item())


def length_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error on lattice lengths (a, b, c) in Angstrom."""
    return float(torch.abs(pred[..., :3] - target[..., :3]).mean().item())


def angle_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error on lattice angles (alpha, beta, gamma) in degrees."""
    return float(torch.abs(pred[..., 3:] - target[..., 3:]).mean().item())


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
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred_arr.shape[0] != 6 or target_arr.shape[0] != 6:
        return False
    if not (np.isfinite(pred_arr).all() and np.isfinite(target_arr).all()):
        return False
    # Degenerate / near-flat cells can make find_mapping extremely expensive.
    if float(np.min(pred_arr[:3])) < 0.4 or float(np.max(pred_arr[:3])) > 250.0:
        return False
    if float(np.min(pred_arr[3:])) < 15.0 or float(np.max(pred_arr[3:])) > 165.0:
        return False
    try:
        if lattice_volume(pred_arr) < 1e-6:
            return False
        pred_lat = lattice_params_to_pmg_lattice(pred_arr)
        target_lat = lattice_params_to_pmg_lattice(target_arr)
        return pred_lat.find_mapping(target_lat, ltol=ltol, atol=atol_deg) is not None
    except Exception:
        return False


def lattice_volume(params: Sequence[float] | np.ndarray) -> float:
    """Absolute unit-cell volume from six lattice parameters."""
    matrix = lattice_params_to_matrix(
        torch.tensor(np.asarray(params, dtype=np.float64).reshape(6), dtype=torch.float64)
    )
    return float(abs(torch.linalg.det(matrix).item()))


def volume_log_ratio(
    pred: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> float:
    """Absolute log volume ratio ``|log(V_pred / V_truth)|`` (inf if either volume ~0)."""
    v_pred = lattice_volume(pred)
    v_target = lattice_volume(target)
    if v_pred < 1e-12 or v_target < 1e-12:
        return float("inf")
    return float(abs(np.log(v_pred / v_target)))


def volume_within_guard(
    pred: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    max_log_volume_ratio: float = DEFAULT_VOLUME_LOG_RATIO_MAX,
) -> bool:
    """True if volumes are within a multiplicative factor (via log-ratio)."""
    return volume_log_ratio(pred, target) <= max_log_volume_ratio


def lattice_match_elementwise(
    pred: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> bool:
    """True if a,b,c and α,β,γ each fall within relative/absolute tolerances.

    Unlike ``lattice_match_pymatgen``, this does **not** accept subcell/supercell
    mappings — it requires elementwise agreement on the six parameters (after
    matrix round-trip for numerical stability, matching ``lattice_match_proxy``).
    """
    pred_t = torch.tensor(np.asarray(pred, dtype=np.float64).reshape(1, 6), dtype=torch.float64)
    target_t = torch.tensor(
        np.asarray(target, dtype=np.float64).reshape(1, 6), dtype=torch.float64
    )
    return bool(
        lattice_match_proxy(pred_t, target_t, ltol=ltol, atol_deg=atol_deg).reshape(-1)[0].item()
    )


def lattice_match_volume_guarded(
    pred: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
    max_log_volume_ratio: float = DEFAULT_VOLUME_LOG_RATIO_MAX,
) -> bool:
    """``find_mapping`` hit AND volume within guard (rejects many subcell/supercell hits)."""
    return lattice_match_pymatgen(
        pred, target, ltol=ltol, atol_deg=atol_deg
    ) and volume_within_guard(pred, target, max_log_volume_ratio=max_log_volume_ratio)


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


def top1_elementwise_match_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Top-1 rate under elementwise a,b,c,α,β,γ tolerances (no subcell mapping)."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    matches = [
        lattice_match_elementwise(
            pred_array[idx], target_array[idx], ltol=ltol, atol_deg=atol_deg
        )
        for idx in range(pred_array.shape[0])
    ]
    return float(np.mean(matches)) if matches else 0.0


def top1_volume_guarded_match_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
    max_log_volume_ratio: float = DEFAULT_VOLUME_LOG_RATIO_MAX,
) -> float:
    """Top-1 ``find_mapping`` rate with volume-ratio guard."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    matches = [
        lattice_match_volume_guarded(
            pred_array[idx],
            target_array[idx],
            ltol=ltol,
            atol_deg=atol_deg,
            max_log_volume_ratio=max_log_volume_ratio,
        )
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


def _candidate_six(candidate: LatticeCandidate) -> list[float]:
    return [
        candidate.a,
        candidate.b,
        candidate.c,
        candidate.alpha,
        candidate.beta,
        candidate.gamma,
    ]


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
                _candidate_six(candidate),
                truth,
                ltol=ltol,
                atol_deg=atol_deg,
            )
            for candidate in candidates
        )
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def topk_elementwise_match_rate(
    candidate_lists: Sequence[Sequence[LatticeCandidate]],
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Oracle Top-K recall under elementwise tolerances (no subcell mapping)."""
    target_array = _as_2d_array(targets)
    hits: list[bool] = []
    for idx, candidates in enumerate(candidate_lists):
        truth = target_array[idx]
        hit = any(
            lattice_match_elementwise(
                _candidate_six(candidate),
                truth,
                ltol=ltol,
                atol_deg=atol_deg,
            )
            for candidate in candidates
        )
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def topk_volume_guarded_match_rate(
    candidate_lists: Sequence[Sequence[LatticeCandidate]],
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
    max_log_volume_ratio: float = DEFAULT_VOLUME_LOG_RATIO_MAX,
) -> float:
    """Oracle Top-K recall: ``find_mapping`` hit with volume within guard."""
    target_array = _as_2d_array(targets)
    hits: list[bool] = []
    for idx, candidates in enumerate(candidate_lists):
        truth = target_array[idx]
        hit = any(
            lattice_match_volume_guarded(
                _candidate_six(candidate),
                truth,
                ltol=ltol,
                atol_deg=atol_deg,
                max_log_volume_ratio=max_log_volume_ratio,
            )
            for candidate in candidates
        )
        hits.append(hit)
    return float(np.mean(hits)) if hits else 0.0


def mapping_vs_elementwise_gap_rate(
    preds: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Fraction where ``find_mapping`` hits but elementwise fails (pseudo-hit rate)."""
    pred_array = _as_2d_array(preds)
    target_array = _as_2d_array(targets)
    if pred_array.shape[0] == 0:
        return 0.0
    pseudo = 0
    for idx in range(pred_array.shape[0]):
        mapped = lattice_match_pymatgen(
            pred_array[idx], target_array[idx], ltol=ltol, atol_deg=atol_deg
        )
        elementwise = lattice_match_elementwise(
            pred_array[idx], target_array[idx], ltol=ltol, atol_deg=atol_deg
        )
        if mapped and not elementwise:
            pseudo += 1
    return float(pseudo / pred_array.shape[0])


def topk_mapping_vs_elementwise_gap_rate(
    candidate_lists: Sequence[Sequence[LatticeCandidate]],
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """Fraction where Top-K has a ``find_mapping`` hit but no elementwise hit."""
    target_array = _as_2d_array(targets)
    if not candidate_lists:
        return 0.0
    pseudo = 0
    for idx, candidates in enumerate(candidate_lists):
        truth = target_array[idx]
        mapped = any(
            lattice_match_pymatgen(
                _candidate_six(c), truth, ltol=ltol, atol_deg=atol_deg
            )
            for c in candidates
        )
        elementwise = any(
            lattice_match_elementwise(
                _candidate_six(c), truth, ltol=ltol, atol_deg=atol_deg
            )
            for c in candidates
        )
        if mapped and not elementwise:
            pseudo += 1
    return float(pseudo / len(candidate_lists))


def oracle_hyp_elementwise_rate(
    hyp_preds: Sequence[Sequence[Sequence[float]]] | torch.Tensor | np.ndarray,
    targets: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    *,
    ltol: float = DEFAULT_LTOL,
    atol_deg: float = DEFAULT_ATOL_DEG,
) -> float:
    """R10: oracle recall among K raw multi-hypothesis candidates (any-hit).

    ``hyp_preds``: ``[N, K, 6]`` physical lattice params per sample's own
    (ground-truth) crystal-system head. Isolates candidate-generation quality
    from ranking/routing -- this is the "can at least one hypothesis fit the
    truth" upper bound the R10 gate checks against the K=1 raw baseline.
    """
    if torch.is_tensor(hyp_preds):
        hyp_array = hyp_preds.detach().cpu().numpy()
    else:
        hyp_array = np.asarray(hyp_preds, dtype=np.float64)
    if hyp_array.ndim != 3 or hyp_array.shape[-1] != 6:
        raise ValueError(f"expected hyp_preds [N, K, 6], got {hyp_array.shape}")
    target_array = _as_2d_array(targets)
    hits: list[bool] = []
    for idx in range(hyp_array.shape[0]):
        truth = target_array[idx]
        hit = any(
            lattice_match_elementwise(hyp_array[idx, k], truth, ltol=ltol, atol_deg=atol_deg)
            for k in range(hyp_array.shape[1])
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
    metrics: dict[str, float] = {
        # Historical key = lattice-inferred CS (Bravais snap), NOT classifier.
        "crystal_system_accuracy": crystal_system_accuracy_from_lattice(
            pred, batch["crystal_system_idx"]
        ),
        "lattice_inferred_cs_accuracy": crystal_system_accuracy_from_lattice(
            pred, batch["crystal_system_idx"]
        ),
        "lattice_mae": lattice_mae(pred, target),
        "length_mae": length_mae(pred, target),
        "angle_mae": angle_mae(pred, target),
        "length_mape": length_mape(pred, target),
        "top1_lattice_match_proxy": top1_lattice_match_proxy(pred, target),
    }
    logits = outputs.get("crystal_system_logits")
    if logits is not None:
        metrics["classifier_cs_accuracy"] = crystal_system_accuracy(
            logits, batch["crystal_system_idx"]
        )
    return metrics


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
