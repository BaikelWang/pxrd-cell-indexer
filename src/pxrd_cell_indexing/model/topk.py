"""Top-K candidate generation via Bravais geometry snap hypotheses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from pxrd_cell_indexing.geometry import lattice_params_to_matrix
from pxrd_cell_indexing.model.bravais import BravaisHypothesis, generate_bravais_hypotheses
from pxrd_cell_indexing.types import TOP_K_DEFAULT, LatticeCandidate

if TYPE_CHECKING:
    from pxrd_cell_indexing.model.heads import IndexingModel

LENGTH_SCALE_FACTORS: tuple[float, ...] = (
    2.0,
    0.5,
    math.sqrt(2),
    1.0 / math.sqrt(2),
    math.sqrt(3),
    1.0 / math.sqrt(3),
)

# Default scales plus common integer supercell/subcell factors (A2 extended set).
EXTENDED_LENGTH_SCALE_FACTORS: tuple[float, ...] = LENGTH_SCALE_FACTORS + (
    3.0,
    4.0,
    1.0 / 3.0,
    1.0 / 4.0,
)

SCALE_SET_REGISTRY: dict[str, tuple[float, ...]] = {
    "none": (),
    "default": LENGTH_SCALE_FACTORS,
    "extended": EXTENDED_LENGTH_SCALE_FACTORS,
}

VARIANT_CONFIDENCE_DISCOUNT = 0.5


@dataclass(frozen=True)
class TopKConfig:
    k: int = TOP_K_DEFAULT
    length_scale_factors: tuple[float, ...] = LENGTH_SCALE_FACTORS
    variant_confidence_discount: float = VARIANT_CONFIDENCE_DISCOUNT
    identity_penalty_score: float = 1.0
    mc_dropout_samples: int = 0
    dedupe_length_tol: float = 0.05
    dedupe_angle_tol: float = 2.0
    top_hypotheses_for_variants: int = 3
    # Drop variants whose |log(V / V_ref)| exceeds this (ref = raw lattice prediction).
    # None disables. log(2)≈0.693 keeps ~factor-2 volume; isotropic ×2 (V×8) is rejected.
    max_log_volume_ratio_vs_base: float | None = None
    # If True, also emit single-axis ×2/×0.5 variants (legacy behavior).
    include_axis_scale_variants: bool = True
    # default = Decision A 8 hypotheses; extended = + mono + hex_strict (A4)
    bravais_set: str = "default"


def _params_array_to_candidate(
    params: torch.Tensor | list[float],
    *,
    crystal_system: str,
    confidence: float,
    bravais_key: str | None = None,
) -> LatticeCandidate:
    if torch.is_tensor(params):
        values = [float(v) for v in params.reshape(6).tolist()]
    else:
        values = [float(v) for v in params]
    a, b, c, alpha, beta, gamma = values
    return LatticeCandidate(
        crystal_system=crystal_system,
        a=max(a, 1e-6),
        b=max(b, 1e-6),
        c=max(c, 1e-6),
        alpha=min(max(alpha, 1e-3), 179.9),
        beta=min(max(beta, 1e-3), 179.9),
        gamma=min(max(gamma, 1e-3), 179.9),
        confidence=max(confidence, 0.0),
        bravais_key=bravais_key,
    )


def _hypothesis_to_candidate(hypothesis: BravaisHypothesis) -> LatticeCandidate:
    label = hypothesis.crystal_system_label or "unknown"
    return _params_array_to_candidate(
        list(hypothesis.snapped_params),
        crystal_system=label,
        confidence=hypothesis.confidence,
        bravais_key=hypothesis.bravais_key,
    )


def scale_lattice_lengths(params: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale a,b,c by ``factor`` while keeping angles unchanged."""
    scaled = params.clone()
    scaled[..., :3] = scaled[..., :3] * factor
    return scaled


def resolve_length_scale_factors(scale_set: str) -> tuple[float, ...]:
    """Resolve a named scale set (``none`` / ``default`` / ``extended``)."""
    key = scale_set.strip().lower()
    if key not in SCALE_SET_REGISTRY:
        raise ValueError(
            f"Unknown scale set {scale_set!r}; expected one of {sorted(SCALE_SET_REGISTRY)}"
        )
    return SCALE_SET_REGISTRY[key]


def parse_length_scale_factors(spec: str | Sequence[float] | None) -> tuple[float, ...]:
    """Parse CLI scale spec: named set, or comma-separated floats."""
    if spec is None:
        return LENGTH_SCALE_FACTORS
    if isinstance(spec, (list, tuple)):
        return tuple(float(x) for x in spec)
    text = str(spec).strip()
    if not text:
        return ()
    if text.lower() in SCALE_SET_REGISTRY:
        return resolve_length_scale_factors(text)
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def candidate_volume(candidate: LatticeCandidate) -> float:
    """Absolute unit-cell volume of a candidate."""
    params = torch.tensor(
        [candidate.a, candidate.b, candidate.c, candidate.alpha, candidate.beta, candidate.gamma],
        dtype=torch.float64,
    )
    matrix = lattice_params_to_matrix(params)
    return float(abs(torch.linalg.det(matrix).item()))


def lattice_params_volume(params: Sequence[float] | torch.Tensor | np.ndarray) -> float:
    """Absolute unit-cell volume from six lattice parameters."""
    if torch.is_tensor(params):
        values = params.detach().cpu().to(dtype=torch.float64).reshape(6)
    else:
        values = torch.tensor(np.asarray(params, dtype=np.float64).reshape(6), dtype=torch.float64)
    matrix = lattice_params_to_matrix(values)
    return float(abs(torch.linalg.det(matrix).item()))


def filter_candidates_by_volume_vs_base(
    candidates: list[LatticeCandidate],
    *,
    base_volume: float,
    max_log_volume_ratio: float,
) -> list[LatticeCandidate]:
    """Keep candidates with ``|log(V / V_base)| <= max_log_volume_ratio``."""
    if base_volume < 1e-12:
        return candidates
    kept: list[LatticeCandidate] = []
    for candidate in candidates:
        volume = candidate_volume(candidate)
        if volume < 1e-12:
            continue
        if abs(math.log(volume / base_volume)) <= max_log_volume_ratio:
            kept.append(candidate)
    return kept


def dedupe_candidates(
    candidates: list[LatticeCandidate],
    *,
    length_tol: float = 0.05,
    angle_tol: float = 2.0,
) -> list[LatticeCandidate]:
    """Drop near-duplicate candidates while preserving highest-confidence entry."""
    kept: list[LatticeCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in kept:
            length_close = (
                abs(existing.a - candidate.a) <= length_tol
                and abs(existing.b - candidate.b) <= length_tol
                and abs(existing.c - candidate.c) <= length_tol
            )
            angle_close = (
                abs(existing.alpha - candidate.alpha) <= angle_tol
                and abs(existing.beta - candidate.beta) <= angle_tol
                and abs(existing.gamma - candidate.gamma) <= angle_tol
            )
            if length_close and angle_close:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def build_top_k_candidates(
    lattice_params: torch.Tensor,
    *,
    k: int = TOP_K_DEFAULT,
    config: TopKConfig | None = None,
) -> list[list[LatticeCandidate]]:
    """Build Top-K candidates from Bravais snap hypotheses + lattice variants.

    Variants are generated fully (no early ``len >= k`` cut), then optionally
    volume-filtered vs the raw prediction, deduped, sorted, and truncated to ``k``.
    """
    cfg = config or TopKConfig(k=k)
    batch_size = lattice_params.shape[0]
    results: list[list[LatticeCandidate]] = []

    for batch_idx in range(batch_size):
        lattice = lattice_params[batch_idx]
        base_volume = lattice_params_volume(lattice)
        hypotheses = generate_bravais_hypotheses(
            lattice,
            identity_penalty_score=cfg.identity_penalty_score,
            bravais_set=cfg.bravais_set,
        )
        candidates: list[LatticeCandidate] = [
            _hypothesis_to_candidate(hypothesis) for hypothesis in hypotheses
        ]

        top_hypotheses = hypotheses[: cfg.top_hypotheses_for_variants]
        for hypothesis in top_hypotheses:
            base_conf = hypothesis.confidence
            snapped_tensor = torch.tensor(
                hypothesis.snapped_params,
                dtype=lattice.dtype,
                device=lattice.device,
            )
            for factor in cfg.length_scale_factors:
                scaled = scale_lattice_lengths(snapped_tensor, factor)
                candidates.append(
                    _params_array_to_candidate(
                        scaled,
                        crystal_system=hypothesis.crystal_system_label or "unknown",
                        confidence=base_conf * cfg.variant_confidence_discount,
                        bravais_key=f"{hypothesis.bravais_key}:scale={factor:.4g}",
                    )
                )

            if cfg.include_axis_scale_variants:
                axis_factors = (2.0, 0.5)
                for axis in range(3):
                    for factor in axis_factors:
                        variant = snapped_tensor.clone()
                        variant[axis] = variant[axis] * factor
                        candidates.append(
                            _params_array_to_candidate(
                                variant,
                                crystal_system=hypothesis.crystal_system_label or "unknown",
                                confidence=base_conf * cfg.variant_confidence_discount * 0.8,
                                bravais_key=f"{hypothesis.bravais_key}:axis{axis}={factor:.4g}",
                            )
                        )

        if cfg.max_log_volume_ratio_vs_base is not None:
            candidates = filter_candidates_by_volume_vs_base(
                candidates,
                base_volume=base_volume,
                max_log_volume_ratio=cfg.max_log_volume_ratio_vs_base,
            )

        candidates = dedupe_candidates(
            candidates,
            length_tol=cfg.dedupe_length_tol,
            angle_tol=cfg.dedupe_angle_tol,
        )
        candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)

        if not candidates:
            # Degenerate guard: keep raw lattice so downstream never sees an empty pool.
            candidates = [
                _params_array_to_candidate(
                    lattice,
                    crystal_system="unknown",
                    confidence=0.0,
                    bravais_key="identity",
                )
            ]

        perturb_idx = 1
        while len(candidates) < cfg.k:
            base = candidates[0]
            jitter = 0.02 * perturb_idx
            candidates.append(
                _params_array_to_candidate(
                    [
                        base.a * (1.0 + jitter),
                        base.b * (1.0 + jitter),
                        base.c * (1.0 + jitter),
                        base.alpha,
                        base.beta,
                        base.gamma,
                    ],
                    crystal_system=base.crystal_system,
                    confidence=base.confidence * (0.2 / perturb_idx),
                    bravais_key=base.bravais_key,
                )
            )
            perturb_idx += 1

        results.append(candidates[: cfg.k])

    return results


def build_multi_anchor_top_k_candidates(
    lattice_params_multi: torch.Tensor,
    *,
    k: int = TOP_K_DEFAULT,
    config: TopKConfig | None = None,
    per_anchor_k: int | None = None,
) -> list[list[LatticeCandidate]]:
    """Merge Top-K pools from multiple raw lattice anchors per sample.

    Args:
        lattice_params_multi: physical lattices with shape ``[B, A, 6]`` (A anchors).
        k: final pool size after merge + dedupe.
        config: shared Top-K config applied to each anchor.
        per_anchor_k: optional per-anchor truncate before merge (default = ``k``).

    Does not change ``build_top_k_candidates`` (single-anchor) API; this is the
    R5-A wrapper for ``lattice_norm_all`` / multi-head anchors.
    """
    if lattice_params_multi.ndim != 3 or lattice_params_multi.shape[-1] != 6:
        raise ValueError(
            f"expected lattice_params_multi [B, A, 6], got {tuple(lattice_params_multi.shape)}"
        )
    cfg = config or TopKConfig(k=k)
    batch_size, n_anchors, _ = lattice_params_multi.shape
    if n_anchors == 0:
        raise ValueError("need at least one anchor")
    anchor_k = per_anchor_k if per_anchor_k is not None else cfg.k
    anchor_cfg = TopKConfig(
        k=anchor_k,
        length_scale_factors=cfg.length_scale_factors,
        variant_confidence_discount=cfg.variant_confidence_discount,
        identity_penalty_score=cfg.identity_penalty_score,
        mc_dropout_samples=0,
        dedupe_length_tol=cfg.dedupe_length_tol,
        dedupe_angle_tol=cfg.dedupe_angle_tol,
        top_hypotheses_for_variants=cfg.top_hypotheses_for_variants,
        max_log_volume_ratio_vs_base=cfg.max_log_volume_ratio_vs_base,
        include_axis_scale_variants=cfg.include_axis_scale_variants,
        bravais_set=cfg.bravais_set,
    )

    merged: list[list[LatticeCandidate]] = [[] for _ in range(batch_size)]
    for anchor_idx in range(n_anchors):
        pools = build_top_k_candidates(
            lattice_params_multi[:, anchor_idx, :],
            k=anchor_k,
            config=anchor_cfg,
        )
        for batch_idx, pool in enumerate(pools):
            # Tag bravais_key so merged pools remain traceable.
            for cand in pool:
                key = cand.bravais_key or "unknown"
                merged[batch_idx].append(
                    LatticeCandidate(
                        crystal_system=cand.crystal_system,
                        a=cand.a,
                        b=cand.b,
                        c=cand.c,
                        alpha=cand.alpha,
                        beta=cand.beta,
                        gamma=cand.gamma,
                        confidence=cand.confidence,
                        bravais_key=f"a{anchor_idx}:{key}",
                    )
                )

    final: list[list[LatticeCandidate]] = []
    for sample_candidates in merged:
        deduped = dedupe_candidates(
            sample_candidates,
            length_tol=cfg.dedupe_length_tol,
            angle_tol=cfg.dedupe_angle_tol,
        )
        deduped = sorted(deduped, key=lambda item: item.confidence, reverse=True)
        if not deduped:
            raise RuntimeError("multi-anchor merge produced empty pool")
        final.append(deduped[: cfg.k])
    return final


def build_top_k_with_mc_dropout(
    model: IndexingModel,
    pxrd_x: torch.Tensor,
    pxrd_y: torch.Tensor,
    peak_num: torch.Tensor,
    *,
    normalizer: Any,
    config: TopKConfig | None = None,
) -> list[list[LatticeCandidate]]:
    """Optional MC-Dropout augmentation merged into Top-K pool."""
    cfg = config or TopKConfig()
    model.eval()
    dropout_modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    for module in dropout_modules:
        module.train()

    merged_candidates: list[list[LatticeCandidate]] = []
    with torch.no_grad():
        base_outputs = model(pxrd_x, pxrd_y, peak_num)
        base_lattice = normalizer.denormalize(base_outputs["lattice_norm"])
        merged_candidates = build_top_k_candidates(base_lattice, config=cfg)

        for _ in range(cfg.mc_dropout_samples):
            outputs = model(pxrd_x, pxrd_y, peak_num)
            lattice = normalizer.denormalize(outputs["lattice_norm"])
            sampled = build_top_k_candidates(
                lattice,
                config=TopKConfig(
                    k=min(5, cfg.k),
                    length_scale_factors=cfg.length_scale_factors,
                    variant_confidence_discount=cfg.variant_confidence_discount,
                    identity_penalty_score=cfg.identity_penalty_score,
                    dedupe_length_tol=cfg.dedupe_length_tol,
                    dedupe_angle_tol=cfg.dedupe_angle_tol,
                    top_hypotheses_for_variants=cfg.top_hypotheses_for_variants,
                    max_log_volume_ratio_vs_base=cfg.max_log_volume_ratio_vs_base,
                    include_axis_scale_variants=cfg.include_axis_scale_variants,
                    bravais_set=cfg.bravais_set,
                ),
            )
            for batch_idx, sample_candidates in enumerate(sampled):
                merged_candidates[batch_idx].extend(sample_candidates)

    for module in dropout_modules:
        module.eval()

    final_results: list[list[LatticeCandidate]] = []
    for sample_candidates in merged_candidates:
        deduped = dedupe_candidates(sample_candidates)
        deduped = sorted(deduped, key=lambda item: item.confidence, reverse=True)
        final_results.append(deduped[: cfg.k])
    return final_results
