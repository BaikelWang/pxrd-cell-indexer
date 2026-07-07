"""Top-K candidate generation for D25 single-head indexing models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from pxrd_cell_indexing.types import CRYSTAL_SYSTEMS, TOP_K_DEFAULT, LatticeCandidate

if TYPE_CHECKING:
    from pxrd_cell_indexing.model.heads import IndexingModel

# Common indexing ambiguities: integer/fractional supercell and subcell scaling.
LENGTH_SCALE_FACTORS: tuple[float, ...] = (
    2.0,
    0.5,
    math.sqrt(2),
    1.0 / math.sqrt(2),
    math.sqrt(3),
    1.0 / math.sqrt(3),
)

VARIANT_CONFIDENCE_DISCOUNT = 0.5
SECONDARY_CS_DISCOUNT = 0.95


@dataclass(frozen=True)
class TopKConfig:
    k: int = TOP_K_DEFAULT
    length_scale_factors: tuple[float, ...] = LENGTH_SCALE_FACTORS
    variant_confidence_discount: float = VARIANT_CONFIDENCE_DISCOUNT
    secondary_cs_discount: float = SECONDARY_CS_DISCOUNT
    mc_dropout_samples: int = 0
    dedupe_length_tol: float = 0.05
    dedupe_angle_tol: float = 2.0


def _params_array_to_candidate(
    params: torch.Tensor | list[float],
    crystal_system: str,
    confidence: float,
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
    )


def scale_lattice_lengths(params: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale a,b,c by ``factor`` while keeping angles unchanged."""
    scaled = params.clone()
    scaled[..., :3] = scaled[..., :3] * factor
    return scaled


def _candidate_key(candidate: LatticeCandidate) -> tuple[float, ...]:
    return (
        round(candidate.a, 4),
        round(candidate.b, 4),
        round(candidate.c, 4),
        round(candidate.alpha, 2),
        round(candidate.beta, 2),
        round(candidate.gamma, 2),
        candidate.crystal_system,
    )


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
            if (
                existing.crystal_system == candidate.crystal_system
                and length_close
                and angle_close
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def build_top_k_candidates(
    crystal_system_logits: torch.Tensor,
    lattice_params: torch.Tensor,
    *,
    k: int = TOP_K_DEFAULT,
    config: TopKConfig | None = None,
) -> list[list[LatticeCandidate]]:
    """Build Top-K candidates from a single regression head (D25).

    Strategy (D26):
    1. Primary candidate: argmax crystal system + regression lattice + cls prob.
    2. Secondary crystal-system hypotheses: same lattice, other cls-ranked systems.
    3. Supercell/subcell variants of the primary lattice to fill remaining slots.
    """
    cfg = config or TopKConfig(k=k)
    probs = F.softmax(crystal_system_logits, dim=-1)
    batch_size = crystal_system_logits.shape[0]
    results: list[list[LatticeCandidate]] = []

    for batch_idx in range(batch_size):
        lattice = lattice_params[batch_idx]
        prob_row = probs[batch_idx]
        primary_idx = int(prob_row.argmax().item())
        primary_cs = CRYSTAL_SYSTEMS[primary_idx]
        primary_prob = float(prob_row[primary_idx].item())

        candidates: list[LatticeCandidate] = [
            _params_array_to_candidate(lattice, primary_cs, primary_prob)
        ]

        ranked_indices = torch.argsort(prob_row, descending=True).tolist()
        for cs_idx in ranked_indices:
            if cs_idx == primary_idx:
                continue
            candidates.append(
                _params_array_to_candidate(
                    lattice,
                    CRYSTAL_SYSTEMS[cs_idx],
                    float(prob_row[cs_idx].item()) * cfg.secondary_cs_discount,
                )
            )
            if len(candidates) >= 7:
                break

        for factor in cfg.length_scale_factors:
            if len(candidates) >= cfg.k:
                break
            scaled = scale_lattice_lengths(lattice, factor)
            candidates.append(
                _params_array_to_candidate(
                    scaled,
                    primary_cs,
                    primary_prob * cfg.variant_confidence_discount,
                )
            )

        # Fill remaining slots with axis-specific and multi-system variants.
        axis_factors = (2.0, 0.5)
        for cs_idx in ranked_indices[:3]:
            cs_name = CRYSTAL_SYSTEMS[cs_idx]
            cs_prob = float(prob_row[cs_idx].item())
            for axis in range(3):
                for factor in axis_factors:
                    if len(candidates) >= cfg.k:
                        break
                    variant = lattice.clone()
                    variant[axis] = variant[axis] * factor
                    candidates.append(
                        _params_array_to_candidate(
                            variant,
                            cs_name,
                            cs_prob * cfg.variant_confidence_discount * 0.8,
                        )
                    )
            for factor in cfg.length_scale_factors:
                if len(candidates) >= cfg.k:
                    break
                scaled = scale_lattice_lengths(lattice, factor)
                candidates.append(
                    _params_array_to_candidate(
                        scaled,
                        cs_name,
                        cs_prob * cfg.variant_confidence_discount * 0.7,
                    )
                )

        candidates = dedupe_candidates(
            candidates,
            length_tol=cfg.dedupe_length_tol,
            angle_tol=cfg.dedupe_angle_tol,
        )
        candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)

        # Ensure exactly K slots for downstream oracle metrics.
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
                    base.crystal_system,
                    base.confidence * (0.2 / perturb_idx),
                )
            )
            perturb_idx += 1

        results.append(candidates[: cfg.k])

    return results


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
        merged_candidates = build_top_k_candidates(
            base_outputs["crystal_system_logits"],
            base_lattice,
            config=cfg,
        )

        for _ in range(cfg.mc_dropout_samples):
            outputs = model(pxrd_x, pxrd_y, peak_num)
            lattice = normalizer.denormalize(outputs["lattice_norm"])
            sampled = build_top_k_candidates(
                outputs["crystal_system_logits"],
                lattice,
                config=TopKConfig(k=min(5, cfg.k)),
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
