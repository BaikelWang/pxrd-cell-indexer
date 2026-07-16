"""Tests for Top-K candidate generation."""

from __future__ import annotations

import math

import torch

from pxrd_cell_indexing.model.topk import (
    TopKConfig,
    build_multi_anchor_top_k_candidates,
    build_top_k_candidates,
    candidate_volume,
    dedupe_candidates,
    filter_candidates_by_volume_vs_base,
    parse_length_scale_factors,
    resolve_length_scale_factors,
    scale_lattice_lengths,
)
from pxrd_cell_indexing.types import TOP_K_DEFAULT, LatticeCandidate


def test_build_top_k_returns_k_candidates() -> None:
    lattice = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    results = build_top_k_candidates(lattice, k=TOP_K_DEFAULT)
    assert len(results) == 1
    assert len(results[0]) == TOP_K_DEFAULT
    assert results[0][0].crystal_system == "cubic"
    assert results[0][0].bravais_key == "cubic_P"
    assert results[0][0].confidence >= results[0][1].confidence


def test_scale_lattice_lengths_only_affects_lengths() -> None:
    lattice = torch.tensor([4.0, 5.0, 6.0, 90.0, 100.0, 120.0])
    scaled = scale_lattice_lengths(lattice, 2.0)
    assert scaled[:3].tolist() == [8.0, 10.0, 12.0]
    assert scaled[3:].tolist() == [90.0, 100.0, 120.0]


def test_dedupe_candidates_removes_near_duplicates() -> None:
    candidates = [
        LatticeCandidate(
            crystal_system="cubic",
            a=5.0,
            b=5.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            confidence=0.9,
            bravais_key="cubic_P",
        ),
        LatticeCandidate(
            crystal_system="cubic",
            a=5.01,
            b=5.01,
            c=5.01,
            alpha=90.1,
            beta=90.1,
            gamma=90.1,
            confidence=0.5,
            bravais_key="cubic_P",
        ),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9


def test_parse_and_resolve_scale_sets() -> None:
    assert resolve_length_scale_factors("none") == ()
    assert len(resolve_length_scale_factors("default")) == 6
    assert len(resolve_length_scale_factors("extended")) == 10
    assert parse_length_scale_factors("2,0.5") == (2.0, 0.5)


def test_volume_filter_rejects_isotropic_double() -> None:
    base = LatticeCandidate(
        crystal_system="cubic",
        a=5.0,
        b=5.0,
        c=5.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=1.0,
    )
    doubled = LatticeCandidate(
        crystal_system="cubic",
        a=10.0,
        b=10.0,
        c=10.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        confidence=0.5,
    )
    kept = filter_candidates_by_volume_vs_base(
        [base, doubled],
        base_volume=candidate_volume(base),
        max_log_volume_ratio=math.log(2.0),
    )
    assert len(kept) == 1
    assert kept[0].a == 5.0


def test_build_top_k_volume_guard_drops_large_scales() -> None:
    lattice = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    # Without guard, ×2 isotropic (V×8) can appear among scale variants.
    unguarded = build_top_k_candidates(
        lattice,
        k=40,
        config=TopKConfig(
            k=40,
            length_scale_factors=(2.0,),
            include_axis_scale_variants=False,
            max_log_volume_ratio_vs_base=None,
        ),
    )[0]
    assert any(abs(c.a - 10.0) < 1e-6 for c in unguarded)

    guarded = build_top_k_candidates(
        lattice,
        k=40,
        config=TopKConfig(
            k=40,
            length_scale_factors=(2.0,),
            include_axis_scale_variants=False,
            max_log_volume_ratio_vs_base=math.log(2.0),
        ),
    )[0]
    assert all(abs(c.a - 10.0) >= 1e-6 for c in guarded)


def test_multi_anchor_merges_and_truncates() -> None:
    # Two different anchors (cubic-ish and tetragonal-ish).
    anchors = torch.tensor(
        [
            [
                [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                [4.0, 4.0, 6.0, 90.0, 90.0, 90.0],
            ]
        ]
    )
    pools = build_multi_anchor_top_k_candidates(
        anchors,
        k=20,
        config=TopKConfig(
            k=20,
            length_scale_factors=(),
            include_axis_scale_variants=False,
        ),
        per_anchor_k=10,
    )
    assert len(pools) == 1
    assert len(pools[0]) == 20
    # Anchors tagged in bravais_key.
    assert any((c.bravais_key or "").startswith("a0:") for c in pools[0])
    assert any((c.bravais_key or "").startswith("a1:") for c in pools[0])
