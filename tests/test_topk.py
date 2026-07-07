"""Tests for Top-K candidate generation."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.model.topk import (
    build_top_k_candidates,
    dedupe_candidates,
    scale_lattice_lengths,
)
from pxrd_cell_indexing.types import TOP_K_DEFAULT, LatticeCandidate


def test_build_top_k_returns_k_candidates() -> None:
    logits = torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]])
    lattice = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    results = build_top_k_candidates(logits, lattice, k=TOP_K_DEFAULT)
    assert len(results) == 1
    assert len(results[0]) == TOP_K_DEFAULT
    assert results[0][0].crystal_system == "cubic"
    assert results[0][0].confidence > results[0][1].confidence


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
        ),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9
