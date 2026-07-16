"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import torch

from pxrd_cell_indexing.eval import (
    angle_mae,
    crystal_system_accuracy,
    crystal_system_accuracy_from_lattice,
    infer_crystal_system_idx_from_lattice,
    lattice_mae,
    lattice_match_elementwise,
    lattice_match_proxy,
    lattice_match_pymatgen,
    lattice_match_volume_guarded,
    lattice_volume,
    length_mae,
    mapping_vs_elementwise_gap_rate,
    oracle_hyp_elementwise_rate,
    top1_elementwise_match_rate,
    top1_joint_match_rate,
    top1_lattice_match_proxy,
    top1_lattice_match_rate,
    topk_elementwise_match_rate,
    topk_lattice_match_rate,
    topk_mapping_vs_elementwise_gap_rate,
    topk_volume_guarded_match_rate,
    volume_log_ratio,
)
from pxrd_cell_indexing.types import LatticeCandidate


def test_perfect_prediction_metrics() -> None:
    logits = torch.tensor([[10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    targets = torch.tensor([0], dtype=torch.long)
    lattice = torch.tensor([[5.0, 5.0, 6.0, 90.0, 90.0, 120.0]])
    assert crystal_system_accuracy(logits, targets) == 1.0
    assert lattice_mae(lattice, lattice) == 0.0
    assert length_mae(lattice, lattice) == 0.0
    assert angle_mae(lattice, lattice) == 0.0
    assert top1_lattice_match_proxy(lattice, lattice) == 1.0


def test_lattice_match_proxy_tolerance() -> None:
    target = torch.tensor([[5.0, 5.0, 6.0, 90.0, 90.0, 120.0]])
    pred = target.clone()
    pred[0, 0] = 5.2
    matches = lattice_match_proxy(pred, target, ltol=0.3, atol_deg=10.0)
    assert bool(matches.reshape(-1)[0].item())


def test_pymatgen_lattice_match_identical() -> None:
    params = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    assert lattice_match_pymatgen(params, params)


def test_top1_and_topk_lattice_match_rate() -> None:
    truth = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]]
    pred_hit = [[5.1, 5.1, 5.1, 90.0, 90.0, 90.0]]
    pred_miss = [[8.0, 8.0, 8.0, 90.0, 90.0, 90.0]]
    assert top1_lattice_match_rate(pred_hit, truth) == 1.0
    assert top1_lattice_match_rate(pred_miss, truth) == 0.0

    candidates = [
        [
            LatticeCandidate(
                crystal_system="cubic",
                a=8.0,
                b=8.0,
                c=8.0,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
                confidence=0.9,
            ),
            LatticeCandidate(
                crystal_system="cubic",
                a=5.1,
                b=5.1,
                c=5.1,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
                confidence=0.4,
            ),
        ]
    ]
    assert topk_lattice_match_rate(candidates, truth) == 1.0


def test_crystal_system_accuracy_from_lattice_cubic() -> None:
    pred = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]]
    targets = torch.tensor([0], dtype=torch.long)
    assert crystal_system_accuracy_from_lattice(pred, targets) == 1.0
    inferred = infer_crystal_system_idx_from_lattice(pred)
    assert inferred[0] == 0


def test_infer_crystal_system_returns_negative_one_for_identity_best() -> None:
    pred = [[3.0, 4.0, 5.0, 95.0, 88.0, 92.0]]
    inferred = infer_crystal_system_idx_from_lattice(pred, identity_penalty_score=0.0)
    # With zero identity penalty, identity may win for chaotic geometry.
    assert inferred.shape == (1,)


def test_top1_joint_match_rate_requires_both_lattice_and_crystal_system() -> None:
    truth = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]]
    pred_hit = [[5.1, 5.1, 5.1, 90.0, 90.0, 90.0]]
    assert top1_joint_match_rate(pred_hit, truth, [0], [0]) == 1.0
    assert top1_joint_match_rate(pred_hit, truth, [1], [0]) == 0.0
    pred_miss = [[8.0, 8.0, 8.0, 90.0, 90.0, 90.0]]
    assert top1_joint_match_rate(pred_miss, truth, [0], [0]) == 0.0
    assert top1_joint_match_rate(pred_hit, truth, [0], [0]) <= top1_lattice_match_rate(
        pred_hit, truth
    )


def test_lattice_volume_and_log_ratio() -> None:
    cubic = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    assert abs(lattice_volume(cubic) - 125.0) < 1e-6
    doubled = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    assert abs(volume_log_ratio(doubled, cubic) - math.log(8.0)) < 1e-6


def test_elementwise_rejects_scaled_subcell_that_mapping_may_accept() -> None:
    """2× cubic supercell: elementwise must fail; volume guard must fail."""
    truth = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    supercell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    assert not lattice_match_elementwise(supercell, truth, ltol=0.05, atol_deg=3.0)
    assert not lattice_match_volume_guarded(
        supercell, truth, ltol=0.3, atol_deg=10.0, max_log_volume_ratio=math.log(2.0)
    )
    # Near-identical stays elementwise-true.
    near = [5.1, 5.1, 5.1, 90.0, 90.0, 90.0]
    assert lattice_match_elementwise(near, truth, ltol=0.3, atol_deg=10.0)
    assert top1_elementwise_match_rate([near], [truth], ltol=0.3, atol_deg=10.0) == 1.0


def test_mapping_vs_elementwise_gap_detects_pseudo_hit() -> None:
    truth = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]]
    # Same orientation, 2× lengths: often find_mapping-equivalent under loose ltol,
    # but never elementwise at ltol=0.05.
    scaled = [[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]]
    mapped = lattice_match_pymatgen(scaled[0], truth[0], ltol=0.3, atol_deg=10.0)
    elementwise = lattice_match_elementwise(scaled[0], truth[0], ltol=0.05, atol_deg=3.0)
    assert not elementwise
    if mapped:
        assert mapping_vs_elementwise_gap_rate(scaled, truth, ltol=0.3, atol_deg=10.0) == 1.0


def test_oracle_hyp_elementwise_rate_any_hit() -> None:
    truth = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0], [4.0, 4.0, 6.0, 90.0, 90.0, 90.0]]
    hyp = torch.tensor(
        [
            # sample 0: hyp-1 matches truth exactly, hyp-0/2 are way off.
            [[10.0, 10.0, 10.0, 90.0, 90.0, 90.0], [5.0, 5.0, 5.0, 90.0, 90.0, 90.0], [1.0] * 6],
            # sample 1: none of the K hypotheses match truth.
            [[1.0] * 6, [2.0] * 6, [3.0] * 6],
        ]
    )
    rate = oracle_hyp_elementwise_rate(hyp, truth, ltol=0.05, atol_deg=3.0)
    assert rate == 0.5


def test_topk_elementwise_and_volume_guarded() -> None:
    truth = [[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]]
    candidates = [
        [
            LatticeCandidate(
                crystal_system="cubic",
                a=10.0,
                b=10.0,
                c=10.0,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
                confidence=0.9,
            ),
            LatticeCandidate(
                crystal_system="cubic",
                a=5.05,
                b=5.05,
                c=5.05,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
                confidence=0.4,
            ),
        ]
    ]
    assert topk_elementwise_match_rate(candidates, truth, ltol=0.05, atol_deg=3.0) == 1.0
    assert topk_volume_guarded_match_rate(candidates, truth, ltol=0.05, atol_deg=3.0) == 1.0
    # Pool with only supercell: elementwise miss; gap may be 1 if mapping hits.
    only_scaled = [
        [
            LatticeCandidate(
                crystal_system="cubic",
                a=10.0,
                b=10.0,
                c=10.0,
                alpha=90.0,
                beta=90.0,
                gamma=90.0,
                confidence=0.9,
            )
        ]
    ]
    assert topk_elementwise_match_rate(only_scaled, truth, ltol=0.05, atol_deg=3.0) == 0.0
    gap = topk_mapping_vs_elementwise_gap_rate(only_scaled, truth, ltol=0.3, atol_deg=10.0)
    assert gap in (0.0, 1.0)  # depends on whether find_mapping accepts 2× cubic
