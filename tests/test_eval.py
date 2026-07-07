"""Tests for evaluation metrics."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.eval import (
    crystal_system_accuracy,
    lattice_mae,
    lattice_match_proxy,
    lattice_match_pymatgen,
    top1_joint_match_rate,
    top1_lattice_match_proxy,
    top1_lattice_match_rate,
    topk_lattice_match_rate,
)
from pxrd_cell_indexing.types import LatticeCandidate


def test_perfect_prediction_metrics() -> None:
    logits = torch.tensor([[10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    targets = torch.tensor([0], dtype=torch.long)
    lattice = torch.tensor([[5.0, 5.0, 6.0, 90.0, 90.0, 120.0]])
    assert crystal_system_accuracy(logits, targets) == 1.0
    assert lattice_mae(lattice, lattice) == 0.0
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
