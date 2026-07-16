"""Tests for trainer best-metric selection."""

from __future__ import annotations

import pytest

from pxrd_cell_indexing.losses import IndexingLoss, LossWeights, compute_best_metric_score


def test_compute_best_metric_proxy() -> None:
    metrics = {
        "top1_lattice_match_proxy": 0.2,
        "crystal_system_accuracy": 0.5,
    }
    assert compute_best_metric_score(metrics, best_metric="top1_lattice_match_proxy") == 0.2


def test_compute_best_metric_real_match() -> None:
    metrics = {
        "top1_lattice_match_rate": 0.35,
        "top1_lattice_match_proxy": 0.2,
    }
    assert compute_best_metric_score(metrics, best_metric="top1_lattice_match_rate") == 0.35


def test_compute_best_metric_joint_match() -> None:
    metrics = {
        "top1_joint_match_rate": 0.209,
        "top1_lattice_match_rate": 0.354,
        "crystal_system_accuracy": 0.575,
    }
    assert compute_best_metric_score(metrics, best_metric="top1_joint_match_rate") == 0.209


def test_compute_best_metric_composite() -> None:
    metrics = {
        "top1_lattice_match_proxy": 0.2,
        "top1_lattice_match_rate": 0.4,
    }
    assert compute_best_metric_score(metrics, best_metric="composite") == pytest.approx(0.3)


def test_indexing_loss_regression_only() -> None:
    loss_fn = IndexingLoss(LossWeights(regression=1.0))
    import torch

    pred = torch.randn(2, 6)
    target = torch.randn(2, 6)
    out = loss_fn(pred, target)
    assert torch.isfinite(out["loss_total"]).all()
    assert "loss_reg" in out
