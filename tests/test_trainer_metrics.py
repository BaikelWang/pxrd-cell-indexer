"""Tests for trainer best-metric selection and uncertainty loss."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.losses import (
    IndexingLoss,
    LossWeights,
    compute_best_metric_score,
)


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
        "crystal_system_accuracy": 0.6,
    }
    assert compute_best_metric_score(metrics, best_metric="composite") == 0.4


def test_uncertainty_indexing_loss_has_learnable_sigmas() -> None:
    loss_fn = IndexingLoss(LossWeights(use_uncertainty_weighting=True))
    assert len(loss_fn.uncertainty_parameters()) == 2
    logits = torch.randn(2, 7)
    pred = torch.randn(2, 6)
    target_idx = torch.tensor([0, 1], dtype=torch.long)
    target = torch.randn(2, 6)
    out = loss_fn(logits, pred, target_idx, target)
    assert torch.isfinite(out["loss_total"]).all()
    assert "log_sigma_cls" in out
    assert "log_sigma_reg" in out


def test_fixed_weights_ignore_uncertainty_flag_false() -> None:
    loss_fn = IndexingLoss(LossWeights(classification=2.0, regression=3.0))
    logits = torch.zeros(1, 7)
    logits[0, 0] = 10.0
    pred = torch.zeros(1, 6)
    target_idx = torch.tensor([0], dtype=torch.long)
    target = torch.zeros(1, 6)
    out = loss_fn(logits, pred, target_idx, target)
    expected = 2.0 * out["loss_cls"] + 3.0 * out["loss_reg"]
    assert torch.allclose(out["loss_total"], expected)
