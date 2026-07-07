"""Tests for indexing loss functions."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.losses import IndexingLoss, LossWeights, uncertainty_weighted_total


def test_indexing_loss_finite_positive() -> None:
    loss_fn = IndexingLoss()
    logits = torch.randn(4, 7)
    pred = torch.randn(4, 6)
    target_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target = torch.randn(4, 6)
    out = loss_fn(logits, pred, target_idx, target)
    assert torch.isfinite(out["loss_total"]).all()
    assert out["loss_total"].item() > 0


def test_fixed_one_to_one_weights() -> None:
    loss_fn = IndexingLoss(LossWeights(classification=1.0, regression=1.0))
    logits = torch.zeros(2, 7)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    pred = torch.zeros(2, 6)
    target_idx = torch.tensor([0, 1], dtype=torch.long)
    target = torch.zeros(2, 6)
    out = loss_fn(logits, pred, target_idx, target)
    assert out["loss_total"].item() == out["loss_cls"].item() + out["loss_reg"].item()


def test_regression_heavier_weight() -> None:
    loss_fn = IndexingLoss(LossWeights(classification=1.0, regression=2.0))
    logits = torch.zeros(1, 7)
    logits[0, 0] = 10.0
    pred = torch.ones(1, 6)
    target_idx = torch.tensor([0], dtype=torch.long)
    target = torch.zeros(1, 6)
    out = loss_fn(logits, pred, target_idx, target)
    expected = out["loss_cls"].item() + 2.0 * out["loss_reg"].item()
    assert abs(out["loss_total"].item() - expected) < 1e-5


def test_uncertainty_weighting_scaffold() -> None:
    cls_loss = torch.tensor(1.0)
    reg_loss = torch.tensor(2.0)
    log_sigma_cls = torch.tensor(0.0)
    log_sigma_reg = torch.tensor(0.0)
    total = uncertainty_weighted_total(cls_loss, reg_loss, log_sigma_cls, log_sigma_reg)
    assert torch.isfinite(total)
