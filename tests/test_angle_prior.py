"""Unit tests for R2 Bravais angle-prior loss."""

from __future__ import annotations

import torch

from pxrd_cell_indexing.losses import LossWeights, IndexingLoss, bravais_angle_prior_loss


def test_bravais_angle_prior_cubic_prefers_matching_hypothesis() -> None:
    # Near-60° cubic F-like angles should have low prior loss.
    pred = torch.tensor([[4.0, 4.0, 4.0, 60.0, 60.0, 60.0]], dtype=torch.float32)
    cs = torch.tensor([0], dtype=torch.long)
    loss_60 = bravais_angle_prior_loss(pred, cs)
    pred90 = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    loss_90 = bravais_angle_prior_loss(pred90, cs)
    assert float(loss_60) < 1.0
    assert float(loss_90) < 1.0
    # Far from all cubic targets should be worse.
    pred_bad = torch.tensor([[4.0, 4.0, 4.0, 75.0, 75.0, 75.0]], dtype=torch.float32)
    loss_bad = bravais_angle_prior_loss(pred_bad, cs)
    assert float(loss_bad) > float(loss_60)


def test_bravais_angle_prior_hex_only_beta() -> None:
    # Hex: only β=90 is penalized; γ can be anything without increasing prior vs β=90.
    good = torch.tensor([[5.0, 5.0, 8.0, 90.0, 90.0, 120.0]], dtype=torch.float32)
    bad_beta = torch.tensor([[5.0, 5.0, 8.0, 90.0, 100.0, 120.0]], dtype=torch.float32)
    cs = torch.tensor([3], dtype=torch.long)
    assert float(bravais_angle_prior_loss(good, cs)) < float(
        bravais_angle_prior_loss(bad_beta, cs)
    )


def test_angle_prior_mode_keeps_smooth_l1() -> None:
    stats = {
        "mean": [0.0] * 6,
        "std": [1.0] * 6,
    }
    # Minimal normalizer stub via MatrixLatticeNormalizer if constructible;
    # use a simple fake by monkeypatching denormalize.
    class _N:
        def denormalize(self, x):
            return x

    loss_fn = IndexingLoss(
        LossWeights(mode="angle_prior", regression=1.0, angle_prior_weight=0.5),
        normalizer=_N(),  # type: ignore[arg-type]
    )
    pred = torch.zeros(4, 6)
    target = torch.zeros(4, 6)
    phys = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * 4)
    cs = torch.zeros(4, dtype=torch.long)
    out = loss_fn(pred, target, lattice_phys_target=phys, crystal_system_idx=cs)
    assert torch.isfinite(out["loss_total"])
    assert "loss_phys" in out
