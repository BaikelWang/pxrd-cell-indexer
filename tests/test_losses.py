"""Tests for indexing loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pxrd_cell_indexing.data.normalization import MatrixLatticeNormalizer
from pxrd_cell_indexing.losses import (
    IndexingLoss,
    LossWeights,
    _length_angle_physical_loss,
    compute_best_metric_score,
    manifold_consistency_loss,
    mcl_min_loss,
)


def test_indexing_loss_finite_positive() -> None:
    loss_fn = IndexingLoss()
    pred = torch.randn(4, 6)
    target = torch.randn(4, 6)
    out = loss_fn(pred, target)
    assert torch.isfinite(out["loss_total"]).all()
    assert out["loss_total"].item() > 0


def test_regression_weight_applied() -> None:
    loss_fn = IndexingLoss(LossWeights(regression=2.0))
    pred = torch.ones(1, 6)
    target = torch.zeros(1, 6)
    out = loss_fn(pred, target)
    expected = 2.0 * out["loss_reg"]
    assert torch.allclose(out["loss_total"], expected)


def test_length_angle_loss_requires_normalizer() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(LossWeights(mode="length_angle"), normalizer=stats)
    pred = torch.zeros(2, 6)
    target_norm = torch.zeros(2, 6)
    target_phys = torch.tensor(
        [
            [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
            [4.0, 4.0, 6.0, 90.0, 90.0, 120.0],
        ]
    )
    cs_idx = torch.tensor([0, 1], dtype=torch.long)
    out = loss_fn(
        pred,
        target_norm,
        lattice_phys_target=target_phys,
        crystal_system_idx=cs_idx,
    )
    assert torch.isfinite(out["loss_total"]).all()
    assert out["loss_phys"].item() >= 0.0


def test_cs_reweight_changes_loss() -> None:
    loss_fn = IndexingLoss(LossWeights(mode="cs_reweight"))
    pred = torch.ones(2, 6)
    target = torch.zeros(2, 6)
    cs_easy = torch.tensor([0, 0], dtype=torch.long)
    cs_hard = torch.tensor([3, 3], dtype=torch.long)
    easy = loss_fn(pred, target, crystal_system_idx=cs_easy)["loss_total"]
    hard = loss_fn(pred, target, crystal_system_idx=cs_hard)["loss_total"]
    assert hard > easy


def test_masked_physical_loss_normalizes_by_active_dims() -> None:
    pred = torch.tensor([[2.0, 2.0, 2.0, 100.0, 100.0, 100.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0, 1.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    got = _length_angle_physical_loss(
        pred,
        target,
        param_mask=mask,
        sample_weight=None,
        length_weight=0.0,
        angle_weight=1.0,
        huber_delta=5.0,
    )

    angle_loss = F.huber_loss(pred[..., 3:], target[..., 3:], reduction="none", delta=5.0)
    expected = ((angle_loss * mask[..., 3:]).sum(dim=-1) / mask[..., 3:].sum(dim=-1)).mean()
    assert torch.allclose(got, expected)


def test_compute_best_metric_composite_uses_lattice_proxy() -> None:
    metrics = {
        "top1_lattice_match_proxy": 0.2,
        "top1_lattice_match_rate": 0.35,
    }
    assert compute_best_metric_score(metrics, best_metric="composite") == pytest.approx(0.275)


def test_strict_hinge_zero_inside_tolerance() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(
        LossWeights(mode="strict_hinge", regression=0.0, physical_weight=1.0),
        normalizer=stats,
    )
    target_phys = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    # Within 0.05 / 3° → hinge phys term should be ~0
    pred_phys = torch.tensor([[5.1, 5.1, 5.1, 92.0, 91.0, 90.5]])
    pred_norm = stats.normalize(pred_phys)
    target_norm = stats.normalize(target_phys)
    out = loss_fn(pred_norm, target_norm, lattice_phys_target=target_phys)
    assert out["loss_phys"].item() == pytest.approx(0.0, abs=1e-6)


def test_strict_hinge_penalizes_beyond_tolerance() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(
        LossWeights(mode="strict_hinge", regression=0.0, physical_weight=1.0),
        normalizer=stats,
    )
    target_phys = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    pred_phys = torch.tensor([[6.0, 5.0, 5.0, 100.0, 90.0, 90.0]])  # 20% length, 10° angle
    pred_norm = stats.normalize(pred_phys)
    target_norm = stats.normalize(target_phys)
    out = loss_fn(pred_norm, target_norm, lattice_phys_target=target_phys)
    assert out["loss_phys"].item() > 0.0


def test_angle_heavy_uses_elevated_angle_weight() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    target_phys = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    pred_phys = torch.tensor([[5.0, 5.0, 5.0, 100.0, 90.0, 90.0]])  # angle-only error
    pred_norm = stats.normalize(pred_phys)
    target_norm = stats.normalize(target_phys)
    light = IndexingLoss(
        LossWeights(mode="length_angle", angle_weight=1.0, length_weight=0.0),
        normalizer=stats,
    )(pred_norm, target_norm, lattice_phys_target=target_phys)["loss_total"]
    heavy = IndexingLoss(
        LossWeights(mode="angle_heavy", angle_weight=1.0, length_weight=0.0),
        normalizer=stats,
    )(pred_norm, target_norm, lattice_phys_target=target_phys)["loss_total"]
    assert heavy.item() > light.item()


def test_compute_best_metric_strict_composite() -> None:
    metrics = {
        "strict_raw_top1_lattice_match_rate": 0.1,
        "strict_raw_top1_elementwise_rate": 0.05,
    }
    assert compute_best_metric_score(metrics, best_metric="strict_composite") == pytest.approx(
        0.075
    )


def test_setting_classification_loss_on_cubic_only() -> None:
    loss_fn = IndexingLoss(LossWeights(setting_classification=1.0, regression=0.0))
    pred = torch.zeros(3, 6)
    target = torch.zeros(3, 6)
    phys = torch.tensor(
        [
            [4.0, 4.0, 4.0, 90.0, 90.0, 90.0],
            [4.0, 4.0, 4.0, 60.0, 60.0, 60.0],
            [5.0, 5.0, 6.0, 90.0, 90.0, 90.0],  # non-cubic
        ]
    )
    cs = torch.tensor([0, 0, 1], dtype=torch.long)
    # Perfect logits for first two cubic settings (0=90, 1=60)
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    out = loss_fn(
        pred,
        target,
        lattice_phys_target=phys,
        crystal_system_idx=cs,
        cubic_setting_logits=logits,
    )
    assert out["loss_setting"].item() < 0.05
    assert out["loss_total"].item() < 0.05


def test_manifold_consistency_zero_when_angles_already_fixed() -> None:
    cs = torch.tensor([1, 2, 3, 6], dtype=torch.long)  # tet, ortho, hex, triclinic
    phys = torch.tensor(
        [
            [4.0, 4.0, 5.0, 90.0, 90.0, 90.0],  # tetragonal: alpha,beta already 90
            [4.0, 5.0, 6.0, 90.0, 90.0, 75.0],  # orthorhombic: alpha,beta already 90
            [4.0, 4.0, 6.0, 90.0, 90.0, 120.0],  # hexagonal: beta already 90
            [4.0, 5.0, 6.0, 70.0, 80.0, 95.0],  # triclinic: untouched, not in table
        ]
    )
    loss = manifold_consistency_loss(phys, cs)
    assert loss.item() < 1e-4


def test_manifold_consistency_penalizes_deviation() -> None:
    cs = torch.tensor([1], dtype=torch.long)  # tetragonal
    bad = torch.tensor([[4.0, 4.0, 5.0, 95.0, 88.0, 90.0]])
    good = torch.tensor([[4.0, 4.0, 5.0, 90.0, 90.0, 90.0]])
    bad_loss = manifold_consistency_loss(bad, cs)
    good_loss = manifold_consistency_loss(good, cs)
    assert bad_loss.item() > good_loss.item()


def test_manifold_consistency_loss_mode_end_to_end() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(
        LossWeights(mode="manifold_consistency", manifold_consistency_weight=0.2),
        normalizer=stats,
    )
    pred = torch.zeros(2, 6)
    target_norm = torch.zeros(2, 6)
    target_phys = torch.tensor(
        [
            [4.0, 4.0, 5.0, 90.0, 90.0, 90.0],
            [4.0, 5.0, 6.0, 90.0, 90.0, 75.0],
        ]
    )
    cs_idx = torch.tensor([1, 2], dtype=torch.long)
    out = loss_fn(pred, target_norm, lattice_phys_target=target_phys, crystal_system_idx=cs_idx)
    assert torch.isfinite(out["loss_total"]).all()


def test_peak_consistency_near_zero_on_truth_lattice() -> None:
    from pxrd_cell_indexing.model.fom import theoretical_two_theta
    from pxrd_cell_indexing.losses import peak_consistency_loss

    truth = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    observed = theoretical_two_theta(truth[0].tolist())[:15]
    pxrd_x = torch.tensor(observed, dtype=torch.float32).view(-1, 1)
    peak_num = torch.tensor([observed.size], dtype=torch.long)
    loss = peak_consistency_loss(truth, pxrd_x, peak_num, scale=1.0, n_lines=12)
    assert loss.item() < 1e-6


def test_peak_consistency_penalizes_wrong_lattice() -> None:
    from pxrd_cell_indexing.model.fom import theoretical_two_theta
    from pxrd_cell_indexing.losses import peak_consistency_loss

    truth = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
    observed = theoretical_two_theta(truth)[:15]
    pxrd_x = torch.tensor(observed, dtype=torch.float32).view(-1, 1)
    peak_num = torch.tensor([observed.size], dtype=torch.long)
    good = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    bad = torch.tensor([[6.5, 6.5, 6.5, 90.0, 90.0, 90.0]])
    good_loss = peak_consistency_loss(good, pxrd_x, peak_num, scale=1.0)
    bad_loss = peak_consistency_loss(bad, pxrd_x, peak_num, scale=1.0)
    assert bad_loss.item() > good_loss.item()


def test_peak_consistency_is_differentiable() -> None:
    from pxrd_cell_indexing.model.fom import theoretical_two_theta
    from pxrd_cell_indexing.losses import peak_consistency_loss

    truth = [4.0, 4.0, 6.0, 90.0, 90.0, 90.0]
    observed = theoretical_two_theta(truth)[:12]
    pxrd_x = torch.tensor(observed, dtype=torch.float32).view(-1, 1)
    peak_num = torch.tensor([observed.size], dtype=torch.long)
    pred = torch.tensor([[4.2, 4.2, 6.1, 90.0, 90.0, 90.0]], requires_grad=True)
    loss = peak_consistency_loss(pred, pxrd_x, peak_num, scale=1000.0)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0


def test_mcl_min_loss_picks_closest_hypothesis() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    hyp = torch.tensor(
        [[[5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [9.0] * 6]]
    )
    loss, winner_idx = mcl_min_loss(hyp, target)
    assert winner_idx.tolist() == [1]
    # min-over-K loss should equal SmoothL1 against the closest (index 1) hypothesis only.
    expected = F.smooth_l1_loss(hyp[:, 1, :], target)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_mcl_min_loss_lower_than_plain_smoothl1_on_ambiguous_targets() -> None:
    # Two different truths, but the *same* pair of hypotheses {1s, 9s} offered
    # for every sample (a single-point head could not fit both). MCL should
    # find near-zero loss by routing each sample to its matching hypothesis;
    # a plain single-point head (always hyp-0) cannot fit both.
    targets = torch.tensor([[1.0] * 6, [9.0] * 6])
    hyp = torch.tensor([[1.0] * 6, [9.0] * 6]).unsqueeze(0).expand(2, -1, -1)  # [B=2, K=2, D=6]
    mcl_loss, winner_idx = mcl_min_loss(hyp, targets)
    plain_loss = F.smooth_l1_loss(hyp[:, 0, :], targets)
    assert winner_idx.tolist() == [0, 1]
    assert mcl_loss.item() < plain_loss.item()
    assert mcl_loss.item() < 1e-6


def test_mcl_loss_mode_end_to_end() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(LossWeights(mode="mcl"), normalizer=stats)
    pred_primary = torch.tensor([[5.0] * 6])
    target = torch.tensor([[5.2] * 6])
    hyp = torch.tensor([[[5.0] * 6, [1.0] * 6, [9.0] * 6]])
    cs_idx = torch.tensor([3])  # non-cubic -> real K=3
    out = loss_fn(
        pred_primary,
        target,
        crystal_system_idx=cs_idx,
        lattice_hyp_pred=hyp,
    )
    assert torch.isfinite(out["loss_total"]).all()
    assert out["loss_total"].item() > 0
    # hyp-0 (5.0) is the closest to target 5.2, so mcl loss_reg should be small.
    assert out["loss_reg"].item() < F.smooth_l1_loss(pred_primary, target).item() + 1e-6


def test_mcl_loss_mode_requires_lattice_hyp_pred() -> None:
    loss_fn = IndexingLoss(LossWeights(mode="mcl"))
    pred = torch.randn(2, 6)
    target = torch.randn(2, 6)
    with pytest.raises(ValueError, match="mcl"):
        loss_fn(pred, target)


def test_physical_length_angle_diagnostics_always_reported() -> None:
    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(LossWeights(mode="baseline"), normalizer=stats)
    # Identity normalizer: norm == phys for these synthetic values.
    pred = torch.tensor([[5.2, 5.1, 5.0, 90.0, 91.0, 89.0]])
    target = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    out = loss_fn(pred, target, lattice_phys_target=target)
    assert "loss_matrix6" in out
    assert "loss_length_phys" in out
    assert "loss_angle_phys" in out
    assert out["loss_length_phys"].item() > 0
    assert out["loss_angle_phys"].item() > 0
    assert torch.allclose(out["loss_matrix6"], out["loss_reg"])


def test_peak_consistency_loss_mode_end_to_end() -> None:
    from pxrd_cell_indexing.model.fom import theoretical_two_theta

    stats = MatrixLatticeNormalizer(
        component_mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        component_std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    loss_fn = IndexingLoss(
        LossWeights(mode="peak_consistency", peak_consistency_weight=0.15),
        normalizer=stats,
    )
    # Identity normalizer: pred_norm ≈ pred_phys
    pred = torch.tensor([[5.2, 5.2, 5.2, 90.0, 90.0, 90.0]])
    target_norm = torch.tensor([[5.0, 5.0, 5.0, 90.0, 90.0, 90.0]])
    observed = theoretical_two_theta([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])[:10]
    pxrd_x = torch.tensor(observed, dtype=torch.float32).view(-1, 1)
    peak_num = torch.tensor([observed.size], dtype=torch.long)
    out = loss_fn(
        pred,
        target_norm,
        lattice_phys_target=target_norm,
        pxrd_x=pxrd_x,
        peak_num=peak_num,
    )
    assert torch.isfinite(out["loss_total"]).all()
    assert out["loss_phys"].item() > 0
    assert out["loss_phys"].item() >= 0.0

