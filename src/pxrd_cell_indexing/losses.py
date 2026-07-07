"""Loss functions for crystal-system classification and lattice regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    """Task loss weights; uncertainty mode ignores fixed classification/regression."""

    classification: float = 1.0
    regression: float = 1.0
    use_uncertainty_weighting: bool = False


class IndexingLoss(nn.Module):
    """Combined CE + full-parameter SmoothL1 on normalized lattice targets."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        if self.weights.use_uncertainty_weighting:
            self.log_sigma_cls = nn.Parameter(torch.zeros(()))
            self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        else:
            self.log_sigma_cls = None
            self.log_sigma_reg = None

    def forward(
        self,
        crystal_system_logits: torch.Tensor,
        lattice_norm_pred: torch.Tensor,
        crystal_system_idx: torch.Tensor,
        lattice_norm_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_cls = F.cross_entropy(crystal_system_logits, crystal_system_idx)
        loss_reg = F.smooth_l1_loss(lattice_norm_pred, lattice_norm_target)

        if self.weights.use_uncertainty_weighting:
            assert self.log_sigma_cls is not None and self.log_sigma_reg is not None
            loss_total = uncertainty_weighted_total(
                loss_cls,
                loss_reg,
                self.log_sigma_cls,
                self.log_sigma_reg,
            )
        else:
            loss_total = (
                self.weights.classification * loss_cls
                + self.weights.regression * loss_reg
            )

        result = {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
        }
        if self.weights.use_uncertainty_weighting:
            result["log_sigma_cls"] = self.log_sigma_cls
            result["log_sigma_reg"] = self.log_sigma_reg
        return result

    def uncertainty_parameters(self) -> list[nn.Parameter]:
        if not self.weights.use_uncertainty_weighting:
            return []
        assert self.log_sigma_cls is not None and self.log_sigma_reg is not None
        return [self.log_sigma_cls, self.log_sigma_reg]


def uncertainty_weighted_total(
    classification_loss: torch.Tensor,
    regression_loss: torch.Tensor,
    log_sigma_cls: torch.Tensor,
    log_sigma_reg: torch.Tensor,
) -> torch.Tensor:
    """Kendall-style uncertainty weighting."""
    return (
        torch.exp(-log_sigma_cls) * classification_loss
        + log_sigma_cls
        + torch.exp(-log_sigma_reg) * regression_loss
        + log_sigma_reg
    )


def compute_best_metric_score(
    valid_metrics: dict[str, float],
    *,
    best_metric: str = "top1_lattice_match_proxy",
) -> float:
    """Select checkpoint score from validation metrics."""
    if best_metric == "composite":
        proxy = valid_metrics.get("top1_lattice_match_proxy", 0.0)
        cls_acc = valid_metrics.get("crystal_system_accuracy", 0.0)
        return 0.5 * proxy + 0.5 * cls_acc
    return float(valid_metrics.get(best_metric, 0.0))
