"""Smooth L1 (Huber) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def smooth_l1_loss(
    pred: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """
    Smooth L1 loss (a.k.a. Huber loss with delta=beta).

    Args:
        pred:      predictions (any shape)
        target:    targets (same shape as pred)
        beta:      threshold between L1 and L2 regimes
        reduction: 'none' | 'mean' | 'sum'
    """
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)
