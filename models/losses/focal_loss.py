"""Sigmoid Focal Loss (Lin et al., 2017)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute sigmoid focal loss.

    Args:
        inputs:    (N,) raw logits
        targets:   (N,) binary labels {0, 1}
        alpha:     weight for positives (higher = сильнее штрафуем пропуск дерева)
        gamma:     focusing exponent (higher = сильнее штрафуем уверенные ошибки)
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        loss scalar or tensor
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
