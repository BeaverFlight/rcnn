"""
models/losses/dice_loss.py — Dice Loss для бинарной классификации

Дополняет Focal Loss в stage2_loss_v2:
  BCE/Focal работает поточечно → стабильные градиенты с первых эпох.
  Dice оптимизирует F1 напрямую → устойчив к дисбалансу pos/neg.

Формула:
    Dice = 1 - (2·Σ(p·t) + ε) / (Σp + Σt + ε)

На дисбалансированных данных (мало деревьев) Dice резко снижает
количество false negatives, что критично для recall в задаче обнаружения
деревьев (лучше лишнее дерево, чем пропустить).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(
    inputs: Tensor,
    targets: Tensor,
    smooth: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Dice Loss для бинарной классификации.

    Args:
        inputs:    (...) raw logits (до sigmoid)
        targets:   (...) бинарные метки {0, 1}, того же shape
        smooth:    сглаживающая константа (избегает деления на ноль)
        reduction: 'mean' | 'sum' | 'none' (по батчу, если inputs 2D)

    Returns:
        scalar loss
    """
    probs = torch.sigmoid(inputs)

    # Flatten по всем измерениям кроме batch
    if probs.dim() > 1:
        probs   = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        intersection = (probs * targets).sum(dim=1)          # (B,)
        denom        = probs.sum(dim=1) + targets.sum(dim=1) # (B,)
        loss = 1.0 - (2.0 * intersection + smooth) / (denom + smooth)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss
    else:
        # 1D: весь тензор как один «батч»
        intersection = (probs * targets).sum()
        denom        = probs.sum() + targets.sum()
        return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def bce_dice_loss(
    inputs: Tensor,
    targets: Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
) -> Tensor:
    """
    Комбинированный BCE + Dice Loss.

    BCE даёт стабильные поточечные градиенты на старте.
    Dice оптимизирует F1 и устойчив к дисбалансу классов.
    Соотношение 50/50 — стандарт в задачах с дисбалансом (ForestFormer3D).

    Args:
        inputs:      (...) raw logits
        targets:     (...) {0, 1} labels
        bce_weight:  вес BCE-компоненты (из cfg: training.bce_weight)
        dice_weight: вес Dice-компоненты (из cfg: training.dice_weight)
        smooth:      Dice smoothing constant

    Returns:
        scalar loss
    """
    bce  = F.binary_cross_entropy_with_logits(inputs, targets)
    dice = dice_loss(inputs, targets, smooth=smooth)
    return bce_weight * bce + dice_weight * dice
