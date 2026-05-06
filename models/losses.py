"""
models/losses.py — Loss-функции для TreeRCNN v2.0

Содержит:
  - offset_loss       : Smooth L1 по смещениям точек к GT-центру
  - centerness_loss   : BCE по принадлежности точки к GT-боксу
  - relation_loss     : BCE для финального score в Stage 3
  - stage2_loss_v2    : Объединённый loss Stage 2 (cls + reg + offset + centerness)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def offset_loss(
    pred_offsets: torch.Tensor,  # (B, N, 3) — предсказанные смещения
    points_xyz: torch.Tensor,    # (B, N, 3) — координаты точек
    gt_centers: torch.Tensor,    # (B, 3)    — GT-центры деревьев
    point_mask: torch.Tensor,    # (B, N)    — 1 если точка внутри GT-бокса
    lambda_off: float = 0.5,
) -> torch.Tensor:
    """
    Smooth L1 loss между предсказанным смещением точки и реальным
    смещением к GT-центру дерева.

    GT для каждой точки: gt_offset = gt_center - point_xyz
    Обучаем только точки внутри GT-бокса (point_mask == 1).
    """
    # GT смещения: от каждой точки до центра её дерева
    gt_offsets = gt_centers.unsqueeze(1) - points_xyz  # (B, N, 3)

    # Считаем loss только по точкам внутри бокса
    mask = point_mask.unsqueeze(-1).float()  # (B, N, 1)
    loss = F.smooth_l1_loss(pred_offsets * mask, gt_offsets * mask, reduction='sum')
    n_pos = mask.sum().clamp(min=1)
    return lambda_off * loss / n_pos


def centerness_loss(
    pred_centerness: torch.Tensor,  # (B, N, 1) — предсказанная уверенность
    point_mask: torch.Tensor,       # (B, N)    — 1 если точка внутри GT-бокса
    lambda_cent: float = 0.5,
) -> torch.Tensor:
    """
    BCE: точки внутри GT-бокса должны иметь centerness=1,
    точки снаружи — centerness=0.
    """
    gt_cent = point_mask.unsqueeze(-1).float()  # (B, N, 1)
    loss = F.binary_cross_entropy(pred_centerness, gt_cent, reduction='mean')
    return lambda_cent * loss


def relation_loss(
    pred_scores: torch.Tensor,  # (B, N, 1) — scores из Relation Head
    gt_labels: torch.Tensor,    # (B, N)    — 1 если IoU с GT > 0.5
    lambda_rel: float = 0.5,
) -> torch.Tensor:
    """
    BCE loss для Stage 3 Relation Head.
    gt_labels: 1 если кандидат является истинным деревом (IoU > 0.5 с GT).
    """
    gt = gt_labels.unsqueeze(-1).float()  # (B, N, 1)
    loss = F.binary_cross_entropy(pred_scores, gt, reduction='mean')
    return lambda_rel * loss


def stage2_loss_v2(
    cls_score: torch.Tensor,     # (B, 1)
    reg_delta: torch.Tensor,     # (B, 6)
    pred_offsets: torch.Tensor,  # (B, N, 3)
    pred_centerness: torch.Tensor,  # (B, N, 1)
    points_xyz: torch.Tensor,    # (B, N, 3)
    gt_box: torch.Tensor,        # (B, 6)  — GT бокс для позитивных proposals
    gt_label: torch.Tensor,      # (B,)    — 1=позитивный, 0=негативный
    point_mask: torch.Tensor,    # (B, N)  — маска точек внутри GT
    lambdas: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Объединённый loss Stage 2.

    Возвращает словарь с отдельными компонентами и 'total'.
    """
    if lambdas is None:
        lambdas = {
            'cls': 1.0, 'reg': 1.0,
            'offset': 0.5, 'centerness': 0.5
        }

    # Классификация (BCE)
    l_cls = F.binary_cross_entropy_with_logits(
        cls_score.squeeze(-1),
        gt_label.float(),
        reduction='mean'
    ) * lambdas['cls']

    # Регрессия боксов (Smooth L1, только позитивные proposals)
    pos_mask = gt_label.bool()
    if pos_mask.any():
        l_reg = F.smooth_l1_loss(
            reg_delta[pos_mask], gt_box[pos_mask], reduction='mean'
        ) * lambdas['reg']
    else:
        l_reg = reg_delta.sum() * 0.0

    # Offset loss
    gt_centers = gt_box[:, :3]  # cx, cy, cz
    l_off = offset_loss(
        pred_offsets, points_xyz, gt_centers, point_mask,
        lambda_off=lambdas['offset']
    )

    # Center-ness loss
    l_cent = centerness_loss(
        pred_centerness, point_mask,
        lambda_cent=lambdas['centerness']
    )

    total = l_cls + l_reg + l_off + l_cent

    return {
        'total': total,
        'cls': l_cls,
        'reg': l_reg,
        'offset': l_off,
        'centerness': l_cent,
    }
