"""
models/losses.py — Loss-функции для TreeRCNN v2.0

Фиксы:
  - centerness_loss: явно документировано использование F.binary_cross_entropy
    (не with_logits) т.к. centerness_head в Stage2Head уже пропускает sigmoid.
  - Добавлена функция compute_point_mask() — вычисление point_mask
    было отсутствующим звеном в предыдущей версии.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_point_mask(
    points_xyz: torch.Tensor,  # (B, N, 3) — координаты точек в proposal
    gt_boxes: torch.Tensor,    # (B, 6)    — GT-бокс (cx, cy, cz, w, l, h)
) -> torch.Tensor:             # (B, N) bool — 1 если точка внутри GT-бокса
    """
    Вычисляет point_mask: какая из N точек попадает внутрь GT-бокса.

    Используется для генерации GT для offset_loss и centerness_loss.
    GT для offset: смещение от точки до GT-центра.
    GT для centerness: 1 если внутри, 0 если снаружи.
    """
    cx = gt_boxes[:, 0].unsqueeze(1)  # (B, 1)
    cy = gt_boxes[:, 1].unsqueeze(1)
    cz = gt_boxes[:, 2].unsqueeze(1)  # нижний центр (h/2 выше земли)
    w  = gt_boxes[:, 3].unsqueeze(1)
    l  = gt_boxes[:, 4].unsqueeze(1)
    h  = gt_boxes[:, 5].unsqueeze(1)

    px = points_xyz[:, :, 0]  # (B, N)
    py = points_xyz[:, :, 1]
    pz = points_xyz[:, :, 2]

    mask = (
        (px >= cx - w / 2) & (px <= cx + w / 2) &
        (py >= cy - l / 2) & (py <= cy + l / 2) &
        (pz >= 0)          & (pz <= h)
    )
    return mask  # (B, N) bool


def offset_loss(
    pred_offsets: torch.Tensor,  # (B, N, 3) — предсказанные смещения
    points_xyz: torch.Tensor,    # (B, N, 3) — координаты точек
    gt_centers: torch.Tensor,    # (B, 3)    — GT-центры деревьев (cx, cy, cz)
    point_mask: torch.Tensor,    # (B, N)    — 1 если точка внутри GT-бокса
    lambda_off: float = 0.5,
) -> torch.Tensor:
    """
    Smooth L1 между предсказанным смещением точки и реальным
    смещением к GT-центру дерева. Считается только по точкам внутри GT-бокса.
    """
    gt_offsets = gt_centers.unsqueeze(1) - points_xyz  # (B, N, 3)
    mask = point_mask.unsqueeze(-1).float()             # (B, N, 1)
    loss = F.smooth_l1_loss(pred_offsets * mask, gt_offsets * mask, reduction='sum')
    n_pos = mask.sum().clamp(min=1)
    return lambda_off * loss / n_pos


def centerness_loss(
    pred_centerness: torch.Tensor,  # (B, N, 1) — уже sigmoid [0..1]
    point_mask: torch.Tensor,       # (B, N)    — 1 если точка внутри GT-бокса
    lambda_cent: float = 0.5,
) -> torch.Tensor:
    """
    BCE с уже применённым sigmoid (F.binary_cross_entropy, не with_logits).
    pred_centerness выходит из nn.Sigmoid() в centerness_head.
    """
    gt_cent = point_mask.unsqueeze(-1).float()  # (B, N, 1), GT = 1 внутри, 0 снаружи
    loss = F.binary_cross_entropy(pred_centerness, gt_cent, reduction='mean')
    return lambda_cent * loss


def relation_loss(
    pred_scores: torch.Tensor,  # (B, N, 1) — scores из RelationHead (sigmoid)
    gt_labels: torch.Tensor,    # (B, N)    — 1 если IoU с GT > 0.5
    lambda_rel: float = 0.5,
) -> torch.Tensor:
    """
    BCE loss для Stage 3 Relation Head.
    pred_scores уже прошёл sigmoid в RelationHead.final_cls.
    """
    gt = gt_labels.unsqueeze(-1).float()  # (B, N, 1)
    loss = F.binary_cross_entropy(pred_scores, gt, reduction='mean')
    return lambda_rel * loss


def stage2_loss_v2(
    cls_score: torch.Tensor,        # (B, 1)    — logit (без sigmoid)
    reg_delta: torch.Tensor,        # (B, 6)
    pred_offsets: torch.Tensor,     # (B, N, 3)
    pred_centerness: torch.Tensor,  # (B, N, 1) — sigmoid [0..1]
    points_xyz: torch.Tensor,       # (B, N, 3)
    gt_box: torch.Tensor,           # (B, 6)    — GT бокс
    gt_label: torch.Tensor,         # (B,)      — 1=позитивный, 0=негативный
    lambdas: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Объединённый loss Stage 2.

    point_mask вычисляется автоматически через compute_point_mask().
    Возвращает словарь с отдельными компонентами и 'total'.
    """
    if lambdas is None:
        lambdas = {'cls': 1.0, 'reg': 1.0, 'offset': 0.5, 'centerness': 0.5}

    # point_mask: точки внутри GT-бокса (вычисляется здесь, не передаётся вне)
    point_mask = compute_point_mask(points_xyz, gt_box)  # (B, N) bool

    # Классификация (BCE with logits — cls_head возвращает logit)
    l_cls = F.binary_cross_entropy_with_logits(
        cls_score.squeeze(-1), gt_label.float(), reduction='mean'
    ) * lambdas['cls']

    # Регрессия боксов (только позитивные)
    pos_mask = gt_label.bool()
    if pos_mask.any():
        l_reg = F.smooth_l1_loss(
            reg_delta[pos_mask], gt_box[pos_mask], reduction='mean'
        ) * lambdas['reg']
    else:
        l_reg = reg_delta.sum() * 0.0

    # Offset и centerness loss
    gt_centers = gt_box[:, :3]  # cx, cy, cz
    l_off  = offset_loss(pred_offsets, points_xyz, gt_centers, point_mask,
                         lambda_off=lambdas['offset'])
    l_cent = centerness_loss(pred_centerness, point_mask,
                             lambda_cent=lambdas['centerness'])

    total = l_cls + l_reg + l_off + l_cent
    return {
        'total': total,
        'cls': l_cls, 'reg': l_reg,
        'offset': l_off, 'centerness': l_cent,
    }
