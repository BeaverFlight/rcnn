"""
Assign positive / negative labels to anchors based on IoU with GT boxes.

Changed:
  - reg_targets now 6D (tx, ty, tz, tw, tl, th)
  - hard-negative mining: instead of random negatives, select top-n_neg
    by cls_scores when provided (OHEM / Hard Negative Mining)
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import torch
from torch import Tensor
from ops.iou3d import iou_volume, iou_bottom, iou_height
from utils.box_coder import encode_boxes

logger = logging.getLogger(__name__)
POS = 1
NEG = 0
IGNORE = -1


def assign_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    pos_iouv: float = 0.4,
    pos_ioub: float = 0.65,
    pos_iouh: float = 0.7,
    n_pos: int = 256,
    n_neg: int = 512,
    cls_scores: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assign targets with optional Hard Negative Mining.

    Args:
        cls_scores: (N,) sigmoid scores from Stage-1; when provided the
                    hardest n_neg negatives (highest score) are selected
                    instead of random ones.  Pass None to fall back to
                    random sampling (first epoch / Stage-2 path).

    Returns:
        labels, reg_targets (6D), matched_gt, sampled_indices
    """
    while anchors.dim() > 2:
        anchors = anchors.squeeze(0)
    while gt_boxes.dim() > 2:
        gt_boxes = gt_boxes.squeeze(0)

    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    device = anchors.device

    labels     = torch.full((N,), IGNORE, dtype=torch.long, device=device)
    matched_gt = torch.full((N,), -1,     dtype=torch.long, device=device)

    if M == 0:
        labels[:] = NEG
        reg_targets = torch.zeros(N, 6, device=device)
        sampled = _sample(labels, n_pos, n_neg, cls_scores)
        return labels, reg_targets, matched_gt, sampled

    iouv     = iou_volume(anchors, gt_boxes)   # (N, M)
    ioub     = iou_bottom(anchors, gt_boxes)   # (N, M)
    iouh_mat = iou_height(anchors, gt_boxes)   # (N, M)

    # Best anchor per GT → always positive
    best_anchor_per_gt = iouv.argmax(dim=0)    # (M,)
    labels[best_anchor_per_gt] = POS
    for i, anc_i in enumerate(best_anchor_per_gt):
        matched_gt[anc_i] = i

    # Condition-based positives
    cond     = (ioub > pos_ioub) & (iouh_mat > pos_iouh) & (iouv > pos_iouv)
    cond_any = cond.any(dim=1)
    new_pos  = cond_any & (labels != POS)
    if new_pos.any():
        labels[new_pos]     = POS
        matched_gt[new_pos] = iouv[new_pos].argmax(dim=1)

    labels[labels == IGNORE] = NEG

    # 6D regression targets
    reg_targets = torch.zeros(N, 6, device=device)
    pos_mask = labels == POS
    if pos_mask.any():
        reg_targets[pos_mask] = encode_boxes(
            gt_boxes[matched_gt[pos_mask]], anchors[pos_mask]
        )

    sampled = _sample(labels, n_pos, n_neg, cls_scores)
    logger.debug(
        "Label assignment: %d pos, %d neg",
        pos_mask.sum().item(), (labels == NEG).sum().item(),
    )
    return labels, reg_targets, matched_gt, sampled


def _sample(
    labels: Tensor,
    n_pos: int,
    n_neg: int,
    cls_scores: Optional[Tensor] = None,
) -> Tensor:
    """Sample positives (random) + negatives (hard or random)."""
    pos_idx = torch.where(labels == POS)[0]
    neg_idx = torch.where(labels == NEG)[0]

    n_pos_actual = min(n_pos, len(pos_idx))
    n_neg_actual = min(n_neg, len(neg_idx))

    if len(pos_idx) > n_pos_actual:
        perm    = torch.randperm(len(pos_idx), device=labels.device)
        pos_idx = pos_idx[perm[:n_pos_actual]]

    if len(neg_idx) > n_neg_actual:
        if cls_scores is not None:
            # Hard Negative Mining: pick negatives with highest cls score
            neg_scores = cls_scores[neg_idx]
            _, top_idx = neg_scores.topk(n_neg_actual)
            neg_idx    = neg_idx[top_idx]
        else:
            perm    = torch.randperm(len(neg_idx), device=labels.device)
            neg_idx = neg_idx[perm[:n_neg_actual]]

    return torch.cat([pos_idx, neg_idx], dim=0)
