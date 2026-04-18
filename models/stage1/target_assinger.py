"""
Assign positive / negative labels to anchors based on IoU with GT boxes.
"""

from __future__ import annotations

import logging
from typing import Tuple

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
    n_neg: int = 256,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Assign labels and regression targets to anchors.

    Positive assignment (OR rule):
      - Anchor with max IoUv for a given GT box, OR
      - IoUb > pos_ioub AND IoUh > pos_iouh AND IoUv > pos_iouv

    Args:
        anchors:  (N, 6)
        gt_boxes: (M, 6)
        pos_iouv, pos_ioub, pos_iouh: IoU thresholds
        n_pos, n_neg: balanced sampling counts

    Returns:
        labels:       (N,) — 1 positive, 0 negative, -1 ignore
        reg_targets:  (N, 4) — encoded box deltas (valid for positive only)
        matched_gt:   (N,) — index of matched GT box (-1 if no match)
        sampled_idx:  (S,) — indices of sampled anchors (n_pos + n_neg)
    """
    N = anchors.shape[0]
    M = gt_boxes.shape[0]
    device = anchors.device

    labels = torch.full((N,), IGNORE, dtype=torch.long, device=device)
    matched_gt = torch.full((N,), -1, dtype=torch.long, device=device)

    if M == 0:
        labels[:] = NEG
        reg_targets = torch.zeros(N, 4, device=device)
        sampled = _balanced_sample(labels, n_pos, n_neg)
        return labels, reg_targets, matched_gt, sampled

    iouv = iou_volume(anchors, gt_boxes)  # (N, M)
    ioub = iou_bottom(anchors, gt_boxes)  # (N, M)
    iouh_mat = iou_height(anchors, gt_boxes)  # (N, M)

    # Best anchor per GT → positive
    best_anchor_per_gt = iouv.argmax(dim=0)  # (M,)
    labels[best_anchor_per_gt] = POS
    for i, anc_i in enumerate(best_anchor_per_gt):
        matched_gt[anc_i] = i

    # Condition-based positives
    cond = (ioub > pos_ioub) & (iouh_mat > pos_iouh) & (iouv > pos_iouv)  # (N, M)
    cond_any = cond.any(dim=1)
    new_pos = cond_any & (labels != POS)
    if new_pos.any():
        labels[new_pos] = POS
        best_gt = iouv[new_pos].argmax(dim=1)
        matched_gt[new_pos] = best_gt

    # All remaining anchors → negative
    labels[labels == IGNORE] = NEG

    # Encode regression targets for positives
    reg_targets = torch.zeros(N, 4, device=device)
    pos_mask = labels == POS
    if pos_mask.any():
        pos_anchors = anchors[pos_mask]
        pos_gt = gt_boxes[matched_gt[pos_mask]]
        reg_targets[pos_mask] = encode_boxes(pos_gt, pos_anchors)

    sampled = _balanced_sample(labels, n_pos, n_neg)
    logger.debug(
        "Label assignment: %d pos, %d neg",
        pos_mask.sum().item(),
        (labels == NEG).sum().item(),
    )
    return labels, reg_targets, matched_gt, sampled


def _balanced_sample(labels: Tensor, n_pos: int, n_neg: int) -> Tensor:
    """Randomly sample n_pos positive and n_neg negative anchor indices."""
    pos_idx = torch.where(labels == POS)[0]
    neg_idx = torch.where(labels == NEG)[0]

    n_pos_actual = min(n_pos, len(pos_idx))
    n_neg_actual = min(n_neg, len(neg_idx))

    if len(pos_idx) > n_pos_actual:
        perm = torch.randperm(len(pos_idx), device=labels.device)
        pos_idx = pos_idx[perm[:n_pos_actual]]

    if len(neg_idx) > n_neg_actual:
        perm = torch.randperm(len(neg_idx), device=labels.device)
        neg_idx = neg_idx[perm[:n_neg_actual]]

    return torch.cat([pos_idx, neg_idx], dim=0)
