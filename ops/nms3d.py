"""3D Non-Maximum Suppression using IoUv."""

from __future__ import annotations

import torch
from torch import Tensor

from ops.iou3d import iou_volume


def nms3d(
    boxes: Tensor, scores: Tensor, iou_threshold: float, max_output: int | None = None
) -> Tensor:
    """
    3D NMS on axis-aligned boxes sorted by score.

    Args:
        boxes:         (N, 6) boxes [x, y, z_c, w, l, h]
        scores:        (N,) confidence scores
        iou_threshold: suppress box if IoUv > threshold with higher-scored box
        max_output:    max number of boxes to keep (None = keep all)

    Returns:
        keep: (K,) indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    boxes = boxes[order]

    keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
        if i + 1 >= len(boxes):
            break
        remaining = torch.where(keep_mask)[0]
        remaining = remaining[remaining > i]
        if len(remaining) == 0:
            break
        ious = iou_volume(boxes[i : i + 1], boxes[remaining])  # (1, R)
        suppress = remaining[ious[0] > iou_threshold]
        keep_mask[suppress] = False

    kept_local = torch.where(keep_mask)[0]
    if max_output is not None:
        kept_local = kept_local[:max_output]

    return order[kept_local]
