"""Encode / decode bounding boxes for regression targets."""

from __future__ import annotations

import torch
from torch import Tensor


def encode_boxes(gt_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Encode GT boxes relative to anchors.

    Parametrization (paper eq.):
        tx = (x_gt - x_a) / w_a
        ty = (y_gt - y_a) / w_a
        tw = log(w_gt / w_a)
        th = log(h_gt / h_a)

    Args:
        gt_boxes: (N, 6) [x, y, z_c, w, l, h]
        anchors:  (N, 6) same format

    Returns:
        deltas: (N, 4) [tx, ty, tw, th]
    """
    wa = anchors[:, 3].clamp(min=1e-4)
    ha = anchors[:, 5].clamp(min=1e-4)

    tx = (gt_boxes[:, 0] - anchors[:, 0]) / wa
    ty = (gt_boxes[:, 1] - anchors[:, 1]) / wa
    tw = torch.log((gt_boxes[:, 3] / wa).clamp(min=1e-4))
    th = torch.log((gt_boxes[:, 5] / ha).clamp(min=1e-4))

    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_boxes(deltas: Tensor, anchors: Tensor) -> Tensor:
    """
    Decode regression deltas back to absolute boxes.

    Args:
        deltas:  (N, 4) [tx, ty, tw, th]
        anchors: (N, 6) [x, y, z_c, w, l, h]

    Returns:
        boxes: (N, 6)
    """
    wa = anchors[:, 3]
    ha = anchors[:, 5]

    x = deltas[:, 0] * wa + anchors[:, 0]
    y = deltas[:, 1] * wa + anchors[:, 1]
    w = torch.exp(deltas[:, 2]) * wa
    h = torch.exp(deltas[:, 3]) * ha
    z_c = h / 2.0

    boxes = torch.stack([x, y, z_c, w, w, h], dim=-1)
    return boxes
