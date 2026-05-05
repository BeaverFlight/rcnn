"""Encode / decode bounding boxes for regression targets.

Changed: 4D → 6D parametrisation.
  tx  = (x_gt  - x_a)  / w_a
  ty  = (y_gt  - y_a)  / w_a
  tz  = (zc_gt - zc_a) / h_a
  tw  = log(w_gt / w_a)
  tl  = log(l_gt / l_a)    (l_a == w_a for square anchors)
  th  = log(h_gt / h_a)
"""

from __future__ import annotations

import torch
from torch import Tensor


def encode_boxes(gt_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Encode GT boxes relative to anchors (6D).

    Args:
        gt_boxes: (N, 6) [x, y, z_c, w, l, h]
        anchors:  (N, 6) same format

    Returns:
        deltas: (N, 6) [tx, ty, tz, tw, tl, th]
    """
    wa = anchors[:, 3].clamp(min=1e-4)
    la = anchors[:, 4].clamp(min=1e-4)
    ha = anchors[:, 5].clamp(min=1e-4)

    tx = (gt_boxes[:, 0] - anchors[:, 0]) / wa
    ty = (gt_boxes[:, 1] - anchors[:, 1]) / wa
    tz = (gt_boxes[:, 2] - anchors[:, 2]) / ha
    tw = torch.log((gt_boxes[:, 3] / wa).clamp(min=1e-4))
    tl = torch.log((gt_boxes[:, 4] / la).clamp(min=1e-4))
    th = torch.log((gt_boxes[:, 5] / ha).clamp(min=1e-4))

    return torch.stack([tx, ty, tz, tw, tl, th], dim=-1)


def decode_boxes(deltas: Tensor, anchors: Tensor) -> Tensor:
    """
    Decode regression deltas back to absolute boxes (6D).

    Args:
        deltas:  (N, 6) [tx, ty, tz, tw, tl, th]
        anchors: (N, 6) [x, y, z_c, w, l, h]

    Returns:
        boxes: (N, 6)
    """
    wa = anchors[:, 3]
    la = anchors[:, 4]
    ha = anchors[:, 5]

    x   = deltas[:, 0] * wa + anchors[:, 0]
    y   = deltas[:, 1] * wa + anchors[:, 1]
    z_c = deltas[:, 2] * ha + anchors[:, 2]
    w   = torch.exp(deltas[:, 3]) * wa
    l   = torch.exp(deltas[:, 4]) * la
    h   = torch.exp(deltas[:, 5]) * ha

    return torch.stack([x, y, z_c, w, l, h], dim=-1)
