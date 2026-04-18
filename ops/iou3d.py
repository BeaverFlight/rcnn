"""
IoU operations for 3D axis-aligned bounding boxes.
Box format: (x, y, z_center, w, l, h) where z_center = h/2 (tree stands on ground).
In TreeRCNN notation: (x, y, h/2, w, w, h) — square crown, w == l.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _box_to_corners_2d(boxes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (x_min, x_max, y_min, y_max) for N boxes shaped (N, 6)."""
    x, y, w = boxes[:, 0], boxes[:, 1], boxes[:, 3]
    half = w / 2.0
    return x - half, x + half, y - half, y + half


def _box_to_height_range(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Return (z_min=0, z_max=h) for N boxes. Trees start at ground (z=0)."""
    h = boxes[:, 5]
    z_min = torch.zeros_like(h)
    z_max = h
    return z_min, z_max


def iou_bottom(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise 2D IoU of bottom (horizontal) faces.

    Args:
        boxes1: (N, 6) — [x, y, z_c, w, l, h]
        boxes2: (M, 6)

    Returns:
        iou_b: (N, M)
    """
    x1_min, x1_max, y1_min, y1_max = _box_to_corners_2d(boxes1)
    x2_min, x2_max, y2_min, y2_max = _box_to_corners_2d(boxes2)

    # (N, M) intersections
    inter_x = (
        torch.min(x1_max[:, None], x2_max[None, :])
        - torch.max(x1_min[:, None], x2_min[None, :])
    ).clamp(min=0)
    inter_y = (
        torch.min(y1_max[:, None], y2_max[None, :])
        - torch.max(y1_min[:, None], y2_min[None, :])
    ).clamp(min=0)

    inter_area = inter_x * inter_y
    area1 = (x1_max - x1_min) * (y1_max - y1_min)  # (N,)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)  # (M,)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


def iou_height(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise 1D IoU along the height axis [0, h].

    Args:
        boxes1: (N, 6)
        boxes2: (M, 6)

    Returns:
        iou_h: (N, M)
    """
    z1_min, z1_max = _box_to_height_range(boxes1)
    z2_min, z2_max = _box_to_height_range(boxes2)

    inter = (
        torch.min(z1_max[:, None], z2_max[None, :])
        - torch.max(z1_min[:, None], z2_min[None, :])
    ).clamp(min=0)
    len1 = z1_max - z1_min  # (N,)
    len2 = z2_max - z2_min  # (M,)
    union = len1[:, None] + len2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def iou_volume(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise 3D volumetric IoU (axis-aligned boxes).

    Args:
        boxes1: (N, 6) — [x, y, z_c, w, l, h]
        boxes2: (M, 6)

    Returns:
        iou_v: (N, M)
    """
    ioub = iou_bottom(boxes1, boxes2)  # (N, M)
    iouh = iou_height(boxes1, boxes2)  # (N, M)

    # IoUv = inter_vol / union_vol
    # inter_vol = inter_area * inter_height
    x1_min, x1_max, y1_min, y1_max = _box_to_corners_2d(boxes1)
    x2_min, x2_max, y2_min, y2_max = _box_to_corners_2d(boxes2)
    z1_min, z1_max = _box_to_height_range(boxes1)
    z2_min, z2_max = _box_to_height_range(boxes2)

    inter_x = (
        torch.min(x1_max[:, None], x2_max[None, :])
        - torch.max(x1_min[:, None], x2_min[None, :])
    ).clamp(min=0)
    inter_y = (
        torch.min(y1_max[:, None], y2_max[None, :])
        - torch.max(y1_min[:, None], y2_min[None, :])
    ).clamp(min=0)
    inter_z = (
        torch.min(z1_max[:, None], z2_max[None, :])
        - torch.max(z1_min[:, None], z2_min[None, :])
    ).clamp(min=0)

    inter_vol = inter_x * inter_y * inter_z

    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)  # (N,)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)  # (M,)

    union_vol = vol1[:, None] + vol2[None, :] - inter_vol
    return inter_vol / union_vol.clamp(min=1e-6)
