"""Data augmentation: rotation, translation, scale, horizontal flip."""

from __future__ import annotations

import numpy as np


def random_rotation_z(
    points: np.ndarray,
    boxes: np.ndarray,
    angle_range: float = 360.0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.uniform(0, np.deg2rad(angle_range))
    if center is None:
        center = points[:, :2].mean(axis=0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts        = points.copy()
    pts[:, :2] = (pts[:, :2] - center) @ R.T + center
    bxs        = boxes.copy()
    bxs[:, :2] = (bxs[:, :2] - center) @ R.T + center
    return pts, bxs


def random_translation(
    points: np.ndarray,
    boxes: np.ndarray,
    translation_range: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.random.uniform(-translation_range, translation_range)
    dy = np.random.uniform(-translation_range, translation_range)
    pts        = points.copy()
    pts[:, 0] += dx
    pts[:, 1] += dy
    bxs        = boxes.copy()
    bxs[:, 0] += dx
    bxs[:, 1] += dy
    return pts, bxs


def random_scale(
    points: np.ndarray,
    boxes: np.ndarray,
    local_maxima: np.ndarray,
    scale_range: float = 0.1,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform XY scale in [1-scale_range, 1+scale_range].

    Applies the SAME scale factor s to:
      - point XY positions
      - box XY centres + crown dimensions (w, l)
      - local_maxima XY positions

    Z and height (h) are NOT scaled — tree height is independent of crown width.
    """
    s = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    if center is None:
        center = points[:, :2].mean(axis=0)

    pts        = points.copy()
    pts[:, :2] = (pts[:, :2] - center) * s + center

    bxs        = boxes.copy()
    bxs[:, :2] = (bxs[:, :2] - center) * s + center
    bxs[:, 3] *= s   # w
    bxs[:, 4] *= s   # l

    lm        = local_maxima.copy()
    lm[:, :2] = (lm[:, :2] - center) * s + center   # same s, not a new draw

    return pts, bxs, lm


def random_flip(
    points: np.ndarray,
    boxes: np.ndarray,
    local_maxima: np.ndarray,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random horizontal flip along X or Y axis (50% probability each)."""
    if center is None:
        center = points[:, :2].mean(axis=0)

    pts = points.copy()
    bxs = boxes.copy()
    lm  = local_maxima.copy()

    if np.random.rand() < 0.5:
        pts[:, 0] = 2 * center[0] - pts[:, 0]
        bxs[:, 0] = 2 * center[0] - bxs[:, 0]
        lm[:, 0]  = 2 * center[0] - lm[:, 0]

    if np.random.rand() < 0.5:
        pts[:, 1] = 2 * center[1] - pts[:, 1]
        bxs[:, 1] = 2 * center[1] - bxs[:, 1]
        lm[:, 1]  = 2 * center[1] - lm[:, 1]

    return pts, bxs, lm


def augment(
    points: np.ndarray,
    boxes: np.ndarray,
    local_maxima: np.ndarray,
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply configured augmentation pipeline.

    Args:
        points:       (N, 3)
        boxes:        (M, 6) GT boxes
        local_maxima: (K, 3) [x, y, height]
        cfg:          augmentation sub-config (cfg.training.augmentation)

    Returns:
        aug_points, aug_boxes, aug_maxima
    """
    center = points[:, :2].mean(axis=0)

    if cfg.get("random_rotation", False):
        angle        = np.random.uniform(0, np.deg2rad(cfg.get("rotation_range", 360)))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        points               = points.copy()
        points[:, :2]        = (points[:, :2] - center) @ R.T + center
        boxes                = boxes.copy()
        boxes[:, :2]         = (boxes[:, :2] - center) @ R.T + center
        local_maxima         = local_maxima.copy()
        local_maxima[:, :2]  = (local_maxima[:, :2] - center) @ R.T + center

    if cfg.get("random_translation", False):
        t  = cfg.get("translation_range", 0.5)
        dx = np.random.uniform(-t, t)
        dy = np.random.uniform(-t, t)
        points[:, 0]       += dx;  points[:, 1]       += dy
        boxes[:, 0]        += dx;  boxes[:, 1]        += dy
        local_maxima[:, 0] += dx;  local_maxima[:, 1] += dy

    if cfg.get("random_scale", False):
        s = cfg.get("scale_range", 0.1)
        points, boxes, local_maxima = random_scale(
            points, boxes, local_maxima, scale_range=s, center=center
        )

    if cfg.get("random_flip", False):
        points, boxes, local_maxima = random_flip(
            points, boxes, local_maxima, center=center
        )

    return points, boxes, local_maxima
