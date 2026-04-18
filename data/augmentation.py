"""Data augmentation: random rotation and translation for point clouds and boxes."""

from __future__ import annotations

import numpy as np


def random_rotation_z(
    points: np.ndarray,
    boxes: np.ndarray,
    angle_range: float = 360.0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate points and boxes around vertical (Z) axis.

    Args:
        points:      (N, 3) [x, y, z]
        boxes:       (M, 6) [x, y, z_c, w, l, h]
        angle_range: maximum rotation in degrees (symmetric around 0 → uniform [0, 360])
        center:      (2,) horizontal center of rotation; defaults to centroid of points

    Returns:
        rotated_points, rotated_boxes
    """
    angle = np.random.uniform(0, np.deg2rad(angle_range))
    if center is None:
        center = points[:, :2].mean(axis=0)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    pts = points.copy()
    pts[:, :2] = (pts[:, :2] - center) @ R.T + center

    bxs = boxes.copy()
    bxs[:, :2] = (bxs[:, :2] - center) @ R.T + center  # x, y
    return pts, bxs


def random_translation(
    points: np.ndarray,
    boxes: np.ndarray,
    translation_range: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random horizontal translation.

    Args:
        points:            (N, 3)
        boxes:             (M, 6)
        translation_range: max offset in meters (uniform [-t, t])

    Returns:
        translated_points, translated_boxes
    """
    dx = np.random.uniform(-translation_range, translation_range)
    dy = np.random.uniform(-translation_range, translation_range)

    pts = points.copy()
    pts[:, 0] += dx
    pts[:, 1] += dy

    bxs = boxes.copy()
    bxs[:, 0] += dx
    bxs[:, 1] += dy
    return pts, bxs


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
        cfg:          augmentation sub-config

    Returns:
        aug_points, aug_boxes, aug_maxima
    """
    center = points[:, :2].mean(axis=0)

    if cfg.random_rotation:
        angle = np.random.uniform(0, np.deg2rad(cfg.rotation_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        points = points.copy()
        points[:, :2] = (points[:, :2] - center) @ R.T + center

        boxes = boxes.copy()
        boxes[:, :2] = (boxes[:, :2] - center) @ R.T + center

        local_maxima = local_maxima.copy()
        local_maxima[:, :2] = (local_maxima[:, :2] - center) @ R.T + center

    if cfg.random_translation:
        dx = np.random.uniform(-cfg.translation_range, cfg.translation_range)
        dy = np.random.uniform(-cfg.translation_range, cfg.translation_range)
        points[:, 0] += dx
        points[:, 1] += dy
        boxes[:, 0] += dx
        boxes[:, 1] += dy
        local_maxima[:, 0] += dx
        local_maxima[:, 1] += dy

    return points, boxes, local_maxima
