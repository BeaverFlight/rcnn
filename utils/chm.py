"""
Canopy Height Model (CHM) generation from normalized LiDAR point clouds.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter, grey_closing

logger = logging.getLogger(__name__)


def generate_chm(
    points: np.ndarray,
    resolution: float = 0.5,
    x_min: float | None = None,
    y_min: float | None = None,
    x_max: float | None = None,
    y_max: float | None = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate CHM raster from normalized point cloud.

    Points are assumed to already have normalized heights (z = height above ground).

    Args:
        points:     (N, 3) array [x, y, z_normalized]
        resolution: pixel size in meters
        x_min, y_min, x_max, y_max: optional bounding box (computed from data if None)

    Returns:
        chm:    2D numpy array of maximum heights per cell
        x_orig: x-coordinate of the top-left corner of the CHM
        y_orig: y-coordinate of the top-left corner of the CHM
    """
    if x_min is None:
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
    if x_max is None:
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    chm = np.zeros((rows, cols), dtype=np.float32)

    px = np.floor((points[:, 0] - x_min) / resolution).astype(int)
    py = np.floor((points[:, 1] - y_min) / resolution).astype(int)

    # Clip to valid range
    px = np.clip(px, 0, cols - 1)
    py = np.clip(py, 0, rows - 1)

    np.maximum.at(chm, (py, px), points[:, 2])
    logger.debug("CHM generated: shape=%s, resolution=%.2f m", chm.shape, resolution)
    return chm, x_min, y_min


def apply_closing_filter(chm: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply morphological closing to fill small gaps in CHM."""
    from scipy.ndimage import grey_closing

    structure = np.ones((window, window), dtype=bool)
    closed = grey_closing(chm, structure=structure)
    return closed.astype(np.float32)


def extract_local_maxima(
    chm: np.ndarray,
    window: int = 3,
    min_height: float = 5.0,
    resolution: float = 0.5,
    x_orig: float = 0.0,
    y_orig: float = 0.0,
) -> np.ndarray:
    """
    Extract local maxima positions from CHM.

    Args:
        chm:        closed CHM array
        window:     local maxima filter window size (pixels)
        min_height: minimum height threshold (meters)
        resolution: CHM resolution (m/pixel)
        x_orig:     x-coordinate of top-left corner
        y_orig:     y-coordinate of top-left corner

    Returns:
        maxima: (K, 3) array of [x, y, height] for each local maximum
    """
    local_max = maximum_filter(chm, size=window)
    is_max = (chm == local_max) & (chm >= min_height)
    rows, cols = np.where(is_max)

    x = x_orig + (cols + 0.5) * resolution
    y = y_orig + (rows + 0.5) * resolution
    heights = chm[rows, cols]

    maxima = np.column_stack([x, y, heights])
    logger.info("Found %d local maxima (height >= %.1f m)", len(maxima), min_height)
    return maxima
