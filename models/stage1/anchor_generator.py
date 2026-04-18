"""
Generate dense (Ad) and local-maxima (Al) 3D anchor boxes.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def crown_size(
    height: float, slope: float = 0.0512, intercept: float = 1.1048
) -> float:
    return slope * height + intercept


def _generate_grid(cx: float, cy: float, slsa: float, si: float) -> np.ndarray:
    """
    Generate a uniform 2D grid of location indices centred at (cx, cy).

    Args:
        cx, cy: grid centre
        slsa:   side length of the square area
        si:     grid spacing

    Returns:
        (K, 2) array of [x, y] positions
    """
    half = slsa / 2.0
    xs = np.arange(cx - half, cx + half + si * 0.5, si)
    ys = np.arange(cy - half, cy + half + si * 0.5, si)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    return grid


class AnchorGenerator:
    """
    Generates Ad (dense) and Al (local-maxima) anchors.

    All anchors use format (x, y, h/2, w, w, h).
    """

    def __init__(self, cfg) -> None:
        self.height_levels: List[float] = list(cfg.anchors.height_levels)
        self.sw_stride_ratio: float = cfg.anchors.sw_stride_ratio
        self.si_ratio: float = cfg.anchors.si_ratio
        self.boundary_ext: float = cfg.anchors.boundary_extension_ratio
        self.slope: float = cfg.crown_regression.slope
        self.intercept: float = cfg.crown_regression.intercept

    def _max_crown(self) -> float:
        return crown_size(max(self.height_levels), self.slope, self.intercept)

    def _sl_at_height(self, h: float) -> float:
        return crown_size(h, self.slope, self.intercept)

    def _anchors_for_window(self, cx: float, cy: float, height: float) -> np.ndarray:
        """
        Generate LN anchors for one sliding window centred at (cx, cy).
        """
        sl = self._sl_at_height(height)
        si = sl * self.si_ratio  # SI = SL / 5
        slsa_base = self._sl_at_height(height)
        slsa = round(slsa_base / 4.0 / si) * 2 * si  # round(SW/4/SI) * 2 * SI
        slsa = max(slsa, si)  # at least one cell

        locs = _generate_grid(cx, cy, slsa, si)  # (LN, 2)
        h_total = height
        z_c = h_total / 2.0
        w = sl

        n = len(locs)
        anchors = np.zeros((n, 6), dtype=np.float32)
        anchors[:, 0] = locs[:, 0]
        anchors[:, 1] = locs[:, 1]
        anchors[:, 2] = z_c
        anchors[:, 3] = w
        anchors[:, 4] = w
        anchors[:, 5] = h_total
        return anchors

    def generate_dense_anchors(
        self, plot_bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Generate Ad anchors over the plot.

        Args:
            plot_bounds: (x_min, y_min, x_max, y_max)

        Returns:
            anchors_ad: (N_ad, 6)
        """
        x_min, y_min, x_max, y_max = plot_bounds
        max_cs = self._max_crown()
        ext = max_cs * self.boundary_ext

        stride = max_cs * self.sw_stride_ratio
        all_anchors: list[np.ndarray] = []

        for h in self.height_levels:
            xs = np.arange(x_min - ext, x_max + ext + stride * 0.5, stride)
            ys = np.arange(y_min - ext, y_max + ext + stride * 0.5, stride)
            for cx in xs:
                for cy in ys:
                    anch = self._anchors_for_window(cx, cy, h)
                    all_anchors.append(anch)

        if not all_anchors:
            return np.zeros((0, 6), dtype=np.float32)
        ad = np.concatenate(all_anchors, axis=0)
        logger.debug("Dense anchors (Ad): %d", len(ad))
        return ad

    def generate_maxima_anchors(self, local_maxima: np.ndarray) -> list[np.ndarray]:
        """
        Generate Al anchors for each local maxima point.

        Args:
            local_maxima: (K, 3) [x, y, height]

        Returns:
            anchors_al: list of K arrays, each (LN, 6)
        """
        anchors_al: list[np.ndarray] = []
        for pt in local_maxima:
            cx, cy, h = float(pt[0]), float(pt[1]), float(pt[2])
            # Use the height level closest to the point's height
            h_level = min(self.height_levels, key=lambda lv: abs(lv - h))
            anch = self._anchors_for_window(cx, cy, h_level)
            anchors_al.append(anch)
        logger.debug("Maxima anchor groups (Al): %d groups", len(anchors_al))
        return anchors_al

    def generate_all(
        self,
        plot_bounds: tuple[float, float, float, float],
        local_maxima: np.ndarray,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Generate Ad and Al anchors as PyTorch tensors.

        Returns:
            ad_tensor: (N_ad, 6)
            al_tensors: list of (LN_k, 6) — one per local maxima point
        """
        ad = self.generate_dense_anchors(plot_bounds)
        al_list = self.generate_maxima_anchors(local_maxima)

        ad_t = torch.from_numpy(ad).float()
        al_t = [torch.from_numpy(a).float() for a in al_list]
        return ad_t, al_t
