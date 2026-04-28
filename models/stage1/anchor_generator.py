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


def _to_flat_list(x) -> list:
    """Любой тип (Tensor / ndarray / list / tuple) → плоский list float."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float64).flatten().tolist()


def _generate_grid(cx: float, cy: float, slsa: float, si: float) -> np.ndarray:
    half = slsa / 2.0
    xs = np.arange(cx - half, cx + half + si * 0.5, si)
    ys = np.arange(cy - half, cy + half + si * 0.5, si)
    return np.array([[x, y] for x in xs for y in ys], dtype=np.float32)


class AnchorGenerator:
    """Generates Ad (dense) and Al (local-maxima) anchors (x, y, z_c, w, w, h)."""

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
        sl = self._sl_at_height(height)
        si = sl * self.si_ratio
        slsa = round(self._sl_at_height(height) / 4.0 / si) * 2 * si
        slsa = max(slsa, si)
        locs = _generate_grid(cx, cy, slsa, si)
        n = len(locs)
        anchors = np.zeros((n, 6), dtype=np.float32)
        anchors[:, 0] = locs[:, 0]
        anchors[:, 1] = locs[:, 1]
        anchors[:, 2] = height / 2.0
        anchors[:, 3] = sl
        anchors[:, 4] = sl
        anchors[:, 5] = height
        return anchors

    def generate_dense_anchors(self, plot_bounds) -> np.ndarray:
        vals = _to_flat_list(plot_bounds)
        if len(vals) != 4:
            raise ValueError(
                f"plot_bounds должен содержать 4 значения, получено {len(vals)}: {vals}"
            )
        x_min, y_min, x_max, y_max = vals[0], vals[1], vals[2], vals[3]

        max_cs = self._max_crown()
        ext = max_cs * self.boundary_ext
        stride = max_cs * self.sw_stride_ratio
        all_anchors: list[np.ndarray] = []

        for h in self.height_levels:
            xs = np.arange(x_min - ext, x_max + ext + stride * 0.5, stride)
            ys = np.arange(y_min - ext, y_max + ext + stride * 0.5, stride)
            for cx in xs:
                for cy in ys:
                    all_anchors.append(
                        self._anchors_for_window(float(cx), float(cy), h)
                    )

        if not all_anchors:
            return np.zeros((0, 6), dtype=np.float32)
        ad = np.concatenate(all_anchors, axis=0)
        logger.debug("Dense anchors (Ad): %d", len(ad))
        return ad

    def generate_maxima_anchors(self, local_maxima) -> list[np.ndarray]:
        lm = np.array(local_maxima, dtype=np.float32)
        if lm.ndim == 1:
            lm = lm.reshape(1, -1)
        elif lm.ndim > 2:
            lm = lm.reshape(-1, lm.shape[-1])

        anchors_al: list[np.ndarray] = []
        for pt in lm:
            cx, cy, h = float(pt[0]), float(pt[1]), float(pt[2])
            h_level = min(self.height_levels, key=lambda lv: abs(lv - h))
            anchors_al.append(self._anchors_for_window(cx, cy, h_level))

        logger.debug("Maxima anchor groups (Al): %d groups", len(anchors_al))
        return anchors_al

    def generate_all(self, plot_bounds, local_maxima) -> tuple[Tensor, list[Tensor]]:
        ad = self.generate_dense_anchors(plot_bounds)
        al_list = self.generate_maxima_anchors(local_maxima)
        return torch.from_numpy(ad).float(), [
            torch.from_numpy(a).float() for a in al_list
        ]
