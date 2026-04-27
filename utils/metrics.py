"""
NEWFOR evaluation metrics.
Protocol: matching detected trees to reference trees by nearest-neighbour
within a search radius, then computing RMS rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

SEARCH_RADIUS = 3.0  # meters (NEWFOR default)


@dataclass
class PlotMetrics:
    """Per-plot matching statistics."""

    plot_id: int
    n_test: int  # detected trees
    n_ref: int  # reference trees
    n_match: int  # correctly matched
    h_mean: float  # mean horizontal displacement of matched pairs
    v_mean: float  # mean vertical (height) displacement of matched pairs

    @property
    def rmr(self) -> float:
        """Matching rate (recall)."""
        return self.n_match / self.n_ref if self.n_ref > 0 else 0.0

    @property
    def rer(self) -> float:
        """Extraction rate: unmatched detections / n_ref."""
        return (self.n_test - self.n_match) / self.n_ref if self.n_ref > 0 else 0.0

    @property
    def rcr(self) -> float:
        """Commission rate: unmatched reference / n_ref."""
        return (self.n_ref - self.n_match) / self.n_ref if self.n_ref > 0 else 0.0

    @property
    def ror(self) -> float:
        """Overall error: |n_test - n_ref| / n_ref."""
        return abs(self.n_test - self.n_ref) / self.n_ref if self.n_ref > 0 else 0.0


@dataclass
class GlobalMetrics:
    rms_ass: float
    rms_extr: float
    rms_com: float
    rms_om: float
    rms_h: float
    rms_v: float


def extract_tree_positions(
    boxes: np.ndarray,
    points: np.ndarray,
    max_points_per_box: int = 512,
) -> np.ndarray:
    """
    Extract (x, y, height) from predicted 3D boxes.

    Height = highest LiDAR point inside box.
    Position = box bottom centre (x, y).

    Args:
        boxes:  (K, 6) predicted boxes [x, y, z_c, w, l, h]
        points: (N, 3) point cloud [x, y, z]

    Returns:
        (K, 3) array [x, y, height], or zeros shape (0, 3) if no boxes.
    """
    results = []
    for box in boxes:
        x, y, z_c, w, l, h = box
        mask = (
            (points[:, 0] >= x - w / 2)
            & (points[:, 0] <= x + w / 2)
            & (points[:, 1] >= y - l / 2)
            & (points[:, 1] <= y + l / 2)
            & (points[:, 2] >= 0)
            & (points[:, 2] <= h)
        )
        inside = points[mask]
        actual_h = float(inside[:, 2].max()) if len(inside) > 0 else float(h)
        results.append([x, y, actual_h])
    return (
        np.array(results, dtype=np.float32)
        if results
        else np.zeros((0, 3), dtype=np.float32)
    )


def newfor_matching(
    detected: np.ndarray,
    reference: np.ndarray,
    search_radius: float = SEARCH_RADIUS,
    plot_id: int = -1,
) -> PlotMetrics:
    """
    Match detected trees to reference trees by nearest-neighbour.

    Args:
        detected:      (Nd, 3+) [x, y, height, ...]
        reference:     (Nr, 3)  [x, y, height]
        search_radius: maximum matching distance in meters
        plot_id:       identifier for logging

    Returns:
        PlotMetrics for this plot
    """
    n_test = len(detected)
    n_ref = len(reference)
    matched_det: set[int] = set()
    h_diffs: list[float] = []
    v_diffs: list[float] = []

    for ref_tree in reference:
        best_dist = search_radius + 1e-9
        best_det = -1
        for det_j, det_tree in enumerate(detected):
            if det_j in matched_det:
                continue
            hdist = np.hypot(det_tree[0] - ref_tree[0], det_tree[1] - ref_tree[1])
            if hdist < best_dist:
                best_dist = hdist
                best_det = det_j
        if best_det >= 0 and best_dist <= search_radius:
            matched_det.add(best_det)
            h_diffs.append(best_dist)
            v_diffs.append(abs(float(detected[best_det, 2]) - float(ref_tree[2])))

    n_match = len(matched_det)
    h_mean = float(np.mean(h_diffs)) if h_diffs else 0.0
    v_mean = float(np.mean(v_diffs)) if v_diffs else 0.0

    logger.info(
        "Plot %d: detected=%d reference=%d matched=%d recall=%.1f%%",
        plot_id,
        n_test,
        n_ref,
        n_match,
        100.0 * n_match / max(n_ref, 1),
    )
    return PlotMetrics(plot_id, n_test, n_ref, n_match, h_mean, v_mean)


def compute_global_metrics(plot_metrics: list[PlotMetrics]) -> GlobalMetrics:
    """
    Aggregate per-plot metrics into global RMS scores.

    Args:
        plot_metrics: list of PlotMetrics

    Returns:
        GlobalMetrics with RMS values
    """

    def rms(vals: list[float]) -> float:
        return float(np.sqrt(np.mean(np.array(vals) ** 2))) if vals else 0.0

    return GlobalMetrics(
        rms_ass=rms([m.rmr for m in plot_metrics]),
        rms_extr=rms([m.rer for m in plot_metrics]),
        rms_com=rms([m.rcr for m in plot_metrics]),
        rms_om=rms([m.ror for m in plot_metrics]),
        rms_h=rms([m.h_mean for m in plot_metrics]),
        rms_v=rms([m.v_mean for m in plot_metrics]),
    )
