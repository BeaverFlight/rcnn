"""3D visualization of point clouds and detections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def visualize_detections(
    points: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: Optional[np.ndarray] = None,
    title: str = "TreeRCNN Detections",
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualize detected and GT trees over the point cloud.

    Uses Open3D for interactive 3D view or matplotlib for 2D top-down.

    Args:
        points:    (N, 3) point cloud
        pred_boxes:(P, 6) predicted boxes [x, y, z_c, w, l, h]
        gt_boxes:  (M, 6) optional ground-truth boxes
        title:     window title
        save_path: if given, save top-down matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            points[:, 0], points[:, 1], s=0.1, c=points[:, 2], cmap="viridis", alpha=0.3
        )

        for box in pred_boxes:
            cx, cy, _, w, _, _ = box
            rect = plt.Rectangle(
                (cx - w / 2, cy - w / 2),
                w,
                w,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                label="Predicted",
            )
            ax.add_patch(rect)

        if gt_boxes is not None:
            for box in gt_boxes:
                cx, cy, _, w, _, _ = box
                rect = plt.Rectangle(
                    (cx - w / 2, cy - w / 2),
                    w,
                    w,
                    linewidth=1,
                    edgecolor="green",
                    facecolor="none",
                    linestyle="--",
                    label="GT",
                )
                ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Saved visualization to %s", save_path)
        else:
            plt.show()
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available; skipping visualization")
