"""
NEWFOR benchmark dataset loader.
Dataset page: http://www.newfor.net/download-newfor-single-tree-detection-benchmark-dataset
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.preprocessing import run_preprocessing, PlotData
from data.augmentation import augment

logger = logging.getLogger(__name__)

PLOT_DIR_PATTERN = "plot_{:02d}"


class NewforDataset(Dataset):
    """
    NEWFOR single-tree detection dataset.

    Expected layout::

        root/
            plot_01/
                plot_01.las
                dem.asc  (or dem.tif)
                reference_trees.txt
            plot_02/
                ...
    """

    def __init__(
        self,
        root: Path | str,
        plot_ids: list[int],
        cfg: DictConfig,
        augment_data: bool = True,
        max_points: int = 40_000,
    ) -> None:
        self.root = Path(root)
        self.plot_ids = plot_ids
        self.cfg = cfg
        self.augment_data = augment_data
        self.max_points = max_points
        self._plots: list[PlotData] = []
        self._load_all()

    def _load_all(self) -> None:
        for pid in self.plot_ids:
            pdir = self.root / PLOT_DIR_PATTERN.format(pid)
            las_files = list(pdir.glob("*.las")) + list(pdir.glob("*.laz"))
            if not las_files:
                logger.warning("No LAS file found in %s, skipping", pdir)
                continue
            las_path = las_files[0]

            dem_candidates = list(pdir.glob("*.asc")) + list(pdir.glob("*.tif"))
            if not dem_candidates:
                raise FileNotFoundError(f"No DEM found in {pdir}")
            dem_path = dem_candidates[0]

            ref_path = pdir / "reference_trees.txt"
            ref_trees = np.loadtxt(str(ref_path), dtype=np.float32)
            if ref_trees.ndim == 1:
                ref_trees = ref_trees[None, :]

            plot_data = run_preprocessing(
                las_path, dem_path, ref_trees, self.cfg, plot_id=pid
            )
            self._plots.append(plot_data)
            logger.info(
                "Loaded plot %02d: %d points, %d GT trees",
                pid,
                len(plot_data.points),
                len(plot_data.gt_boxes),
            )

    def __len__(self) -> int:
        return len(self._plots)

    def __getitem__(self, idx: int) -> dict:
        plot = self._plots[idx]

        points = plot.points.copy()
        gt_boxes = plot.gt_boxes.copy()
        local_maxima = plot.local_maxima.copy()

        if self.augment_data:
            points, gt_boxes, local_maxima = augment(
                points, gt_boxes, local_maxima, self.cfg.training.augmentation
            )

        # Пересчитываем plot_bounds из актуальных точек (после аугментации)
        # ВАЖНО: возвращаем как список float, не как тензор,
        # чтобы избежать проблем с batch-stacking в collate_fn
        x_min = float(points[:, 0].min())
        y_min = float(points[:, 1].min())
        x_max = float(points[:, 0].max())
        y_max = float(points[:, 1].max())
        plot_bounds = [x_min, y_min, x_max, y_max]

        logger.debug(
            "plot %02d bounds: x=[%.1f, %.1f]  y=[%.1f, %.1f]",
            plot.plot_id, x_min, x_max, y_min, y_max,
        )

        # Subsample
        if len(points) > self.max_points:
            idx_sub = np.random.choice(len(points), self.max_points, replace=False)
            points = points[idx_sub]

        return {
            "points":       torch.from_numpy(points).float(),       # (N, 3)
            "gt_boxes":     torch.from_numpy(gt_boxes).float(),     # (M, 6)
            "local_maxima": torch.from_numpy(local_maxima).float(), # (K, 3)
            "plot_bounds":  plot_bounds,   # list[float] — не тензор!
            "plot_id":      plot.plot_id,
        }
