"""
NEWFOR benchmark dataset loader.
Dataset page: http://www.newfor.net/download-newfor-single-tree-detection-benchmark-dataset

Changes vs v1:
  - __getitem__ (training mode): использует TrainingTiler для случайной нарезки
    тайлов из плота вместо глобальной random-subsampling.
    Если все max_attempts тайлов не набрали min_trees — откат к полному плоту
    с обычной подвыборкой (fallback).
  - Тайлинг активируется только при augment_data=True И cfg.tiling.enabled=True.
    При augment_data=False (валидация/инференс) поведение не меняется.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.preprocessing import run_preprocessing, PlotData
from data.augmentation import augment

# Тайлинг — мягкая зависимость (не ломает v1 без секции tiling в конфиге)
try:
    from utils.tiling import TrainingTiler, TileConfig
    _TILING_AVAILABLE = True
except ImportError:
    _TILING_AVAILABLE = False

logger = logging.getLogger(__name__)

PLOT_DIR_PATTERN = "plot_{:02d}"


def _build_tiler(cfg: DictConfig) -> "Optional[TrainingTiler]":
    """Создаёт TrainingTiler из секции cfg.tiling или возвращает None."""
    if not _TILING_AVAILABLE:
        logger.warning("utils.tiling не найден — тайлинг отключён.")
        return None
    t = cfg.tiling
    tile_cfg = TileConfig(
        tile_size=float(t.get("tile_size",    40.0)),
        overlap=float(t.get("overlap",        10.0)),
        min_trees=int(t.get("min_trees",       3)),
        max_points=int(t.get("max_points",     32768)),
        min_points=int(t.get("min_points",     100)),
        center_weight=float(t.get("center_weight", 1.0)),
        border_weight=float(t.get("border_weight", 0.5)),
    )
    max_att = int(t.get("max_attempts", 10))
    return TrainingTiler(tile_cfg, max_attempts=max_att)


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
        max_points: int | None = 40_000,
    ) -> None:
        self.root = Path(root)
        self.plot_ids = plot_ids
        self.cfg = cfg
        self.augment_data = augment_data
        self.max_points = max_points
        self._plots: list[PlotData] = []

        # Тайлинг только для обучения и только если явно включён в конфиге
        tiling_cfg = getattr(cfg, "tiling", None)
        tiling_on  = bool(tiling_cfg.get("enabled", False)) if tiling_cfg else False
        self._tiler: Optional[TrainingTiler] = (
            _build_tiler(cfg) if (augment_data and tiling_on) else None
        )
        if self._tiler is not None:
            logger.info(
                "[Dataset] Тайлинг ВКЛЮЧЁН (tile_size=%.1f, overlap=%.1f, min_trees=%d)",
                tiling_cfg.get("tile_size", 40.0),
                tiling_cfg.get("overlap",   10.0),
                tiling_cfg.get("min_trees",  3),
            )
        else:
            logger.info("[Dataset] Тайлинг ОТКЛЮЧЁН (random subsampling).")

        self._load_all()

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._plots)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        plot = self._plots[idx]

        points      = plot.points.copy()       # (N, 3+)
        gt_boxes    = plot.gt_boxes.copy()     # (M, 6)
        local_maxima = plot.local_maxima.copy() # (K, 3)

        if self.augment_data:
            points, gt_boxes, local_maxima = augment(
                points, gt_boxes, local_maxima, self.cfg.training.augmentation
            )

        # ── Тайлинг (только обучение) ──────────────────────────────────
        if self._tiler is not None:
            tile_result = self._tiler.random_tile(points, gt_boxes)
            if tile_result is not None:
                tile_pts, tile_gt = tile_result
                # local_maxima пересчитываем из gt_boxes тайла:
                # берём центры GT-деревьев как proxy для local_maxima внутри тайла
                # (аналог того, что делает find_local_maxima из CHM)
                lm_mask = (
                    (local_maxima[:, 0] >= tile_pts[:, 0].min()) &
                    (local_maxima[:, 0] <= tile_pts[:, 0].max()) &
                    (local_maxima[:, 1] >= tile_pts[:, 1].min()) &
                    (local_maxima[:, 1] <= tile_pts[:, 1].max())
                )
                tile_lm = local_maxima[lm_mask].copy()
                # смещаем local_maxima в локальные координаты тайла
                if len(tile_lm) > 0:
                    # центр тайла = 0 (TrainingTiler уже сдвинул pts и gt)
                    tile_cx = float(tile_pts[:, 0].mean())
                    tile_cy = float(tile_pts[:, 1].mean())
                    tile_lm[:, 0] -= tile_cx
                    tile_lm[:, 1] -= tile_cy
                else:
                    # fallback: local_maxima из GT-центров тайла
                    tile_lm = tile_gt[:, :3].copy()

                points       = tile_pts
                gt_boxes     = tile_gt
                local_maxima = tile_lm
                logger.debug(
                    "plot %02d → tile: %d pts, %d GT, %d maxima",
                    plot.plot_id, len(points), len(gt_boxes), len(local_maxima),
                )
            else:
                logger.debug(
                    "plot %02d: тайл не найден за max_attempts попыток — использую полный плот",
                    plot.plot_id,
                )
                # Fallback: обычная подвыборка без тайлинга
                if self.max_points is not None and len(points) > self.max_points:
                    idx_sub = np.random.choice(len(points), self.max_points, replace=False)
                    points = points[idx_sub]
        else:
            # ── Обычная подвыборка (валидация или тайлинг выключен) ────
            if self.max_points is not None and len(points) > self.max_points:
                idx_sub = np.random.choice(len(points), self.max_points, replace=False)
                points = points[idx_sub]

        # ── Plot bounds (по актуальным точкам) ─────────────────────────
        x_min = float(points[:, 0].min())
        y_min = float(points[:, 1].min())
        x_max = float(points[:, 0].max())
        y_max = float(points[:, 1].max())
        plot_bounds = [x_min, y_min, x_max, y_max]

        logger.debug(
            "plot %02d bounds: x=[%.1f, %.1f]  y=[%.1f, %.1f]",
            plot.plot_id, x_min, x_max, y_min, y_max,
        )

        return {
            "points":       torch.from_numpy(points).float(),        # (N, 3+)
            "gt_boxes":     torch.from_numpy(gt_boxes).float(),      # (M, 6)
            "local_maxima": torch.from_numpy(local_maxima).float(),  # (K, 3)
            "plot_bounds":  plot_bounds,
            "plot_id":      plot.plot_id,
        }
