"""
utils/tiling.py — Модуль тайлинга для TreeRCNN v2.0

Фиксы:
  - InferenceTiler.merge(): заменён неверный Python `and` на numpy поэлементный `&`
  - TrainingTiler: добавлен fallback-поведение в random_tile (max_attempts)
  - rich_features fallback: добавлен warning если Open3D не установлен
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TileConfig:
    tile_size: float = 40.0
    overlap: float = 10.0
    min_trees: int = 3
    max_points: int = 32768
    min_points: int = 100
    center_weight: float = 1.0
    border_weight: float = 0.5


@dataclass
class TileMeta:
    origin_x: float
    origin_y: float
    tile_idx: int
    is_border: bool = False


class TrainingTiler:
    """
    Случайная нарезка тайлов во время обучения.

    Пример использования в Dataset.__getitem__:
        tiler = TrainingTiler(cfg)
        result = tiler.random_tile(plot_points, gt_boxes)  # max_attempts попыток внутри
        if result is None:
            # весь плот пустой — пропустить или взять другой плот
            ...
        tile_pts, tile_gt = result
    """

    def __init__(self, cfg: TileConfig, max_attempts: int = 10):
        self.cfg = cfg
        self.max_attempts = max_attempts

    def random_tile(
        self,
        points: np.ndarray,    # (N, 9) — обогащённые точки
        gt_boxes: np.ndarray,  # (M, 6) — cx, cy, cz, w, l, h
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Делает до max_attempts попыток найти непустой тайл.
        Возвращает (tile_points, tile_gt_boxes) с локализованными
        координатами (центр тайла = (0, 0)) или None если все попытки неудачны.
        """
        for _ in range(self.max_attempts):
            result = self._try_tile(points, gt_boxes)
            if result is not None:
                return result
        logger.warning(
            "TrainingTiler: не удалось найти непустой тайл за %d попыток (min_trees=%d)",
            self.max_attempts, self.cfg.min_trees
        )
        return None

    def _try_tile(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cfg = self.cfg
        half = cfg.tile_size / 2.0

        anchor = gt_boxes[np.random.randint(len(gt_boxes))]
        cx = anchor[0] + np.random.uniform(-10.0, 10.0)
        cy = anchor[1] + np.random.uniform(-10.0, 10.0)

        mask_pts = (
            (points[:, 0] >= cx - half) & (points[:, 0] <= cx + half) &
            (points[:, 1] >= cy - half) & (points[:, 1] <= cy + half)
        )
        tile_pts = points[mask_pts].copy()

        if len(tile_pts) < cfg.min_points:
            return None

        mask_gt = (
            (gt_boxes[:, 0] >= cx - half) & (gt_boxes[:, 0] <= cx + half) &
            (gt_boxes[:, 1] >= cy - half) & (gt_boxes[:, 1] <= cy + half)
        )
        tile_gt = gt_boxes[mask_gt].copy()

        if len(tile_gt) < cfg.min_trees:
            return None

        if len(tile_pts) > cfg.max_points:
            idx = np.random.choice(len(tile_pts), cfg.max_points, replace=False)
            tile_pts = tile_pts[idx]

        tile_pts[:, 0] -= cx
        tile_pts[:, 1] -= cy
        tile_gt[:, 0] -= cx
        tile_gt[:, 1] -= cy

        return tile_pts, tile_gt


class InferenceTiler:
    """
    Скользящее окно с перекрытием для инференса.

    Пример:
        tiler = InferenceTiler(cfg)
        for tile_pts, meta in tiler.tiles(las_points):
            boxes = model.predict(tile_pts)
            tiler.collect(boxes, meta)
        final_boxes = tiler.merge()
    """

    def __init__(self, cfg: TileConfig):
        self.cfg = cfg
        self._collected: List[Tuple[np.ndarray, TileMeta]] = []

    def tiles(
        self, points: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, TileMeta]]:
        cfg  = self.cfg
        half = cfg.tile_size / 2.0
        step = cfg.tile_size - cfg.overlap

        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())

        xs = np.arange(x_min + half, x_max, step)
        ys = np.arange(y_min + half, y_max, step)

        tile_idx = 0
        for cx in xs:
            for cy in ys:
                mask = (
                    (points[:, 0] >= cx - half) & (points[:, 0] <= cx + half) &
                    (points[:, 1] >= cy - half) & (points[:, 1] <= cy + half)
                )
                tile_pts = points[mask].copy()

                if len(tile_pts) < cfg.min_points:
                    continue

                if len(tile_pts) > cfg.max_points:
                    idx = np.random.choice(len(tile_pts), cfg.max_points, replace=False)
                    tile_pts = tile_pts[idx]

                is_border = (
                    (cx - half <= x_min + step * 0.1) or
                    (cx + half >= x_max - step * 0.1) or
                    (cy - half <= y_min + step * 0.1) or
                    (cy + half >= y_max - step * 0.1)
                )

                meta = TileMeta(
                    origin_x=float(cx), origin_y=float(cy),
                    tile_idx=tile_idx, is_border=is_border
                )
                tile_pts[:, 0] -= cx
                tile_pts[:, 1] -= cy

                tile_idx += 1
                yield tile_pts, meta

    def collect(self, boxes: np.ndarray, meta: TileMeta) -> None:
        if boxes is None or len(boxes) == 0:
            return
        global_boxes = boxes.copy()
        global_boxes[:, 0] += meta.origin_x
        global_boxes[:, 1] += meta.origin_y
        self._collected.append((global_boxes, meta))

    def merge(self, score_col: int = 6) -> np.ndarray:
        """
        Объединяет боксы из всех тайлов с взвешением скоров.
        Фикс: заменён неверный Python `and` на numpy поэлементный `&`.

        score_col: индекс колонки score в матрице боксов (по умолчанию 6)
        """
        if not self._collected:
            return np.empty((0, 7))

        cfg    = self.cfg
        buffer = cfg.overlap / 2.0
        all_boxes = []

        for boxes, meta in self._collected:
            weighted = boxes.copy()

            if weighted.shape[1] > score_col:
                # Поэлементный `&` вместо Python `and` — исправлено ValueError
                half = cfg.tile_size / 2.0
                in_center = (
                    (np.abs(weighted[:, 0] - meta.origin_x) <= (half - buffer)) &
                    (np.abs(weighted[:, 1] - meta.origin_y) <= (half - buffer))
                )
                weight = cfg.border_weight if meta.is_border else cfg.center_weight
                # центральная зона получает полный вес, буферная — пониженный
                weighted[~in_center, score_col] *= cfg.border_weight
                weighted[in_center, score_col]  *= weight

            all_boxes.append(weighted)

        return np.concatenate(all_boxes, axis=0)
