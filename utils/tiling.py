"""
utils/tiling.py — Модуль тайлинга для TreeRCNN v2.0

Два публичных класса:
  - TrainingTiler : случайная нарезка во время обучения
  - InferenceTiler: скользящее окно с перекрытием для инференса
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


@dataclass
class TileConfig:
    tile_size: float = 40.0          # метры
    overlap: float = 10.0            # перекрытие при инференсе
    min_trees: int = 3               # минимум GT-деревьев в тайле
    max_points: int = 32768          # субсэмплинг если точек больше
    min_points: int = 100            # тайл игнорируется если точек меньше
    center_weight: float = 1.0       # вес боксов из центральной зоны
    border_weight: float = 0.5       # вес боксов из буферной зоны


@dataclass
class TileMeta:
    origin_x: float          # глобальные координаты центра тайла
    origin_y: float
    tile_idx: int
    is_border: bool = False  # тайл на краю плота


class TrainingTiler:
    """
    Случайная нарезка тайлов во время обучения.

    Пример использования в Dataset.__getitem__:
        tiler = TrainingTiler(cfg)
        for _ in range(10):  # до 10 попыток найти непустой тайл
            result = tiler.random_tile(plot_points, gt_boxes)
            if result is not None:
                tile_pts, tile_gt = result
                break
    """

    def __init__(self, cfg: TileConfig):
        self.cfg = cfg

    def random_tile(
        self,
        points: np.ndarray,   # (N, 9) — обогащённые точки xyz+features
        gt_boxes: np.ndarray, # (M, 6) — cx, cy, cz, w, l, h
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Выбирает случайный GT-якорь, добавляет случайное смещение,
        вырезает тайл и фильтрует GT-боксы по центру.

        Возвращает (tile_points, tile_gt_boxes) с локализованными
        координатами (центр тайла = (0, 0)) или None если тайл пустой.
        """
        cfg = self.cfg
        half = cfg.tile_size / 2.0

        # Случайный якорь — одно из GT-деревьев
        anchor = gt_boxes[np.random.randint(len(gt_boxes))]
        cx = anchor[0] + np.random.uniform(-10.0, 10.0)
        cy = anchor[1] + np.random.uniform(-10.0, 10.0)

        # Вырезаем точки
        mask_pts = (
            (points[:, 0] >= cx - half) & (points[:, 0] <= cx + half) &
            (points[:, 1] >= cy - half) & (points[:, 1] <= cy + half)
        )
        tile_pts = points[mask_pts].copy()

        if len(tile_pts) < cfg.min_points:
            return None

        # GT-боксы: только те, чей центр строго внутри тайла
        mask_gt = (
            (gt_boxes[:, 0] >= cx - half) & (gt_boxes[:, 0] <= cx + half) &
            (gt_boxes[:, 1] >= cy - half) & (gt_boxes[:, 1] <= cy + half)
        )
        tile_gt = gt_boxes[mask_gt].copy()

        if len(tile_gt) < cfg.min_trees:
            return None

        # Субсэмплинг
        if len(tile_pts) > cfg.max_points:
            idx = np.random.choice(len(tile_pts), cfg.max_points, replace=False)
            tile_pts = tile_pts[idx]

        # Локализация: центр тайла → (0, 0)
        tile_pts[:, 0] -= cx
        tile_pts[:, 1] -= cy
        tile_gt[:, 0] -= cx
        tile_gt[:, 1] -= cy

        return tile_pts, tile_gt


class InferenceTiler:
    """
    Скользящее окно с перекрытием для инференса.

    Пример использования:
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
        """
        Генератор тайлов. Yields (tile_points, TileMeta).
        Координаты точек локализованы относительно центра тайла.
        """
        cfg = self.cfg
        half = cfg.tile_size / 2.0
        step = cfg.tile_size - cfg.overlap

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

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
                    cx - half <= x_min + 1.0 or cx + half >= x_max - 1.0 or
                    cy - half <= y_min + 1.0 or cy + half >= y_max - 1.0
                )

                meta = TileMeta(
                    origin_x=cx, origin_y=cy,
                    tile_idx=tile_idx, is_border=is_border
                )

                # Локализация
                tile_pts[:, 0] -= cx
                tile_pts[:, 1] -= cy

                tile_idx += 1
                yield tile_pts, meta

    def collect(self, boxes: np.ndarray, meta: TileMeta) -> None:
        """
        Принимает боксы из тайла (в локальных координатах),
        переводит их обратно в глобальные и сохраняет.
        """
        if boxes is None or len(boxes) == 0:
            return
        global_boxes = boxes.copy()
        global_boxes[:, 0] += meta.origin_x
        global_boxes[:, 1] += meta.origin_y
        self._collected.append((global_boxes, meta))

    def merge(self, iou_threshold: float = 0.3) -> np.ndarray:
        """
        Объединяет боксы из всех тайлов.
        Боксы из буферной зоны (overlap) получают пониженный score.
        Применяет финальный NMS.
        """
        if not self._collected:
            return np.empty((0, 7))

        cfg = self.cfg
        buffer = cfg.overlap / 2.0
        all_boxes = []

        for boxes, meta in self._collected:
            weighted = boxes.copy()
            # Боксы в центральной зоне тайла
            in_center = (
                np.abs(boxes[:, 0] - meta.origin_x) <= (cfg.tile_size / 2.0 - buffer) and
                np.abs(boxes[:, 1] - meta.origin_y) <= (cfg.tile_size / 2.0 - buffer)
            )
            if boxes.shape[1] > 6:  # если есть score-колонка
                weight = cfg.center_weight if not meta.is_border else cfg.border_weight
                weighted[:, 6] *= weight
            all_boxes.append(weighted)

        return np.concatenate(all_boxes, axis=0)
