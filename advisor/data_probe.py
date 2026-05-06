"""
advisor/data_probe.py — анализ данных (dataset)

API:
    probe_dataset(data_root, cfg) -> DatasetStats

Анализирует:
  - число плотов / деревьев
  - распределение плотности поинтов
  - распределение высот деревьев (h)
  - имбаланс классов (pos/neg ratio)
  - количество точек на дерево (редкие плоты)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DatasetStats:
    n_plots:           int
    n_trees_total:     int
    n_trees_per_plot:  list[int]     = field(default_factory=list)
    pts_per_plot:      list[int]     = field(default_factory=list)  # число точек
    height_mean:       float         = 0.0
    height_std:        float         = 0.0
    height_min:        float         = 0.0
    height_max:        float         = 0.0
    pts_per_tree_mean: float         = 0.0   # mean точек на дерево
    pts_per_tree_min:  float         = 0.0
    sparse_plots:      list[int]     = field(default_factory=list)  # plot_id < 50 pts/tree
    warning:           Optional[str] = None


def probe_dataset(data_root: Path, cfg) -> DatasetStats:
    """
    Сканирует data_root и возвращает DatasetStats.
    Работает с форматом .npz конвертера (points, gt_boxes).
    """
    data_root = Path(data_root)
    files = sorted(data_root.glob("*.npz"))

    if not files:
        return DatasetStats(
            n_plots=0, n_trees_total=0,
            warning="Нет .npz-файлов в data_root."
        )

    n_trees_list:    list[int]   = []
    pts_list:        list[int]   = []
    heights:         list[float] = []
    pts_per_tree_list: list[float] = []
    sparse_plots:    list[int]   = []

    for f in files:
        try:
            data     = np.load(str(f))
            pts      = data["points"]   if "points"   in data else None
            gt_boxes = data["gt_boxes"] if "gt_boxes" in data else None
        except Exception:
            continue

        n_pts   = len(pts)      if pts      is not None else 0
        n_trees = len(gt_boxes) if gt_boxes is not None else 0
        plot_id = int(f.stem.split("_")[-1]) if "_" in f.stem else 0

        n_trees_list.append(n_trees)
        pts_list.append(n_pts)

        if gt_boxes is not None and len(gt_boxes) > 0:
            hs = gt_boxes[:, 5].tolist()   # h-компонента бокса
            heights.extend(hs)

        ppt = n_pts / max(n_trees, 1)
        pts_per_tree_list.append(ppt)
        if ppt < 50:
            sparse_plots.append(plot_id)

    stats = DatasetStats(
        n_plots          = len(files),
        n_trees_total    = sum(n_trees_list),
        n_trees_per_plot = n_trees_list,
        pts_per_plot     = pts_list,
    )

    if heights:
        stats.height_mean = float(np.mean(heights))
        stats.height_std  = float(np.std(heights))
        stats.height_min  = float(np.min(heights))
        stats.height_max  = float(np.max(heights))

    if pts_per_tree_list:
        stats.pts_per_tree_mean = float(np.mean(pts_per_tree_list))
        stats.pts_per_tree_min  = float(np.min(pts_per_tree_list))
        stats.sparse_plots      = sparse_plots

    return stats
