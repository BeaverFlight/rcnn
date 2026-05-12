"""
utils/rich_features.py — Предрасчёт обогащённых геометрических признаков

Фикс: добавлен warning если Open3D не установлен —
в этом случае verticality будет нулём для всех точек, что бесполезно для сети.
Установка: pip install open3d
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning(
        "open3d не установлен. Признаки нормалей и verticality будут нулевыми — "
        "это сильно снизит полезность богатого вектора. Установите: pip install open3d"
    )


def compute_rich_features(
    points_xyz: np.ndarray,
    k_neighbors: int = 20,
    intensity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Вычисляет обогащённые признаки для облака точек.

    Args:
        points_xyz  : (N, 3) — координаты после нормализации DEM
        k_neighbors : число соседей для расчёта нормалей
        intensity   : (N,) — интенсивность LiDAR (опционально)

    Returns:
        (N, 9): [x, y, z, nx, ny, nz, verticality, z_norm, intensity_norm]
    """
    N = len(points_xyz)

    if HAS_OPEN3D and N > k_neighbors:
        normals = _compute_normals_open3d(points_xyz, k_neighbors)
    else:
        if not HAS_OPEN3D:
            pass  # warning уже выведен при импорте
        normals = np.zeros((N, 3), dtype=np.float32)
        normals[:, 2] = 1.0

    # verticality: 0=горизонталь (земля), 1=вертикаль (ствол)
    verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)

    z = points_xyz[:, 2].astype(np.float32)
    z_norm = z / (z.max() + 1e-6)

    if intensity is not None:
        i_min, i_max = intensity.min(), intensity.max() + 1e-6
        intensity_norm = ((intensity - i_min) / (i_max - i_min)).astype(np.float32)
    else:
        intensity_norm = np.zeros(N, dtype=np.float32)

    return np.column_stack([
        points_xyz.astype(np.float32),
        normals.astype(np.float32),
        verticality,
        z_norm,
        intensity_norm,
    ]).astype(np.float32)


def _compute_normals_open3d(points_xyz: np.ndarray, k: int) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    return np.asarray(pcd.normals, dtype=np.float32)
