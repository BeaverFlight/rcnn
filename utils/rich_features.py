"""
utils/rich_features.py — Предрасчёт обогащённых геометрических признаков

Добавляет к точкам xyz:
  - нормали (nx, ny, nz)
  - verticality (0=горизонталь/земля, 1=вертикаль/ствол)
  - z_norm (нормализованная высота над землёй)
  - intensity_norm (нормализованная интенсивность, если есть)

Итого: (N, 9) вместо (N, 3)

Пример использования в convert_dataset.py:
    from utils.rich_features import compute_rich_features
    points_9d = compute_rich_features(points_xyz, k_neighbors=20)
"""
from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def compute_rich_features(
    points_xyz: np.ndarray,
    k_neighbors: int = 20,
    intensity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Вычисляет обогащённые признаки для облака точек.

    Args:
        points_xyz   : (N, 3) — координаты точек после нормализации DEM
        k_neighbors  : число соседей для расчёта нормалей
        intensity    : (N,) — интенсивность возврата LiDAR (опционально)

    Returns:
        (N, 9) — [x, y, z, nx, ny, nz, verticality, z_norm, intensity_norm]
    """
    N = len(points_xyz)

    if HAS_OPEN3D and N > k_neighbors:
        normals = _compute_normals_open3d(points_xyz, k_neighbors)
    else:
        # Fallback: нулевые нормали (сеть всё равно выучит из xyz)
        normals = np.zeros((N, 3), dtype=np.float32)
        normals[:, 2] = 1.0  # ориентация вверх по умолчанию

    # Verticality: 0 = горизонтальная поверхность (земля)
    #              1 = вертикальная поверхность (ствол)
    verticality = (1.0 - np.abs(normals[:, 2])).astype(np.float32)

    # Нормализованная высота над землёй [0..1]
    z = points_xyz[:, 2].astype(np.float32)
    z_max = z.max() + 1e-6
    z_norm = z / z_max

    # Нормализованная интенсивность
    if intensity is not None:
        i_min, i_max = intensity.min(), intensity.max() + 1e-6
        intensity_norm = ((intensity - i_min) / (i_max - i_min)).astype(np.float32)
    else:
        intensity_norm = np.zeros(N, dtype=np.float32)

    return np.column_stack([
        points_xyz.astype(np.float32),   # x, y, z       (3)
        normals.astype(np.float32),       # nx, ny, nz    (3)
        verticality,                      # verticality   (1)
        z_norm,                           # z_norm        (1)
        intensity_norm,                   # intensity     (1)
    ]).astype(np.float32)                 # → (N, 9)


def _compute_normals_open3d(points_xyz: np.ndarray, k: int) -> np.ndarray:
    """Вычисляет нормали через Open3D KNN PCA."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    # Ориентируем нормали вверх (nz > 0)
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    return np.asarray(pcd.normals, dtype=np.float32)
