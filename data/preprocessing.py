"""
Full preprocessing pipeline: LAS → normalized points → CHM → local maxima → GT boxes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlotData:
    """Container for a single forest plot."""

    points: np.ndarray  # (N, 3) normalized [x, y, z]
    local_maxima: np.ndarray  # (K, 3) [x, y, height]
    gt_boxes: np.ndarray  # (M, 6) [x, y, h/2, w, w, h]
    chm: np.ndarray  # 2D CHM array
    chm_x_orig: float
    chm_y_orig: float
    plot_bounds: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    plot_id: int = -1


def crown_size_from_height(
    height: float, slope: float = 0.0512, intercept: float = 1.1048
) -> float:
    """
    Predict crown diameter from tree height via linear regression.

    crown_size = slope * height + intercept
    """
    return slope * height + intercept


def normalize_heights(
    points: np.ndarray,
    dem: np.ndarray,
    dem_x_orig: float,
    dem_y_orig: float,
    dem_resolution: float,
) -> np.ndarray:
    """
    Replace absolute point heights with height above DEM.

    Args:
        points:         (N, 4+) array where col 0=X, 1=Y, 2=Z_abs
        dem:            2D DEM raster
        dem_x_orig:     x origin of DEM
        dem_y_orig:     y origin of DEM
        dem_resolution: DEM cell size in meters

    Returns:
        normalized: (N, 3) [x, y, z_normalized]
    """
    rows_dem, cols_dem = dem.shape
    px = np.floor((points[:, 0] - dem_x_orig) / dem_resolution).astype(int)
    py = np.floor((points[:, 1] - dem_y_orig) / dem_resolution).astype(int)
    px = np.clip(px, 0, cols_dem - 1)
    py = np.clip(py, 0, rows_dem - 1)
    ground_z = dem[py, px]
    z_norm = points[:, 2] - ground_z
    return np.column_stack([points[:, 0], points[:, 1], z_norm]).astype(np.float32)


def build_gt_boxes(
    ref_trees: np.ndarray, slope: float = 0.0512, intercept: float = 1.1048
) -> np.ndarray:
    """
    Build ground-truth bounding boxes from reference tree positions.

    Args:
        ref_trees: (M, 3) [x, y, height]

    Returns:
        gt_boxes: (M, 6) [x, y, h/2, w, w, h]
    """
    n = len(ref_trees)
    gt = np.zeros((n, 6), dtype=np.float32)
    gt[:, 0] = ref_trees[:, 0]
    gt[:, 1] = ref_trees[:, 1]
    gt[:, 5] = ref_trees[:, 2]
    gt[:, 2] = ref_trees[:, 2] / 2.0
    crown = slope * ref_trees[:, 2] + intercept
    gt[:, 3] = crown
    gt[:, 4] = crown
    return gt


def load_las_file(las_path: Path) -> np.ndarray:
    """
    Read .las/.laz file and return (N, 3+) array [X, Y, Z, ...].
    """
    import laspy

    with laspy.open(str(las_path)) as f:
        las = f.read()
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    return np.column_stack([x, y, z]).astype(np.float32)


def load_dem_asc(dem_path: Path) -> Tuple[np.ndarray, float, float, float]:
    """
    Load DEM from ASCII raster (.asc) or GeoTIFF (.tif).

    Returns:
        dem: 2D array
        x_orig, y_orig: lower-left corner coordinates
        resolution: cell size in meters
    """
    suffix = dem_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(str(dem_path)) as src:
                dem = src.read(1).astype(np.float32)
                transform = src.transform
                x_orig = transform.c
                y_orig = transform.f + src.height * transform.e  # lower-left y
                resolution = abs(transform.a)
            return dem, x_orig, y_orig, resolution
        except ImportError:
            pass  # fall through to gdal
    # ASCII grid fallback
    header = {}
    data_start = 0
    with open(dem_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 2 and parts[0].lower() in (
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "cellsize",
            "nodata_value",
        ):
            header[parts[0].lower()] = float(parts[1])
        else:
            data_start = i
            break
    dem = np.array(
        [[float(v) for v in line.split()] for line in lines[data_start:]],
        dtype=np.float32,
    )
    dem = np.flipud(dem)  # ASC stores top row first
    return (
        dem,
        float(header["xllcorner"]),
        float(header["yllcorner"]),
        float(header["cellsize"]),
    )


def run_preprocessing(
    las_path: Path,
    dem_path: Path,
    ref_trees: np.ndarray,
    cfg,
    roi_mask: Optional[np.ndarray] = None,
    plot_id: int = -1,
) -> PlotData:
    """
    Full preprocessing pipeline for one plot.

    Args:
        las_path:  path to .las/.laz
        dem_path:  path to DEM raster
        ref_trees: (M, 3) [x, y, height] reference trees
        cfg:       OmegaConf config object
        roi_mask:  optional boolean mask for ROI filtering

    Returns:
        PlotData with all derived products
    """
    from utils.chm import generate_chm, apply_closing_filter, extract_local_maxima

    logger.info("Loading LAS: %s", las_path)
    raw_points = load_las_file(las_path)

    logger.info("Loading DEM: %s", dem_path)
    dem, dem_x, dem_y, dem_res = load_dem_asc(dem_path)

    logger.info("Normalizing heights")
    pts = normalize_heights(raw_points, dem, dem_x, dem_y, dem_res)

    # Filter low vegetation
    pts = pts[pts[:, 2] >= cfg.preprocessing.min_height]
    logger.info("Points after height filter: %d", len(pts))

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    chm, chm_x, chm_y = generate_chm(
        pts,
        resolution=cfg.preprocessing.chm_resolution,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )
    chm_closed = apply_closing_filter(chm, window=cfg.preprocessing.closing_window)
    local_maxima = extract_local_maxima(
        chm_closed,
        window=cfg.preprocessing.local_maxima_window,
        min_height=cfg.preprocessing.local_maxima_min_height,
        resolution=cfg.preprocessing.chm_resolution,
        x_orig=chm_x,
        y_orig=chm_y,
    )

    gt_boxes = build_gt_boxes(
        ref_trees,
        slope=cfg.crown_regression.slope,
        intercept=cfg.crown_regression.intercept,
    )

    return PlotData(
        points=pts,
        local_maxima=local_maxima,
        gt_boxes=gt_boxes,
        chm=chm_closed,
        chm_x_orig=chm_x,
        chm_y_orig=chm_y,
        plot_bounds=(x_min, y_min, x_max, y_max),
        plot_id=plot_id,
    )
