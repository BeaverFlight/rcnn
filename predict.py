"""
TreeRCNN inference on a single .las / .laz file.

Supports full large point clouds via sliding-window tiling:
  each tile is processed independently, then results are merged with global NMS.

DEM is optional:
  - If --dem is given, heights are normalised using the external raster.
  - If omitted, a DEM is estimated from classification=2 (ground) points
    inside the LAS file itself via bilinear-interpolated raster.

Outputs (in --out_dir):
  trees.csv       — x, y, height_m, score
  trees.geojson   — GeoJSON FeatureCollection (for QGIS / GIS tools)

Usage:
  # with external DEM
  python predict.py \\
      --config configs/tree_rcnn_auto.yaml \\
      --ckpt   outputs/fold_0/best.pth \\
      --las    data/forest.las \\
      --dem    data/dem.asc

  # DEM from LAS ground points (classification=2)
  python predict.py \\
      --config configs/tree_rcnn_auto.yaml \\
      --ckpt   outputs/fold_0/best.pth \\
      --las    data/forest.las

  # all options
  python predict.py \\
      --config           configs/tree_rcnn_auto.yaml \\
      --ckpt             outputs/fold_0/best.pth \\
      --las              data/forest.las \\
      --out_dir          results/ \\
      --tile_size        100 \\
      --tile_overlap     15 \\
      --score_threshold  0.5 \\
      --max_points_tile  200000 \\
      --dem_resolution   0.5 \\
      --seed             42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.preprocessing import (
    load_las_file,
    load_dem_asc,
    normalize_heights,
)
from models.tree_rcnn import TreeRCNN
from ops.nms3d import nms3d
from train import set_seed
from utils.metrics import extract_tree_positions
from utils.chm import generate_chm, apply_closing_filter, extract_local_maxima

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# DEM from LAS ground points (classification == 2)
# ---------------------------------------------------------------------------

def _dem_from_las_ground(
    las_path: Path,
    resolution: float = 0.5,
) -> tuple[np.ndarray, float, float, float]:
    """
    Build a DEM raster from LAS classification=2 (ground) points.

    Returns:
        dem        — 2D float32 array (rows=y, cols=x)
        x_orig     — left edge of raster
        y_orig     — bottom edge of raster
        resolution — cell size in metres (same as input)
    """
    import laspy

    with laspy.open(str(las_path)) as f:
        las = f.read()

    classification = np.array(las.classification, dtype=np.uint8)
    ground_mask    = classification == 2

    if ground_mask.sum() < 10:
        raise RuntimeError(
            f"Less than 10 ground points (class=2) found in {las_path}. "
            "Provide an external DEM with --dem."
        )

    x = np.array(las.x, dtype=np.float64)[ground_mask]
    y = np.array(las.y, dtype=np.float64)[ground_mask]
    z = np.array(las.z, dtype=np.float64)[ground_mask]

    x_orig = x.min()
    y_orig = y.min()

    cols = int(np.ceil((x.max() - x_orig) / resolution)) + 1
    rows = int(np.ceil((y.max() - y_orig) / resolution)) + 1

    dem_sum = np.zeros((rows, cols), dtype=np.float64)
    dem_cnt = np.zeros((rows, cols), dtype=np.int32)

    ci = np.floor((x - x_orig) / resolution).astype(int)
    ri = np.floor((y - y_orig) / resolution).astype(int)
    ci = np.clip(ci, 0, cols - 1)
    ri = np.clip(ri, 0, rows - 1)

    np.add.at(dem_sum, (ri, ci), z)
    np.add.at(dem_cnt, (ri, ci), 1)

    # Cells with no ground points: fill with nearest neighbour
    dem = np.where(dem_cnt > 0, dem_sum / dem_cnt, np.nan).astype(np.float32)
    nan_mask = np.isnan(dem)
    if nan_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        dem[nan_mask] = dem[idx[0][nan_mask], idx[1][nan_mask]]

    logger.info(
        "DEM from ground points: %d pts → %dx%d raster (res=%.2fm)",
        ground_mask.sum(), rows, cols, resolution,
    )
    return dem, float(x_orig), float(y_orig), float(resolution)


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

@dataclass
class Tile:
    idx:         int
    x_min:       float
    y_min:       float
    x_max:       float
    y_max:       float
    points:      np.ndarray      # (N, 3) height-normalised
    local_maxima: np.ndarray     # (K, 3)


def _build_tiles(
    points: np.ndarray,
    tile_size: float,
    overlap: float,
) -> list[tuple[float, float, float, float]]:
    x_min_g, y_min_g = float(points[:, 0].min()), float(points[:, 1].min())
    x_max_g, y_max_g = float(points[:, 0].max()), float(points[:, 1].max())
    stride = tile_size - overlap
    tiles: list[tuple[float, float, float, float]] = []
    y = y_min_g
    while y < y_max_g:
        x = x_min_g
        while x < x_max_g:
            tiles.append((x, y, x + tile_size, y + tile_size))
            x += stride
        y += stride
    return tiles


def _make_tile(
    idx: int,
    bounds: tuple[float, float, float, float],
    points: np.ndarray,
    cfg,
) -> Tile | None:
    x_min, y_min, x_max, y_max = bounds
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max)
    )
    pts = points[mask]
    if len(pts) < 10:
        return None
    chm, chm_x, chm_y = generate_chm(
        pts,
        resolution=cfg.preprocessing.chm_resolution,
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max,
    )
    chm_closed   = apply_closing_filter(chm, window=cfg.preprocessing.closing_window)
    local_maxima = extract_local_maxima(
        chm_closed,
        window=cfg.preprocessing.local_maxima_window,
        min_height=cfg.preprocessing.local_maxima_min_height,
        resolution=cfg.preprocessing.chm_resolution,
        x_orig=chm_x,
        y_orig=chm_y,
    )
    return Tile(idx=idx, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                points=pts, local_maxima=local_maxima)


# ---------------------------------------------------------------------------
# Per-tile inference
# ---------------------------------------------------------------------------

def _infer_tile(
    tile: Tile,
    model: TreeRCNN,
    device: torch.device,
    score_threshold: float,
    max_points: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pts = tile.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    plot_bounds = (tile.x_min, tile.y_min, tile.x_max, tile.y_max)
    points_t    = torch.from_numpy(pts).float().to(device)
    maxima_t    = torch.from_numpy(tile.local_maxima).float().to(device)
    gt_dummy    = torch.zeros(0, 6, device=device)

    orig_thr = float(model.cfg.stage2_nms.score_threshold)
    model.cfg.stage2_nms.score_threshold = score_threshold

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(points_t, gt_dummy, maxima_t, plot_bounds, training=False)

    model.cfg.stage2_nms.score_threshold = orig_thr

    boxes  = out["boxes"].cpu().float().numpy()
    scores = out["scores"].cpu().float().numpy()
    return boxes, scores


# ---------------------------------------------------------------------------
# Merge tiles + global NMS
# ---------------------------------------------------------------------------

def _merge_tiles(
    all_boxes:  list[np.ndarray],
    all_scores: list[np.ndarray],
    tiles:      list[Tile],
    overlap:    float,
    iouv_thr:   float,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    core_boxes:  list[np.ndarray] = []
    core_scores: list[np.ndarray] = []
    half = overlap / 2.0

    for tile, boxes, scores in zip(tiles, all_boxes, all_scores):
        if len(boxes) == 0:
            continue
        cx, cy = boxes[:, 0], boxes[:, 1]
        in_core = (
            (cx >= tile.x_min + half) & (cx < tile.x_max - half) &
            (cy >= tile.y_min + half) & (cy < tile.y_max - half)
        )
        core_boxes.append(boxes[in_core])
        core_scores.append(scores[in_core])

    if not core_boxes:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_b = np.concatenate(core_boxes,  axis=0)
    all_s = np.concatenate(core_scores, axis=0)
    b_t   = torch.from_numpy(all_b).float().to(device)
    s_t   = torch.from_numpy(all_s).float().to(device)
    keep  = nms3d(b_t, s_t, iouv_thr)
    keep_np = keep.cpu().numpy()
    return all_b[keep_np], all_s[keep_np]


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save_csv(trees: np.ndarray, scores: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "height_m", "score"])
        for (x, y, h), s in zip(trees, scores):
            writer.writerow([
                round(float(x), 3), round(float(y), 3),
                round(float(h), 3), round(float(s), 4),
            ])
    logger.info("Saved %d trees \u2192 %s", len(trees), path)


def _save_geojson(trees: np.ndarray, scores: np.ndarray, path: Path) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(x), 3), round(float(y), 3)],
            },
            "properties": {
                "height_m": round(float(h), 3),
                "score":    round(float(s), 4),
            },
        }
        for (x, y, h), s in zip(trees, scores)
    ]
    path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=2
    ))
    logger.info("Saved GeoJSON \u2192 %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TreeRCNN inference with sliding-window tiling"
    )
    parser.add_argument("--config",           required=True)
    parser.add_argument("--ckpt",             required=True)
    parser.add_argument("--las",              required=True)
    parser.add_argument("--dem",              default=None,
                        help="DEM raster (.asc/.tif). Omit to build DEM from "
                             "LAS classification=2 (ground) points.")
    parser.add_argument("--dem_resolution",   type=float, default=0.5,
                        help="Cell size in metres for auto-DEM [0.5]")
    parser.add_argument("--out_dir",          default="results/")
    parser.add_argument("--tile_size",        type=float, default=120.0)
    parser.add_argument("--tile_overlap",     type=float, default=15.0)
    parser.add_argument("--score_threshold",  type=float, default=None)
    parser.add_argument("--max_points_tile",  type=int,   default=250_000)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--no_amp",           action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    logger.info("Device: %s  |  AMP: %s", device, amp_enabled)

    score_thr = args.score_threshold if args.score_threshold is not None \
                else float(cfg.stage2_nms.score_threshold)
    logger.info("Score threshold: %.2f", score_thr)

    # ── Load model ─────────────────────────────────────────────────────────
    model = TreeRCNN(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing)
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected)

    # If checkpoint contains config, log it for reference
    if "cfg" in ckpt:
        logger.info("Checkpoint trained with config: %s",
                    json.dumps(ckpt["cfg"], ensure_ascii=False))

    model.eval()
    model.set_epoch(9999)  # ensure stage2 is unfrozen
    logger.info("Loaded checkpoint %s  (epoch %s)", args.ckpt, ckpt.get("epoch", "?"))

    # ── Load raw point cloud ───────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("Loading LAS: %s", args.las)
    raw_points = load_las_file(Path(args.las))

    # ── DEM: external file or auto from ground points ──────────────────────
    if args.dem is not None:
        logger.info("Loading DEM from file: %s", args.dem)
        dem, dem_x, dem_y, dem_res = load_dem_asc(Path(args.dem))
    else:
        logger.info("Building DEM from LAS ground points (class=2) …")
        dem, dem_x, dem_y, dem_res = _dem_from_las_ground(
            Path(args.las), resolution=args.dem_resolution
        )

    # ── Height normalisation & filter ───────────────────────────────────
    logger.info("Normalising heights")
    points = normalize_heights(raw_points, dem, dem_x, dem_y, dem_res)
    min_h  = float(cfg.preprocessing.min_height)
    points = points[points[:, 2] >= min_h]
    logger.info("Points after height filter: %d  (%.1fs)",
                len(points), time.perf_counter() - t0)

    # ── Build tiles ─────────────────────────────────────────────────────────
    tile_bounds = _build_tiles(points, args.tile_size, args.tile_overlap)
    logger.info("Tiling: %d tiles  (size=%.0fm  overlap=%.0fm)",
                len(tile_bounds), args.tile_size, args.tile_overlap)

    tiles: list[Tile] = []
    for i, bounds in enumerate(tile_bounds):
        tile = _make_tile(i, bounds, points, cfg)
        if tile is not None:
            tiles.append(tile)
    logger.info("Non-empty tiles: %d / %d", len(tiles), len(tile_bounds))

    # ── Per-tile inference ──────────────────────────────────────────────────
    all_boxes:  list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for tile in tiles:
        t_tile = time.perf_counter()
        logger.info("Tile %d/%d  pts=%d  maxima=%d",
                    tile.idx + 1, len(tiles), len(tile.points), len(tile.local_maxima))
        boxes, scores = _infer_tile(
            tile, model, device, score_thr, args.max_points_tile, amp_enabled
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        logger.info("  \u2192 %d detections  (%.2fs)", len(boxes),
                    time.perf_counter() - t_tile)

    # ── Merge & global NMS ──────────────────────────────────────────────────
    logger.info("Merging tiles with global NMS …")
    iouv_thr = float(cfg.stage2_nms.iouv_threshold)
    final_boxes, final_scores = _merge_tiles(
        all_boxes, all_scores, tiles, args.tile_overlap, iouv_thr, device
    )
    logger.info("After merge + NMS: %d trees  (total %.1fs)",
                len(final_boxes), time.perf_counter() - t0)

    # ── Extract apex XYZ ───────────────────────────────────────────────────
    trees_xyz = (
        extract_tree_positions(final_boxes, points)
        if len(final_boxes) > 0
        else np.zeros((0, 3), dtype=np.float32)
    )

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(trees_xyz, final_scores, out_dir / "trees.csv")
    _save_geojson(trees_xyz, final_scores, out_dir / "trees.geojson")
    logger.info("Done. Detected %d trees.", len(trees_xyz))


if __name__ == "__main__":
    main()
