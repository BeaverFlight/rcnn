"""
TreeRCNN inference on a single .las / .laz file.

Supports full large point clouds via sliding-window tiling:
  each tile is processed independently, then results are merged with global NMS.

Outputs (in --out_dir):
  trees.csv       — x, y, height_m, score, tile_id
  trees.geojson   — GeoJSON FeatureCollection (same data, for QGIS / GIS tools)

Usage:
  # minimal — DEM required for proper height normalisation
  python predict.py \\
      --config  configs/tree_rcnn_auto.yaml \\
      --ckpt    outputs/fold_0/best.pth \\
      --las     data/forest.las \\
      --dem     data/dem.asc

  # with all options
  python predict.py \\
      --config           configs/tree_rcnn_auto.yaml \\
      --ckpt             outputs/fold_0/best.pth \\
      --las              data/forest.las \\
      --dem              data/dem.asc \\
      --out_dir          results/ \\
      --tile_size        100 \\
      --tile_overlap     15 \\
      --score_threshold  0.5 \\
      --max_points_tile  200000 \\
      --seed             42

CLI arguments:
  --config           Path to OmegaConf YAML config (required)
  --ckpt             Path to .pth checkpoint (required)
  --las              Input .las / .laz file (required)
  --dem              Path to DEM raster .asc / .tif (required for accurate heights)
  --out_dir          Output directory [default: results/]
  --tile_size        Tile side length in metres [default: 120]
  --tile_overlap     Overlap between adjacent tiles in metres [default: 15]
  --score_threshold  Override stage2_nms.score_threshold from config
  --max_points_tile  Max points per tile sent to the model [default: 250000]
  --seed             Random seed [default: 42]
  --no_amp           Disable AMP even on CUDA
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
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
# Tile helpers
# ---------------------------------------------------------------------------

@dataclass
class Tile:
    idx:    int
    x_min:  float
    y_min:  float
    x_max:  float
    y_max:  float
    points: np.ndarray          # (N, 3) already height-normalised
    local_maxima: np.ndarray    # (K, 3)


def _build_tiles(
    points: np.ndarray,
    tile_size: float,
    overlap: float,
) -> list[tuple[float, float, float, float]]:
    """Return list of (x_min, y_min, x_max, y_max) tile bounds with overlap."""
    x_min_g, y_min_g = points[:, 0].min(), points[:, 1].min()
    x_max_g, y_max_g = points[:, 0].max(), points[:, 1].max()

    stride = tile_size - overlap
    tiles  = []
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
    """Extract points for one tile and compute local maxima."""
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
    """
    Run full TreeRCNN inference on one tile.
    Returns (boxes_np [K,6], scores_np [K]).
    """
    pts = tile.points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    plot_bounds = (tile.x_min, tile.y_min, tile.x_max, tile.y_max)

    points_t  = torch.from_numpy(pts).float().to(device)
    maxima_t  = torch.from_numpy(tile.local_maxima).float().to(device)
    gt_dummy  = torch.zeros(0, 6, device=device)

    # Override score threshold for this run
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
# Merge tiles: keep only detections in the non-overlapping "core" of each tile,
# then run global NMS to clean up boundary duplicates.
# ---------------------------------------------------------------------------

def _merge_tiles(
    all_boxes:  list[np.ndarray],
    all_scores: list[np.ndarray],
    tiles:      list[Tile],
    overlap:    float,
    iouv_thr:   float,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Filter each tile's detections to its core (inset by overlap/2).
    2. Concatenate and run global NMS.
    """
    core_boxes:  list[np.ndarray] = []
    core_scores: list[np.ndarray] = []
    half = overlap / 2.0

    for tile, boxes, scores in zip(tiles, all_boxes, all_scores):
        if len(boxes) == 0:
            continue
        cx = boxes[:, 0]
        cy = boxes[:, 1]
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

    b_t = torch.from_numpy(all_b).float().to(device)
    s_t = torch.from_numpy(all_s).float().to(device)
    keep = nms3d(b_t, s_t, iouv_thr)
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
    logger.info("Saved %d trees → %s", len(trees), path)


def _save_geojson(trees: np.ndarray, scores: np.ndarray, path: Path) -> None:
    features = []
    for (x, y, h), s in zip(trees, scores):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(x), 3), round(float(y), 3)],
            },
            "properties": {
                "height_m": round(float(h), 3),
                "score":    round(float(s), 4),
            },
        })
    path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        indent=2,
    ))
    logger.info("Saved GeoJSON → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TreeRCNN inference on .las/.laz file with sliding-window tiling"
    )
    parser.add_argument("--config",           required=True)
    parser.add_argument("--ckpt",             required=True)
    parser.add_argument("--las",              required=True)
    parser.add_argument("--dem",              required=True,
                        help="Path to DEM raster (.asc or .tif)")
    parser.add_argument("--out_dir",          default="results/")
    parser.add_argument("--tile_size",        type=float, default=120.0,
                        help="Tile side length in metres [120]")
    parser.add_argument("--tile_overlap",     type=float, default=15.0,
                        help="Overlap between tiles in metres [15]")
    parser.add_argument("--score_threshold",  type=float, default=None,
                        help="Override stage2_nms.score_threshold")
    parser.add_argument("--max_points_tile",  type=int,   default=250_000,
                        help="Max points per tile sent to model [250000]")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--no_amp",           action="store_true",
                        help="Disable AMP (autocast) even on CUDA")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(args.seed)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    logger.info("Device: %s  |  AMP: %s", device, amp_enabled)

    score_thr = args.score_threshold if args.score_threshold is not None \
                else float(cfg.stage2_nms.score_threshold)
    logger.info("Score threshold: %.2f", score_thr)

    # ── Load model ──────────────────────────────────────────────────────────
    model = TreeRCNN(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing)
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected)

    model.eval()
    model.set_epoch(9999)   # ensure stage2 is unfrozen
    logger.info("Loaded checkpoint %s  (epoch %s)", args.ckpt, ckpt.get("epoch", "?"))

    # ── Load & preprocess full point cloud ──────────────────────────────────
    t0 = time.perf_counter()
    logger.info("Loading LAS: %s", args.las)
    raw_points = load_las_file(Path(args.las))

    logger.info("Loading DEM: %s", args.dem)
    dem, dem_x, dem_y, dem_res = load_dem_asc(Path(args.dem))

    logger.info("Normalising heights")
    points = normalize_heights(raw_points, dem, dem_x, dem_y, dem_res)

    # Filter low vegetation
    min_h  = float(cfg.preprocessing.min_height)
    points = points[points[:, 2] >= min_h]
    logger.info("Points after height filter: %d  (%.1fs)", len(points),
                time.perf_counter() - t0)

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
        logger.info("  → %d detections  (%.2fs)", len(boxes),
                    time.perf_counter() - t_tile)

    # ── Merge & global NMS ──────────────────────────────────────────────────
    logger.info("Merging tiles with global NMS …")
    iouv_thr   = float(cfg.stage2_nms.iouv_threshold)
    final_boxes, final_scores = _merge_tiles(
        all_boxes, all_scores, tiles, args.tile_overlap, iouv_thr, device
    )
    logger.info("After merge + NMS: %d trees  (total %.1fs)",
                len(final_boxes), time.perf_counter() - t0)

    # ── Extract apex XYZ ────────────────────────────────────────────────────
    if len(final_boxes) > 0:
        trees_xyz = extract_tree_positions(final_boxes, points)
    else:
        trees_xyz = np.zeros((0, 3), dtype=np.float32)

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(trees_xyz, final_scores, out_dir / "trees.csv")
    _save_geojson(trees_xyz, final_scores, out_dir / "trees.geojson")
    logger.info("Done. Detected %d trees.", len(trees_xyz))


if __name__ == "__main__":
    main()
