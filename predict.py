"""
TreeRCNN inference on a single .las file.

Outputs:
  <out_dir>/trees.csv     — x, y, height, score for every detected tree
  <out_dir>/trees.geojson — GeoJSON FeatureCollection (same data)

Usage example:
  python predict.py \\
      --config  configs/tree_rcnn_auto.yaml \\
      --ckpt    outputs/fold_0/best.pth \\
      --las     /data/forest_plot.las \\
      --out_dir results/

Optional:
  --score_threshold 0.5   override stage2_nms.score_threshold from config
  --chunk_size 64         proposals per Stage-2 batch (lower = less VRAM)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.newfor_dataset import NewforDataset
from models.tree_rcnn import TreeRCNN, _subsample_points_in_box
from train import set_seed
from utils.metrics import extract_tree_positions
from ops.nms3d import nms3d
from utils.box_coder import decode_boxes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# Stage-2 chunked inference (без OOM)
# ---------------------------------------------------------------------------

def _stage2_chunked(
    model: TreeRCNN,
    points: torch.Tensor,
    proposals: torch.Tensor,
    score_threshold: float,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Прогоняем Stage-2 чанками чтобы не получить OOM."""
    cfg_nms = model.cfg.stage2_nms
    iouv_thr = float(cfg_nms.iouv_threshold)

    all_scores:  list[torch.Tensor] = []
    all_refined: list[torch.Tensor] = []

    for start in range(0, len(proposals), chunk_size):
        chunk = proposals[start: start + chunk_size]
        pts_list = [_subsample_points_in_box(points, p) for p in chunk]

        with torch.no_grad():
            cls_logits, reg_deltas = model.stage2(pts_list, chunk)

        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, chunk)
        all_scores.append(scores)
        all_refined.append(refined)

        del cls_logits, reg_deltas, pts_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores  = torch.cat(all_scores,  dim=0)
    refined = torch.cat(all_refined, dim=0)

    mask = scores >= score_threshold
    refined, scores = refined[mask], scores[mask]
    if len(refined) == 0:
        return refined, scores

    keep = nms3d(refined, scores, iouv_thr)
    return refined[keep], scores[keep]


# ---------------------------------------------------------------------------
# Load raw .las and preprocess
# ---------------------------------------------------------------------------

def _load_las_points(las_path: Path, cfg, max_points: int) -> tuple[np.ndarray, tuple]:
    """
    Читаем .las, возвращаем:
      points     — (N, 3) float32  [x, y, z_normalized]
      plot_bounds — (xmin, ymin, xmax, ymax)
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy not installed: pip install laspy")

    las = laspy.read(str(las_path))
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()

    # нормализация: вычитаем z_ground через 0.5-перцентиль
    z_ground = float(np.percentile(z, 0.5))
    z_norm   = z - z_ground

    # фильтруем низкие точки
    min_h = float(cfg.preprocessing.min_height)
    mask  = z_norm >= min_h
    x, y, z_norm = x[mask], y[mask], z_norm[mask]

    points = np.stack([x, y, z_norm], axis=1).astype(np.float32)

    # subsampling если точек слишком много
    if len(points) > max_points:
        idx    = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    logger.info(
        "Loaded %s: %d points after filter (bounds %.1f×%.1f m)",
        las_path.name, len(points), xmax - xmin, ymax - ymin,
    )
    return points, (xmin, ymin, xmax, ymax)


def _get_local_maxima(points: np.ndarray, cfg) -> np.ndarray:
    """Ищем локальные максимумы CHM через scipy."""
    try:
        from scipy.ndimage import maximum_filter, label
    except ImportError:
        raise ImportError("scipy not installed: pip install scipy")

    res  = float(cfg.preprocessing.chm_resolution)
    minh = float(cfg.preprocessing.local_maxima_min_height)
    win  = int(cfg.preprocessing.local_maxima_window)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    cols = int((xmax - xmin) / res) + 1
    rows = int((ymax - ymin) / res) + 1
    chm  = np.zeros((rows, cols), dtype=np.float32)

    ci = ((x - xmin) / res).astype(int)
    ri = ((y - ymin) / res).astype(int)
    np.maximum.at(chm, (ri, ci), z)

    local_max = maximum_filter(chm, size=win) == chm
    high_mask = chm >= minh
    peaks     = local_max & high_mask

    ri_peaks, ci_peaks = np.where(peaks)
    if len(ri_peaks) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    px = ci_peaks * res + xmin
    py = ri_peaks * res + ymin
    pz = chm[ri_peaks, ci_peaks]

    maxima = np.stack([px, py, pz], axis=1).astype(np.float32)
    logger.info("Found %d local maxima", len(maxima))
    return maxima


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save_csv(trees: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Сохраняем x, y, height, score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "height_m", "score"])
        for (x, y, h), s in zip(trees, scores):
            writer.writerow([round(float(x), 3), round(float(y), 3),
                             round(float(h), 3), round(float(s), 4)])
    logger.info("Saved %d trees → %s", len(trees), path)


def _save_geojson(trees: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Сохраняем GeoJSON FeatureCollection с точками-вершинами."""
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
    geojson = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(geojson, indent=2))
    logger.info("Saved GeoJSON → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TreeRCNN inference on .las file")
    parser.add_argument("--config",          required=True,  help="Path to config YAML")
    parser.add_argument("--ckpt",            required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--las",             required=True,  help="Input .las file")
    parser.add_argument("--out_dir",         default="results", help="Output directory")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="Override stage2_nms.score_threshold")
    parser.add_argument("--chunk_size",      type=int,   default=64,
                        help="Proposals per Stage-2 batch (lower = less VRAM)")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # оверрайд порога если задан
    score_thr = args.score_threshold if args.score_threshold is not None \
                else float(cfg.stage2_nms.score_threshold)
    logger.info("Score threshold: %.2f", score_thr)

    # --- загрузка модели ---
    model = TreeRCNN(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint: %s (epoch %d)", args.ckpt, ckpt.get("epoch", "?"))

    # --- загрузка облака точек ---
    las_path  = Path(args.las)
    max_pts   = int(cfg.training.get("max_points", 150_000))
    points_np, plot_bounds = _load_las_points(las_path, cfg, max_pts)
    local_maxima_np        = _get_local_maxima(points_np, cfg)

    points_t       = torch.from_numpy(points_np).to(device)
    local_maxima_t = torch.from_numpy(local_maxima_np).to(device)

    # --- Stage-1: proposals ---
    with torch.no_grad():
        ad, al_list = model.anchor_gen.generate_all(
            plot_bounds, local_maxima_t.cpu().numpy()
        )
        ad = ad.to(device)
        al = [a.to(device) for a in al_list]
        proposals = model._stage1_proposals(points_t, ad, al, device)

    logger.info("Stage-1 proposals: %d", len(proposals))

    if len(proposals) == 0:
        logger.warning("No proposals from Stage-1 — output will be empty")
        trees_xyz = np.zeros((0, 3), dtype=np.float32)
        scores_np  = np.zeros(0, dtype=np.float32)
    else:
        # --- Stage-2: refinement (чанками) ---
        final_boxes, final_scores = _stage2_chunked(
            model, points_t, proposals, score_thr, chunk_size=args.chunk_size
        )
        logger.info("After Stage-2 + NMS: %d trees", len(final_boxes))

        trees_xyz = extract_tree_positions(
            final_boxes.cpu().numpy(), points_np
        )
        scores_np = final_scores.cpu().numpy()

    # --- сохраняем ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(trees_xyz, scores_np, out_dir / "trees.csv")
    _save_geojson(trees_xyz, scores_np, out_dir / "trees.geojson")

    logger.info("Done. Detected %d trees.", len(trees_xyz))


if __name__ == "__main__":
    main()
