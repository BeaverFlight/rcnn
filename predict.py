"""
predict.py — Инференс TreeRCNN (v1/v2) на LAS/LAZ файлах.

Version controlled via cfg.model_version (default: v1).
Все остальные аргументы CLI неизменны.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models.build_model import build_model
from train import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")


def _load_model(cfg, ckpt_path: Path, device: torch.device):
    """Loads v1 or v2 model from checkpoint."""
    model = build_model(cfg).to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:    logger.warning("Missing keys:    %d", len(missing))
    if unexpected: logger.warning("Unexpected keys: %d", len(unexpected))
    logger.info("Loaded checkpoint: %s (epoch %d)", ckpt_path, ckpt.get("epoch", "?"))
    model.eval()
    return model


def predict_file(
    las_path: Path,
    model,
    cfg,
    device: torch.device,
    out_path: Path | None = None,
) -> np.ndarray:
    """
    Инференс на одном LAS/LAZ-файле.
    Возвращает (N, 6) боксов [cx, cy, cz, w, l, h].
    Если out_path задан — сохраняет .npy.
    """
    import laspy
    from utils.preprocessing import (
        normalize_dem,
        compute_chm,
        find_local_maxima,
        get_plot_bounds,
    )

    logger.info("Predicting: %s", las_path)
    las = laspy.read(str(las_path))
    x   = np.array(las.x, dtype=np.float32)
    y   = np.array(las.y, dtype=np.float32)
    z   = np.array(las.z, dtype=np.float32)

    pts_raw = np.column_stack([x, y, z])
    pts_norm, z_ground = normalize_dem(pts_raw, cfg)
    chm    = compute_chm(pts_norm, cfg)
    maxima = find_local_maxima(chm, cfg)
    bounds = get_plot_bounds(pts_norm)

    max_pts = int(cfg.training.get("val_max_points", 200_000))
    if len(pts_norm) > max_pts:
        idx      = np.random.choice(len(pts_norm), max_pts, replace=False)
        pts_norm = pts_norm[idx]

    pts_t    = torch.from_numpy(pts_norm).to(device)
    maxima_t = torch.from_numpy(maxima).to(device)
    dummy_gt = torch.zeros(0, 6, device=device)

    with torch.inference_mode():
        out = model(pts_t, dummy_gt, maxima_t, bounds, training=False)

    boxes = out["boxes"].cpu().numpy()  # (N, 6)
    logger.info("%s → %d деревьев", las_path.name, len(boxes))

    if out_path is not None:
        np.save(str(out_path), boxes)
        logger.info("Сохранено: %s", out_path)

    return boxes


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Predict trees (v1/v2)")
    parser.add_argument("--config",   default="configs/tree_rcnn.yaml")
    parser.add_argument("--ckpt",     required=True,  help="Путь к .pth-чекпоинту")
    parser.add_argument("--input",    required=True,  help="LAS/LAZ файл или директория")
    parser.add_argument("--out_dir",  default="predictions/")
    parser.add_argument("--seed",     type=int, default=None)
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    seed   = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Model: %s", device, getattr(cfg, 'model_version', 'v1'))

    model    = _load_model(cfg, Path(args.ckpt), device)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    las_files  = (
        sorted(input_path.glob("*.las")) + sorted(input_path.glob("*.laz"))
        if input_path.is_dir()
        else [input_path]
    )

    if not las_files:
        logger.warning("Файлы LAS/LAZ не найдены: %s", args.input)
        return

    for las_path in las_files:
        out_path = out_dir / (las_path.stem + "_pred.npy")
        predict_file(las_path, model, cfg, device, out_path=out_path)

    logger.info("Готово. Обработано %d файлов.", len(las_files))


if __name__ == "__main__":
    main()
