"""
4-fold cross-validation evaluation of TreeRCNN.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.newfor_dataset import NewforDataset
from models.tree_rcnn import TreeRCNN
from train import _extract_tree_positions, set_seed
from utils.metrics import newfor_matching, compute_global_metrics, PlotMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("eval")


def eval_fold(
    model: TreeRCNN,
    val_ds: NewforDataset,
    device: torch.device,
) -> list[PlotMetrics]:
    model.eval()
    results = []

    with torch.no_grad():
        for sample in val_ds:
            points = sample["points"].to(device)
            gt_boxes = sample["gt_boxes"].to(device)
            local_maxima = sample["local_maxima"].to(device)
            plot_bounds = sample["plot_bounds"].to(device)
            pid = sample["plot_id"]

            out = model(points, gt_boxes, local_maxima, plot_bounds, training=False)
            pred = out["boxes"].cpu().numpy()
            detected = _extract_tree_positions(pred, points.cpu().numpy())

            ref = gt_boxes.cpu().numpy()
            ref_xyz = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])

            pm = newfor_matching(detected, ref_xyz, plot_id=pid)
            results.append(pm)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--ckpt_dir", required=True, help="Directory containing fold_X/ subdirectories"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = [list(f) for f in cfg.cross_validation.folds]
    all_ids = [pid for fold in folds for pid in fold]
    all_metrics: list[PlotMetrics] = []

    for fi, val_ids in enumerate(folds):
        ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "latest.pth"
        if not ckpt_path.exists():
            logger.warning("No checkpoint for fold %d, skipping", fi)
            continue

        model = TreeRCNN(cfg).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded fold %d checkpoint: %s", fi, ckpt_path)

        val_ds = NewforDataset(
            Path(args.data_root),
            val_ids,
            cfg,
            augment_data=False,
            max_points=cfg.training.max_points,
        )
