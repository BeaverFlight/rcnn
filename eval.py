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
from train import set_seed
from utils.metrics import (
    extract_tree_positions,
    newfor_matching,
    compute_global_metrics,
    PlotMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
            detected = extract_tree_positions(pred, points.cpu().numpy())

            ref = gt_boxes.cpu().numpy()
            ref_xyz = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])

            pm = newfor_matching(detected, ref_xyz, plot_id=pid)
            results.append(pm)

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TreeRCNN (4-fold CV)")
    parser.add_argument("--config", default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Directory containing fold_0/, fold_1/, ... subdirectories",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    folds = [list(f) for f in cfg.cross_validation.folds]
    all_metrics: list[PlotMetrics] = []

    for fi, val_ids in enumerate(folds):
        ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "latest.pth"
        if not ckpt_path.exists():
            logger.warning("No checkpoint for fold %d, skipping", fi)
            continue

        model = TreeRCNN(cfg).to(device)
        # map_location обеспечивает корректную загрузку при смене GPU/CPU
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

        fold_metrics = eval_fold(model, val_ds, device)
        all_metrics.extend(fold_metrics)

        fold_global = compute_global_metrics(fold_metrics)
        logger.info(
            "Fold %d | RMS_matching=%.4f RMS_extr=%.4f RMS_com=%.4f",
            fi,
            fold_global.rms_ass,
            fold_global.rms_extr,
            fold_global.rms_com,
        )

    if all_metrics:
        gm = compute_global_metrics(all_metrics)
        logger.info("=== GLOBAL METRICS ===")
        logger.info("  RMS_matching  : %.4f", gm.rms_ass)
        logger.info("  RMS_extraction: %.4f", gm.rms_extr)
        logger.info("  RMS_commission: %.4f", gm.rms_com)
        logger.info("  RMS_overall   : %.4f", gm.rms_om)
        logger.info("  RMS_h_error   : %.4f m", gm.rms_h)
        logger.info("  RMS_v_error   : %.4f m", gm.rms_v)
    else:
        logger.warning("No metrics collected — check checkpoints and data.")


if __name__ == "__main__":
    main()
