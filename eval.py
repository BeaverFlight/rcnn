"""
4-fold cross-validation evaluation of TreeRCNN (v1/v2).

Version controlled via cfg.model_version (default: v1).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.newfor_dataset import NewforDataset
from models.build_model import build_model
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


def eval_fold(model, val_ds, device) -> list[PlotMetrics]:
    model.eval()
    results: list[PlotMetrics] = []
    with torch.inference_mode():
        for sample in val_ds:
            points       = sample["points"].to(device)
            gt_boxes     = sample["gt_boxes"].to(device)
            local_maxima = sample["local_maxima"].to(device)
            plot_bounds  = sample["plot_bounds"]
            pid          = sample["plot_id"]

            out      = model(points, gt_boxes, local_maxima, plot_bounds, training=False)
            detected = extract_tree_positions(out["boxes"].cpu().numpy(), points.cpu().numpy())
            ref      = gt_boxes.cpu().numpy()
            ref_xyz  = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])
            pm       = newfor_matching(detected, ref_xyz, plot_id=pid)
            results.append(pm)
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TreeRCNN (v1/v2)")
    parser.add_argument("--config",    default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt_dir",  required=True,
                        help="Директория с fold_0/, fold_1/, ...")
    parser.add_argument("--out_json",  default=None,
                        help="Куда сохранить сводный результат (eval.json)")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Model: %s", device, getattr(cfg, 'model_version', 'v1'))

    folds       = [list(f) for f in cfg.cross_validation.folds]
    all_metrics: list[PlotMetrics] = []
    fold_results: list[dict] = []

    for fi, val_ids in enumerate(folds):
        ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(args.ckpt_dir) / f"fold_{fi}" / "latest.pth"
        if not ckpt_path.exists():
            logger.warning("No checkpoint for fold %d — skip", fi)
            continue

        # ── Фабрика: автоматически v1 или v2 через cfg.model_version ──
        model = build_model(cfg).to(device)
        ckpt  = torch.load(str(ckpt_path), map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:    logger.warning("Missing keys: %d", len(missing))
        if unexpected: logger.warning("Unexpected:  %d", len(unexpected))
        logger.info("Loaded fold %d: %s", fi, ckpt_path)

        val_ds = NewforDataset(
            Path(args.data_root), val_ids, cfg,
            augment_data=False, max_points=cfg.training.max_points,
        )

        fold_metrics = eval_fold(model, val_ds, device)
        all_metrics.extend(fold_metrics)

        gm = compute_global_metrics(fold_metrics)
        n_det = sum(pm.n_test  for pm in fold_metrics)
        n_ref = sum(pm.n_ref   for pm in fold_metrics)
        n_mat = sum(pm.n_match for pm in fold_metrics)
        prec  = n_mat / n_det if n_det > 0 else 0.0
        rec   = n_mat / n_ref if n_ref > 0 else 0.0
        f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        logger.info(
            "Fold %d | F1=%.4f P=%.4f R=%.4f | RMS_match=%.4f h=%.4f v=%.4f",
            fi, f1, prec, rec, gm.rms_ass, gm.rms_h, gm.rms_v,
        )
        fold_results.append({
            "fold": fi, "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "rms_matching": round(gm.rms_ass, 4),
            "rms_h": round(gm.rms_h, 4), "rms_v": round(gm.rms_v, 4),
        })

    if not all_metrics:
        logger.warning("Нет метрик — проверьте чекпоинты и данные.")
        return

    gm = compute_global_metrics(all_metrics)
    n_det = sum(pm.n_test  for pm in all_metrics)
    n_ref = sum(pm.n_ref   for pm in all_metrics)
    n_mat = sum(pm.n_match for pm in all_metrics)
    prec  = n_mat / n_det if n_det > 0 else 0.0
    rec   = n_mat / n_ref if n_ref > 0 else 0.0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

    logger.info("=== GLOBAL (all folds) ===")
    logger.info("  F1            : %.4f", f1)
    logger.info("  Precision     : %.4f", prec)
    logger.info("  Recall        : %.4f", rec)
    logger.info("  RMS_matching  : %.4f", gm.rms_ass)
    logger.info("  RMS_extraction: %.4f", gm.rms_extr)
    logger.info("  RMS_commission: %.4f", gm.rms_com)
    logger.info("  RMS_overall   : %.4f", gm.rms_om)
    logger.info("  RMS_h_error   : %.4f m", gm.rms_h)
    logger.info("  RMS_v_error   : %.4f m", gm.rms_v)

    if args.out_json:
        result = {
            "model_version": str(getattr(cfg, 'model_version', 'v1')),
            "global": {
                "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                "rms_matching": round(gm.rms_ass, 4),
                "rms_h": round(gm.rms_h, 4), "rms_v": round(gm.rms_v, 4),
            },
            "folds": fold_results,
        }
        Path(args.out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Сводные результаты сохранены: %s", args.out_json)


if __name__ == "__main__":
    main()
