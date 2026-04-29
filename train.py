"""
TreeRCNN training script with 4-fold cross-validation.

Checkpoint policy:
  - every val_interval epochs: outputs/fold_N/epoch_<E>.pth  +  epoch_<E>.json
  - latest.pth  — always the most recent checkpoint (for resume)
  - best.pth    — epoch with highest quality_score (F1-like, NOT pure recall)

quality_score = harmonic_mean(precision, recall)
  precision = matched / detected   (штраф за спам)
  recall    = matched / reference  (штраф за пропуски)
  F1        = 2 * P * R / (P + R)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.collate_fn import collate_tree_rcnn
from data.newfor_dataset import NewforDataset
from models.tree_rcnn import TreeRCNN
from utils.metrics import (
    PlotMetrics,
    extract_tree_positions,
    newfor_matching,
    compute_global_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _f1_score(matched: int, detected: int, reference: int) -> float:
    """Micro-averaged F1 across all plots."""
    if detected == 0 and reference == 0:
        return 0.0
    precision = matched / detected if detected > 0 else 0.0
    recall    = matched / reference if reference > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _quality_score(plot_metrics: list[PlotMetrics]) -> tuple[float, dict]:
    """
    Compute F1-based quality score + full metrics dict for JSON report.

    Returns
    -------
    score : float   — primary metric used to select best.pth
    info  : dict    — all numbers written to the JSON report
    """
    total_detected  = sum(pm.n_test   for pm in plot_metrics)
    total_reference = sum(pm.n_ref    for pm in plot_metrics)
    total_matched   = sum(pm.n_match  for pm in plot_metrics)

    precision = total_matched / total_detected  if total_detected  > 0 else 0.0
    recall    = total_matched / total_reference if total_reference > 0 else 0.0
    f1        = _f1_score(total_matched, total_detected, total_reference)

    gm = compute_global_metrics(plot_metrics)

    info = {
        "total_detected":  total_detected,
        "total_reference": total_reference,
        "total_matched":   total_matched,
        "precision":       round(precision,      4),
        "recall":          round(recall,          4),
        "f1":              round(f1,              4),
        "rms_matching":    round(gm.rms_ass,      4),
        "rms_extraction":  round(gm.rms_extr,     4),
        "rms_commission":  round(gm.rms_com,      4),
        "rms_overall":     round(gm.rms_om,       4),
        "rms_h_error_m":   round(gm.rms_h,        4),
        "rms_v_error_m":   round(gm.rms_v,        4),
        "per_plot": [
            {
                "plot_id":   pm.plot_id,
                "detected":  pm.n_test,
                "reference": pm.n_ref,
                "matched":   pm.n_match,
                "recall":    round(pm.rmr, 4),
                "precision": round(pm.n_match / pm.n_test if pm.n_test > 0 else 0.0, 4),
                "h_mean_m":  round(pm.h_mean, 4),
                "v_mean_m":  round(pm.v_mean, 4),
            }
            for pm in plot_metrics
        ],
    }
    return f1, info


def save_checkpoint(
    model: TreeRCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
        },
        str(path),
    )
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: TreeRCNN,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # поддержка старых чекпоинтов где ключ был best_rms
    score = ckpt.get("best_score", ckpt.get("best_rms", 0.0))
    logger.info("Resumed from %s (epoch %d, best_score=%.4f)", path, ckpt["epoch"], score)
    return ckpt["epoch"], score


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_fold(
    model: TreeRCNN,
    val_ds: NewforDataset,
    cfg,
    device: torch.device,
) -> list[PlotMetrics]:
    model.eval()
    results: list[PlotMetrics] = []

    with torch.no_grad():
        for sample in val_ds:
            points       = sample["points"].to(device)
            gt_boxes     = sample["gt_boxes"].to(device)
            local_maxima = sample["local_maxima"].to(device)
            plot_bounds  = sample["plot_bounds"]   # list[float] — не нужен .to()
            plot_id      = sample["plot_id"]

            out = model(points, gt_boxes, local_maxima, plot_bounds, training=False)
            detected = extract_tree_positions(
                out["boxes"].cpu().numpy(), points.cpu().numpy()
            )
            ref     = gt_boxes.cpu().numpy()
            ref_xyz = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])

            results.append(newfor_matching(detected, ref_xyz, plot_id=plot_id))

    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    train_ids: list[int],
    val_ids: list[int],
    data_root: Path,
    cfg,
    out_dir: Path,
    device: torch.device,
) -> None:
    assert cfg.training.batch_size == 1, (
        "batch_size > 1 is not supported: the current collate/unwrap logic "
        "assumes a single sample per batch."
    )

    logger.info("=== Fold %d | Train: %s | Val: %s ===", fold_idx, train_ids, val_ids)

    train_ds = NewforDataset(
        data_root, train_ids, cfg,
        augment_data=True, max_points=cfg.training.max_points,
    )
    val_ds = NewforDataset(
        data_root, val_ids, cfg,
        augment_data=False, max_points=cfg.training.max_points,
    )

    num_workers: int = cfg.training.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_tree_rcnn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0 and device.type == "cuda"),
    )

    model = TreeRCNN(cfg).to(device)

    lr = cfg.training.get("learning_rate", 1e-3)
    wd = cfg.training.get("weight_decay", 1e-4)

    opt_name: str = cfg.training.get("optimizer", "adam").lower()
    if opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=lr * 1e-3,
    )

    ckpt_dir = out_dir / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score  = 0.0
    start_epoch = 0

    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        start_epoch, best_score = load_checkpoint(model, optimizer, resume, device)

    val_interval:  int   = cfg.training.get("val_interval",  10)
    log_interval:  int   = cfg.training.get("log_interval",  3)
    max_grad_norm: float = cfg.training.get("max_grad_norm", 1.0)
    ckpt_interval: int   = cfg.training.get("checkpoint_interval", 10)

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_losses: list[float] = []
        nan_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            points       = batch["points"]
            gt_boxes     = batch["gt_boxes"]
            local_maxima = batch["local_maxima"]
            plot_bounds  = batch["plot_bounds"]

            if isinstance(points,       list): points       = points[0]
            if isinstance(gt_boxes,     list): gt_boxes     = gt_boxes[0]
            if isinstance(local_maxima, list): local_maxima = local_maxima[0]
            if isinstance(plot_bounds, (list, tuple)):
                plot_bounds = plot_bounds[0]
            if isinstance(plot_bounds, torch.Tensor):
                plot_bounds = plot_bounds.squeeze().tolist()

            points       = points.to(device)
            gt_boxes     = gt_boxes.to(device)
            local_maxima = local_maxima.to(device)

            optimizer.zero_grad()
            loss_dict  = model(points, gt_boxes, local_maxima, plot_bounds, training=True)
            total_loss: torch.Tensor = loss_dict["total_loss"]

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batches += 1
                logger.warning(
                    "Epoch %d, batch %d: NaN/Inf loss — skipped (total: %d)",
                    epoch + 1, batch_idx, nan_batches,
                )
                optimizer.zero_grad()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_losses.append(total_loss.item())

            if (batch_idx + 1) % log_interval == 0:
                recent = np.mean(epoch_losses[-log_interval:])
                logger.info(
                    "Epoch %d | Batch %d | Loss: %.4f | LR: %.2e",
                    epoch + 1, batch_idx + 1, recent,
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()

        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        logger.info(
            "Epoch %d/%d | Loss: %.4f | NaN batches: %d | LR: %.2e",
            epoch + 1, cfg.training.epochs, mean_loss, nan_batches,
            optimizer.param_groups[0]["lr"],
        )

        # ---- сохраняем latest каждые checkpoint_interval эпох ----------
        if (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, best_score,
                ckpt_dir / "latest.pth",
            )

        # ---- валидация ---------------------------------------------------
        if (epoch + 1) % val_interval == 0 and len(val_ds) > 0:
            plot_metrics = _evaluate_fold(model, val_ds, cfg, device)
            score, info  = _quality_score(plot_metrics)

            info["epoch"]      = epoch + 1
            info["fold"]       = fold_idx
            info["train_loss"] = round(float(mean_loss), 6)

            # --- epoch_<E>.pth + epoch_<E>.json (сохраняем ВСЕГДА) ------
            epoch_ckpt = ckpt_dir / f"epoch_{epoch + 1:04d}.pth"
            epoch_json = ckpt_dir / f"epoch_{epoch + 1:04d}.json"

            save_checkpoint(
                model, optimizer, epoch + 1, best_score, epoch_ckpt
            )
            epoch_json.write_text(
                json.dumps(info, indent=2, ensure_ascii=False)
            )

            logger.info(
                "Fold %d | Epoch %d | F1=%.4f | P=%.4f | R=%.4f | "
                "det=%d ref=%d match=%d",
                fold_idx, epoch + 1,
                info["f1"], info["precision"], info["recall"],
                info["total_detected"], info["total_reference"], info["total_matched"],
            )

            # --- best.pth обновляем только если F1 вырос ----------------
            if score > best_score:
                best_score = score
                best_ckpt  = ckpt_dir / "best.pth"
                best_json  = ckpt_dir / "best.json"
                shutil.copy2(epoch_ckpt, best_ckpt)
                shutil.copy2(epoch_json, best_json)
                logger.info(
                    "  ★ New best! F1=%.4f — saved as best.pth", best_score
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TreeRCNN")
    parser.add_argument("--config",    default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir",   default="outputs/")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Single fold index (0-indexed); default: all folds",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[list[int]] = [list(f) for f in cfg.cross_validation.folds]
    all_ids = [pid for fold in folds for pid in fold]

    fold_range = range(len(folds)) if args.fold is None else [args.fold]

    for fi in fold_range:
        val_ids   = folds[fi]
        train_ids = [pid for pid in all_ids if pid not in val_ids]
        train_fold(fi, train_ids, val_ids, data_root, cfg, out_dir, device)


if __name__ == "__main__":
    main()
