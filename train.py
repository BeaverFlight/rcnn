"""
TreeRCNN training script with 4-fold cross-validation.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.collate_fn import collate_tree_rcnn
from data.newfor_dataset import NewforDataset
from models.tree_rcnn import TreeRCNN
from utils.metrics import (
    extract_tree_positions,
    newfor_matching,
    compute_global_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: TreeRCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_rms: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_rms": best_rms,
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
    logger.info("Resumed from %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt.get("best_rms", 0.0)


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
        data_root, train_ids, cfg, augment_data=True,
        max_points=cfg.training.max_points,
    )
    val_ds = NewforDataset(
        data_root, val_ids, cfg, augment_data=False,
        max_points=cfg.training.max_points,
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=lr * 1e-3,
    )

    ckpt_dir = out_dir / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_rms = 0.0
    start_epoch = 0

    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        start_epoch, best_rms = load_checkpoint(model, optimizer, resume, device)

    val_interval: int = cfg.training.get("val_interval", 100)
    log_interval: int = cfg.training.get("log_interval", 10)
    max_grad_norm: float = cfg.training.get("max_grad_norm", 1.0)

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_losses: list[float] = []
        nan_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            points       = batch["points"]
            gt_boxes     = batch["gt_boxes"]
            local_maxima = batch["local_maxima"]
            # plot_bounds — список [list[float]] после collate (batch_size=1)
            plot_bounds  = batch["plot_bounds"]

            # Разворачиваем batch=1
            if isinstance(points,       list): points       = points[0]
            if isinstance(gt_boxes,     list): gt_boxes     = gt_boxes[0]
            if isinstance(local_maxima, list): local_maxima = local_maxima[0]
            # plot_bounds – всегда list после collate, берём [0]
            if isinstance(plot_bounds, (list, tuple)):
                plot_bounds = plot_bounds[0]
            # Если по какой-то причине всё же тензор (1,4) — сплющиваем
            if isinstance(plot_bounds, torch.Tensor):
                plot_bounds = plot_bounds.squeeze().tolist()

            points       = points.to(device)
            gt_boxes     = gt_boxes.to(device)
            local_maxima = local_maxima.to(device)
            # plot_bounds — простой list[float], не нуждается в .to(device)

            optimizer.zero_grad()
            loss_dict = model(points, gt_boxes, local_maxima, plot_bounds, training=True)
            total_loss: torch.Tensor = loss_dict["total_loss"]

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batches += 1
                logger.warning(
                    "Epoch %d, batch %d: NaN/Inf loss — batch skipped (total: %d)",
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

        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, best_rms, ckpt_dir / "latest.pth"
            )

        if (epoch + 1) % val_interval == 0 and len(val_ds) > 0:
            rms = _evaluate_fold(model, val_ds, cfg, device)
            logger.info(
                "Fold %d | Epoch %d | RMS_matching=%.4f", fold_idx, epoch + 1, rms
            )
            if rms > best_rms:
                best_rms = rms
                save_checkpoint(
                    model, optimizer, epoch + 1, best_rms, ckpt_dir / "best.pth"
                )


def _evaluate_fold(
    model: TreeRCNN,
    val_ds: NewforDataset,
    cfg,
    device: torch.device,
) -> float:
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for sample in val_ds:
            points       = sample["points"].to(device)
            gt_boxes     = sample["gt_boxes"].to(device)
            local_maxima = sample["local_maxima"].to(device)
            plot_bounds  = sample["plot_bounds"]  # list[float]
            plot_id      = sample["plot_id"]

            out = model(points, gt_boxes, local_maxima, plot_bounds, training=False)
            pred_boxes = out["boxes"].cpu().numpy()
            pts_np = points.cpu().numpy()

            detected = extract_tree_positions(pred_boxes, pts_np)
            ref = gt_boxes.cpu().numpy()
            ref_xyz = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])

            pm = newfor_matching(detected, ref_xyz, plot_id=plot_id)
            all_metrics.append(pm)

    gm = compute_global_metrics(all_metrics)
    return gm.rms_ass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TreeRCNN")
    parser.add_argument("--config", default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="outputs/")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Single fold index to train (0-indexed); default: all folds",
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
