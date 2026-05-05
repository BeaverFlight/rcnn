"""
TreeRCNN training script with 4-fold cross-validation.

Checkpoint policy:
  - every val_interval epochs: outputs/fold_N/epoch_<E>.pth  +  epoch_<E>.json
  - latest.pth  — always the most recent checkpoint (for resume)
  - best.pth    — epoch with highest quality_score (F1-like)

Performance additions:
  - AMP: torch.autocast + GradScaler, enabled when device=cuda.
    Controlled by cfg.training.amp (default True for cuda).
  - GradScaler handles fp16 underflow; disabled on CPU automatically.
  - Checkpoint saves/restores scaler state for seamless resume.
  - torch.set_num_threads / set_num_interop_threads configured at startup
    so that OpenMP-backed CPU ops (anchor generation, assign_targets)
    use multiple cores without GIL contention.

JSON output per val epoch now includes full training config under key 'config'.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
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


def _configure_cpu_threads(cfg) -> None:
    """
    Set PyTorch intra/inter-op thread counts for CPU operations.

    - intra_op  (set_num_threads):        controls OpenMP parallelism inside
      a single op, e.g. matrix multiply, anchor generation numpy loops.
    - inter_op  (set_num_interop_threads): controls how many independent ops
      PyTorch may run in parallel (useful when multiple CPU ops are queued).

    Defaults: 6 intra / 2 inter — leaves 2 physical cores free for the
    DataLoader workers and OS, avoiding contention on a typical 8-core CPU.
    Override via cfg.training.cpu_threads / cfg.training.cpu_interop_threads.
    """
    intra = int(cfg.training.get("cpu_threads",        6))
    inter = int(cfg.training.get("cpu_interop_threads", 2))
    torch.set_num_threads(intra)
    torch.set_num_interop_threads(inter)
    logger.info("CPU threads: intra_op=%d  inter_op=%d", intra, inter)


def _log_vram(label: str) -> None:
    """Log current and peak VRAM usage. No-op if CUDA is not available."""
    if not torch.cuda.is_available():
        return
    alloc   = torch.cuda.memory_allocated()  / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(
        "VRAM [%s]: alloc=%.2f GB  reserved=%.2f GB  total=%.2f GB  (alloc %.0f%%)",
        label, alloc, reserved, total, alloc / total * 100,
    )


def _f1_score(matched: int, detected: int, reference: int) -> float:
    if detected == 0 and reference == 0:
        return 0.0
    precision = matched / detected  if detected  > 0 else 0.0
    recall    = matched / reference if reference > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _quality_score(plot_metrics: list[PlotMetrics]) -> tuple[float, dict]:
    total_detected  = sum(pm.n_test  for pm in plot_metrics)
    total_reference = sum(pm.n_ref   for pm in plot_metrics)
    total_matched   = sum(pm.n_match for pm in plot_metrics)

    precision = total_matched / total_detected  if total_detected  > 0 else 0.0
    recall    = total_matched / total_reference if total_reference > 0 else 0.0
    f1        = _f1_score(total_matched, total_detected, total_reference)
    gm        = compute_global_metrics(plot_metrics)

    info = {
        "total_detected":  total_detected,
        "total_reference": total_reference,
        "total_matched":   total_matched,
        "precision":       round(precision,   4),
        "recall":          round(recall,       4),
        "f1":              round(f1,           4),
        "rms_matching":    round(gm.rms_ass,   4),
        "rms_extraction":  round(gm.rms_extr,  4),
        "rms_commission":  round(gm.rms_com,   4),
        "rms_overall":     round(gm.rms_om,    4),
        "rms_h_error_m":   round(gm.rms_h,     4),
        "rms_v_error_m":   round(gm.rms_v,     4),
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
    cfg=None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score":           best_score,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if cfg is not None:
        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    torch.save(payload, str(path))
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: TreeRCNN,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(str(path), map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning("load_checkpoint: %d missing keys: %s", len(missing), missing)
    if unexpected:
        logger.warning("load_checkpoint: %d unexpected keys: %s", len(unexpected), unexpected)

    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except (ValueError, KeyError) as exc:
        logger.warning("load_checkpoint: optimizer incompatible (%s) — fresh start.", exc)

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

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

    with torch.inference_mode():
        for sample in val_ds:
            points       = sample["points"].to(device)
            gt_boxes     = sample["gt_boxes"].to(device)
            local_maxima = sample["local_maxima"].to(device)
            plot_bounds  = sample["plot_bounds"]
            plot_id      = sample["plot_id"]

            logger.info("Val plot %d: %d points, %d GT trees — inference...",
                        plot_id, len(points), len(gt_boxes))
            t_plot = time.perf_counter()
            out    = model(points, gt_boxes, local_maxima, plot_bounds, training=False)
            logger.info("Val plot %d: %.2fs, boxes=%d",
                        plot_id, time.perf_counter() - t_plot, len(out["boxes"]))

            detected = extract_tree_positions(
                out["boxes"].cpu().numpy(), points.cpu().numpy()
            )
            ref     = gt_boxes.cpu().numpy()
            ref_xyz = np.column_stack([ref[:, 0], ref[:, 1], ref[:, 5]])

            pm = newfor_matching(detected, ref_xyz, plot_id=plot_id)
            logger.info(
                "Val plot %d: det=%d ref=%d match=%d recall=%.1f%% prec=%.1f%%",
                plot_id, pm.n_test, pm.n_ref, pm.n_match,
                pm.rmr * 100,
                (pm.n_match / pm.n_test * 100) if pm.n_test > 0 else 0.0,
            )
            results.append(pm)

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
    assert cfg.training.batch_size == 1, "batch_size > 1 is not supported."

    logger.info("=== Fold %d | Train: %s | Val: %s ===", fold_idx, train_ids, val_ids)

    train_ds = NewforDataset(data_root, train_ids, cfg, augment_data=True,
                             max_points=cfg.training.max_points)
    val_max_points: int = cfg.training.get("val_max_points", 200_000)
    val_ds = NewforDataset(data_root, val_ids, cfg, augment_data=False,
                           max_points=val_max_points)
    logger.info("Val dataset: %d plots, val_max_points=%d", len(val_ds), val_max_points)

    num_workers: int = cfg.training.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        collate_fn=collate_tree_rcnn, num_workers=num_workers,
        pin_memory=(num_workers > 0 and device.type == "cuda"),
    )

    model = TreeRCNN(cfg).to(device)

    lr       = cfg.training.get("learning_rate", 1e-3)
    wd       = cfg.training.get("weight_decay",  1e-4)
    opt_name = cfg.training.get("optimizer", "adam").lower()

    if opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=lr * 0.1,
    )

    use_amp: bool = (device.type == "cuda" and bool(cfg.training.get("amp", True)))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if use_amp:
        logger.info("AMP enabled (autocast + GradScaler).")
    else:
        logger.info("AMP disabled (CPU or cfg.training.amp=False).")

    ckpt_dir = out_dir / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score  = 0.0
    start_epoch = 0
    resume = ckpt_dir / "latest.pth"
    if resume.exists():
        start_epoch, best_score = load_checkpoint(model, optimizer, resume, device, scaler=scaler)

    val_interval:  int   = cfg.training.get("val_interval",  10)
    log_interval:  int   = cfg.training.get("log_interval",  3)
    max_grad_norm: float = cfg.training.get("max_grad_norm", 1.0)
    ckpt_interval: int   = cfg.training.get("checkpoint_interval", 10)
    freeze_s2_epochs: int = cfg.training.get("freeze_stage2_epochs", 0)
    if freeze_s2_epochs > 0:
        logger.info("Two-phase: Stage-2 frozen for first %d epochs.", freeze_s2_epochs)

    # Serialise cfg once — reused in every JSON write
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for epoch in range(start_epoch, cfg.training.epochs):
        model.set_epoch(epoch)
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
            if isinstance(plot_bounds, (list, tuple)): plot_bounds = plot_bounds[0]
            if isinstance(plot_bounds, torch.Tensor):  plot_bounds = plot_bounds.squeeze().tolist()

            points       = points.to(device, non_blocking=True)
            gt_boxes     = gt_boxes.to(device, non_blocking=True)
            local_maxima = local_maxima.to(device, non_blocking=True)

            # --- диагностика VRAM и размера сэмпла перед forward ---
            logger.info(
                "Batch e%d/b%d: points=%d  gt_boxes=%d",
                epoch + 1, batch_idx + 1,
                points.shape[0], gt_boxes.shape[0],
            )
            _log_vram(f"e{epoch+1}/b{batch_idx+1} pre-forward")

            optimizer.zero_grad(set_to_none=True)

            loss_dict  = model(points, gt_boxes, local_maxima, plot_bounds, training=True)
            total_loss = loss_dict["total_loss"]

            _log_vram(f"e{epoch+1}/b{batch_idx+1} post-forward")

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batches += 1
                logger.warning("Epoch %d, batch %d: NaN/Inf loss — skipped (%d total)",
                                epoch + 1, batch_idx, nan_batches)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(total_loss.item())

            if (batch_idx + 1) % log_interval == 0:
                recent = np.mean(epoch_losses[-log_interval:])
                logger.info("Epoch %d | Batch %d | Loss: %.4f | LR: %.2e | scale: %.0f",
                            epoch + 1, batch_idx + 1, recent,
                            optimizer.param_groups[0]["lr"], scaler.get_scale())

        scheduler.step()
        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        logger.info("Epoch %d/%d | Loss: %.4f | NaN: %d | LR: %.2e",
                    epoch + 1, cfg.training.epochs, mean_loss, nan_batches,
                    optimizer.param_groups[0]["lr"])

        if (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_score,
                            ckpt_dir / "latest.pth", cfg=cfg, scaler=scaler)

        if (epoch + 1) % val_interval == 0 and len(val_ds) > 0:
            logger.info("--- Val (epoch %d, %d plots) ---", epoch + 1, len(val_ds))
            t_val = time.perf_counter()
            plot_metrics = _evaluate_fold(model, val_ds, cfg, device)
            score, info  = _quality_score(plot_metrics)
            logger.info("--- Val done in %.2fs ---", time.perf_counter() - t_val)

            info["epoch"]      = epoch + 1
            info["fold"]       = fold_idx
            info["train_loss"] = round(float(mean_loss), 6)
            info["config"]     = cfg_dict          # <-- full training config

            epoch_ckpt = ckpt_dir / f"epoch_{epoch + 1:04d}.pth"
            epoch_json = ckpt_dir / f"epoch_{epoch + 1:04d}.json"
            save_checkpoint(model, optimizer, epoch + 1, best_score,
                            epoch_ckpt, cfg=cfg, scaler=scaler)
            epoch_json.write_text(json.dumps(info, indent=2, ensure_ascii=False))

            logger.info("Fold %d | Epoch %d | F1=%.4f P=%.4f R=%.4f det=%d ref=%d match=%d",
                        fold_idx, epoch + 1, info["f1"], info["precision"], info["recall"],
                        info["total_detected"], info["total_reference"], info["total_matched"])

            if score > best_score:
                best_score = score
                shutil.copy2(epoch_ckpt, ckpt_dir / "best.pth")
                shutil.copy2(epoch_json, ckpt_dir / "best.json")
                logger.info("  \u2605 New best! F1=%.4f — best.pth", best_score)

            model.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TreeRCNN")
    parser.add_argument("--config",    default="configs/tree_rcnn.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir",   default="outputs/")
    parser.add_argument("--fold",      type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _configure_cpu_threads(cfg)

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
