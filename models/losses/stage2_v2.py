"""
models/losses/stage2_v2.py — Stage-2 loss для TreeRCNN v2

Компоненты:
  cls_loss       = λ_focal·FocalLoss + λ_dice·DiceLoss
  reg_loss       = SmoothL1 (xy + wl отдельно, zh с λ_v_reg)
  offset_loss    = SmoothL1 (voting offsets для каждой точки)
  centerness_loss= λ_bce·BCE + λ_dice·DiceLoss (point-wise centerness)

Веса управляются через словарь lambdas (все значения опциональны, дефолты ниже).
Дефолтные веса из конфига cfg.training — передаются в tree_rcnn_v2.

Fix: centerness BCE — используется F.binary_cross_entropy (не with_logits),
     т.к. Stage2Head.centerness_head уже применяет Sigmoid на выходе.
     pred_centerness содержит значения [0..1], а не logits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.losses.focal_loss import sigmoid_focal_loss
from models.losses.smooth_l1  import smooth_l1_loss
from models.losses.dice_loss   import dice_loss

_DEFAULTS = {
    "cls":          1.0,
    "reg":          1.0,
    "offset":       0.5,
    "centerness":   0.5,
    # Dice-веса
    "cls_focal_w":  0.5,   # доля Focal в cls_loss
    "cls_dice_w":   0.5,   # доля Dice  в cls_loss
    "cent_bce_w":   0.5,   # доля BCE   в centerness_loss
    "cent_dice_w":  0.5,   # доля Dice  в centerness_loss
    # Вертикальный вес
    "v_reg":        2.0,
    # Focal гиперпараметры
    "focal_alpha":  0.25,
    "focal_gamma":  2.0,
}


def _merge(defaults: dict, overrides: dict | None) -> dict:
    d = dict(defaults)
    if overrides:
        d.update(overrides)
    return d


def stage2_loss_v2(
    cls_score:       Tensor,         # (S, 1)      logits
    reg_delta:       Tensor,         # (S, 6)      регрессионные дельты
    pred_offsets:    Tensor | None,  # (S, N2, 3)  voting offsets (CPU или GPU)
    pred_centerness: Tensor | None,  # (S, N2, 1)  centerness scores [0..1] (уже sigmoid!)
    points_xyz:      Tensor | None,  # (S, N2, 3)  координаты точек
    gt_box:          Tensor,         # (S, 6)      target регрессия
    gt_label:        Tensor,         # (S,)        {0, 1}
    lambdas:         dict | None = None,
) -> dict[str, Tensor]:
    """
    Возвращает словарь:
        cls        — cls_loss (Focal + Dice)
        reg        — SmoothL1 regression
        offset     — voting offset loss
        centerness — BCE + Dice centerness
        total      — взвешенная сумма

    ВАЖНО: pred_centerness должен содержать значения [0..1] (уже после Sigmoid),
    т.к. Stage2Head.centerness_head применяет nn.Sigmoid() на выходе.
    Для loss используется F.binary_cross_entropy (не with_logits).
    """
    L = _merge(_DEFAULTS, lambdas)
    device = cls_score.device
    zero   = torch.tensor(0.0, device=device)

    # -----------------------------------------------------------------
    # 1. CLS loss = Focal + Dice
    # -----------------------------------------------------------------
    logits_1d = cls_score.squeeze(-1)            # (S,)
    labels    = gt_label.float().to(device)

    focal = sigmoid_focal_loss(
        logits_1d, labels,
        alpha=L["focal_alpha"], gamma=L["focal_gamma"],
    )
    dice_cls = dice_loss(logits_1d, labels)
    cls_loss = L["cls_focal_w"] * focal + L["cls_dice_w"] * dice_cls

    # -----------------------------------------------------------------
    # 2. REG loss (только позитивные)
    # -----------------------------------------------------------------
    pos_mask = labels == 1
    if pos_mask.any():
        pred = reg_delta[pos_mask]
        tgt  = gt_box[pos_mask].to(device)
        loss_xy = smooth_l1_loss(pred[:, [0, 1, 3, 4]], tgt[:, [0, 1, 3, 4]])
        loss_zh = smooth_l1_loss(pred[:, [2, 5]],       tgt[:, [2, 5]])
        reg_loss = loss_xy + L["v_reg"] * loss_zh
    else:
        reg_loss = zero

    # -----------------------------------------------------------------
    # 3. Offset loss — voting направления для каждой точки
    # -----------------------------------------------------------------
    off_loss = zero
    if pred_offsets is not None and pos_mask.any() and points_xyz is not None:
        # gt offset = центр бокса - координата точки
        # gt_box[pos, :3] = (cx, cy, cz_centre)
        off_pts   = pred_offsets[pos_mask].to(device)   # (P, N2, 3)
        xyz_pts   = points_xyz[pos_mask].to(device)     # (P, N2, 3)
        gt_ctr    = gt_box[pos_mask, :3].to(device)     # (P, 3)
        # gt_offset: (P, N2, 3) = centre broadcast - point_xyz
        gt_off    = gt_ctr.unsqueeze(1) - xyz_pts       # (P, N2, 3)
        # маскируем нулевые (padded) точки
        valid_pts = (xyz_pts.abs().sum(-1) > 0).float().unsqueeze(-1)  # (P, N2, 1)
        off_loss  = (smooth_l1_loss(off_pts, gt_off, reduction='none') * valid_pts).mean()

    # -----------------------------------------------------------------
    # 4. Centerness loss = BCE + Dice
    #
    # Fix: pred_centerness уже прошёл через nn.Sigmoid() в Stage2Head,
    # поэтому используем F.binary_cross_entropy (не with_logits).
    # Это исправляет ошибку двойного sigmoid (logits → sigmoid → sigmoid),
    # которая приводила к неверным градиентам и зажатию весов centerness_head.
    # -----------------------------------------------------------------
    cent_loss = zero
    if pred_centerness is not None and pos_mask.any() and points_xyz is not None:
        # pred_centerness: значения в [0..1] (после Sigmoid в centerness_head)
        cent_probs  = pred_centerness[pos_mask].to(device).squeeze(-1)  # (P, N2)
        xyz_pos     = points_xyz[pos_mask].to(device)                   # (P, N2, 3)
        gt_ctr      = gt_box[pos_mask, :3].to(device)                   # (P, 3)
        gt_wlh      = gt_box[pos_mask, 3:].to(device)                   # (P, 3): w, l, h

        # gt centerness = exp(-dist / (0.5 * min(w, l)))
        # → 1.0 в центре, ~0 на краях кроны
        dist2d  = torch.norm(xyz_pos[:, :, :2] - gt_ctr[:, :2].unsqueeze(1), dim=-1)  # (P, N2)
        half_r  = (gt_wlh[:, :2].min(dim=-1).values / 2.0).clamp(min=0.1)            # (P,)
        gt_cent = torch.exp(-dist2d / half_r.unsqueeze(1))                            # (P, N2)

        valid_pts = (xyz_pos.abs().sum(-1) > 0).float()  # (P, N2)

        # BCE с уже-sigmoid предсказаниями (не with_logits!)
        bce_c  = F.binary_cross_entropy(
            cent_probs.clamp(1e-6, 1.0 - 1e-6),  # числовая стабильность
            gt_cent,
            reduction='none',
        )
        # dice_loss ожидает predictions в [0..1] — cent_probs уже корректны
        dice_c = dice_loss(cent_probs, gt_cent, reduction='mean')
        cent_loss = (
            L["cent_bce_w"]  * (bce_c * valid_pts).mean()
            + L["cent_dice_w"] * dice_c
        )

    # -----------------------------------------------------------------
    # Total
    # -----------------------------------------------------------------
    total = (
        L["cls"]        * cls_loss
        + L["reg"]        * reg_loss
        + L["offset"]     * off_loss
        + L["centerness"] * cent_loss
    )

    return {
        "cls":        cls_loss,
        "reg":        reg_loss,
        "offset":     off_loss,
        "centerness": cent_loss,
        "total":      total,
    }
