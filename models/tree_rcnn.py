"""
TreeRCNN: two-stage 3D tree detection network.

Fixes vs original:
  - _run_stage1_on_anchors: якоря обрабатываются одним батчем (pad → stack → forward)
    вместо Python-цикла. Ускорение на GPU — в разы.
  - Добавлен guard на NaN в total_loss перед return (в training).
  - _subsample_points_in_box: добавлен guard min 4 точки для BatchNorm/LayerNorm.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.stage1.anchor_generator import AnchorGenerator
from models.stage1.proposal_head import ProposalHead
from models.stage1.target_assigner import assign_targets
from models.stage2.refinement_head import RefinementHead
from models.losses.focal_loss import sigmoid_focal_loss
from models.losses.smooth_l1 import smooth_l1_loss
from ops.nms3d import nms3d
from utils.box_coder import decode_boxes

logger = logging.getLogger(__name__)

_MAX_POINTS_PER_BOX = 512
_MIN_POINTS_FOR_NET = 4   # меньше — нет смысла гонять через PointNet++


def _subsample_points_in_box(
    points: Tensor, box: Tensor, n: int = _MAX_POINTS_PER_BOX
) -> Tensor:
    """Return up to n points contained in an axis-aligned 3D box."""
    x, y, z_c, w, l, h = box.unbind()
    mask = (
        (points[:, 0] >= x - w / 2)
        & (points[:, 0] <= x + w / 2)
        & (points[:, 1] >= y - l / 2)
        & (points[:, 1] <= y + l / 2)
        & (points[:, 2] >= 0)
        & (points[:, 2] <= h)
    )
    inside = points[mask]
    if len(inside) > n:
        idx = torch.randperm(len(inside), device=inside.device)[:n]
        inside = inside[idx]
    return inside


def _pad_windows_to_batch(
    pts_list: list[Tensor],
    device: torch.device,
    min_pts: int = _MIN_POINTS_FOR_NET,
) -> tuple[Tensor, list[int]]:
    """
    Паддируем список тензоров точек до единого max_n через repeat-padding.
    Возвращает (batch: (B, max_n, 3), valid_indices: список индексов с >= min_pts точек).
    """
    valid_idx = [i for i, p in enumerate(pts_list) if p.shape[0] >= min_pts]
    if not valid_idx:
        return torch.zeros(0, min_pts, 3, device=device), valid_idx

    valid_pts = [pts_list[i] for i in valid_idx]
    max_n = max(p.shape[0] for p in valid_pts)

    batched = torch.zeros(len(valid_pts), max_n, 3, device=device)
    for k, pts in enumerate(valid_pts):
        n = pts.shape[0]
        if n < max_n:
            reps = (max_n + n - 1) // n
            pts = pts.repeat(reps, 1)[:max_n]
        batched[k] = pts
    return batched, valid_idx


class TreeRCNN(nn.Module):
    """
    Full TreeRCNN model.
    Forward pass supports both training (returns loss dict) and
    inference (returns detected boxes).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.anchor_gen = AnchorGenerator(cfg)
        self.stage1 = ProposalHead(cfg)
        self.stage2 = RefinementHead(cfg)
        self.lambda_reg: float = cfg.training.lambda_reg

    # ------------------------------------------------------------------
    def forward(
        self,
        points: Tensor,
        gt_boxes: Tensor,
        local_maxima: Tensor,
        plot_bounds,
        training: bool = True,
    ) -> dict:
        # Убираем batch-измерения
        while points.dim() > 2:
            points = points.squeeze(0)
        while gt_boxes.dim() > 2:
            gt_boxes = gt_boxes.squeeze(0)
        while local_maxima.dim() > 2:
            local_maxima = local_maxima.squeeze(0)

        device = points.device
        pb = (
            tuple(plot_bounds.tolist())
            if isinstance(plot_bounds, Tensor)
            else plot_bounds
        )

        # ---- Stage 1 ------------------------------------------------
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad = ad.to(device)
        al_flat = [a.to(device) for a in al_list]

        if training:
            loss_s1 = self._stage1_loss(points, ad, al_flat, gt_boxes)
        else:
            loss_s1 = {}

        proposals = self._stage1_proposals(points, ad, al_flat, device)

        if len(proposals) == 0:
            zero = torch.tensor(0.0, device=device)
            if training:
                return {
                    **loss_s1,
                    "loss_stage2_cls": zero,
                    "loss_stage2_reg": zero,
                    "total_loss": loss_s1.get("total_loss_stage1", zero),
                }
            return {
                "boxes": torch.zeros(0, 6, device=device),
                "scores": torch.zeros(0, device=device),
            }

        # ---- Stage 2 ------------------------------------------------
        if training:
            loss_s2 = self._stage2_loss(points, proposals, gt_boxes)
            total = (
                loss_s1.get("total_loss_stage1", torch.tensor(0.0, device=device))
                + loss_s2["total_loss_stage2"]
            )
            # NaN guard: если loss взорвался — возвращаем нули, не ломаем обучение
            if torch.isnan(total) or torch.isinf(total):
                logger.warning("NaN/Inf in total_loss — skipping batch")
                total = torch.tensor(0.0, device=device, requires_grad=True)
            return {**loss_s1, **loss_s2, "total_loss": total}

        final_boxes, final_scores = self._stage2_inference(points, proposals)
        return {"boxes": final_boxes, "scores": final_scores}

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------
    def _stage1_loss(
        self,
        points: Tensor,
        ad: Tensor,
        al_list: list[Tensor],
        gt_boxes: Tensor,
    ) -> dict:
        all_anchors = torch.cat([ad] + al_list, dim=0) if al_list else ad
        cfg_la = self.cfg.label_assignment

        labels, reg_targets, _, sampled = assign_targets(
            all_anchors,
            gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )

        sampled_anchors = all_anchors[sampled]
        cls_logits, reg_deltas = self._run_stage1_on_anchors(points, sampled_anchors)

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels)

        pos_mask = sampled_labels == 1
        if pos_mask.any():
            reg_loss = smooth_l1_loss(
                reg_deltas[pos_mask], reg_targets[sampled][pos_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=points.device)

        total = cls_loss + self.lambda_reg * reg_loss
        return {
            "loss_stage1_cls": cls_loss,
            "loss_stage1_reg": reg_loss,
            "total_loss_stage1": total,
        }

    def _run_stage1_on_anchors(
        self, points: Tensor, anchors: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Обрабатывает все якоря одним батчем через GPU.

        Алгоритм:
          1. Для каждого якоря: вырезаем точки, нормализуем относительно центра якоря.
          2. Паддируем все окна до max_n через repeat-padding.
          3. Один вызов self.stage1(batch) → (B, 1), (B, 4).
          4. Для невалидных якорей (< _MIN_POINTS_FOR_NET точек) выставляем нули.
        """
        device = points.device
        A = len(anchors)

        # Вырезаем и нормализуем точки для каждого якоря
        pts_list: list[Tensor] = []
        for anchor in anchors:
            pts = _subsample_points_in_box(points, anchor)
            if pts.shape[0] >= _MIN_POINTS_FOR_NET:
                norm = pts.clone()
                norm[:, 0] -= anchor[0]
                norm[:, 1] -= anchor[1]
                pts_list.append(norm)
            else:
                pts_list.append(pts)  # <4 — будет отфильтровано

        batch, valid_idx = _pad_windows_to_batch(pts_list, device)

        cls_out = torch.zeros(A, 1, device=device)
        reg_out = torch.zeros(A, 4, device=device)

        if len(valid_idx) == 0:
            return cls_out, reg_out

        # Один батчевый форвард-пасс
        c, r = self.stage1(batch)                 # (V, 1), (V, 4)
        for k, orig_i in enumerate(valid_idx):
            cls_out[orig_i] = c[k]
            reg_out[orig_i] = r[k]

        return cls_out, reg_out

    def _stage1_proposals(
        self,
        points: Tensor,
        ad: Tensor,
        al_list: list[Tensor],
        device: torch.device,
    ) -> Tensor:
        cfg_nms = self.cfg.stage1_nms

        with torch.no_grad():
            if len(ad) > 0:
                cls_ad, reg_ad = self._run_stage1_on_anchors(points, ad)
                scores_ad = torch.sigmoid(cls_ad.squeeze(-1))
                boxes_ad = decode_boxes(reg_ad, ad)
                keep_ad = nms3d(
                    boxes_ad, scores_ad,
                    cfg_nms.ad_iouv_threshold,
                    cfg_nms.ad_max_proposals,
                )
                props_ad = boxes_ad[keep_ad]
            else:
                props_ad = torch.zeros(0, 6, device=device)

            props_al_parts = []
            for al in al_list:
                if len(al) == 0:
                    continue
                cls_al, reg_al = self._run_stage1_on_anchors(points, al)
                scores_al = torch.sigmoid(cls_al.squeeze(-1))
                boxes_al = decode_boxes(reg_al, al)
                keep_al = nms3d(
                    boxes_al, scores_al,
                    cfg_nms.al_iouv_threshold,
                    cfg_nms.al_max_proposals_per_maxima,
                )
                props_al_parts.append(boxes_al[keep_al])

            props_al = (
                torch.cat(props_al_parts, dim=0)
                if props_al_parts
                else torch.zeros(0, 6, device=device)
            )

        proposals = torch.cat([props_ad, props_al], dim=0)
        logger.debug("Stage-1 proposals: %d", len(proposals))
        return proposals

    # ------------------------------------------------------------------
    # Stage 2 helpers
    # ------------------------------------------------------------------
    def _stage2_loss(
        self, points: Tensor, proposals: Tensor, gt_boxes: Tensor
    ) -> dict:
        cfg_la = self.cfg.label_assignment
        labels, reg_targets, _, sampled = assign_targets(
            proposals,
            gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )

        sampled_props = proposals[sampled]
        pts_list = [_subsample_points_in_box(points, p) for p in sampled_props]
        cls_logits, reg_deltas = self.stage2(pts_list, sampled_props)

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels)

        pos_mask = sampled_labels == 1
        if pos_mask.any():
            reg_loss = smooth_l1_loss(
                reg_deltas[pos_mask], reg_targets[sampled][pos_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=points.device)

        total = cls_loss + self.lambda_reg * reg_loss
        return {
            "loss_stage2_cls": cls_loss,
            "loss_stage2_reg": reg_loss,
            "total_loss_stage2": total,
        }

    def _stage2_inference(
        self, points: Tensor, proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        cfg_nms = self.cfg.stage2_nms
        pts_list = [_subsample_points_in_box(points, p) for p in proposals]
        with torch.no_grad():
            cls_logits, reg_deltas = self.stage2(pts_list, proposals)
        scores = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)

        score_mask = scores >= cfg_nms.score_threshold
        refined = refined[score_mask]
        scores = scores[score_mask]

        if len(refined) == 0:
            return refined, scores

        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        return refined[keep], scores[keep]
