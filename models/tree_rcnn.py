"""
TreeRCNN: two-stage 3D tree detection network.
"""

from __future__ import annotations

import logging
import time

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
_MIN_POINTS_FOR_NET = 4
_ANCHOR_CHUNK = 1024


def _subsample_points_in_box(
    points: Tensor, box: Tensor, n: int = _MAX_POINTS_PER_BOX
) -> Tensor:
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


def _subsample_points_batch(
    points: Tensor,
    anchors: Tensor,
    n: int = _MAX_POINTS_PER_BOX,
    chunk: int = _ANCHOR_CHUNK,
) -> list[Tensor]:
    device = points.device
    A = anchors.shape[0]
    result: list[Tensor] = [None] * A  # type: ignore[list-item]

    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    for start in range(0, A, chunk):
        end = min(start + chunk, A)
        anc = anchors[start:end]
        C = end - start

        cx = anc[:, 0].unsqueeze(1)
        cy = anc[:, 1].unsqueeze(1)
        w  = anc[:, 3].unsqueeze(1)
        l  = anc[:, 4].unsqueeze(1)
        h  = anc[:, 5].unsqueeze(1)

        mask = (
            (px.unsqueeze(0) >= cx - w / 2)
            & (px.unsqueeze(0) <= cx + w / 2)
            & (py.unsqueeze(0) >= cy - l / 2)
            & (py.unsqueeze(0) <= cy + l / 2)
            & (pz.unsqueeze(0) >= 0)
            & (pz.unsqueeze(0) <= h)
        )

        counts = mask.sum(dim=1)
        anchor_idx, point_idx = mask.nonzero(as_tuple=True)
        counts_list = counts.tolist()
        groups = torch.split(point_idx, counts_list)

        for i, grp in enumerate(groups):
            if len(grp) == 0:
                result[start + i] = points.new_zeros(0, 3)
                continue
            if len(grp) > n:
                perm = torch.randperm(len(grp), device=device)[:n]
                grp = grp[perm]
            pts = points[grp]
            pts = pts.clone()
            pts[:, 0] -= anchors[start + i, 0]
            pts[:, 1] -= anchors[start + i, 1]
            result[start + i] = pts

    return result  # type: ignore[return-value]


def _pad_windows_to_batch(
    pts_list: list[Tensor],
    device: torch.device,
    min_pts: int = _MIN_POINTS_FOR_NET,
) -> tuple[Tensor, list[int]]:
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
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.anchor_gen = AnchorGenerator(cfg)
        self.stage1 = ProposalHead(cfg)
        self.stage2 = RefinementHead(cfg)
        self.lambda_reg: float = cfg.training.lambda_reg

    def forward(
        self,
        points: Tensor,
        gt_boxes: Tensor,
        local_maxima: Tensor,
        plot_bounds,
        training: bool = True,
    ) -> dict:
        t0 = time.perf_counter()

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

        logger.info(
            "Forward: %d pts | %d GT | %d maxima",
            len(points), len(gt_boxes), len(local_maxima),
        )

        # ---- Anchor generation ----------------------------------------
        t1 = time.perf_counter()
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad = ad.to(device)
        al_flat = [a.to(device) for a in al_list]
        n_al = sum(len(a) for a in al_flat)
        logger.info(
            "Anchors: ad=%d  al=%d  (%.2fs)",
            len(ad), n_al, time.perf_counter() - t1,
        )

        # ---- Stage 1 --------------------------------------------------
        if training:
            t2 = time.perf_counter()
            loss_s1 = self._stage1_loss(points, ad, al_flat, gt_boxes)
            logger.info("S1 loss done (%.2fs)", time.perf_counter() - t2)
        else:
            loss_s1 = {}

        t3 = time.perf_counter()
        proposals = self._stage1_proposals(points, ad, al_flat, device)
        logger.info("S1 proposals: %d  (%.2fs)", len(proposals), time.perf_counter() - t3)

        if len(proposals) == 0:
            zero = torch.tensor(0.0, device=device)
            logger.warning("No S1 proposals — zero loss")
            if training:
                return {
                    **loss_s1,
                    "loss_stage2_cls": zero,
                    "loss_stage2_reg": zero,
                    "total_loss": loss_s1.get("total_loss_stage1", zero),
                }
            return {"boxes": torch.zeros(0, 6, device=device), "scores": torch.zeros(0, device=device)}

        # ---- Stage 2 --------------------------------------------------
        if training:
            t4 = time.perf_counter()
            loss_s2 = self._stage2_loss(points, proposals, gt_boxes)
            logger.info("S2 loss done (%.2fs)", time.perf_counter() - t4)

            total = (
                loss_s1.get("total_loss_stage1", torch.tensor(0.0, device=device))
                + loss_s2["total_loss_stage2"]
            )
            if torch.isnan(total) or torch.isinf(total):
                logger.warning("NaN/Inf in total_loss — skipping batch")
                total = torch.tensor(0.0, device=device, requires_grad=True)

            logger.info(
                "Forward done %.2fs | loss=%.4f",
                time.perf_counter() - t0, total.item(),
            )
            return {**loss_s1, **loss_s2, "total_loss": total}

        final_boxes, final_scores = self._stage2_inference(points, proposals)
        logger.info("Inference done (%.2fs)", time.perf_counter() - t0)
        return {"boxes": final_boxes, "scores": final_scores}

    # ------------------------------------------------------------------
    def _stage1_loss(self, points, ad, al_list, gt_boxes):
        all_anchors = torch.cat([ad] + al_list, dim=0) if al_list else ad
        cfg_la = self.cfg.label_assignment

        labels, reg_targets, _, sampled = assign_targets(
            all_anchors, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )
        sampled_anchors = all_anchors[sampled]
        logger.info("S1 loss: %d sampled anchors", len(sampled_anchors))

        cls_logits, reg_deltas = self._run_stage1_on_anchors(points, sampled_anchors)
        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels)

        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage1_cls": cls_loss, "loss_stage1_reg": reg_loss, "total_loss_stage1": total}

    def _run_stage1_on_anchors(self, points, anchors):
        device = points.device
        A = len(anchors)
        t = time.perf_counter()
        logger.info("  stage1 anchors=%d pts=%d — batch sampling...", A, len(points))

        # --- ДИАГНОСТИКА: сравниваем диапазоны координат ---
        with torch.no_grad():
            px_min, px_max = points[:, 0].min().item(), points[:, 0].max().item()
            py_min, py_max = points[:, 1].min().item(), points[:, 1].max().item()
            pz_min, pz_max = points[:, 2].min().item(), points[:, 2].max().item()
            ax_min, ax_max = anchors[:, 0].min().item(), anchors[:, 0].max().item()
            ay_min, ay_max = anchors[:, 1].min().item(), anchors[:, 1].max().item()
            ah_min, ah_max = anchors[:, 5].min().item(), anchors[:, 5].max().item()
        logger.info(
            "  PTS  x=[%.1f, %.1f]  y=[%.1f, %.1f]  z=[%.2f, %.2f]",
            px_min, px_max, py_min, py_max, pz_min, pz_max,
        )
        logger.info(
            "  ANCH x=[%.1f, %.1f]  y=[%.1f, %.1f]  h=[%.2f, %.2f]",
            ax_min, ax_max, ay_min, ay_max, ah_min, ah_max,
        )
        # --- конец диагностики ---

        pts_list = _subsample_points_batch(points, anchors)
        logger.info("  sampling done %.2fs", time.perf_counter() - t)

        batch, valid_idx = _pad_windows_to_batch(pts_list, device)
        logger.info("  valid=%d/%d → stage1 fwd...", len(valid_idx), A)

        cls_out = torch.zeros(A, 1, device=device)
        reg_out = torch.zeros(A, 4, device=device)
        if len(valid_idx) == 0:
            logger.warning("  0 valid anchors (all < %d pts)", _MIN_POINTS_FOR_NET)
            return cls_out, reg_out

        t2 = time.perf_counter()
        c, r = self.stage1(batch)
        logger.info("  stage1 fwd done %.2fs", time.perf_counter() - t2)

        for k, orig_i in enumerate(valid_idx):
            cls_out[orig_i] = c[k]
            reg_out[orig_i] = r[k]
        return cls_out, reg_out

    def _stage1_proposals(self, points, ad, al_list, device):
        cfg_nms = self.cfg.stage1_nms
        with torch.no_grad():
            if len(ad) > 0:
                logger.info("S1 proposals: ad=%d", len(ad))
                cls_ad, reg_ad = self._run_stage1_on_anchors(points, ad)
                scores_ad = torch.sigmoid(cls_ad.squeeze(-1))
                boxes_ad  = decode_boxes(reg_ad, ad)
                keep_ad   = nms3d(boxes_ad, scores_ad, cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals)
                props_ad  = boxes_ad[keep_ad]
                logger.info("  ad after NMS: %d", len(props_ad))
            else:
                props_ad = torch.zeros(0, 6, device=device)

            parts = []
            for i, al in enumerate(al_list):
                if len(al) == 0:
                    continue
                cls_al, reg_al = self._run_stage1_on_anchors(points, al)
                scores_al = torch.sigmoid(cls_al.squeeze(-1))
                boxes_al  = decode_boxes(reg_al, al)
                keep_al   = nms3d(boxes_al, scores_al, cfg_nms.al_iouv_threshold, cfg_nms.al_max_proposals_per_maxima)
                parts.append(boxes_al[keep_al])

            props_al = torch.cat(parts, dim=0) if parts else torch.zeros(0, 6, device=device)

        proposals = torch.cat([props_ad, props_al], dim=0)
        logger.info("S1 total proposals: %d", len(proposals))
        return proposals

    def _stage2_loss(self, points, proposals, gt_boxes):
        cfg_la = self.cfg.label_assignment
        labels, reg_targets, _, sampled = assign_targets(
            proposals, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )
        sampled_props = proposals[sampled]
        logger.info("S2 loss: %d sampled proposals", len(sampled_props))
        pts_list = [_subsample_points_in_box(points, p) for p in sampled_props]
        cls_logits, reg_deltas = self.stage2(pts_list, sampled_props)

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels)
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage2_cls": cls_loss, "loss_stage2_reg": reg_loss, "total_loss_stage2": total}

    def _stage2_inference(self, points, proposals):
        cfg_nms = self.cfg.stage2_nms
        pts_list = [_subsample_points_in_box(points, p) for p in proposals]
        with torch.no_grad():
            cls_logits, reg_deltas = self.stage2(pts_list, proposals)
        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)
        score_mask = scores >= cfg_nms.score_threshold
        refined, scores = refined[score_mask], scores[score_mask]
        if len(refined) == 0:
            return refined, scores
        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        return refined[keep], scores[keep]
