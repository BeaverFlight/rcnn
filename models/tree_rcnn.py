"""
TreeRCNN: two-stage 3D tree detection network.

Changes:
  - 6D regression throughout.
  - Two-phase training via set_epoch().
  - Hard Negative Mining: single-pass — infer all anchors once (no_grad),
    assign HNM targets, then one gradient forward on sampled subset.
  - Stage-2 proposals are detached.
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn
from torch import Tensor

from models.stage1.anchor_generator import AnchorGenerator
from models.stage1.proposal_head import ProposalHead
from models.stage1.target_assinger import assign_targets
from models.stage2.refinement_head import RefinementHead
from models.losses.focal_loss import sigmoid_focal_loss
from models.losses.smooth_l1 import smooth_l1_loss
from ops.nms3d import nms3d
from utils.box_coder import decode_boxes

logger = logging.getLogger(__name__)

_MAX_POINTS_PER_BOX   = 512
_MIN_POINTS_FOR_NET   = 4
_ANCHOR_CHUNK         = 1024
_STAGE1_INFER_BATCH   = 64
_STAGE2_INFER_CHUNK   = 2048


# ---------------------------------------------------------------------------
# Point-in-box sampling helpers
# ---------------------------------------------------------------------------

def _subsample_points_in_box(
    points: Tensor, box: Tensor, n: int = _MAX_POINTS_PER_BOX
) -> Tensor:
    x, y, z_c, w, l, h = box.unbind()
    mask = (
        (points[:, 0] >= x - w / 2) & (points[:, 0] <= x + w / 2)
        & (points[:, 1] >= y - l / 2) & (points[:, 1] <= y + l / 2)
        & (points[:, 2] >= 0) & (points[:, 2] <= h)
    )
    inside = points[mask]
    if len(inside) > n:
        inside = inside[torch.randperm(len(inside), device=inside.device)[:n]]
    return inside


def _subsample_points_batch(
    points: Tensor,
    anchors: Tensor,
    n: int     = _MAX_POINTS_PER_BOX,
    chunk: int = _ANCHOR_CHUNK,
) -> list[Tensor]:
    device = points.device
    A      = anchors.shape[0]
    result: list[Tensor] = [None] * A  # type: ignore[list-item]

    px, py, pz = points[:, 0], points[:, 1], points[:, 2]

    for start in range(0, A, chunk):
        end = min(start + chunk, A)
        anc = anchors[start:end]
        cx  = anc[:, 0].unsqueeze(1)
        cy  = anc[:, 1].unsqueeze(1)
        w   = anc[:, 3].unsqueeze(1)
        l   = anc[:, 4].unsqueeze(1)
        h   = anc[:, 5].unsqueeze(1)

        mask = (
            (px.unsqueeze(0) >= cx - w / 2) & (px.unsqueeze(0) <= cx + w / 2)
            & (py.unsqueeze(0) >= cy - l / 2) & (py.unsqueeze(0) <= cy + l / 2)
            & (pz.unsqueeze(0) >= 0) & (pz.unsqueeze(0) <= h)
        )
        counts      = mask.sum(dim=1)
        _, point_idx = mask.nonzero(as_tuple=True)
        groups      = torch.split(point_idx, counts.tolist())

        for i, grp in enumerate(groups):
            if len(grp) == 0:
                result[start + i] = points.new_zeros(0, 3)
                continue
            if len(grp) > n:
                grp = grp[torch.randperm(len(grp), device=device)[:n]]
            pts        = points[grp].clone()
            pts[:, 0] -= anchors[start + i, 0]
            pts[:, 1] -= anchors[start + i, 1]
            result[start + i] = pts

    return result  # type: ignore[return-value]


def _subsample_points_batch_proposals(
    points: Tensor,
    proposals: Tensor,
    n: int     = _MAX_POINTS_PER_BOX,
    chunk: int = _STAGE2_INFER_CHUNK,
) -> list[Tensor]:
    device = points.device
    if proposals.device != device:
        proposals = proposals.to(device)

    P      = proposals.shape[0]
    result: list[Tensor] = [None] * P  # type: ignore[list-item]
    px, py, pz = points[:, 0].to(device), points[:, 1].to(device), points[:, 2].to(device)

    total_chunks = (P + chunk - 1) // chunk
    logger.info(
        "  Stage2 sampling: %d proposals / %d points / chunk=%d / total_chunks=%d",
        P, len(points), chunk, total_chunks,
    )

    for chunk_idx, start in enumerate(range(0, P, chunk), start=1):
        end  = min(start + chunk, P)
        prop = proposals[start:end]
        t_chunk = time.perf_counter()

        cx = prop[:, 0].unsqueeze(1)
        cy = prop[:, 1].unsqueeze(1)
        w  = prop[:, 3].unsqueeze(1)
        l  = prop[:, 4].unsqueeze(1)
        h  = prop[:, 5].unsqueeze(1)

        mask = (
            (px.unsqueeze(0) >= cx - w / 2) & (px.unsqueeze(0) <= cx + w / 2)
            & (py.unsqueeze(0) >= cy - l / 2) & (py.unsqueeze(0) <= cy + l / 2)
            & (pz.unsqueeze(0) >= 0) & (pz.unsqueeze(0) <= h)
        )
        counts      = mask.sum(dim=1)
        _, point_idx = mask.nonzero(as_tuple=True)
        groups      = torch.split(point_idx, counts.tolist())

        non_empty = total_kept = 0
        for i, grp in enumerate(groups):
            if len(grp) == 0:
                result[start + i] = points.new_zeros(0, 3)
                continue
            non_empty += 1
            if len(grp) > n:
                grp = grp[torch.randperm(len(grp), device=device)[:n]]
            total_kept        += len(grp)
            result[start + i]  = points[grp].clone()

        logger.info(
            "  Sampling chunk %d/%d (props %d-%d): non_empty=%d kept_pts=%d elapsed=%.2fs",
            chunk_idx, total_chunks, start, end - 1,
            non_empty, total_kept, time.perf_counter() - t_chunk,
        )

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
    max_n     = max(p.shape[0] for p in valid_pts)
    batched   = torch.zeros(len(valid_pts), max_n, 3, device=device)
    for k, pts in enumerate(valid_pts):
        n = pts.shape[0]
        if n < max_n:
            pts = pts.repeat((max_n + n - 1) // n, 1)[:max_n]
        batched[k] = pts
    return batched, valid_idx


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TreeRCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg        = cfg
        self.anchor_gen = AnchorGenerator(cfg)
        self.stage1     = ProposalHead(cfg)
        self.stage2     = RefinementHead(cfg)
        self.lambda_reg: float = cfg.training.lambda_reg

        fl = cfg.training.get("focal_loss", {})
        self._focal_alpha: float = float(fl.get("alpha", 0.25) if fl else 0.25)
        self._focal_gamma: float = float(fl.get("gamma", 2.0)  if fl else 2.0)

        self._s2_chunk: int         = int(cfg.training.get("stage2_infer_chunk", _STAGE2_INFER_CHUNK))
        self._freeze_s2_epochs: int = int(cfg.training.get("freeze_stage2_epochs", 0))
        self._current_epoch: int    = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        freeze = epoch < self._freeze_s2_epochs
        for p in self.stage2.parameters():
            p.requires_grad = not freeze
        if freeze:
            logger.debug("Epoch %d: Stage-2 FROZEN", epoch)
        elif epoch == self._freeze_s2_epochs and self._freeze_s2_epochs > 0:
            logger.info("Epoch %d: Stage-2 UNFROZEN — two-phase transition.", epoch)

    # ------------------------------------------------------------------

    def forward(
        self,
        points: Tensor,
        gt_boxes: Tensor,
        local_maxima: Tensor,
        plot_bounds,
        training: bool = True,
    ) -> dict:
        t0 = time.perf_counter()

        while points.dim()       > 2: points       = points.squeeze(0)
        while gt_boxes.dim()     > 2: gt_boxes     = gt_boxes.squeeze(0)
        while local_maxima.dim() > 2: local_maxima = local_maxima.squeeze(0)

        device = points.device
        pb = tuple(plot_bounds.tolist()) if isinstance(plot_bounds, Tensor) else plot_bounds

        t1 = time.perf_counter()
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad      = ad.to(device)
        al_flat = [a.to(device) for a in al_list]
        logger.debug("Anchors: ad=%d al=%d (%.2fs)",
                     len(ad), sum(len(a) for a in al_flat), time.perf_counter() - t1)

        if training:
            loss_s1 = self._stage1_loss(points, ad, al_flat, gt_boxes)
        else:
            loss_s1 = {}

        proposals = self._stage1_proposals(points, ad, al_flat, device)
        if not training:
            logger.info("  Stage1 -> %d proposals", len(proposals))

        if len(proposals) == 0:
            zero = torch.tensor(0.0, device=device)
            logger.warning("No S1 proposals — zero loss")
            if training:
                return {**loss_s1,
                        "loss_stage2_cls": zero, "loss_stage2_reg": zero,
                        "total_loss": loss_s1.get("total_loss_stage1", zero)}
            return {"boxes": torch.zeros(0, 6, device=device),
                    "scores": torch.zeros(0, device=device)}

        if training:
            loss_s2 = self._stage2_loss(points, proposals.detach(), gt_boxes)
            total   = (
                loss_s1.get("total_loss_stage1", torch.tensor(0.0, device=device))
                + loss_s2["total_loss_stage2"]
            )
            if torch.isnan(total) or torch.isinf(total):
                logger.warning("NaN/Inf total_loss — skipping batch")
                total = torch.tensor(0.0, device=device, requires_grad=True)
            logger.debug("Forward %.2fs | loss=%.4f",
                         time.perf_counter() - t0, total.item())
            return {**loss_s1, **loss_s2, "total_loss": total}

        final_boxes, final_scores = self._stage2_inference(points, proposals)
        logger.info("  Stage2 inference done: %d final boxes (%.2fs total)",
                    len(final_boxes), time.perf_counter() - t0)
        return {"boxes": final_boxes, "scores": final_scores}

    # ------------------------------------------------------------------

    def _stage1_loss(self, points: Tensor, ad: Tensor, al_list: list[Tensor], gt_boxes: Tensor) -> dict:
        """
        Single-pass HNM:
          1. Infer ALL anchors with no_grad  → get cls_scores for HNM.
          2. assign_targets with cls_scores  → hard-negative sampled indices.
          3. One gradient forward on sampled subset only.
        """
        all_anchors = torch.cat([ad] + al_list, dim=0) if al_list else ad
        cfg_la      = self.cfg.label_assignment

        # Step 1: score all anchors cheaply (no gradients, no point sampling twice)
        with torch.no_grad():
            all_cls, _ = self._run_stage1_on_anchors(
                points, all_anchors, infer_mode=True, tag="s1_hnm_scan"
            )
            cls_scores_all = torch.sigmoid(all_cls.squeeze(-1))

        # Step 2: assign labels + HNM sampling in one call
        labels, reg_targets, _, sampled = assign_targets(
            all_anchors, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
            cls_scores=cls_scores_all,
        )

        # Step 3: gradient forward only on the sampled subset
        sampled_anchors          = all_anchors[sampled]
        cls_logits, reg_deltas   = self._run_stage1_on_anchors(
            points, sampled_anchors, infer_mode=False, tag="s1_grad"
        )
        sampled_labels = labels[sampled].float()

        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha, gamma=self._focal_gamma,
        )
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage1_cls": cls_loss, "loss_stage1_reg": reg_loss,
                "total_loss_stage1": total}

    def _run_stage1_on_anchors(
        self,
        points: Tensor,
        anchors: Tensor,
        infer_mode: bool = True,
        tag: str = "",
    ) -> tuple[Tensor, Tensor]:
        device    = points.device
        A         = len(anchors)
        t         = time.perf_counter()
        pts_list  = _subsample_points_batch(points, anchors)
        t_sample  = time.perf_counter() - t

        valid_idx = [i for i, p in enumerate(pts_list) if p.shape[0] >= _MIN_POINTS_FOR_NET]
        V         = len(valid_idx)
        cls_out   = torch.zeros(A, 1, device=device)
        reg_out   = torch.zeros(A, 6, device=device)

        if not valid_idx:
            logger.warning("  [%s] 0 valid anchors/%d", tag, A)
            return cls_out, reg_out

        valid_pts = [pts_list[i] for i in valid_idx]
        mb        = int(getattr(self.cfg.training, "stage1_infer_batch", _STAGE1_INFER_BATCH))
        t_fwd     = time.perf_counter()

        ctx = torch.no_grad() if infer_mode else torch.enable_grad()
        with ctx:  # type: ignore[attr-defined]
            for start in range(0, V, mb):
                end      = min(start + mb, V)
                batch, _ = _pad_windows_to_batch(valid_pts[start:end], device)
                c, r     = self.stage1(batch)
                for k, orig_i in enumerate(valid_idx[start:end]):
                    cls_out[orig_i] = c[k].detach() if infer_mode else c[k]
                    reg_out[orig_i] = r[k].detach() if infer_mode else r[k]

        logger.debug("  [%s] anchors=%d valid=%d | sample=%.2fs fwd=%.2fs (%d mb)",
                     tag, A, V, t_sample, time.perf_counter() - t_fwd,
                     (V + mb - 1) // mb)
        return cls_out, reg_out

    def _stage1_proposals(self, points: Tensor, ad: Tensor, al_list: list[Tensor], device) -> Tensor:
        cfg_nms      = self.cfg.stage1_nms
        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))

        if len(ad) > 0:
            cls_ad, reg_ad = self._run_stage1_on_anchors(points, ad, infer_mode=True, tag="ad")
            scores_ad = torch.sigmoid(cls_ad.squeeze(-1))
            boxes_ad  = decode_boxes(reg_ad, ad)
            keep_ad   = nms3d(boxes_ad, scores_ad,
                              cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals,
                              score_threshold=ad_score_thr)
            props_ad  = boxes_ad[keep_ad]
        else:
            props_ad = torch.zeros(0, 6, device=device)

        if al_list:
            al_all    = torch.cat(al_list, dim=0)
            sizes     = [len(a) for a in al_list]
            cls_al, reg_al = self._run_stage1_on_anchors(points, al_all, infer_mode=True, tag="al")
            scores_al = torch.sigmoid(cls_al.squeeze(-1))
            boxes_al  = decode_boxes(reg_al, al_all)

            parts, offset = [], 0
            for sz in sizes:
                if sz == 0:
                    offset += sz
                    continue
                sl   = slice(offset, offset + sz)
                keep = nms3d(boxes_al[sl], scores_al[sl],
                             cfg_nms.al_iouv_threshold, cfg_nms.al_max_proposals_per_maxima)
                parts.append(boxes_al[sl][keep])
                offset += sz
            props_al = torch.cat(parts, dim=0) if parts else torch.zeros(0, 6, device=device)
        else:
            props_al = torch.zeros(0, 6, device=device)

        return torch.cat([props_ad, props_al], dim=0)

    def _stage2_loss(self, points: Tensor, proposals: Tensor, gt_boxes: Tensor) -> dict:
        cfg_la = self.cfg.label_assignment
        labels, reg_targets, _, sampled = assign_targets(
            proposals, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )
        sampled_props          = proposals[sampled]
        pts_list               = [_subsample_points_in_box(points, p) for p in sampled_props]
        cls_logits, reg_deltas = self.stage2(pts_list, sampled_props)

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha, gamma=self._focal_gamma,
        )
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage2_cls": cls_loss, "loss_stage2_reg": reg_loss,
                "total_loss_stage2": total}

    def _stage2_inference(self, points: Tensor, proposals: Tensor) -> tuple[Tensor, Tensor]:
        cfg_nms  = self.cfg.stage2_nms
        t_sample = time.perf_counter()
        pts_list = _subsample_points_batch_proposals(
            points, proposals, n=_MAX_POINTS_PER_BOX, chunk=self._s2_chunk
        )
        logger.info("  Stage2 sampling done in %.2fs", time.perf_counter() - t_sample)

        with torch.no_grad():
            cls_logits, reg_deltas = self.stage2(pts_list, proposals)

        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)

        score_mask      = scores >= cfg_nms.score_threshold
        refined, scores = refined[score_mask], scores[score_mask]
        logger.info("  Stage2 score filter: %d -> %d boxes",
                    len(score_mask), len(refined))

        if len(refined) == 0:
            return refined, scores

        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        logger.info("  Stage2 NMS: %d -> %d boxes", len(refined), len(keep))
        return refined[keep], scores[keep]
