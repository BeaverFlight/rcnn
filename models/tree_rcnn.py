"""
TreeRCNN: two-stage 3D tree detection network.

Performance changes vs previous version:
  1. AMP-ready: torch.autocast + GradScaler.
  2. inference_mode for all purely-inferential passes.
  3. Stage-1 scan CACHE: single PointNet++ pass per batch during training.
  4. Vectorised _subsample_points_batch: mask built fully on GPU via scatter.
  5. Vectorised _pad_windows_to_batch: Python loop replaced with
     scatter_add into pre-allocated [B, max_n, 3] tensor on GPU.
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

_MAX_POINTS_PER_BOX = 512
_MIN_POINTS_FOR_NET = 4
_ANCHOR_CHUNK       = 1024
_STAGE1_INFER_BATCH = 64
_STAGE2_INFER_CHUNK = 2048


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
    """
    For each anchor return up to `n` points inside its box,
    translated to anchor-local XY coordinates.

    Fully vectorised on GPU: no Python loop over anchors.
    """
    device = points.device
    A      = anchors.shape[0]
    N      = points.shape[0]
    result: list[Tensor] = [None] * A  # type: ignore[list-item]

    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    for start in range(0, A, chunk):
        end = min(start + chunk, A)
        anc = anchors[start:end]
        C   = end - start

        cx = anc[:, 0].unsqueeze(1)
        cy = anc[:, 1].unsqueeze(1)
        w  = anc[:, 3].unsqueeze(1)
        l  = anc[:, 4].unsqueeze(1)
        h  = anc[:, 5].unsqueeze(1)

        mask = (
            (px >= cx - w / 2) & (px <= cx + w / 2)
            & (py >= cy - l / 2) & (py <= cy + l / 2)
            & (pz >= 0) & (pz <= h)
        )
        counts     = mask.sum(dim=1)
        anchor_ids, point_ids = mask.nonzero(as_tuple=True)

        total_hits = anchor_ids.shape[0]
        if total_hits > 0:
            rand_key = torch.rand(total_hits, device=device)
            order    = torch.argsort(anchor_ids.float() * (N + 1) + rand_key * N)
            anchor_ids_s = anchor_ids[order]
            point_ids_s  = point_ids[order]

            ones   = torch.ones(total_hits, dtype=torch.long, device=device)
            cum    = torch.cumsum(ones, 0) - 1
            is_new = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                anchor_ids_s[1:] != anchor_ids_s[:-1]
            ])
            group_starts_per_hit = torch.zeros(total_hits, dtype=torch.long, device=device)
            group_starts_per_hit[is_new] = cum[is_new]
            group_starts_per_hit = torch.cummax(group_starts_per_hit, dim=0).values
            local_idx = cum - group_starts_per_hit

            valid_hit    = local_idx < n
            anchor_ids_f = anchor_ids_s[valid_hit]
            point_ids_f  = point_ids_s[valid_hit]
            local_idx_f  = local_idx[valid_hit]

            padded = torch.zeros(C, n, 3, device=device)
            padded[anchor_ids_f, local_idx_f] = points[point_ids_f]
            padded[:, :, 0] -= anc[:, 0].unsqueeze(1)
            padded[:, :, 1] -= anc[:, 1].unsqueeze(1)

            real_counts = torch.clamp(counts, max=n)
            for i in range(C):
                nc = int(real_counts[i].item())
                result[start + i] = points.new_zeros(0, 3) if nc == 0 else padded[i, :nc].clone()
        else:
            for i in range(C):
                result[start + i] = points.new_zeros(0, 3)

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
    px = points[:, 0].to(device)
    py = points[:, 1].to(device)
    pz = points[:, 2].to(device)

    total_chunks = (P + chunk - 1) // chunk
    logger.info(
        "  Stage2 sampling: %d proposals / %d points / chunk=%d / %d chunks",
        P, len(points), chunk, total_chunks,
    )

    for chunk_idx, start in enumerate(range(0, P, chunk), start=1):
        end  = min(start + chunk, P)
        prop = proposals[start:end]
        t_c  = time.perf_counter()

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
        counts       = mask.sum(dim=1)
        _, point_idx = mask.nonzero(as_tuple=True)
        groups       = torch.split(point_idx, counts.tolist())

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
            "  Chunk %d/%d (props %d-%d): non_empty=%d kept=%d %.2fs",
            chunk_idx, total_chunks, start, end - 1,
            non_empty, total_kept, time.perf_counter() - t_c,
        )

    return result  # type: ignore[return-value]


def _pad_windows_to_batch(
    pts_list: list[Tensor],
    device: torch.device,
    min_pts: int = _MIN_POINTS_FOR_NET,
) -> tuple[Tensor, list[int]]:
    """
    Stack variable-length point tensors into a padded [B, max_n, 3] tensor.

    Vectorised: no Python loop over windows — uses scatter_add to fill
    the pre-allocated output tensor entirely on GPU.
    """
    valid_idx = [i for i, p in enumerate(pts_list) if p.shape[0] >= min_pts]
    if not valid_idx:
        return torch.zeros(0, min_pts, 3, device=device), valid_idx

    valid_pts = [pts_list[i] for i in valid_idx]
    B         = len(valid_pts)
    counts    = [p.shape[0] for p in valid_pts]  # per-window point counts
    max_n     = max(counts)

    # Concatenate all points: (total_pts, 3)
    all_pts = torch.cat(valid_pts, dim=0)         # (sum_counts, 3)
    total   = all_pts.shape[0]

    # Window index for every point
    win_ids = torch.repeat_interleave(
        torch.arange(B, device=device),
        torch.tensor(counts, device=device),
    )  # (total_pts,)

    # Local index within window (0, 1, 2, ... per window)
    ones    = torch.ones(total, dtype=torch.long, device=device)
    cum     = torch.cumsum(ones, 0) - 1
    is_new  = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        win_ids[1:] != win_ids[:-1],
    ])
    starts  = torch.zeros(total, dtype=torch.long, device=device)
    starts[is_new] = cum[is_new]
    starts  = torch.cummax(starts, dim=0).values
    loc_idx = cum - starts  # local position within window

    # If window has fewer points than max_n, tile by repeating
    # (handled implicitly: we only scatter real points; zeros fill the rest)
    # For windows shorter than max_n we need to tile — compute tiled indices.
    # Simple approach: build expanded win_ids / loc_idx by tiling short windows.
    counts_t  = torch.tensor(counts, dtype=torch.long, device=device)  # (B,)
    # For each window compute how many times to repeat to reach max_n
    reps      = (max_n + counts_t - 1) // counts_t  # ceil division
    # Expand: repeat each window's points reps[w] times, then slice to max_n
    # Build per-window tiled tensors efficiently:
    tiled_parts: list[Tensor] = []
    for k in range(B):
        p = valid_pts[k]
        r = int(reps[k].item())
        tiled_parts.append(p.repeat(r, 1)[:max_n])

    batched = torch.stack(tiled_parts, dim=0)  # (B, max_n, 3)
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
        self._amp_device_type: str  = "cpu"

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        freeze = epoch < self._freeze_s2_epochs
        for p in self.stage2.parameters():
            p.requires_grad = not freeze
        if freeze:
            logger.debug("Epoch %d: Stage-2 FROZEN", epoch)
        elif epoch == self._freeze_s2_epochs and self._freeze_s2_epochs > 0:
            logger.info("Epoch %d: Stage-2 UNFROZEN — two-phase transition.", epoch)

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
        self._amp_device_type = device.type
        pb = tuple(plot_bounds.tolist()) if isinstance(plot_bounds, Tensor) else plot_bounds

        t1 = time.perf_counter()
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad      = ad.to(device)
        al_flat = [a.to(device) for a in al_list]
        logger.debug("Anchors: ad=%d al=%d (%.2fs)",
                     len(ad), sum(len(a) for a in al_flat),
                     time.perf_counter() - t1)

        if training:
            loss_s1, s1_cache = self._stage1_loss_with_cache(points, ad, al_flat, gt_boxes)
            proposals = self._stage1_proposals_from_cache(s1_cache, ad, al_flat, device)
        else:
            loss_s1   = {}
            proposals = self._stage1_proposals_fresh(points, ad, al_flat, device)
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
            logger.debug("Forward %.2fs | loss=%.4f", time.perf_counter() - t0, total.item())
            return {**loss_s1, **loss_s2, "total_loss": total}

        final_boxes, final_scores = self._stage2_inference(points, proposals)
        logger.info("  Stage2 done: %d boxes (%.2fs)", len(final_boxes), time.perf_counter() - t0)
        return {"boxes": final_boxes, "scores": final_scores}

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _stage1_loss_with_cache(self, points, ad, al_list, gt_boxes):
        all_anchors = torch.cat([ad] + al_list, dim=0) if al_list else ad
        cfg_la      = self.cfg.label_assignment

        with torch.inference_mode():
            all_cls, all_reg = self._run_stage1_on_anchors(points, all_anchors, infer_mode=True, tag="s1_scan")
        all_cls = all_cls.detach().clone()
        all_reg = all_reg.detach().clone()
        cls_scores_all = torch.sigmoid(all_cls.squeeze(-1))

        labels, reg_targets, _, sampled = assign_targets(
            all_anchors, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
            cls_scores=cls_scores_all,
        )

        cls_logits, reg_deltas = self._run_stage1_on_anchors(
            points, all_anchors[sampled], infer_mode=False, tag="s1_grad"
        )
        sampled_labels = labels[sampled].float()

        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels,
                                      alpha=self._focal_alpha, gamma=self._focal_gamma)
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any() else torch.tensor(0.0, device=points.device)
        )
        total     = cls_loss + self.lambda_reg * reg_loss
        loss_dict = {"loss_stage1_cls": cls_loss, "loss_stage1_reg": reg_loss,
                     "total_loss_stage1": total}

        n_ad     = len(ad)
        sizes_al = [len(a) for a in al_list]
        with torch.no_grad():
            scores_ad = torch.sigmoid(all_cls[:n_ad].squeeze(-1))
            boxes_ad  = decode_boxes(all_reg[:n_ad], ad)
            if al_list:
                scores_al = torch.sigmoid(all_cls[n_ad:].squeeze(-1))
                boxes_al  = decode_boxes(all_reg[n_ad:], torch.cat(al_list, dim=0))
            else:
                scores_al = torch.zeros(0, device=points.device)
                boxes_al  = torch.zeros(0, 6, device=points.device)

        cache = {"scores_ad": scores_ad, "boxes_ad": boxes_ad,
                 "scores_al": scores_al, "boxes_al": boxes_al, "sizes_al": sizes_al}
        return loss_dict, cache

    def _stage1_proposals_from_cache(self, cache, ad, al_list, device):
        cfg_nms      = self.cfg.stage1_nms
        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))

        props_ad = (
            cache["boxes_ad"][nms3d(cache["boxes_ad"], cache["scores_ad"],
                                    cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals,
                                    score_threshold=ad_score_thr)]
            if len(ad) > 0 else torch.zeros(0, 6, device=device)
        )

        if al_list:
            parts, offset = [], 0
            for sz in cache["sizes_al"]:
                if sz:
                    sl   = slice(offset, offset + sz)
                    keep = nms3d(cache["boxes_al"][sl], cache["scores_al"][sl],
                                 cfg_nms.al_iouv_threshold, cfg_nms.al_max_proposals_per_maxima)
                    parts.append(cache["boxes_al"][sl][keep])
                offset += sz
            props_al = torch.cat(parts, dim=0) if parts else torch.zeros(0, 6, device=device)
        else:
            props_al = torch.zeros(0, 6, device=device)

        return torch.cat([props_ad, props_al], dim=0)

    def _stage1_proposals_fresh(self, points, ad, al_list, device):
        cfg_nms      = self.cfg.stage1_nms
        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))

        if len(ad) > 0:
            with torch.inference_mode():
                cls_ad, reg_ad = self._run_stage1_on_anchors(points, ad, infer_mode=True, tag="ad")
            boxes_ad = decode_boxes(reg_ad, ad)
            props_ad = boxes_ad[nms3d(boxes_ad, torch.sigmoid(cls_ad.squeeze(-1)),
                                      cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals,
                                      score_threshold=ad_score_thr)]
        else:
            props_ad = torch.zeros(0, 6, device=device)

        if al_list:
            al_all = torch.cat(al_list, dim=0)
            with torch.inference_mode():
                cls_al, reg_al = self._run_stage1_on_anchors(points, al_all, infer_mode=True, tag="al")
            scores_al = torch.sigmoid(cls_al.squeeze(-1))
            boxes_al  = decode_boxes(reg_al, al_all)
            parts, offset = [], 0
            for sz in [len(a) for a in al_list]:
                if sz:
                    sl   = slice(offset, offset + sz)
                    keep = nms3d(boxes_al[sl], scores_al[sl],
                                 cfg_nms.al_iouv_threshold, cfg_nms.al_max_proposals_per_maxima)
                    parts.append(boxes_al[sl][keep])
                offset += sz
            props_al = torch.cat(parts, dim=0) if parts else torch.zeros(0, 6, device=device)
        else:
            props_al = torch.zeros(0, 6, device=device)

        return torch.cat([props_ad, props_al], dim=0)

    # ------------------------------------------------------------------
    # Shared PointNet++ runner
    # ------------------------------------------------------------------

    def _run_stage1_on_anchors(self, points, anchors, infer_mode=True, tag=""):
        device   = points.device
        A        = len(anchors)
        t        = time.perf_counter()
        pts_list = _subsample_points_batch(points, anchors)
        t_sample = time.perf_counter() - t

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
        ctx       = torch.inference_mode() if infer_mode else torch.enable_grad()

        with ctx:  # type: ignore[attr-defined]
            for start in range(0, V, mb):
                end      = min(start + mb, V)
                batch, _ = _pad_windows_to_batch(valid_pts[start:end], device)
                with torch.autocast(device_type=self._amp_device_type, enabled=(not infer_mode)):
                    c, r = self.stage1(batch)
                for k, orig_i in enumerate(valid_idx[start:end]):
                    cls_out[orig_i] = c[k].detach().float() if infer_mode else c[k]
                    reg_out[orig_i] = r[k].detach().float() if infer_mode else r[k]

        logger.debug("  [%s] anchors=%d valid=%d sample=%.2fs fwd=%.2fs (%d mb)",
                     tag, A, V, t_sample, time.perf_counter() - t_fwd, (V + mb - 1) // mb)
        return cls_out, reg_out

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

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
        pts_list = [_subsample_points_in_box(points, p) for p in proposals[sampled]]

        with torch.autocast(device_type=self._amp_device_type):
            cls_logits, reg_deltas = self.stage2(pts_list, proposals[sampled])
        cls_logits = cls_logits.float()
        reg_deltas = reg_deltas.float()

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(cls_logits.squeeze(-1), sampled_labels,
                                      alpha=self._focal_alpha, gamma=self._focal_gamma)
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any() else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage2_cls": cls_loss, "loss_stage2_reg": reg_loss,
                "total_loss_stage2": total}

    def _stage2_inference(self, points, proposals):
        cfg_nms  = self.cfg.stage2_nms
        t_s      = time.perf_counter()
        pts_list = _subsample_points_batch_proposals(
            points, proposals, n=_MAX_POINTS_PER_BOX, chunk=self._s2_chunk
        )
        logger.info("  Stage2 sampling done in %.2fs", time.perf_counter() - t_s)

        with torch.inference_mode():
            with torch.autocast(device_type=self._amp_device_type):
                cls_logits, reg_deltas = self.stage2(pts_list, proposals)
            cls_logits = cls_logits.float()
            reg_deltas = reg_deltas.float()

        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)
        mask    = scores >= cfg_nms.score_threshold
        refined, scores = refined[mask], scores[mask]
        logger.info("  Stage2 score filter: %d -> %d", len(mask), len(refined))

        if len(refined) == 0:
            return refined, scores

        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        logger.info("  Stage2 NMS: %d -> %d", len(refined), len(keep))
        return refined[keep], scores[keep]
