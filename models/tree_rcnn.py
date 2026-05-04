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
from models.stage1.target_assinger import assign_targets
from models.stage2.refinement_head import RefinementHead
from models.losses.focal_loss import sigmoid_focal_loss
from models.losses.smooth_l1 import smooth_l1_loss
from ops.nms3d import nms3d
from utils.box_coder import decode_boxes

logger = logging.getLogger(__name__)

_MAX_POINTS_PER_BOX = 512
_MIN_POINTS_FOR_NET = 4
_ANCHOR_CHUNK = 1024
_STAGE1_INFER_BATCH = 64
# Увеличен с 256 до 2048: при ~1800 proposals даёт 1 итерацию вместо 7.
# Маска (2048, N_points) строится в CUDA, CPU лишь собирает результат.
# Снизь до 512-1024 если получаешь OOM при val_max_points > 300_000.
_STAGE2_INFER_CHUNK = 2048


def _subsample_points_in_box(
    points: Tensor, box: Tensor, n: int = _MAX_POINTS_PER_BOX
) -> Tensor:
    """Single-box fallback (used only in _stage2_loss during training)."""
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
    """
    Vectorised point-in-box sampling for a batch of anchors.
    Processes anchors in chunks to avoid OOM on large point clouds.
    Returns list of (N_i, 3) tensors with points centred on each anchor.
    """
    device = points.device
    A = anchors.shape[0]
    result: list[Tensor] = [None] * A  # type: ignore[list-item]

    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    for start in range(0, A, chunk):
        end = min(start + chunk, A)
        anc = anchors[start:end]

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
        _, point_idx = mask.nonzero(as_tuple=True)
        counts_list = counts.tolist()
        groups = torch.split(point_idx, counts_list)

        for i, grp in enumerate(groups):
            if len(grp) == 0:
                result[start + i] = points.new_zeros(0, 3)
                continue
            if len(grp) > n:
                perm = torch.randperm(len(grp), device=device)[:n]
                grp = grp[perm]
            pts = points[grp].clone()
            pts[:, 0] -= anchors[start + i, 0]
            pts[:, 1] -= anchors[start + i, 1]
            result[start + i] = pts

    return result  # type: ignore[return-value]


def _subsample_points_batch_proposals(
    points: Tensor,
    proposals: Tensor,
    n: int = _MAX_POINTS_PER_BOX,
    chunk: int = _STAGE2_INFER_CHUNK,
) -> list[Tensor]:
    """
    Vectorised point-in-box sampling for Stage 2 proposals.

    Identical logic to _subsample_points_batch but WITHOUT centring
    (Stage 2 RefinementHead receives absolute coordinates).

    Key optimisation: points и proposals явно переносятся на один device
    перед построением маски, чтобы все тензорные операции выполнялись
    в CUDA, а не на CPU. Без этого broadcasting (chunk, N_points) мог
    незаметно упасть на CPU и вызвать 100% загрузку процессора.

    chunk=2048 по умолчанию: при ~1800 proposals = 1 итерация вместо 7.
    Снизь до 512-1024 при OOM (val_max_points > 300_000 + мало VRAM).
    """
    # --- Гарантируем GPU-выполнение -----------------------------------
    device = points.device
    if proposals.device != device:
        proposals = proposals.to(device)

    P = proposals.shape[0]
    result: list[Tensor] = [None] * P  # type: ignore[list-item]

    # Все координатные тензоры явно на device
    px = points[:, 0].to(device)
    py = points[:, 1].to(device)
    pz = points[:, 2].to(device)

    total_chunks = (P + chunk - 1) // chunk
    logger.info(
        "  Stage2 sampling: %d proposals / %d points / chunk=%d / total_chunks=%d",
        P, len(points), chunk, total_chunks,
    )

    for chunk_idx, start in enumerate(range(0, P, chunk), start=1):
        end = min(start + chunk, P)
        prop = proposals[start:end]  # (chunk, 6) — уже на device

        t_chunk = time.perf_counter()

        cx = prop[:, 0].unsqueeze(1)   # (chunk, 1) on device
        cy = prop[:, 1].unsqueeze(1)
        w  = prop[:, 3].unsqueeze(1)
        l  = prop[:, 4].unsqueeze(1)
        h  = prop[:, 5].unsqueeze(1)

        # mask: (chunk, N_points) — строится в CUDA
        mask = (
            (px.unsqueeze(0) >= cx - w / 2)
            & (px.unsqueeze(0) <= cx + w / 2)
            & (py.unsqueeze(0) >= cy - l / 2)
            & (py.unsqueeze(0) <= cy + l / 2)
            & (pz.unsqueeze(0) >= 0)
            & (pz.unsqueeze(0) <= h)
        )

        counts = mask.sum(dim=1)           # (chunk,) on device
        _, point_idx = mask.nonzero(as_tuple=True)
        # .tolist() — единственный CPU-переход, нужен для torch.split
        groups = torch.split(point_idx, counts.tolist())

        non_empty = 0
        total_kept = 0

        for i, grp in enumerate(groups):
            if len(grp) == 0:
                result[start + i] = points.new_zeros(0, 3)
                continue
            non_empty += 1
            if len(grp) > n:
                perm = torch.randperm(len(grp), device=device)[:n]
                grp = grp[perm]
            total_kept += len(grp)
            result[start + i] = points[grp].clone()  # absolute coords, on device

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

        fl = cfg.training.get("focal_loss", {})
        self._focal_alpha: float = float(fl.get("alpha", 0.25) if fl else 0.25)
        self._focal_gamma: float = float(fl.get("gamma", 2.0) if fl else 2.0)

        self._s2_chunk: int = int(
            cfg.training.get("stage2_infer_chunk", _STAGE2_INFER_CHUNK)
        )

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

        # ---- Anchor generation ----------------------------------------
        t1 = time.perf_counter()
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad = ad.to(device)
        al_flat = [a.to(device) for a in al_list]
        n_al = sum(len(a) for a in al_flat)
        logger.debug(
            "Anchors: ad=%d  al=%d  (%.2fs)",
            len(ad), n_al, time.perf_counter() - t1,
        )

        # ---- Stage 1 --------------------------------------------------
        if training:
            t2 = time.perf_counter()
            loss_s1 = self._stage1_loss(points, ad, al_flat, gt_boxes)
            logger.debug("S1 loss done (%.2fs)", time.perf_counter() - t2)
        else:
            loss_s1 = {}

        t3 = time.perf_counter()
        proposals = self._stage1_proposals(points, ad, al_flat, device)
        logger.debug("S1 proposals done: %d  (%.2fs)", len(proposals), time.perf_counter() - t3)

        if not training:
            logger.info(
                "  Stage1 -> %d proposals (%.2fs)",
                len(proposals), time.perf_counter() - t3,
            )

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
            logger.debug("S2 loss done (%.2fs)", time.perf_counter() - t4)

            total = (
                loss_s1.get("total_loss_stage1", torch.tensor(0.0, device=device))
                + loss_s2["total_loss_stage2"]
            )
            if torch.isnan(total) or torch.isinf(total):
                logger.warning("NaN/Inf in total_loss — skipping batch")
                total = torch.tensor(0.0, device=device, requires_grad=True)

            logger.debug(
                "Forward done %.2fs | loss=%.4f",
                time.perf_counter() - t0, total.item(),
            )
            return {**loss_s1, **loss_s2, "total_loss": total}

        t5 = time.perf_counter()
        final_boxes, final_scores = self._stage2_inference(points, proposals)
        logger.info(
            "  Stage2 inference done: %d final boxes (%.2fs total)",
            len(final_boxes), time.perf_counter() - t5,
        )
        logger.debug("Inference done (%.2fs)", time.perf_counter() - t0)
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
        logger.debug("S1 loss: %d sampled anchors", len(sampled_anchors))

        cls_logits, reg_deltas = self._run_stage1_on_anchors(
            points, sampled_anchors, infer_mode=False, tag="s1_loss"
        )
        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha,
            gamma=self._focal_gamma,
        )

        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage1_cls": cls_loss, "loss_stage1_reg": reg_loss, "total_loss_stage1": total}

    def _run_stage1_on_anchors(
        self,
        points: Tensor,
        anchors: Tensor,
        infer_mode: bool = True,
        tag: str = "",
    ) -> tuple[Tensor, Tensor]:
        device = points.device
        A = len(anchors)
        t = time.perf_counter()

        pts_list = _subsample_points_batch(points, anchors)
        t_sample = time.perf_counter() - t

        valid_idx = [i for i, p in enumerate(pts_list) if p.shape[0] >= _MIN_POINTS_FOR_NET]
        V = len(valid_idx)

        cls_out = torch.zeros(A, 1, device=device)
        reg_out = torch.zeros(A, 4, device=device)

        if not valid_idx:
            logger.warning("  [%s] 0 valid anchors/%d (all < %d pts)", tag, A, _MIN_POINTS_FOR_NET)
            return cls_out, reg_out

        valid_pts = [pts_list[i] for i in valid_idx]
        mb = int(getattr(self.cfg.training, "stage1_infer_batch", _STAGE1_INFER_BATCH))
        t_fwd = time.perf_counter()

        ctx = torch.no_grad() if infer_mode else torch.enable_grad()
        with ctx:  # type: ignore[attr-defined]
            for start in range(0, V, mb):
                end = min(start + mb, V)
                batch, _ = _pad_windows_to_batch(valid_pts[start:end], device)
                c, r = self.stage1(batch)
                for k, orig_i in enumerate(valid_idx[start:end]):
                    cls_out[orig_i] = c[k].detach() if infer_mode else c[k]
                    reg_out[orig_i] = r[k].detach() if infer_mode else r[k]

        n_mb = (V + mb - 1) // mb
        logger.debug(
            "  [%s] anchors=%d valid=%d/%d | sample=%.2fs fwd=%.2fs (%d mb)",
            tag, A, V, A,
            t_sample, time.perf_counter() - t_fwd, n_mb,
        )
        return cls_out, reg_out

    def _stage1_proposals(self, points, ad, al_list, device):
        cfg_nms = self.cfg.stage1_nms
        t0 = time.perf_counter()

        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))

        # --- ad proposals ---
        if len(ad) > 0:
            cls_ad, reg_ad = self._run_stage1_on_anchors(points, ad, infer_mode=True, tag="ad")
            scores_ad = torch.sigmoid(cls_ad.squeeze(-1))
            boxes_ad  = decode_boxes(reg_ad, ad)
            keep_ad   = nms3d(
                boxes_ad, scores_ad,
                cfg_nms.ad_iouv_threshold,
                cfg_nms.ad_max_proposals,
                score_threshold=ad_score_thr,
            )
            props_ad  = boxes_ad[keep_ad]
            logger.debug("  ad NMS: %d → %d", len(ad), len(props_ad))
        else:
            props_ad = torch.zeros(0, 6, device=device)

        # --- al proposals ---
        if al_list:
            al_all = torch.cat(al_list, dim=0)
            sizes = [len(a) for a in al_list]

            cls_al_all, reg_al_all = self._run_stage1_on_anchors(
                points, al_all, infer_mode=True, tag="al"
            )
            scores_al_all = torch.sigmoid(cls_al_all.squeeze(-1))
            boxes_al_all  = decode_boxes(reg_al_all, al_all)

            parts = []
            offset = 0
            for sz in sizes:
                if sz == 0:
                    offset += sz
                    continue
                sl = slice(offset, offset + sz)
                keep = nms3d(
                    boxes_al_all[sl], scores_al_all[sl],
                    cfg_nms.al_iouv_threshold,
                    cfg_nms.al_max_proposals_per_maxima,
                )
                parts.append(boxes_al_all[sl][keep])
                offset += sz

            props_al = torch.cat(parts, dim=0) if parts else torch.zeros(0, 6, device=device)
            logger.debug("  al NMS: %d → %d", len(al_all), len(props_al))
        else:
            props_al = torch.zeros(0, 6, device=device)

        proposals = torch.cat([props_ad, props_al], dim=0)
        logger.debug(
            "  proposals total: %d  (%.2fs)",
            len(proposals), time.perf_counter() - t0,
        )
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
        logger.debug("S2 loss: %d sampled proposals", len(sampled_props))
        # Training path: Python loop допустим — sampled proposals << всех proposals
        pts_list = [_subsample_points_in_box(points, p) for p in sampled_props]
        cls_logits, reg_deltas = self.stage2(pts_list, sampled_props)

        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha,
            gamma=self._focal_gamma,
        )
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any()
            else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        return {"loss_stage2_cls": cls_loss, "loss_stage2_reg": reg_loss, "total_loss_stage2": total}

    def _stage2_inference(self, points: Tensor, proposals: Tensor):
        """
        Stage 2 inference with vectorised point sampling.

        BEFORE: Python loop `[_subsample_points_in_box(points, p) for p in proposals]`
                → O(P * N) where P = number of proposals, N = number of points
                → hangs on large point clouds (e.g. 6063 proposals * 1.1M points)

        AFTER:  _subsample_points_batch_proposals processes `chunk` proposals at
                a time using broadcasting; the boolean mask is (chunk, N) instead
                of being rebuilt from scratch for every single proposal.
                → O(ceil(P / chunk) * chunk * N)  — same asymptotic but without
                  Python-loop overhead; runs mostly in CUDA tensor ops.
        """
        cfg_nms = self.cfg.stage2_nms

        # ---------- sampling -------------------------------------------
        t_sample = time.perf_counter()
        pts_list = _subsample_points_batch_proposals(
            points, proposals, n=_MAX_POINTS_PER_BOX, chunk=self._s2_chunk
        )
        logger.info(
            "  Stage2 sampling done in %.2fs",
            time.perf_counter() - t_sample,
        )

        # ---------- network forward ------------------------------------
        logger.info("  Stage2 forward pass started...")
        t_fwd = time.perf_counter()
        with torch.no_grad():
            cls_logits, reg_deltas = self.stage2(pts_list, proposals)
        logger.info(
            "  Stage2 forward done in %.2fs",
            time.perf_counter() - t_fwd,
        )

        # ---------- score filtering ------------------------------------
        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)

        before_thr = len(scores)
        score_mask = scores >= cfg_nms.score_threshold
        refined, scores = refined[score_mask], scores[score_mask]
        logger.info(
            "  Stage2 score filter: %d -> %d boxes (thr=%.2f)",
            before_thr, len(refined), cfg_nms.score_threshold,
        )

        if len(refined) == 0:
            logger.info("  Stage2 NMS skipped (0 boxes after score filter)")
            return refined, scores

        # ---------- NMS ------------------------------------------------
        t_nms = time.perf_counter()
        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        logger.info(
            "  Stage2 NMS done in %.2fs: %d -> %d boxes",
            time.perf_counter() - t_nms, len(refined), len(keep),
        )
        return refined[keep], scores[keep]
