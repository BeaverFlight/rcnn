"""
models/tree_rcnn_v2.py — TreeRCNN v2.0

Отличия от v1 (tree_rcnn.py):
  - Backbone: дополнительный SA-extra слой + global SA
  - FPN поверх SA-иерархии (полностью включен, не dead code)
  - Stage 1: ProposalHead получает FPN-признаки из p2-уровня
  - Stage 2: RefinementHeadV2 (Offset + Center-ness + FC-neck)
  - Stage 3: RelationHead (блендинг Stage2+Stage3 scores)
  - stage2_loss_v2 вместо _stage2_loss
  - Все мемори-оптимизации v1 сохранены

Fix (Stage-3 grad graph):
  Старый код писал inplace в feats_buf[start:end] = x.float(),
  что разрывало autograd между SA-слоями и feat_proj.
  Теперь: чанк-выходы копятся в list, torch.cat вне цикла,
  затем feat_proj применяется к целому тензору без разрыва.

Fix (FPN):
  PointNeXtEncoder запускается один раз на всё облако.
  TreeFPN строит p2/p3/p4 из SA2/SA3/global.
  p2 передаётся в _run_stage1_on_anchors → ProposalHead.

Fix (Stage-3 blending):
  inference возвращает alpha*s3 + (1-alpha)*s2
  (cfg.training.stage3_blend_alpha, дефолт 0.7).
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
from models.stage2.refinement_head_v2 import RefinementHeadV2
from models.fpn import TreeFPN
from models.backbone.pointnext_modules import PointNeXtEncoder
from models.relation_head import RelationHead
from models.losses import stage2_loss_v2
from models.losses.focal_loss import sigmoid_focal_loss
from models.losses.smooth_l1 import smooth_l1_loss
from ops.nms3d import nms3d
from utils.box_coder import decode_boxes

from models.tree_rcnn import (
    _subsample_points_batch,
    _subsample_points_batch_proposals,
    _subsample_points_loss,
    _pad_windows_to_batch,
    _MAX_POINTS_PER_BOX,
    _MIN_POINTS_FOR_NET,
    _ANCHOR_CHUNK,
    _STAGE1_INFER_BATCH,
    _STAGE2_INFER_CHUNK,
)

logger = logging.getLogger(__name__)


class TreeRCNNV2(nn.Module):
    """
    TreeRCNN v2.0.

    Forward-пасс:
      1. PointNeXtEncoder(points) → [f1, f2, f3, f4]   (SA-иерархия)
      2. TreeFPN(f2, f3, f4)     → p2, p3, p4
      3. Stage-1 с p2 как дополнительным признаком
      4. NMS → proposals
      5. Stage-2 (тренировка: loss_v2 / инференс: refined boxes + s2_scores)
      6. Stage-3 RelationHead (тренировка: loss / инференс: blended scores)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg        = cfg
        self.anchor_gen = AnchorGenerator(cfg)

        # ─ FPN backbone (запускается один раз в forward, даёт f2/f3/f4) ──────────
        self.encoder = PointNeXtEncoder(cfg)
        enc_dims     = self.encoder.out_dims          # [C1, C2, C3, C4]

        fpn_cfg = getattr(cfg, 'fpn', None)
        fpn_out = int(fpn_cfg.out_channels) if fpn_cfg else 256
        self.fpn = TreeFPN(
            in_channels=[enc_dims[1], enc_dims[2], enc_dims[3]],
            out_channels=fpn_out,
        )
        self._fpn_out = fpn_out

        # ─ Stage 1 ───────────────────────────────────────────────────
        self.stage1 = ProposalHead(cfg)

        # ─ Stage 2 ───────────────────────────────────────────────────
        self.stage2 = RefinementHeadV2(cfg)

        # ─ Stage 3 (Relation Head) ───────────────────────────────────
        rel_cfg  = getattr(cfg, 'relation_head', None)
        rel_fdim = int(rel_cfg.feat_dim)  if rel_cfg else self.stage2.extractor.out_dim
        rel_cdim = int(rel_cfg.coord_dim) if rel_cfg else 5
        rel_nh   = int(rel_cfg.n_heads)   if rel_cfg else 8
        rel_nl   = int(rel_cfg.n_layers)  if rel_cfg else 2
        self.relation_head = RelationHead(
            feat_dim=rel_fdim, coord_dim=rel_cdim,
            n_heads=rel_nh, n_layers=rel_nl,
        )
        s2_out = self.stage2.extractor.out_dim
        self.feat_proj = (
            nn.Linear(s2_out, rel_fdim)
            if s2_out != rel_fdim else nn.Identity()
        )

        # ─ Гиперпараметры ───────────────────────────────────────────
        self.lambda_reg:     float = cfg.training.lambda_reg
        self.lambda_v_reg:   float = float(cfg.training.get("lambda_v_reg",  1.0))
        self.lambda_stage3:  float = float(cfg.training.get("lambda_stage3", 0.5))
        self._s3_blend_alpha: float = float(
            cfg.training.get("stage3_blend_alpha", 0.7)
        )
        self._stage3_enabled: bool = True

        fl = cfg.training.get("focal_loss", {})
        self._focal_alpha: float = float(fl.get("alpha", 0.25) if fl else 0.25)
        self._focal_gamma: float = float(fl.get("gamma", 2.0)  if fl else 2.0)

        self._s2_chunk:      int = int(cfg.training.get("stage2_infer_chunk",   _STAGE2_INFER_CHUNK))
        self._s2_fwd_chunk:  int = int(cfg.training.get("stage2_forward_chunk", 256))
        self._freeze_s2_epochs: int = int(cfg.training.get("freeze_stage2_epochs", 0))
        self._freeze_s3_epochs: int = int(cfg.training.get("freeze_stage3_epochs", 50))
        self._current_epoch: int   = 0
        self._amp_device_type: str = "cpu"
        self._cuda: bool           = False

    # ------------------------------------------------------------------
    # Epoch callback
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

        freeze_s2 = epoch < self._freeze_s2_epochs
        for p in self.stage2.parameters():
            p.requires_grad = not freeze_s2

        freeze_s3 = epoch < self._freeze_s3_epochs
        for p in self.relation_head.parameters():
            p.requires_grad = not freeze_s3
        for p in self.feat_proj.parameters():
            p.requires_grad = not freeze_s3
        self._stage3_enabled = not freeze_s3

        if freeze_s2:
            logger.debug("Epoch %d: Stage-2 FROZEN", epoch)
        if freeze_s3:
            logger.debug("Epoch %d: Stage-3 FROZEN", epoch)
        elif epoch == self._freeze_s3_epochs:
            logger.info("Epoch %d: Stage-3 ENABLED (blend_alpha=%.2f)",
                        epoch, self._s3_blend_alpha)

    # ------------------------------------------------------------------
    # Public forward
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
        self._amp_device_type = device.type
        self._cuda = device.type == "cuda"
        pb = tuple(plot_bounds.tolist()) if isinstance(plot_bounds, Tensor) else plot_bounds

        # ─ FPN backbone (1 пасс на всё облако) ───────────────────────
        t_fpn = time.perf_counter()
        xyz_in = points.unsqueeze(0)       # (1, N, 3)
        with torch.autocast(device_type=device.type):
            f1, f2, f3, f4 = self.encoder(xyz_in)           # SA-уровни
            p2, p3, p4     = self.fpn(f2, f3, f4)           # FPN-уровни
        # p2: (1, S2, fpn_out) — используется Stage-1 как context
        fpn_p2 = p2.squeeze(0).detach()   # (S2, fpn_out), no grad in stage1 scan
        logger.debug("FPN done in %.2fs (S2=%d)", time.perf_counter() - t_fpn, fpn_p2.shape[0])

        # ─ Stage 1 ───────────────────────────────────────────────────
        t1 = time.perf_counter()
        ad, al_list = self.anchor_gen.generate_all(pb, local_maxima.cpu().numpy())
        ad      = ad.to(device)
        al_flat = [a.to(device) for a in al_list]
        logger.debug("Anchors: ad=%d (%.2fs)", len(ad), time.perf_counter() - t1)

        if training:
            loss_s1, s1_cache = self._stage1_loss_with_cache(
                points, ad, al_flat, gt_boxes, fpn_p2
            )
            proposals = self._stage1_proposals_from_cache(s1_cache, ad, al_flat, device)
        else:
            loss_s1   = {}
            proposals = self._stage1_proposals_fresh(points, ad, al_flat, device, fpn_p2)
            logger.info("  Stage1 -> %d proposals", len(proposals))

        del ad, al_flat, fpn_p2, f1, f2, f3, f4, p2, p3, p4
        if self._cuda:
            torch.cuda.empty_cache()

        if len(proposals) == 0:
            zero = torch.tensor(0.0, device=device)
            if training:
                return {**loss_s1,
                        "loss_stage2_cls": zero, "loss_stage2_reg": zero,
                        "loss_stage3": zero,
                        "total_loss": loss_s1.get("total_loss_stage1", zero)}
            return {"boxes": torch.zeros(0, 6, device=device),
                    "scores": torch.zeros(0, device=device)}

        # ─ Stage 2 ───────────────────────────────────────────────────
        if training:
            loss_s2 = self._stage2_loss_v2(points, proposals.detach(), gt_boxes)
            loss_s3 = {"loss_stage3": torch.tensor(0.0, device=device)}
            if self._stage3_enabled and len(proposals) > 0:
                loss_s3 = self._stage3_loss(points, proposals.detach(), gt_boxes)

            total = (
                loss_s1.get("total_loss_stage1", torch.tensor(0.0, device=device))
                + loss_s2["total_loss_stage2"]
                + loss_s3["loss_stage3"]
            )
            if torch.isnan(total) or torch.isinf(total):
                logger.warning("NaN/Inf total_loss — skipping batch")
                total = torch.tensor(0.0, device=device, requires_grad=True)
            logger.debug("Forward %.2fs | loss=%.4f",
                         time.perf_counter() - t0, total.item())
            return {**loss_s1, **loss_s2, **loss_s3, "total_loss": total}

        final_boxes, s2_scores = self._stage2_inference(points, proposals)
        final_scores = s2_scores
        if len(final_boxes) > 0 and self._stage3_enabled:
            final_scores = self._stage3_inference(points, final_boxes, s2_scores)
        logger.info("  Final: %d boxes (%.2fs)",
                    len(final_boxes), time.perf_counter() - t0)
        return {"boxes": final_boxes, "scores": final_scores}

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------
    def _stage1_loss_with_cache(self, points, ad, al_list, gt_boxes, fpn_p2):
        all_anchors = torch.cat([ad] + al_list, dim=0) if al_list else ad
        cfg_la = self.cfg.label_assignment

        with torch.inference_mode():
            _cls_raw, _reg_raw = self._run_stage1_on_anchors(
                points, all_anchors, fpn_p2, infer_mode=True, tag="s1_scan"
            )
        all_cls = _cls_raw.detach().clone()
        all_reg = _reg_raw.detach().clone()
        del _cls_raw, _reg_raw
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

        # fpn_p2 для gradient pass без detach (grad проходит через ProposalHead)
        fpn_p2_grad = fpn_p2  # уже detach() в forward(), stage1 всегда detached
        cls_logits, reg_deltas = self._run_stage1_on_anchors(
            points, all_anchors[sampled], fpn_p2_grad, infer_mode=False, tag="s1_grad"
        )
        sampled_labels = labels[sampled].float()
        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha, gamma=self._focal_gamma,
        )
        pos_mask = sampled_labels == 1
        reg_loss = (
            smooth_l1_loss(reg_deltas[pos_mask], reg_targets[sampled][pos_mask])
            if pos_mask.any() else torch.tensor(0.0, device=points.device)
        )
        total = cls_loss + self.lambda_reg * reg_loss
        loss_dict = {
            "loss_stage1_cls":   cls_loss,
            "loss_stage1_reg":   reg_loss,
            "total_loss_stage1": total,
        }

        n_ad = len(ad)
        with torch.no_grad():
            scores_ad = torch.sigmoid(all_cls[:n_ad].squeeze(-1))
            boxes_ad  = decode_boxes(all_reg[:n_ad], ad)
            if al_list:
                scores_al = torch.sigmoid(all_cls[n_ad:].squeeze(-1))
                boxes_al  = decode_boxes(all_reg[n_ad:], torch.cat(al_list, dim=0))
            else:
                scores_al = torch.zeros(0, device=points.device)
                boxes_al  = torch.zeros(0, 6, device=points.device)

        cache = {
            "scores_ad": scores_ad, "boxes_ad": boxes_ad,
            "scores_al": scores_al, "boxes_al": boxes_al,
            "sizes_al":  [len(a) for a in al_list],
        }
        del all_cls, all_reg
        return loss_dict, cache

    def _stage1_proposals_from_cache(self, cache, ad, al_list, device):
        cfg_nms = self.cfg.stage1_nms
        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))
        props_ad = (
            cache["boxes_ad"][nms3d(
                cache["boxes_ad"], cache["scores_ad"],
                cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals,
                score_threshold=ad_score_thr,
            )]
            if len(ad) > 0 else torch.zeros(0, 6, device=device)
        )
        if al_list:
            parts, offset = [], 0
            for sz in cache["sizes_al"]:
                if sz:
                    sl   = slice(offset, offset + sz)
                    keep = nms3d(
                        cache["boxes_al"][sl], cache["scores_al"][sl],
                        cfg_nms.al_iouv_threshold,
                        cfg_nms.al_max_proposals_per_maxima,
                    )
                    parts.append(cache["boxes_al"][sl][keep])
                offset += sz
            props_al = torch.cat(parts) if parts else torch.zeros(0, 6, device=device)
        else:
            props_al = torch.zeros(0, 6, device=device)
        return torch.cat([props_ad, props_al], dim=0)

    def _stage1_proposals_fresh(self, points, ad, al_list, device, fpn_p2):
        cfg_nms = self.cfg.stage1_nms
        ad_score_thr = float(getattr(cfg_nms, "ad_score_threshold", 0.0))
        if len(ad) > 0:
            with torch.inference_mode():
                cls_ad, reg_ad = self._run_stage1_on_anchors(
                    points, ad, fpn_p2, infer_mode=True, tag="ad"
                )
            boxes_ad = decode_boxes(reg_ad, ad)
            props_ad = boxes_ad[
                nms3d(boxes_ad, torch.sigmoid(cls_ad.squeeze(-1)),
                      cfg_nms.ad_iouv_threshold, cfg_nms.ad_max_proposals,
                      score_threshold=ad_score_thr)
            ]
        else:
            props_ad = torch.zeros(0, 6, device=device)

        if al_list:
            al_all = torch.cat(al_list)
            with torch.inference_mode():
                cls_al, reg_al = self._run_stage1_on_anchors(
                    points, al_all, fpn_p2, infer_mode=True, tag="al"
                )
            scores_al = torch.sigmoid(cls_al.squeeze(-1))
            boxes_al  = decode_boxes(reg_al, al_all)
            parts, offset = [], 0
            for sz in [len(a) for a in al_list]:
                if sz:
                    sl   = slice(offset, offset + sz)
                    keep = nms3d(
                        boxes_al[sl], scores_al[sl],
                        cfg_nms.al_iouv_threshold,
                        cfg_nms.al_max_proposals_per_maxima,
                    )
                    parts.append(boxes_al[sl][keep])
                offset += sz
            props_al = torch.cat(parts) if parts else torch.zeros(0, 6, device=device)
        else:
            props_al = torch.zeros(0, 6, device=device)
        return torch.cat([props_ad, props_al], dim=0)

    def _run_stage1_on_anchors(
        self,
        points: Tensor,
        anchors: Tensor,
        fpn_p2: Tensor,
        infer_mode: bool = True,
        tag: str = "",
    ) -> tuple[Tensor, Tensor]:
        """
        Запускает ProposalHead на батче анкоров, передаёт FPN-признаки
        через разовый fpn_context параметр ProposalHead.forward().

        fpn_p2 : (S2, fpn_out) — уже detach'ed (Stage-1 не градиентирует енкодер).
        """
        device   = points.device
        A        = len(anchors)
        pts_list = _subsample_points_batch(points, anchors)

        valid_idx = [
            i for i, p in enumerate(pts_list)
            if p.shape[0] >= _MIN_POINTS_FOR_NET
        ]
        cls_out = torch.zeros(A, 1, device=device)
        reg_out = torch.zeros(A, 6, device=device)
        if not valid_idx:
            logger.warning("  [%s] 0 valid anchors/%d", tag, A)
            return cls_out, reg_out

        valid_pts = [pts_list[i] for i in valid_idx]
        mb  = int(getattr(self.cfg.training, "stage1_infer_batch", _STAGE1_INFER_BATCH))
        ctx = torch.inference_mode() if infer_mode else torch.enable_grad()

        with ctx:
            for start in range(0, len(valid_idx), mb):
                end      = min(start + mb, len(valid_idx))
                batch, _ = _pad_windows_to_batch(valid_pts[start:end], device)
                # Интерполируем p2 до размера батча (B_c, fpn_out)
                # для каждого анкора берём mean по всем S2 точкам —
                # global context всего плота
                B_c = end - start
                fpn_ctx = fpn_p2.mean(0, keepdim=True).expand(B_c, -1)  # (B_c, fpn_out)

                with torch.autocast(
                    device_type=self._amp_device_type,
                    enabled=(not infer_mode),
                ):
                    c, r = self.stage1(batch, fpn_context=fpn_ctx)

                for k, orig_i in enumerate(valid_idx[start:end]):
                    cls_out[orig_i] = c[k].detach().float() if infer_mode else c[k]
                    reg_out[orig_i] = r[k].detach().float() if infer_mode else r[k]

        del pts_list
        return cls_out, reg_out

    # ------------------------------------------------------------------
    # Stage 2 — training
    # ------------------------------------------------------------------
    def _stage2_loss_v2(
        self, points: Tensor, proposals: Tensor, gt_boxes: Tensor
    ) -> dict:
        cfg_la = self.cfg.label_assignment
        device = points.device

        labels, reg_targets, _, sampled = assign_targets(
            proposals, gt_boxes,
            pos_iouv=cfg_la.positive_iouv_overlap,
            pos_ioub=cfg_la.positive_ioub_overlap,
            pos_iouh=cfg_la.positive_iouh_overlap,
            n_pos=self.cfg.training.n_positive,
            n_neg=self.cfg.training.n_negative,
        )

        sampled_proposals = proposals[sampled]
        sampled_labels    = labels[sampled].float()
        sampled_reg_tgt   = reg_targets[sampled]
        S = len(sampled)

        pts_list_cpu = _subsample_points_loss(
            points.cpu(), sampled_proposals.cpu(), n=_MAX_POINTS_PER_BOX
        )

        fwd_chunk  = self._s2_fwd_chunk
        all_cls:  list[Tensor] = []
        all_reg:  list[Tensor] = []
        all_off:  list[Tensor] = []
        all_cent: list[Tensor] = []
        all_xyz:  list[Tensor] = []

        for start in range(0, S, fwd_chunk):
            end = min(start + fwd_chunk, S)
            chunk_pts  = [
                p.to(device, non_blocking=True) for p in pts_list_cpu[start:end]
            ]
            chunk_prop = sampled_proposals[start:end]

            with torch.autocast(device_type=self._amp_device_type):
                c, r, off, cent, xyz = self.stage2.forward_train(chunk_pts, chunk_prop)

            all_cls.append(c.float())
            all_reg.append(r.float())
            if off is not None:
                all_off.append(off)
                all_cent.append(cent)
                all_xyz.append(xyz)

            del chunk_pts
            if self._cuda:
                torch.cuda.empty_cache()

        cls_logits = torch.cat(all_cls, dim=0)
        reg_deltas = torch.cat(all_reg, dim=0)
        del all_cls, all_reg

        cls_loss = sigmoid_focal_loss(
            cls_logits.squeeze(-1), sampled_labels,
            alpha=self._focal_alpha, gamma=self._focal_gamma,
        )
        pos_mask = sampled_labels == 1
        if pos_mask.any():
            pred    = reg_deltas[pos_mask]
            tgt     = sampled_reg_tgt[pos_mask]
            loss_xy = smooth_l1_loss(pred[:, [0, 1, 3, 4]], tgt[:, [0, 1, 3, 4]])
            loss_vh = smooth_l1_loss(pred[:, [2, 5]],       tgt[:, [2, 5]])
            reg_loss = loss_xy + self.lambda_v_reg * loss_vh
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total = cls_loss + self.lambda_reg * reg_loss

        l_off = l_cent = torch.tensor(0.0, device=device)
        if all_off:
            off_cat  = torch.cat(all_off,  dim=0).to(device)
            cent_cat = torch.cat(all_cent, dim=0).to(device)
            xyz_cat  = torch.cat(all_xyz,  dim=0).to(device)
            del all_off, all_cent, all_xyz

            if pos_mask.any():
                ld = stage2_loss_v2(
                    cls_score=cls_logits,
                    reg_delta=reg_deltas,
                    pred_offsets=off_cat,
                    pred_centerness=cent_cat,
                    points_xyz=xyz_cat,
                    gt_box=sampled_reg_tgt,
                    gt_label=sampled_labels,
                    lambdas={
                        'cls':        0.0,
                        'reg':        0.0,
                        'offset':     float(getattr(self.cfg.training, 'lambda_offset',     0.5)),
                        'centerness': float(getattr(self.cfg.training, 'lambda_centerness', 0.5)),
                    },
                )
                l_off  = ld['offset']
                l_cent = ld['centerness']

            del off_cat, cent_cat, xyz_cat
            if self._cuda:
                torch.cuda.empty_cache()

        total = total + l_off + l_cent
        return {
            "loss_stage2_cls":    cls_loss,
            "loss_stage2_reg":    reg_loss,
            "loss_stage2_offset": l_off,
            "loss_stage2_cent":   l_cent,
            "total_loss_stage2":  total,
        }

    # ------------------------------------------------------------------
    # Stage 2 — inference
    # ------------------------------------------------------------------
    def _stage2_inference(
        self, points: Tensor, proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        cfg_nms  = self.cfg.stage2_nms
        pts_list = _subsample_points_batch_proposals(
            points, proposals, n=_MAX_POINTS_PER_BOX, chunk=self._s2_chunk
        )
        with torch.inference_mode():
            with torch.autocast(device_type=self._amp_device_type):
                cls_logits, reg_deltas = self.stage2(pts_list, proposals)
            cls_logits = cls_logits.float()
            reg_deltas = reg_deltas.float()

        scores  = torch.sigmoid(cls_logits.squeeze(-1))
        refined = decode_boxes(reg_deltas, proposals)
        mask    = scores >= cfg_nms.score_threshold
        refined, scores = refined[mask], scores[mask]

        del cls_logits, reg_deltas, mask
        if self._cuda:
            torch.cuda.empty_cache()

        if len(refined) == 0:
            return refined, scores
        keep = nms3d(refined, scores, cfg_nms.iouv_threshold)
        return refined[keep], scores[keep]

    # ------------------------------------------------------------------
    # Stage 3 helpers — извлечение семантических признаков
    # ------------------------------------------------------------------
    def _extract_s3_feats(
        self, points: Tensor, proposals: Tensor, grad: bool
    ) -> Tensor:
        """
        Извлекает семантические признаки каждого proposal.

        Fix (grad graph):
            Старое inplace feats_buf[i:j] = x.float() разрывало autograd
            между SA-слоями и feat_proj.

            Теперь:
              - чанк-выходы собираются в list[Tensor]
              - torch.cat вызывается один раз вне цикла
              - feat_proj применяется к целому тензору
              → градиент идёт сквозь SA → pool → proj → feat_proj без разрывов

        Возвращает: (1, N, rel_feat_dim)
        """
        device = points.device
        N      = len(proposals)

        pts_list_cpu = _subsample_points_loss(
            points.cpu(), proposals.cpu(), n=_MAX_POINTS_PER_BOX
        )

        chunk    = self._s2_fwd_chunk
        feat_buf: list[Tensor] = []   # собираем чанки, не inplace

        ctx = torch.enable_grad() if grad else torch.inference_mode()
        with ctx:
            for start in range(0, N, chunk):
                end        = min(start + chunk, N)
                chunk_pts  = [
                    p.to(device, non_blocking=True)
                    for p in pts_list_cpu[start:end]
                ]
                chunk_prop = proposals[start:end]

                with torch.autocast(
                    device_type=self._amp_device_type, enabled=grad
                ):
                    pw_feats, pw_xyz = self.stage2._extract_pw_features(
                        self._pad_pts(chunk_pts, device), chunk_prop
                    )  # (B_c, S2, C2)

                    head  = self.stage2.stage2_head
                    cent  = head.centerness_head(pw_feats)              # (B_c, S2, 1)
                    w_att = cent / (cent.sum(dim=1, keepdim=True) + 1e-6)
                    concat = torch.cat(
                        [pw_xyz + head.offset_head(pw_feats), pw_feats],
                        dim=-1,
                    )                                                   # (B_c, S2, 3+C2)
                    g_feat = (concat * w_att).sum(dim=1)                # (B_c, 3+C2)
                    x = head._fc_neck(head.proj(g_feat))                # (B_c, feat_dim)

                # если grad=False — detach перед сохранением
                feat_buf.append(x.float() if grad else x.detach().float())

                del chunk_pts, pw_feats, pw_xyz, cent, w_att, concat, g_feat, x
                if self._cuda:
                    torch.cuda.empty_cache()

        # cat один раз — граф цельный
        feats = torch.cat(feat_buf, dim=0)            # (N, stage2_feat_dim)

        # feat_proj — всегда с градиентом если grad=True
        proj_ctx = torch.enable_grad() if grad else torch.inference_mode()
        with proj_ctx:
            box_feats = self.feat_proj(feats)          # (N, rel_feat_dim)

        return box_feats.unsqueeze(0)                  # (1, N, rel_feat_dim)

    @staticmethod
    def _pad_pts(pts_list: list[Tensor], device: torch.device) -> Tensor:
        """Padding списка point-тензоров в батч (B, max_N, 3)."""
        max_n = max(p.shape[0] for p in pts_list)
        B     = len(pts_list)
        out   = torch.zeros(B, max_n, 3, device=device)
        for k, pts in enumerate(pts_list):
            n = pts.shape[0]
            if n < max_n:
                pts = pts.repeat((max_n + n - 1) // n, 1)[:max_n]
            out[k] = pts
        return out

    # ------------------------------------------------------------------
    # Stage 3 — training
    # ------------------------------------------------------------------
    def _stage3_loss(
        self, points: Tensor, proposals: Tensor, gt_boxes: Tensor
    ) -> dict:
        from models.losses import relation_loss
        from ops.iou3d import iou3d_batch

        device  = proposals.device
        MAX_S3  = int(getattr(self.cfg.training, 'stage3_max_proposals', 500))
        if len(proposals) > MAX_S3:
            proposals = proposals[:MAX_S3]
        N = len(proposals)

        with torch.no_grad():
            iou       = iou3d_batch(proposals, gt_boxes)
            gt_labels = (iou.max(dim=1).values >= 0.5).float()
        del iou

        coords    = proposals[:, [0, 1, 2, 3, 5]].unsqueeze(0)   # (1, N, 5)
        box_feats = self._extract_s3_feats(points, proposals, grad=True)

        with torch.autocast(device_type=self._amp_device_type):
            scores = self.relation_head(box_feats, coords)        # (1, N, 1)

        loss = relation_loss(
            pred_scores=scores,
            gt_labels=gt_labels.unsqueeze(0),
            lambda_rel=self.lambda_stage3,
        )

        del box_feats, scores, gt_labels, coords
        if self._cuda:
            torch.cuda.empty_cache()
        return {"loss_stage3": loss}

    # ------------------------------------------------------------------
    # Stage 3 — inference (blended scores)
    # ------------------------------------------------------------------
    def _stage3_inference(
        self, points: Tensor, boxes: Tensor, s2_scores: Tensor
    ) -> Tensor:
        """
        Возвращает blended score:
            final = alpha * s3_score + (1 - alpha) * s2_score

        alpha читается из cfg.training.stage3_blend_alpha (дефолт 0.7).
        При alpha=1.0 — чистый Stage-3 (старое поведение).
        При alpha=0.0 — Stage-3 отключён, остаются Stage-2 scores.
        """
        device = boxes.device
        coords = boxes[:, [0, 1, 2, 3, 5]].unsqueeze(0)

        box_feats = self._extract_s3_feats(points, boxes, grad=False)

        with torch.inference_mode():
            with torch.autocast(device_type=self._amp_device_type):
                new_sc = self.relation_head(box_feats, coords)    # (1, N, 1)
        s3_scores = new_sc.squeeze(0).squeeze(-1)                 # (N,)

        del box_feats, new_sc, coords
        if self._cuda:
            torch.cuda.empty_cache()

        alpha = self._s3_blend_alpha
        return alpha * s3_scores + (1.0 - alpha) * s2_scores
