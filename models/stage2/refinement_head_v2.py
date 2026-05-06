"""
models/stage2/refinement_head_v2.py — Stage 2 v2.0

Отличия от v1 (RefinementHead):
  - MultiPositionExtractor теперь отдаёт point-wise признаки ДО global pooling
  - Stage2Head заменяет cls_head + reg_head в v1
  - Stage2Head добавляет: Offset Head, Center-ness Head, FC-neck
  - Chunked forward с явным del + empty_cache между чанками
  - Возвращает (cls_logits, reg_deltas, offsets, centerness, pw_xyz)
    для использования в stage2_loss_v2 и RelationHead

Memory strategy:
  - Chunked forward: VRAM = chunk * max_n * feat_dim * fp16
  - pw_feats / pw_xyz CPU-оффлоад во время внешнего лосса (тренировка)
  - del промежуточных тензоров + empty_cache после каждого чанка
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.stage2.multi_position import MultiPositionExtractor
from models.stage2_head import Stage2Head

_S2_FORWARD_CHUNK = 256


class RefinementHeadV2(nn.Module):
    """
    Stage 2 v2.0: Voting-based refinement.

    forward() в режиме инференса возвращает:
        cls_logits : (P, 1)
        reg_deltas : (P, 6)

    forward_train() возвращает дополнительно:
        offsets    : (P, N_pad, 3)  — offset predictions (CPU)
        centerness : (P, N_pad, 1)  — centerness scores  (CPU)
        pw_xyz     : (P, N_pad, 3)  — padded point coords (CPU)
    """

    MIN_PTS = 4

    def __init__(self, cfg) -> None:
        super().__init__()
        self.extractor = MultiPositionExtractor(cfg)
        pw_dim   = self._get_pw_dim(cfg)  # размерность point-wise признаков перед global pool
        feat_dim = self.extractor.out_dim  # размерность после global pool
        self.stage2_head = Stage2Head(pw_feat_dim=pw_dim, feat_dim=feat_dim)
        self._fwd_chunk: int = int(
            cfg.training.get("stage2_forward_chunk", _S2_FORWARD_CHUNK)
        )
        self._cuda: bool = False

    @staticmethod
    def _get_pw_dim(cfg) -> int:
        """
        Размерность point-wise фич перед global pooling =
        выход sa2 (sa_layers[1].mlp[-1]).
        """
        return int(cfg.pointnet2.sa_layers[1].mlp[-1])

    # ------------------------------------------------------------------
    # Inference (compatible with tree_rcnn.py _stage2_inference)
    # ------------------------------------------------------------------
    def forward(
        self, points_list: list[Tensor], proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Совместимый интерфейс с v1 RefinementHead.
        Возвращает только cls_logits, reg_deltas.
        """
        cls_logits, reg_deltas, _, _, _ = self._chunked_forward(
            points_list, proposals, return_aux=False
        )
        return cls_logits, reg_deltas

    # ------------------------------------------------------------------
    # Training (returns aux tensors for stage2_loss_v2)
    # ------------------------------------------------------------------
    def forward_train(
        self, points_list: list[Tensor], proposals: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            cls_logits : (P, 1)       GPU
            reg_deltas : (P, 6)       GPU
            offsets    : (P, N, 3)    CPU (offload для loss-вычисления)
            centerness : (P, N, 1)    CPU
            pw_xyz     : (P, N, 3)    CPU
        """
        return self._chunked_forward(points_list, proposals, return_aux=True)

    # ------------------------------------------------------------------
    # Core chunked forward
    # ------------------------------------------------------------------
    def _chunked_forward(
        self,
        points_list: list[Tensor],
        proposals: Tensor,
        return_aux: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        device = proposals.device
        self._cuda = device.type == "cuda"
        P        = len(proposals)
        feat_dim = self.extractor.out_dim

        cls_out = torch.zeros(P, 1, device=device)
        reg_out = torch.zeros(P, 6, device=device)

        # Point-wise выходы (aux) — накапливаем на CPU
        if return_aux:
            pw_dim  = self.stage2_head.pw_feat_dim
            all_off:  list[Tensor] = []
            all_cent: list[Tensor] = []
            all_xyz:  list[Tensor] = []

        valid_idx = [
            i for i, p in enumerate(points_list)
            if p.shape[0] >= self.MIN_PTS
        ]
        if not valid_idx:
            none3 = (None, None, None)
            return cls_out, reg_out, *none3

        valid_pts   = [points_list[i] for i in valid_idx]
        valid_props = proposals[valid_idx]  # (V, 6)
        V     = len(valid_pts)
        chunk = self._fwd_chunk

        for start in range(0, V, chunk):
            end        = min(start + chunk, V)
            c_pts      = valid_pts[start:end]    # list[Tensor]
            c_props    = valid_props[start:end]  # (c, 6)

            # --- Padding внутри чанка ---
            max_n   = max(p.shape[0] for p in c_pts)
            B_c     = len(c_pts)
            batched = torch.zeros(B_c, max_n, 3, device=device)
            for k, pts in enumerate(c_pts):
                n = pts.shape[0]
                if n < max_n:
                    # tiling вместо zero-padding: меньше zeros-балласт
                    pts = pts.repeat((max_n + n - 1) // n, 1)[:max_n]
                batched[k] = pts

            # --- SA1+SA2 — point-wise фичи (до global pool) ---
            pw_feats, pw_xyz = self._extract_pw_features(batched, c_props)  # (B_c, N2, pw_dim), (B_c, N2, 3)

            # --- Stage2Head ---
            cls_c, reg_c, off_c, cent_c = self.stage2_head(pw_feats, pw_xyz)  # GPU tensors

            # --- Записываем выходы в GPU-буферы ---
            vi = valid_idx[start:end]
            for k, orig_i in enumerate(vi):
                cls_out[orig_i] = cls_c[k]
                reg_out[orig_i] = reg_c[k]

            # --- aux: offload на CPU немедленно ---
            if return_aux:
                all_off.append(off_c.detach().cpu())
                all_cent.append(cent_c.detach().cpu())
                all_xyz.append(pw_xyz.detach().cpu())

            # --- Очистка промежуточных тензоров ---
            del batched, pw_feats, pw_xyz, cls_c, reg_c, off_c, cent_c
            if self._cuda:
                torch.cuda.empty_cache()

        if return_aux:
            # Сборка aux-тензоров на CPU
            offsets_cpu    = torch.cat(all_off,  dim=0)  # (V, N2, 3)
            centerness_cpu = torch.cat(all_cent, dim=0)  # (V, N2, 1)
            xyz_cpu        = torch.cat(all_xyz,  dim=0)  # (V, N2, 3)
            del all_off, all_cent, all_xyz

            # Распаковка по valid_idx обратно в (P, N2, *)
            N2 = offsets_cpu.shape[1]
            full_off  = torch.zeros(P, N2, 3)
            full_cent = torch.zeros(P, N2, 1)
            full_xyz  = torch.zeros(P, N2, 3)
            for out_k, orig_i in enumerate(valid_idx):
                full_off[orig_i]  = offsets_cpu[out_k]
                full_cent[orig_i] = centerness_cpu[out_k]
                full_xyz[orig_i]  = xyz_cpu[out_k]
            del offsets_cpu, centerness_cpu, xyz_cpu

            return cls_out, reg_out, full_off, full_cent, full_xyz

        return cls_out, reg_out, None, None, None

    # ------------------------------------------------------------------
    # Point-wise feature extractor (SA1 + SA2, без global pool)
    # ------------------------------------------------------------------
    def _extract_pw_features(
        self, batched_xyz: Tensor, proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Запускает SA1 и SA2 из MultiPositionExtractor до global pooling.
        Возвращает point-wise фичи (B, N2, C2) и координаты (B, N2, 3)
        после SA2 (до SA3 global pool в MultiPositionExtractor).
        """
        ext  = self.extractor
        B, N, _ = batched_xyz.shape
        device  = batched_xyz.device

        x  = proposals[:, 0]
        y  = proposals[:, 1]
        w  = proposals[:, 3]
        hw = w / 2

        # Центро-нормализация (повторяем логику MultiPositionExtractor.forward_batch)
        xy_centre = torch.stack([x, y], dim=-1).unsqueeze(1)  # (B, 1, 2)
        xyz_norm  = batched_xyz.clone()
        xyz_norm[:, :, :2] -= xy_centre

        # 4 смещённых позиции
        offsets4 = torch.stack([
            torch.stack([x + hw, y     ], dim=-1),
            torch.stack([x - hw, y     ], dim=-1),
            torch.stack([x,      y + hw], dim=-1),
            torch.stack([x,      y - hw], dim=-1),
        ], dim=1)  # (B, 4, 2)
        xy_pts      = batched_xyz[:, :, :2].unsqueeze(1)          # (B, 1, N, 2)
        dxy         = xy_pts - offsets4.unsqueeze(2)              # (B, 4, N, 2)
        z_rep       = batched_xyz[:, :, 2:].unsqueeze(1).expand(B, 4, N, 1)
        offset_feat = torch.cat([dxy, z_rep], dim=-1).permute(0, 2, 1, 3).reshape(B, N, 12)

        # SA1 → point-wise
        xyz1, f1 = ext.sa1(xyz_norm, offset_feat)  # (B, S1, 64)
        # SA2 → point-wise (S2 поинтов)
        xyz2, f2 = ext.sa2(xyz1, f1)               # (B, S2, C2)

        # Чистим промежуточные тензоры
        del xyz_norm, offset_feat, dxy, z_rep, xy_pts, f1

        # f2: (B, S2, C2) channel-last — прямо в Stage2Head
        return f2, xyz2
