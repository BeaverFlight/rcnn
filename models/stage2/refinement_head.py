"""Stage 2 refinement: classification + regression on Stage-1 proposals.

Fixes vs original:
  - Обработка всех proposals одним батчем через _pad_and_batch / forward батчем.
    Устраняет Python-цикл `for pts, prop in zip(...)` который блокировал GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.stage2.multi_position import MultiPositionExtractor


class RefinementHead(nn.Module):
    """
    Refines Stage-1 proposals using multi-position features.
    """

    MIN_PTS = 4  # минимум точек для осмысленного извлечения фич

    def __init__(self, cfg) -> None:
        super().__init__()
        self.extractor = MultiPositionExtractor(cfg)
        feat_dim = self.extractor.out_dim
        self.cls_head = nn.Linear(feat_dim, 1)
        self.reg_head = nn.Linear(feat_dim, 4)

    # ------------------------------------------------------------------
    def forward(
        self, points_list: list[Tensor], proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            points_list: list of (Nk, 3) — точки внутри каждого proposal
            proposals:   (P, 6)          — Stage-1 proposals

        Returns:
            cls_logits: (P, 1)
            reg_deltas: (P, 4)
        """
        if len(points_list) == 0:
            dev = proposals.device
            return (
                torch.zeros(0, 1, device=dev),
                torch.zeros(0, 4, device=dev),
            )

        feats = self._extract_batch(points_list, proposals)  # (P, feat_dim)
        return self.cls_head(feats), self.reg_head(feats)

    # ------------------------------------------------------------------
    def _extract_batch(self, points_list: list[Tensor], proposals: Tensor) -> Tensor:
        """
        Извлекает фичи для всех proposals батчем.

        Точки каждого окна паддятся до общего max_n через повторение
        (repeat-padding), чтобы сформировать тензор (P, max_n, 3).
        Пустые / малые окна заменяются нулевым вектором признаков.
        """
        device = proposals.device
        P = len(proposals)

        valid_mask = torch.zeros(P, dtype=torch.bool, device=device)
        padded_list: list[Tensor] = []
        max_n = 0

        for pts in points_list:
            if pts.shape[0] >= self.MIN_PTS:
                valid_mask[len(padded_list)] = True  # NOTE: ниже переприсваиваем
                max_n = max(max_n, pts.shape[0])
            padded_list.append(pts)

        # Пересчитываем valid_mask корректно
        valid_mask = torch.tensor(
            [p.shape[0] >= self.MIN_PTS for p in padded_list],
            dtype=torch.bool, device=device,
        )

        feat_dim = self.extractor.out_dim
        out = torch.zeros(P, feat_dim, device=device)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]  # индексы валидных proposals
        if len(valid_idx) == 0:
            return out

        # Паддируем только валидные окна
        valid_pts = [padded_list[i] for i in valid_idx.tolist()]
        valid_props = proposals[valid_idx]              # (V, 6)

        max_n_valid = max(p.shape[0] for p in valid_pts)
        # repeat-padding: если точек меньше max_n_valid — повторяем их по кругу
        batched = torch.zeros(len(valid_pts), max_n_valid, 3, device=device)
        for k, pts in enumerate(valid_pts):
            n = pts.shape[0]
            if n < max_n_valid:
                reps = (max_n_valid + n - 1) // n
                pts_ext = pts.repeat(reps, 1)[:max_n_valid]
            else:
                pts_ext = pts
            batched[k] = pts_ext

        feats = self.extractor.forward_batch(batched, valid_props)  # (V, feat_dim)
        out[valid_idx] = feats
        return out
