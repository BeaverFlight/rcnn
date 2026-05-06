"""
models/stage2_head.py — Stage 2 Refinement Head для TreeRCNN v2.0

Включает:
  - Multi-position feature extraction (5 ракурсов, как в v1)
  - Offset Head: point-wise предсказание смещения к центру ствола
  - Center-ness Head: уверенность точки в принадлежности к дереву
  - FC-neck: промежуточное представление после взвешенного пулинга
  - Финальные головы: cls_head (BCE) и reg_head (6D бокс)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2Head(nn.Module):
    """
    Args:
        pw_feat_dim : размерность point-wise признаков (выход SA-слоёв Stage2)
        feat_dim    : размерность после глобального пулинга

    Forward:
        proposal_pw_feats : (B, N, pw_feat_dim) — point-wise признаки
        proposal_xyz      : (B, N, 3) — координаты точек в proposal

    Returns:
        cls_score   : (B, 1)   — вероятность дерева
        reg_delta   : (B, 6)   — cx, cy, cz, w, l, h относительно proposal
        offsets     : (B, N, 3)— смещения точек к центру ствола
        centerness  : (B, N, 1)— уверенность точки (sigmoid)
    """

    def __init__(self, pw_feat_dim: int = 256, feat_dim: int = 1024):
        super().__init__()

        # ── Offset Head (point-wise, до пулинга) ──────────────────────────
        self.offset_head = nn.Sequential(
            nn.Linear(pw_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3),  # (dx, dy, dz)
        )

        # ── Center-ness Head (point-wise, до пулинга) ─────────────────────
        self.centerness_head = nn.Sequential(
            nn.Linear(pw_feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Проекция shifted xyz (3D) + pw_feat в пространство feat_dim
        self.proj = nn.Linear(3 + pw_feat_dim, feat_dim)

        # ── FC-neck (после взвешенного пулинга) ───────────────────────────
        self.fc_neck = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        # Near-identity init: последний Linear ведёт себя как identity на старте
        nn.init.eye_(self.fc_neck[-3].weight[:feat_dim, :feat_dim])
        nn.init.zeros_(self.fc_neck[-3].bias)

        # ── Финальные головы ──────────────────────────────────────────────
        self.cls_head = nn.Linear(feat_dim, 1)
        self.reg_head = nn.Linear(feat_dim, 6)

    def forward(
        self,
        proposal_pw_feats: torch.Tensor,  # (B, N, pw_feat_dim)
        proposal_xyz: torch.Tensor,       # (B, N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Шаг 1: Предсказываем offset и centerness для каждой точки
        offsets = self.offset_head(proposal_pw_feats)       # (B, N, 3)
        centerness = self.centerness_head(proposal_pw_feats) # (B, N, 1)

        # Шаг 2: Сдвигаем точки к предсказанным центрам деревьев
        shifted_xyz = proposal_xyz + offsets  # (B, N, 3)

        # Шаг 3: Взвешенный глобальный пулинг
        #   centerness — это веса важности каждой точки
        weights = centerness / (centerness.sum(dim=1, keepdim=True) + 1e-6)  # (B, N, 1)
        concat = torch.cat([shifted_xyz, proposal_pw_feats], dim=-1)  # (B, N, 3+pw)
        weighted_feat = (concat * weights).sum(dim=1)                  # (B, 3+pw)

        # Шаг 4: Проекция в feat_dim
        global_feat = self.proj(weighted_feat)  # (B, feat_dim)

        # Шаг 5: FC-neck
        refined_feat = self.fc_neck(global_feat)  # (B, feat_dim)

        # Шаг 6: Финальные предсказания
        cls_score = self.cls_head(refined_feat)  # (B, 1)
        reg_delta = self.reg_head(refined_feat)  # (B, 6)

        return cls_score, reg_delta, offsets, centerness
