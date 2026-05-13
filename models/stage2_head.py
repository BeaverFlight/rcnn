"""
models/stage2_head.py — Stage 2 Refinement Head для TreeRCNN v2.0

Фиксы:
  - fc_neck near-identity init: неверный индекс [-3] (был Dropout) — заменён на явный self.fc_neck_last
  - centerness: sigmoid уже внутри, используется F.binary_cross_entropy (не with_logits)
  - добавлен публичный метод extract_features() для использования в Stage-3
    (устраняет прямой доступ к приватным слоям в _extract_s3_feats)

Integration note:
  Этот модуль работает поверх внутренней SA-цепочки Stage 2 (RefinementHead).
  RefinementHead должен отдать point-wise признаки до global pooling,
  а Stage2Head заменяет global pooling + head в RefinementHead.
  
  Интерфейс RefinementHead в v2.0:
    pw_feats, xyz = self.sa_layers(batch)    # (B, N, pw_feat_dim)
    cls, reg, offsets, centerness = stage2_head(pw_feats, xyz)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2Head(nn.Module):
    """
    Args:
        pw_feat_dim : размерность point-wise признаков (выход SA-слоёв Stage 2)
        feat_dim    : размерносщ после глобального пулинга

    Forward input:
        proposal_pw_feats : (B, N, pw_feat_dim) — point-wise признаки ДО global pooling
        proposal_xyz      : (B, N, 3) — координаты точек в proposal

    Forward output:
        cls_score   : (B, 1)    — вероятность дерева (logit, без sigmoid)
        reg_delta   : (B, 6)    — cx, cy, cz, w, l, h относительно proposal
        offsets     : (B, N, 3) — смещения точек к центру ствола
        centerness  : (B, N, 1) — уверенность точки (уже sigmoid, [0..1])
    """

    def __init__(self, pw_feat_dim: int = 256, feat_dim: int = 1024):
        super().__init__()
        self.pw_feat_dim = pw_feat_dim
        self.feat_dim = feat_dim

        # ── Offset Head (point-wise, до пулинга) ───────────────────────────────────────
        self.offset_head = nn.Sequential(
            nn.Linear(pw_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3),  # (dx, dy, dz)
        )

        # ── Center-ness Head (point-wise, до пулинга) ──────────────────────────────
        self.centerness_head = nn.Sequential(
            nn.Linear(pw_feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # выход в [0..1], loss = F.binary_cross_entropy (не with_logits)
        )

        # Проекция shifted_xyz + pw_feat в пространство feat_dim
        self.proj = nn.Linear(3 + pw_feat_dim, feat_dim)

        # ── FC-neck (после взвешенного пулинга) ───────────────────────────────
        self.fc_neck_expand = nn.Linear(feat_dim, feat_dim * 2)
        self.fc_neck_norm1  = nn.LayerNorm(feat_dim * 2)
        self.fc_neck_drop   = nn.Dropout(0.2)
        self.fc_neck_last   = nn.Linear(feat_dim * 2, feat_dim)  # near-identity init ниже
        self.fc_neck_norm2  = nn.LayerNorm(feat_dim)
        self.fc_neck_act    = nn.GELU()

        # Near-identity init: fc_neck_last ведёт себя как identity на старте
        # благодаря этому neck не ломает уже выученные признаки при fine-tune
        nn.init.eye_(self.fc_neck_last.weight[:feat_dim, :feat_dim])
        nn.init.zeros_(self.fc_neck_last.weight[:feat_dim, feat_dim:])  # правая половина
        nn.init.zeros_(self.fc_neck_last.bias)

        # ── Финальные головы ───────────────────────────────────────────────────────
        # Примечание: cls_head возвращает logit (без sigmoid).
        # sigmoid применяется в losses.py через binary_cross_entropy_with_logits.
        self.cls_head = nn.Linear(feat_dim, 1)
        self.reg_head = nn.Linear(feat_dim, 6)

    def _fc_neck(self, x: torch.Tensor) -> torch.Tensor:
        """ФФC-neck с явными названиями слоёв (устраняет неверный индекс)."""
        x = self.fc_neck_expand(x)
        x = self.fc_neck_norm1(x)
        x = F.gelu(x)
        x = self.fc_neck_drop(x)
        x = self.fc_neck_last(x)
        x = self.fc_neck_norm2(x)
        x = self.fc_neck_act(x)
        return x

    def _compute_global_feat(
        self,
        proposal_pw_feats: torch.Tensor,  # (B, N, pw_feat_dim)
        proposal_xyz: torch.Tensor,       # (B, N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Общая логика взвешенного пулинга и проекции.
        Используется как forward(), так и extract_features().

        Returns:
            x         : (B, feat_dim) — признак после fc_neck
            offsets   : (B, N, 3)
            centerness: (B, N, 1)
        """
        offsets    = self.offset_head(proposal_pw_feats)       # (B, N, 3)
        centerness = self.centerness_head(proposal_pw_feats)   # (B, N, 1), [0..1]

        shifted_xyz = proposal_xyz + offsets
        weights     = centerness / (centerness.sum(dim=1, keepdim=True) + 1e-6)
        concat      = torch.cat([shifted_xyz, proposal_pw_feats], dim=-1)  # (B, N, 3+pw)
        global_feat = (concat * weights).sum(dim=1)                         # (B, 3+pw)

        x = self.proj(global_feat)   # (B, feat_dim)
        x = self._fc_neck(x)         # (B, feat_dim)
        return x, offsets, centerness

    def extract_features(
        self,
        proposal_pw_feats: torch.Tensor,  # (B, N, pw_feat_dim)
        proposal_xyz: torch.Tensor,       # (B, N, 3)
    ) -> torch.Tensor:                    # (B, feat_dim)
        """
        Публичный метод для Stage-3: извлекает семантические признаки пропозала.

        Устраняет нарушение инкапсуляции: раньше _extract_s3_feats в tree_rcnn_v2.py
        напрямую обращался к head.centerness_head, head.offset_head,
        head._fc_neck, head.proj. Теперь логика инкапсулирована здесь.

        Returns:
            (B, feat_dim) — глобальный признак пропозала без logit/reg голов
        """
        x, _, _ = self._compute_global_feat(proposal_pw_feats, proposal_xyz)
        return x

    def forward(
        self,
        proposal_pw_feats: torch.Tensor,  # (B, N, pw_feat_dim)
        proposal_xyz: torch.Tensor,       # (B, N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        x, offsets, centerness = self._compute_global_feat(proposal_pw_feats, proposal_xyz)

        cls_score = self.cls_head(x)  # (B, 1) — logit
        reg_delta = self.reg_head(x)  # (B, 6)

        return cls_score, reg_delta, offsets, centerness
