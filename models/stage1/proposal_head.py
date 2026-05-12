"""
Stage 1: Proposal Generation Network.

Изменения:
  - reg_head выдаёт 6D дельты (tx, ty, tz, tw, tl, th).
  - forward() принимает опциональный fpn_context: (B, fpn_dim)
    из TreeFPN p2-уровня. Если передан, конкатенируется с SA3-признаком
    через fpn_proj перед cls/reg-головами.
  - Обратная совместимость: fpn_context=None — поведение v1 сохранено.
  - Gradient checkpointing для каждого SA-слоя снижает пик VRAM backward ~60%%.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch import Tensor

from models.backbone.pointnet2_modules import PointNetSetAbstraction

logger = logging.getLogger(__name__)


class ProposalHead(nn.Module):
    """
    Stage-1 head: processes a batch of windows with PointNet++,
    predicts objectness score and 6D box delta per anchor.

    Опционально принимает FPN-контекст (fpn_context) из TreeFPN,
    который конкатенируется с SA3-глобальным признаком.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        pn2 = cfg.pointnet2

        sa0_mlp = list(pn2.sa_layers[0].mlp)
        sa1_mlp = list(pn2.sa_layers[1].mlp)
        sa2_mlp = list(pn2.sa_layers[2].mlp)

        self.sa1 = PointNetSetAbstraction(
            npoint=pn2.sa_layers[0].npoint,
            radius=pn2.sa_layers[0].radius,
            nsample=pn2.sa_layers[0].nsample,
            in_channel=sa0_mlp[0],
            mlp=sa0_mlp[1:],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=pn2.sa_layers[1].npoint,
            radius=pn2.sa_layers[1].radius,
            nsample=pn2.sa_layers[1].nsample,
            in_channel=sa1_mlp[0],
            mlp=sa1_mlp[1:],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=sa2_mlp[0],
            mlp=sa2_mlp[1:],
        )

        feat_dim = sa2_mlp[-1]

        # FPN интеграция: проекция FPN-контекста в feat_dim
        # fpn_dim читается из cfg.fpn.out_channels;
        # если 0 или отсутствует — FPN не используется (v1-совместимость)
        fpn_cfg = getattr(cfg, 'fpn', None)
        fpn_dim = int(fpn_cfg.out_channels) if fpn_cfg else 0
        if fpn_dim > 0:
            self.fpn_proj = nn.Sequential(
                nn.Linear(fpn_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
            )
            # Объединяющий проекцирует [sa3_feat || fpn_feat] → feat_dim
            self.fusion = nn.Linear(feat_dim * 2, feat_dim)
        else:
            self.fpn_proj = None
            self.fusion   = None

        self.cls_head = nn.Linear(feat_dim, 1)   # objectness logit
        self.reg_head = nn.Linear(feat_dim, 6)   # 6D: (tx, ty, tz, tw, tl, th)

    # ------------------------------------------------------------------
    # Статические обёртки для gradient checkpoint
    # ------------------------------------------------------------------

    @staticmethod
    def _sa1_fn(
        sa1: PointNetSetAbstraction, xyz: Tensor
    ) -> tuple[Tensor, Tensor]:
        return sa1(xyz, None)

    @staticmethod
    def _sa2_fn(
        sa2: PointNetSetAbstraction, xyz: Tensor, feats: Tensor
    ) -> tuple[Tensor, Tensor]:
        return sa2(xyz, feats)

    @staticmethod
    def _sa3_fn(
        sa3: PointNetSetAbstraction, xyz: Tensor, feats: Tensor
    ) -> Tensor:
        _, out = sa3(xyz, feats)
        return out   # (B, 1, D)

    def extract_features(self, xyz: Tensor) -> Tensor:
        """
        Запускает PointNet++ на батче виндоу.

        Args:
            xyz: (B, N, 3)

        Returns:
            feat: (B, feat_dim)
        """
        training_pass = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.sa1.parameters()
        )

        if training_pass:
            xyz1, f1 = ckpt.checkpoint(
                self._sa1_fn, self.sa1, xyz, use_reentrant=False,
            )
            xyz2, f2 = ckpt.checkpoint(
                self._sa2_fn, self.sa2, xyz1, f1, use_reentrant=False,
            )
            f3 = ckpt.checkpoint(
                self._sa3_fn, self.sa3, xyz2, f2, use_reentrant=False,
            )
        else:
            xyz1, f1 = self.sa1(xyz, None)
            xyz2, f2 = self.sa2(xyz1, f1)
            _, f3    = self.sa3(xyz2, f2)

        return f3.squeeze(1)   # (B, feat_dim)

    def forward(
        self,
        xyz: Tensor,
        fpn_context: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            xyz:         (B, N, 3)
            fpn_context: (B, fpn_dim) — global FPN p2 context, optional.
                         Если None — поведение v1 без FPN.

        Returns:
            cls_logits: (B, 1)
            reg_deltas: (B, 6)
        """
        feat = self.extract_features(xyz)   # (B, feat_dim)

        # FPN слияние: конкат SA3-признак + FPN-проекция → объединение
        if fpn_context is not None and self.fpn_proj is not None:
            fpn_feat = self.fpn_proj(fpn_context)              # (B, feat_dim)
            feat     = self.fusion(torch.cat([feat, fpn_feat], dim=-1))  # (B, feat_dim)

        return self.cls_head(feat), self.reg_head(feat)
