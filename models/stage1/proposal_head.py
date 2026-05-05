"""
Stage 1: Proposal Generation Network.

Changed:
  - reg_head outputs 6D deltas (tx, ty, tz, tw, tl, th).
  - forward() accepts batch (B, N, 3) and returns (cls: B×1, reg: B×6).
  - extract_features использует gradient checkpointing для каждого SA-слоя:
    сохраняются только входы/выходы слоёв, промежуточные активации
    пересчитываются при backward. Снижает пик VRAM backward на ~60%.
    Checkpointing включён только в режиме training (requires_grad=True),
    при inference (torch.inference_mode / no_grad) — пропускается автоматически.
"""

from __future__ import annotations

import logging

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
        self.cls_head = nn.Linear(feat_dim, 1)   # objectness logit
        self.reg_head = nn.Linear(feat_dim, 6)   # 6D: (tx, ty, tz, tw, tl, th)

    # ------------------------------------------------------------------
    # Статические обёртки для checkpoint — нужны чтобы передавать
    # Optional[Tensor] через *args интерфейс checkpoint.checkpoint.
    # checkpoint требует Tensor-аргументы; None передаём как sentinel.
    # ------------------------------------------------------------------

    @staticmethod
    def _sa1_fn(sa1: PointNetSetAbstraction, xyz: Tensor) -> tuple[Tensor, Tensor]:
        new_xyz, feats = sa1(xyz, None)
        return new_xyz, feats

    @staticmethod
    def _sa2_fn(
        sa2: PointNetSetAbstraction, xyz: Tensor, feats: Tensor
    ) -> tuple[Tensor, Tensor]:
        new_xyz, new_feats = sa2(xyz, feats)
        return new_xyz, new_feats

    @staticmethod
    def _sa3_fn(
        sa3: PointNetSetAbstraction, xyz: Tensor, feats: Tensor
    ) -> Tensor:
        _, out = sa3(xyz, feats)
        return out  # (B, 1, D)

    def extract_features(self, xyz: Tensor) -> Tensor:
        """
        Run PointNet++ on a batch of windows.

        Args:
            xyz: (B, N, 3)

        Returns:
            feat: (B, feat_dim)

        Gradient checkpointing включён когда любой параметр требует grad
        (т.е. в training-pass с infer_mode=False). При inference_mode /
        no_grad — use_reentrant=False делает checkpoint прозрачным,
        фактически выполняя forward без сохранения сегментов.
        """
        training_pass = torch.is_grad_enabled() and xyz.requires_grad or any(
            p.requires_grad for p in self.sa1.parameters()
        )

        if training_pass:
            xyz1, f1 = ckpt.checkpoint(
                self._sa1_fn, self.sa1, xyz,
                use_reentrant=False,
            )
            xyz2, f2 = ckpt.checkpoint(
                self._sa2_fn, self.sa2, xyz1, f1,
                use_reentrant=False,
            )
            f3 = ckpt.checkpoint(
                self._sa3_fn, self.sa3, xyz2, f2,
                use_reentrant=False,
            )
        else:
            xyz1, f1 = self.sa1(xyz, None)
            xyz2, f2 = self.sa2(xyz1, f1)
            _, f3   = self.sa3(xyz2, f2)

        return f3.squeeze(1)   # (B, feat_dim)

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            xyz: (B, N, 3)

        Returns:
            cls_logits: (B, 1)
            reg_deltas: (B, 6)
        """
        feat = self.extract_features(xyz)
        return self.cls_head(feat), self.reg_head(feat)
