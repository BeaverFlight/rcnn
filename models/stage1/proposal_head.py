"""
Stage 1: Proposal Generation Network.
Uses PointNet++ to extract per-window features and predict (cls, reg) per anchor.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.backbone.pointnet2_modules import PointNetSetAbstraction

logger = logging.getLogger(__name__)


class ProposalHead(nn.Module):
    """
    Stage-1 head: processes points per sliding window with PointNet++,
    then predicts objectness score and box delta per anchor.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        pn2 = cfg.pointnet2

        # Layer 0: input is xyz only → in_channel = 3
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
        self.cls_head = nn.Linear(feat_dim, 1)  # objectness logit
        self.reg_head = nn.Linear(feat_dim, 4)  # (tx, ty, tw, th)

    def extract_features(self, xyz: Tensor) -> Tensor:
        """
        Run PointNet++ on a batch of windows.

        Args:
            xyz: (B, N, 3) normalized point coordinates

        Returns:
            feat: (B, feat_dim)
        """
        xyz1, f1 = self.sa1(xyz, None)  # (B, 64, 64)
        xyz2, f2 = self.sa2(xyz1, f1)  # (B, 32, 128)
        _, f3 = self.sa3(xyz2, f2)  # (B, 1, 512)
        return f3.squeeze(1)  # (B, 512)

    def forward(self, xyz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            xyz: (B, N, 3)

        Returns:
            cls_logits: (B, 1)
            reg_deltas: (B, 4)
        """
        feat = self.extract_features(xyz)
        return self.cls_head(feat), self.reg_head(feat)
