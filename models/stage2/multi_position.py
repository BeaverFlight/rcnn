"""
Stage 2: Multi-position feature extraction for proposal refinement.

Fixes vs original:
  - Добавлен forward_batch(xyz_batch, proposals_batch) для обработки
    нескольких proposals одним GPU-вызовом вместо Python-цикла.
  - forward() теперь вызывает forward_batch для единственного proposal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.backbone.pointnet2_modules import PointNetSetAbstraction


class MultiPositionExtractor(nn.Module):
    """
    Extract multi-position features for proposals.

    Запускает PointNet++ на центро-нормализованных координатах;
    4 смещённые нормализации добавляются как дополнительные признаки.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        pn2 = cfg.pointnet2
        sa0_mlp = list(pn2.sa_layers[0].mlp)
        sa1_mlp = list(pn2.sa_layers[1].mlp)
        sa2_mlp = list(pn2.sa_layers[2].mlp)

        # Input: 3 (centre coords) + 12 (4×3 offset coords) = 15
        in_ch = sa0_mlp[0] + 12

        self.sa1 = PointNetSetAbstraction(
            npoint=pn2.sa_layers[0].npoint,
            radius=pn2.sa_layers[0].radius,
            nsample=pn2.sa_layers[0].nsample,
            in_channel=in_ch,
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
        self.out_dim = sa2_mlp[-1]

    # ------------------------------------------------------------------
    # Batched interface (основной путь)
    # ------------------------------------------------------------------
    def forward_batch(self, xyz_batch: Tensor, proposals: Tensor) -> Tensor:
        """
        Обрабатывает B proposals одним GPU-вызовом.

        Args:
            xyz_batch: (B, N, 3) — точки внутри каждого proposal (world coords),
                       одинаковое N (добейтесь паддингом перед вызовом).
            proposals: (B, 6)   — [x, y, z_c, w, l, h]

        Returns:
            feats: (B, feat_dim)
        """
        B, N, _ = xyz_batch.shape
        x  = proposals[:, 0]  # (B,)
        y  = proposals[:, 1]
        w  = proposals[:, 3]

        # Центро-нормализация: (B, N, 3)
        centre_pts = xyz_batch.clone()
        centre_pts[:, :, 0] -= x.unsqueeze(1)
        centre_pts[:, :, 1] -= y.unsqueeze(1)
        xyz_norm = centre_pts  # (B, N, 3)

        # 4 смещённые нормализации → (B, N, 12)
        offsets_xy = [
            torch.stack([ x + w / 2,  y         ], dim=-1),  # (B, 2)
            torch.stack([ x - w / 2,  y         ], dim=-1),
            torch.stack([ x,          y + w / 2 ], dim=-1),
            torch.stack([ x,          y - w / 2 ], dim=-1),
        ]
        offset_parts = []
        for oxy in offsets_xy:
            op = xyz_batch.clone()
            op[:, :, 0] -= oxy[:, 0:1]
            op[:, :, 1] -= oxy[:, 1:2]
            offset_parts.append(op)
        offset_feat = torch.cat(offset_parts, dim=-1)  # (B, N, 12)

        xyz1, f1 = self.sa1(xyz_norm, offset_feat)    # (B, S1, 64)
        xyz2, f2 = self.sa2(xyz1, f1)                 # (B, S2, 128)
        _, f3    = self.sa3(xyz2, f2)                 # (B, 1, 512)
        return f3.squeeze(1)                           # (B, feat_dim)

    # ------------------------------------------------------------------
    # Single-proposal interface (используется снаружи если нужно)
    # ------------------------------------------------------------------
    def forward(self, points: Tensor, proposal: Tensor) -> Tensor:
        """
        Args:
            points:   (N, 3) points inside proposal (world coords)
            proposal: (6,)  [x, y, z_c, w, l, h]

        Returns:
            feature: (feat_dim,)
        """
        return self.forward_batch(
            points.unsqueeze(0),
            proposal.unsqueeze(0),
        ).squeeze(0)
