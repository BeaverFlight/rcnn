"""
Stage 2: Multi-position feature extraction for proposal refinement.

Memory fix:
  - Убраны 4 вызова xyz_batch.clone() в forward_batch — они создавали
    4 полных копии (B, N, 3) на GPU одновременно (до 6+ GB при chunk=256).
    Теперь смещения вычисляются без clone через вычитание precomputed
    offset тензора shape (B, 1, 2), который broadcast по N.
  - forward() по-прежнему вызывает forward_batch.
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
        device = xyz_batch.device

        x = proposals[:, 0]   # (B,)
        y = proposals[:, 1]
        w = proposals[:, 3]

        # --- Центро-нормализация без clone ---
        # xy_centre: (B, 1, 2) → broadcast вычитается по всем N точкам
        xy_centre = torch.stack([x, y], dim=-1).unsqueeze(1)  # (B, 1, 2)
        xyz_norm = xyz_batch.clone()                          # (B, N, 3) — единственный clone
        xyz_norm[:, :, :2] -= xy_centre                       # inplace, нет лишних аллокаций

        # --- 4 смещённые нормализации без clone ---
        # Вычисляем 4 offset-центра: (B, 4, 2)
        hw = w / 2  # (B,)
        # offsets: [[x+hw, y], [x-hw, y], [x, y+hw], [x, y-hw]]
        offsets = torch.stack([
            torch.stack([x + hw, y     ], dim=-1),
            torch.stack([x - hw, y     ], dim=-1),
            torch.stack([x,      y + hw], dim=-1),
            torch.stack([x,      y - hw], dim=-1),
        ], dim=1)  # (B, 4, 2)

        # xyz_batch[:, :, :2]: (B, N, 2)
        # offsets.unsqueeze(2): (B, 4, 1, 2)
        # diff: (B, 4, N, 2) — broadcasting, нет clone
        xy_pts = xyz_batch[:, :, :2].unsqueeze(1)          # (B, 1, N, 2)
        dxy    = xy_pts - offsets.unsqueeze(2)             # (B, 4, N, 2)  broadcast
        z_rep  = xyz_batch[:, :, 2:].unsqueeze(1).expand(
            B, 4, N, 1
        )                                                  # (B, 4, N, 1)
        offset_4d = torch.cat([dxy, z_rep], dim=-1)        # (B, 4, N, 3)

        # Reshape → (B, N, 12) без дополнительных копий
        offset_feat = offset_4d.permute(0, 2, 1, 3).reshape(B, N, 12)  # (B, N, 12)

        xyz1, f1 = self.sa1(xyz_norm, offset_feat)    # (B, S1, 64)
        xyz2, f2 = self.sa2(xyz1, f1)                 # (B, S2, 128)
        _,    f3 = self.sa3(xyz2, f2)                 # (B, 1,  D)
        return f3.squeeze(1)                           # (B, feat_dim)

    # ------------------------------------------------------------------
    # Single-proposal interface
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
