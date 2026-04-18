"""
PointNet++ Set Abstraction (SSG) and Feature Propagation modules.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.backbone.pointnet2_utils import (
    ball_query,
    farthest_point_sample,
    index_points,
    square_distance,
)


class PointNetSetAbstraction(nn.Module):
    """
    Single Set Abstraction layer (Single-Scale Grouping).

    If npoint is None → global abstraction (no spatial grouping).
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: list[int],
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers: list[nn.Module] = []
        last = in_channel
        for out in mlp:
            layers += [nn.Linear(last, out), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last

    def forward(
        self, xyz: Tensor, points: Optional[Tensor]
    ) -> tuple[Tensor | None, Tensor]:
        """
        Args:
            xyz:    (B, N, 3) point coordinates
            points: (B, N, C) point features (or None)

        Returns:
            new_xyz:    (B, npoint, 3) or None if global
            new_points: (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape

        if self.npoint is None:
            # Global abstraction
            if points is not None:
                combined = torch.cat([xyz, points], dim=-1)  # (B, N, 3+C)
            else:
                combined = xyz
            out = self._apply_mlp(combined)  # (B, N, D)
            new_points = out.max(dim=1)[0]  # (B, D)
            new_points = new_points.unsqueeze(1)  # (B, 1, D)
            return None, new_points

        # Sample centroids
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, S)
        new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)

        # Group neighbours
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, S, K)
        grouped_xyz = index_points(xyz, idx)  # (B, S, K, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # local coords

        if points is not None:
            grouped_pts = index_points(points, idx)  # (B, S, K, C)
            combined = torch.cat([grouped_xyz, grouped_pts], dim=-1)
        else:
            combined = grouped_xyz  # (B, S, K, 3)

        out = self._apply_mlp(combined)  # (B, S, K, D)
        new_points = out.max(dim=2)[0]  # (B, S, D)
        return new_xyz, new_points

    def _apply_mlp(self, x: Tensor) -> Tensor:
        """Apply MLP with BN; x may be (B, S, K, C) or (B, N, C)."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        for layer in self.mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x_flat = layer(x_flat)
            else:
                x_flat = layer(x_flat)
        return x_flat.reshape(*shape[:-1], -1)
