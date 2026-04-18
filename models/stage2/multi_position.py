"""
Stage 2: Multi-position feature extraction for proposal refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.backbone.pointnet2_modules import PointNetSetAbstraction


def normalize_to_center(points: Tensor, cx: float, cy: float) -> Tensor:
    """Translate points so that (cx, cy, 0) is the new origin."""
    pts = points.clone()
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    return pts


class MultiPositionExtractor(nn.Module):
    """
    Extract multi-position features for a single proposal.

    Runs PointNet++ on centre-normalized coordinates;
    the 4 offset normalizations are appended as additional input features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        pn2 = cfg.pointnet2
        sa0_mlp = list(pn2.sa_layers[0].mlp)
        sa1_mlp = list(pn2.sa_layers[1].mlp)
        sa2_mlp = list(pn2.sa_layers[2].mlp)

        # Input: 3 (centre coords) + 12 (4×3 offset coords) = 15
        in_ch = sa0_mlp[0] + 12  # original in_ch + offset features

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

    def forward(self, points: Tensor, proposal: Tensor) -> Tensor:
        """
        Args:
            points:   (N, 3) points inside proposal (world coords)
            proposal: (6,)  [x, y, z_c, w, l, h]

        Returns:
            feature: (feat_dim,)
        """
        x, y, _, w, _, _ = proposal.unbind()
        offsets = [
            (x + w / 2, y),
            (x - w / 2, y),
            (x, y + w / 2),
            (x, y - w / 2),
        ]

        # Centre-normalized coords (B=1 for PointNet++)
        centre_pts = points.clone()
        centre_pts[:, 0] -= x
        centre_pts[:, 1] -= y
        xyz = centre_pts.unsqueeze(0)  # (1, N, 3)

        # Offset normalizations concatenated as features
        offset_parts = []
        for ox, oy in offsets:
            op = points.clone()
            op[:, 0] -= ox
            op[:, 1] -= oy
            offset_parts.append(op)
        offset_feat = torch.cat(offset_parts, dim=-1)  # (N, 12)
        feats = offset_feat.unsqueeze(0)  # (1, N, 12)

        # SA layers treat offset_feat as point features at first SA layer
        # We inject feats by combining xyz and feats at input of sa1
        combined_input = torch.cat([xyz, feats], dim=-1)  # (1, N, 15)
        # Reshape for PointNetSetAbstraction: pass xyz separately
        xyz1, f1 = self.sa1(xyz, feats)
        xyz2, f2 = self.sa2(xyz1, f1)
        _, f3 = self.sa3(xyz2, f2)
        return f3.squeeze(0).squeeze(0)  # (feat_dim,)
