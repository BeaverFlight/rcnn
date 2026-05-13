"""
models/backbone/pointnext_modules.py — PointNeXt backbone для TreeRCNN v2.

Архитектура:
    PointNeXt (NeurIPS 2022): https://arxiv.org/abs/2206.04670
    "Revisiting PointNet++ with Improved Training and Scaling Strategies"

Отличия от текущего PointNet++ в pointnet2_modules.py:
  1. Inverted Residual Bottleneck (IRB) — остаточные связи внутри SA-блока.
     Вместо плоского MLP: expand → pointwise → contract + skip.
  2. Separable MLP — обработка пространства (xyz-смещений) и признаков
     разделена, что снижает число параметров при той же ёмкости.
  3. GELU вместо ReLU — лучший сигнал для глубоких сетей.
  4. Residual shortcut через linear projection: если in_channel != out_channel.
  5. Полностью совместимый forward() интерфейс с PointNetSetAbstraction:
       forward(xyz, points) -> (new_xyz, new_points)

Fix: PointNeXtEncoder теперь сохраняет self.sa2_xyz после forward(),
     чтобы _fpn_lookup в tree_rcnn_v2 использовал настоящие SA2 FPS-центроиды
     вместо некорректного fallback points[:S2].
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
)


# ---------------------------------------------------------------------------
# Базовый строительный блок: Inverted Residual Bottleneck MLP
# ---------------------------------------------------------------------------

class InvertedResidualMLP(nn.Module):
    """
    Inverted Residual Bottleneck (IRB) для point features.

    Структура:
        x  →  Linear(in, expand_dim)  →  GELU  →  LayerNorm
           →  Linear(expand_dim, out)
           →  + skip (x если in==out, иначе Linear(in, out))

    expand_ratio=4 стандартный как в ConvNeXt/MobileNetV2.
    """

    def __init__(self, in_dim: int, out_dim: int, expand_ratio: int = 4) -> None:
        super().__init__()
        mid = in_dim * expand_ratio
        self.expand = nn.Linear(in_dim, mid)
        self.norm   = nn.LayerNorm(mid)
        self.act    = nn.GELU()
        self.proj   = nn.Linear(mid, out_dim)

        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm(self.expand(x)))
        return self.proj(h) + self.skip(x)


# ---------------------------------------------------------------------------
# PointNeXt Set Abstraction — drop-in замена PointNetSetAbstraction
# ---------------------------------------------------------------------------

class PointNeXtSetAbstraction(nn.Module):
    """
    PointNeXt Set Abstraction (SA) блок.

    Совместимый интерфейс с PointNetSetAbstraction:
        forward(xyz, points) -> (new_xyz | None, new_points)

    npoint=None → Global SA (maxpool по всем точкам).
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: list[int],
        expand_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample

        self.xyz_encoder = nn.Sequential(
            nn.Linear(3, mlp[0]),
            nn.LayerNorm(mlp[0]),
            nn.GELU(),
        )

        dims = [in_channel + mlp[0]] + mlp
        irb_blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            irb_blocks.append(InvertedResidualMLP(dims[i], dims[i + 1], expand_ratio))
        self.irb = nn.Sequential(*irb_blocks)

        self.out_channels = mlp[-1]

    def forward(
        self, xyz: Tensor, points: Optional[Tensor]
    ) -> tuple[Optional[Tensor], Tensor]:
        """
        Args:
            xyz:    (B, N, 3)
            points: (B, N, C) or None

        Returns:
            new_xyz:    (B, npoint, 3) or None
            new_points: (B, npoint, out_channels)
        """
        B, N, _ = xyz.shape

        if self.npoint is None:
            xyz_feat = self.xyz_encoder(xyz)
            if points is not None:
                combined = torch.cat([points, xyz_feat], dim=-1)
            else:
                combined = torch.cat([xyz_feat, xyz_feat], dim=-1)
            out = self.irb(combined)
            new_points = out.max(dim=1, keepdim=True)[0]
            return None, new_points

        fps_idx = farthest_point_sample(xyz, self.npoint)      # (B, S)
        new_xyz = index_points(xyz, fps_idx)                   # (B, S, 3)

        idx         = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)       # local offset

        xyz_feat = self.xyz_encoder(grouped_xyz)               # (B, S, K, mlp[0])

        if points is not None:
            grouped_pts = index_points(points, idx)
            combined = torch.cat([grouped_pts, xyz_feat], dim=-1)
        else:
            combined = torch.cat([xyz_feat, xyz_feat], dim=-1)

        out = self.irb(combined)                               # (B, S, K, out)
        new_points = out.max(dim=2)[0]                         # (B, S, out)
        return new_xyz, new_points


# ---------------------------------------------------------------------------
# PointNeXt Encoder — полная иерархия SA для TreeRCNN v2
# ---------------------------------------------------------------------------

class PointNeXtEncoder(nn.Module):
    """
    4-уровневый энкодер на PointNeXt SA блоках.

    Иерархия:
        SA1  (Stem) : 512 pt, r=2.0,  nsample=64
        SA2         : 256 pt, r=4.0,  nsample=64
        SA3  (extra): 128 pt, r=8.0,  nsample=64
        SA-global   : global pooling

    Возвращает список признаков для FPN:
        [feat_sa1, feat_sa2, feat_sa3, feat_global]
        shapes: (B, S1, C1), (B, S2, C2), (B, S3, C3), (B, 1, C4)

    Fix: после каждого forward() сохраняет self.sa2_xyz — реальные FPS-центроиды
    SA2-уровня (B, S2, 3). Используется в _fpn_lookup для per-anchor FPN lookup.
    Без этого дерево rcnn_v2 некорректно делало fallback на points[:S2].
    """

    _DEFAULT = dict(
        npoints  = [512, 256, 128],
        radii    = [2.0, 4.0, 8.0],
        nsamples = [64,  64,  64],
        channels = [[32, 64], [64, 128], [128, 256], [256, 512]],
    )

    def __init__(self, cfg=None) -> None:
        super().__init__()

        pnx = getattr(cfg, 'pointnext', None) if cfg else None

        def _g(key):
            if pnx is not None and hasattr(pnx, key):
                v = getattr(pnx, key)
                return list(v) if hasattr(v, '__iter__') else v
            return self._DEFAULT[key]

        npoints  = _g('npoints')
        radii    = _g('radii')
        nsamples = _g('nsamples')
        channels = _g('channels')

        self.sa1 = PointNeXtSetAbstraction(
            npoint=npoints[0], radius=radii[0], nsample=nsamples[0],
            in_channel=0, mlp=channels[0]
        )
        self.sa2 = PointNeXtSetAbstraction(
            npoint=npoints[1], radius=radii[1], nsample=nsamples[1],
            in_channel=channels[0][-1], mlp=channels[1]
        )
        self.sa3 = PointNeXtSetAbstraction(
            npoint=npoints[2], radius=radii[2], nsample=nsamples[2],
            in_channel=channels[1][-1], mlp=channels[2]
        )
        self.sa_global = PointNeXtSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=channels[2][-1], mlp=channels[3]
        )

        self.out_dims = [
            channels[0][-1],
            channels[1][-1],
            channels[2][-1],
            channels[3][-1],
        ]

        # Кеш SA2-центроидов: заполняется в forward(), читается _fpn_lookup.
        # Не является параметром модели — не попадает в state_dict.
        self.sa2_xyz: Optional[Tensor] = None

    def forward(self, xyz: Tensor) -> list[Tensor]:
        """
        Args:
            xyz: (B, N, 3) — облако точек

        Returns:
            [sa1_feat, sa2_feat, sa3_feat, global_feat]
            Shapes: (B, S1, C1), (B, S2, C2), (B, S3, C3), (B, 1, C4)

        Side effect:
            self.sa2_xyz ← (B, S2, 3) FPS-центроиды SA2.
            Используется tree_rcnn_v2._fpn_lookup для per-anchor FPN context.
        """
        xyz1, f1 = self.sa1(xyz,  None)      # (B, S1, C1)
        xyz2, f2 = self.sa2(xyz1, f1)        # (B, S2, C2)
        xyz3, f3 = self.sa3(xyz2, f2)        # (B, S3, C3)
        _,    f4 = self.sa_global(xyz3, f3)  # (B, 1, C4)

        # Сохраняем реальные SA2-центроиды (без grad — только для lookup)
        self.sa2_xyz = xyz2.detach()

        return [f1, f2, f3, f4]
