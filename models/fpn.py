"""
models/fpn.py — Feature Pyramid Network для TreeRCNN v2.0

Объединяет признаки с трёх уровней PointNet++ SA-иерархии:
  p2 (высокое разрешение, мелкие деревья / подлесок)
  p3 (средний масштаб)
  p4 (глобальный контекст тайла)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeFPN(nn.Module):
    """
    Top-down Feature Pyramid Network.

    Принимает выходы SA2, SA3 (SA-extra), SA4 (global).
    Возвращает три уровня признаков выровненной размерности (out_channels).

    Пример:
        fpn = TreeFPN(in_channels=[256, 512, 1024], out_channels=256)
        p2, p3, p4 = fpn(f2, f3, f4)  # все → (B, 256, N_i)
    """

    def __init__(
        self,
        in_channels: list[int] = (256, 512, 1024),
        out_channels: int = 256,
    ):
        super().__init__()

        # Lateral connections: выравниваем каналы до out_channels
        self.lat2 = nn.Conv1d(in_channels[0], out_channels, kernel_size=1)
        self.lat3 = nn.Conv1d(in_channels[1], out_channels, kernel_size=1)
        self.lat4 = nn.Conv1d(in_channels[2], out_channels, kernel_size=1)

        # Output convolutions после слияния
        self.out2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.out3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        f2: torch.Tensor,  # (B, C2, N2)
        f3: torch.Tensor,  # (B, C3, N3)
        f4: torch.Tensor,  # (B, C4, 1) — global
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l2 = self.lat2(f2)  # (B, out, N2)
        l3 = self.lat3(f3)  # (B, out, N3)
        l4 = self.lat4(f4)  # (B, out, 1)

        # Top-down: обогащаем мелкие уровни контекстом крупных
        p3 = self.out3(l3 + F.interpolate(l4, size=l3.shape[-1], mode='nearest'))
        p2 = self.out2(l2 + F.interpolate(p3, size=l2.shape[-1], mode='nearest'))
        p4 = l4

        return p2, p3, p4
