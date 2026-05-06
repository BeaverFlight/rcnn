"""
models/fpn.py — Feature Pyramid Network для TreeRCNN v2.0

Фикс: channel-last → channel-first перед Conv1d (необходим permute)
PointNetSetAbstraction возвращает (B, N, C), Conv1d ожидает (B, C, N).
Все .permute(0, 2, 1) делаются внутри forward().
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeFPN(nn.Module):
    """
    Top-down Feature Pyramid Network.

    Вход: выходы SA2, SA3 (SA-extra), SA4 (global) в формате channel-last (B, N, C).
    Выход: p2, p3, p4 — три уровня в формате channel-last (B, N, out_channels).

    Пример:
        fpn = TreeFPN(in_channels=[256, 512, 1024], out_channels=256)
        # f2, f3 — (B, N_i, C_i) channel-last (выход SA)
        # f4     — (B, 1, C4) channel-last (global)
        p2, p3, p4 = fpn(f2, f3, f4)  # все → (B, N_i, out_channels)
    """

    def __init__(
        self,
        in_channels: list[int] = (256, 512, 1024),
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections: 1×1 конволюция, выравнивает каналы до out_channels
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
        f2: torch.Tensor,  # (B, N2, C2) channel-last — выход SA2
        f3: torch.Tensor,  # (B, N3, C3) channel-last — выход SA3
        f4: torch.Tensor,  # (B, 1, C4)  channel-last — global SA4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # channel-last → channel-first для Conv1d: (B, N, C) → (B, C, N)
        f2_cf = f2.permute(0, 2, 1).contiguous()  # (B, C2, N2)
        f3_cf = f3.permute(0, 2, 1).contiguous()  # (B, C3, N3)
        f4_cf = f4.permute(0, 2, 1).contiguous()  # (B, C4, 1)

        l2 = self.lat2(f2_cf)  # (B, out, N2)
        l3 = self.lat3(f3_cf)  # (B, out, N3)
        l4 = self.lat4(f4_cf)  # (B, out, 1)

        # Top-down: обогащаем мелкие уровни контекстом крупных
        p3_cf = self.out3(l3 + F.interpolate(l4, size=l3.shape[-1], mode='nearest'))  # (B, out, N3)
        p2_cf = self.out2(l2 + F.interpolate(p3_cf, size=l2.shape[-1], mode='nearest'))  # (B, out, N2)
        p4_cf = l4  # (B, out, 1)

        # channel-first → channel-last: (B, C, N) → (B, N, C) — совместимо с остальными слоями
        p2 = p2_cf.permute(0, 2, 1).contiguous()  # (B, N2, out)
        p3 = p3_cf.permute(0, 2, 1).contiguous()  # (B, N3, out)
        p4 = p4_cf.permute(0, 2, 1).contiguous()  # (B, 1,  out)

        return p2, p3, p4
