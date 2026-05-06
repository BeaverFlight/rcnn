"""
models/relation_head.py — Stage 3: Relation Head (Transformer) для TreeRCNN v2.0

Принимает N кандидатов-деревьев из Stage 2,
пропускает их через Self-Attention (каждое дерево «смотрит» на соседей),
выдаёт финальные scores с учётом контекста.

Это заменяет жёсткий NMS на обучаемую фильтрацию дубликатов.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RelationHead(nn.Module):
    """
    Args:
        feat_dim  : размерность признаков кандидата (выход Stage 2)
        coord_dim : размерность координат бокса (по умолчанию 5: cx,cy,cz,w,h)
        n_heads   : число голов внимания
        n_layers  : число TransformerEncoder слоёв

    Forward:
        box_feats  : (B, N, feat_dim) — признаки N кандидатов
        box_coords : (B, N, coord_dim)— координаты боксов (cx,cy,cz,w,h)

    Returns:
        scores : (B, N, 1) — финальные scores с учётом контекста соседей
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        coord_dim: int = 5,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()

        # Позиционное кодирование: координаты бокса → feat_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(coord_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, feat_dim),
        )

        # Self-attention Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN: стабильнее при небольших датасетах
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Финальный пересчёт score с учётом контекста
        self.final_cls = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Dropout на входе
        self.input_drop = nn.Dropout(0.1)

    def forward(
        self,
        box_feats: torch.Tensor,   # (B, N, feat_dim)
        box_coords: torch.Tensor,  # (B, N, coord_dim)
    ) -> torch.Tensor:             # (B, N, 1)

        # Позиционная информация о соседях
        pos = self.pos_embed(box_coords)  # (B, N, feat_dim)

        # Признаки + позиционное кодирование
        x = self.input_drop(box_feats + pos)  # (B, N, feat_dim)

        # Каждое дерево «смотрит» на все остальные
        x = self.transformer(x)  # (B, N, feat_dim)

        # Финальные scores
        scores = self.final_cls(x)  # (B, N, 1)
        return scores
