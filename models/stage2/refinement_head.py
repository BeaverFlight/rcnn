"""Stage 2 refinement: classification + regression on Stage-1 proposals."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.stage2.multi_position import MultiPositionExtractor


class RefinementHead(nn.Module):
    """
    Refines Stage-1 proposals using multi-position features.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.extractor = MultiPositionExtractor(cfg)
        feat_dim = self.extractor.out_dim
        self.cls_head = nn.Linear(feat_dim, 1)
        self.reg_head = nn.Linear(feat_dim, 4)

    def forward(
        self, points_list: list[Tensor], proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            points_list: list of (Nk, 3) point tensors, one per proposal
            proposals:   (P, 6) Stage-1 proposals

        Returns:
            cls_logits: (P, 1)
            reg_deltas: (P, 4)
        """
        cls_list, reg_list = [], []
        for pts, prop in zip(points_list, proposals):
            if pts.shape[0] == 0:
                feat = torch.zeros(self.extractor.out_dim, device=proposals.device)
            else:
                feat = self.extractor(pts, prop)
            cls_list.append(self.cls_head(feat.unsqueeze(0)))
            reg_list.append(self.reg_head(feat.unsqueeze(0)))

        return torch.cat(cls_list, dim=0), torch.cat(reg_list, dim=0)
