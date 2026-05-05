"""Stage 2 refinement: classification + regression on Stage-1 proposals.

Changed:
  - reg_head outputs 6D deltas.
  - Fixed dead-code: first valid_mask assignment inside the loop removed.
  - Batched forward via _extract_batch (no Python loop over proposals).
  - _extract_batch now processes proposals in chunks to avoid CUDA hang
    when val produces 5k+ proposals (inference path).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.stage2.multi_position import MultiPositionExtractor

_S2_FORWARD_CHUNK = 512   # proposals per forward_batch call


class RefinementHead(nn.Module):
    """Refines Stage-1 proposals using multi-position features."""

    MIN_PTS = 4

    def __init__(self, cfg) -> None:
        super().__init__()
        self.extractor = MultiPositionExtractor(cfg)
        feat_dim = self.extractor.out_dim
        self.cls_head = nn.Linear(feat_dim, 1)
        self.reg_head = nn.Linear(feat_dim, 6)
        # chunk size can be tuned via cfg.training.stage2_forward_chunk
        self._fwd_chunk: int = int(
            cfg.training.get("stage2_forward_chunk", _S2_FORWARD_CHUNK)
        )

    def forward(
        self, points_list: list[Tensor], proposals: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            points_list: list of (Nk, 3)
            proposals:   (P, 6)

        Returns:
            cls_logits: (P, 1)
            reg_deltas: (P, 6)
        """
        if len(points_list) == 0:
            dev = proposals.device
            return torch.zeros(0, 1, device=dev), torch.zeros(0, 6, device=dev)

        feats = self._extract_batch(points_list, proposals)   # (P, feat_dim)
        return self.cls_head(feats), self.reg_head(feats)

    def _extract_batch(self, points_list: list[Tensor], proposals: Tensor) -> Tensor:
        """
        Batch feature extraction for all proposals, chunked to avoid
        CUDA memory spikes / hangs on large inference proposal sets.
        """
        device   = proposals.device
        P        = len(proposals)
        feat_dim = self.extractor.out_dim
        out      = torch.zeros(P, feat_dim, device=device)

        valid_mask = torch.tensor(
            [p.shape[0] >= self.MIN_PTS for p in points_list],
            dtype=torch.bool, device=device,
        )
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return out

        valid_pts   = [points_list[i] for i in valid_idx.tolist()]
        valid_props = proposals[valid_idx]   # (V, 6)
        V           = len(valid_pts)
        chunk       = self._fwd_chunk

        feat_chunks: list[Tensor] = []
        for start in range(0, V, chunk):
            end        = min(start + chunk, V)
            chunk_pts  = valid_pts[start:end]
            chunk_prop = valid_props[start:end]

            max_n   = max(p.shape[0] for p in chunk_pts)
            batched = torch.zeros(len(chunk_pts), max_n, 3, device=device)
            for k, pts in enumerate(chunk_pts):
                n = pts.shape[0]
                if n < max_n:
                    pts = pts.repeat((max_n + n - 1) // n, 1)[:max_n]
                batched[k] = pts

            feat_chunks.append(
                self.extractor.forward_batch(batched, chunk_prop)  # (chunk, feat_dim)
            )

        out[valid_idx] = torch.cat(feat_chunks, dim=0)
        return out
