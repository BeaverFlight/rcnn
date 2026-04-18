"""Custom collate function for variable-size point cloud batches."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def collate_tree_rcnn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate a list of plot samples into a batch.

    Each sample dict may contain tensors of varying first-dimension sizes.
    Stacks fixed-size tensors; lists everything else.
    """
    result: dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], Tensor):
            try:
                result[key] = torch.stack(vals, dim=0)
            except RuntimeError:
                # Variable size — keep as list
                result[key] = vals
        else:
            result[key] = vals

    return result
