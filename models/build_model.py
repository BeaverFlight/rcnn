"""
models/build_model.py — фабрика моделей

Единственное место, где решается v1 vs v2.
train.py, eval.py, predict.py импортируют только build_model.
Добавить новую версию — добавить vetвь в MODEL_REGISTRY.

VERSION_KEY в конфиге: cfg.model_version (str, default "v1")
Чтобы перейти на v2 — добавь в config:
    model_version: v2
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_model(cfg) -> nn.Module:
    """
    Создаёт модель по cfg.model_version.

    v1 (default) — TreeRCNN  (исходная архитектура)
    v2           — TreeRCNNV2 (FPN + Voting + RelationHead)
    """
    version: str = str(getattr(cfg, "model_version", "v1")).lower().strip()

    if version == "v2":
        from models.tree_rcnn_v2 import TreeRCNNV2
        model = TreeRCNNV2(cfg)
        logger.info("Model: TreeRCNNV2 (v2)")
    else:
        from models.tree_rcnn import TreeRCNN
        model = TreeRCNN(cfg)
        logger.info("Model: TreeRCNN (v1)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Trainable parameters: {:,}".format(n_params))
    return model
