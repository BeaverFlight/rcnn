"""
models/losses.py — обратно-совместимый шим.

Каноническая реализация всех loss-функций находится в пакете
models/losses/ (focal_loss.py, smooth_l1.py, dice_loss.py,
stage2_v2.py, relation.py).

Этот файл — тонкая обёртка для обратной совместимости:
  from models.losses import stage2_loss_v2  # работает через оба пути

НЕ добавляй сюда новый код — правь соответствующий модуль в losses/.
"""
from models.losses.focal_loss import sigmoid_focal_loss       # noqa: F401
from models.losses.smooth_l1  import smooth_l1_loss           # noqa: F401
from models.losses.dice_loss  import dice_loss, bce_dice_loss # noqa: F401
from models.losses.stage2_v2  import stage2_loss_v2           # noqa: F401
from models.losses.relation   import relation_loss             # noqa: F401

__all__ = [
    "sigmoid_focal_loss",
    "smooth_l1_loss",
    "dice_loss",
    "bce_dice_loss",
    "stage2_loss_v2",
    "relation_loss",
]
