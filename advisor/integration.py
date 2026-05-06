"""
advisor/integration.py — паччи для интеграции в train.py

Изменения в train.py минимальны:
    from advisor.integration import make_advisor, advisor_push, advisor_val_push

    # до цикла
    advisor = make_advisor(cfg, data_root)

    # внутри epoch-цикла
    advisor_push(advisor, epoch, loss_dict, metrics=None)

    # после val
    advisor_val_push(advisor, epoch, loss_dict, score_info)

    # по желанию
    advisor.report()
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from advisor.advisor import TrainingAdvisor


def make_advisor(
    cfg,
    data_root: str | Path,
    window: int = 50,
    report_interval: int = 0,
) -> TrainingAdvisor:
    """
    Создаёт Advisor, одновременно запуская сканирование системы и данных.
    Вызывать до начала цикла.
    """
    adv = TrainingAdvisor(
        cfg=cfg, data_root=data_root,
        window=window, report_interval=report_interval,
    )
    # Делаем probe заранее — кашируется
    adv.refresh_system()
    adv.refresh_data()
    # Печатаем первичный репорт
    adv.report()
    return adv


def advisor_push(
    advisor: Optional[TrainingAdvisor],
    epoch: int,
    loss_dict: dict,
    metrics: Optional[dict] = None,
) -> None:
    """No-op если advisor is None."""
    if advisor is not None:
        advisor.push(epoch, loss_dict, metrics)


def advisor_val_push(
    advisor: Optional[TrainingAdvisor],
    epoch: int,
    loss_dict: dict,
    score_info: dict,
) -> None:
    """
    Вызывается после валидации.
    score_info — дикт из _quality_score(): содержит 'f1', 'precision', 'recall'.
    """
    if advisor is not None:
        metrics = {"f1": score_info.get("f1", 0.0),
                   "precision": score_info.get("precision", 0.0),
                   "recall": score_info.get("recall", 0.0)}
        advisor.push(epoch, loss_dict, metrics)
