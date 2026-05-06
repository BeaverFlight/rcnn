"""
advisor/integration.py — патчи для интеграции в train.py

Изменения в train.py минимальны:
    from advisor.integration import make_advisor, advisor_push, advisor_val_push
    from advisor.integration import advisor_confirm_applied  # ← new

    # до цикла
    advisor = make_advisor(cfg, data_root)

    # внутри epoch-цикла
    advisor_push(advisor, epoch, loss_dict, metrics=None)

    # после val
    advisor_val_push(advisor, epoch, loss_dict, score_info)

    # когда реально меняешь гиперпараметр по совету Advisor'а:
    advisor_confirm_applied(advisor, a.action)
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
    adv.refresh_system()
    adv.refresh_data()
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


def advisor_confirm_applied(
    advisor: Optional[TrainingAdvisor],
    action: dict,
) -> None:
    """
    Вызывать ТОЛЬКО когда совет реально применён (конфиг изменён).

    Создаёт pending-запись в ExperienceDB с текущим f1_before.
    Запись будет финализирована на следующем advisor_val_push().

    Пример использования в train.py:

        advices = advisor.advise()
        for a in advices:
            if a.level in ('critical', 'warning') and a.action:
                logger.info('[Advisor] Применяем: %s', a.action)
                apply_action_to_cfg(cfg, a.action)   # твоя функция
                advisor_confirm_applied(advisor, a.action)
                break  # применяем одно изменение за раз
    """
    if advisor is not None:
        advisor.confirm_applied(action)
