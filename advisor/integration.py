"""
advisor/integration.py — патчи для интеграции в train.py

Изменения в train.py минимальны:
    from advisor.integration import make_advisor, advisor_push, advisor_val_push

    # до цикла
    advisor = make_advisor(cfg, data_root)

    # внутри epoch-цикла (передаём cfg!)
    advisor_push(advisor, epoch, loss_dict, cfg=cfg)

    # после val (передаём cfg и метрики)
    advisor_val_push(advisor, epoch, loss_dict, score_info, cfg=cfg)

ConfigWatcher автоматически детектирует что изменилось и пишет в DB.
Никакого ручного confirm_applied() не нужно.
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
    watch_prefix: str = "training",
) -> TrainingAdvisor:
    """
    Создаёт Advisor с ConfigWatcher.
    Вызывать до начала цикла.
    """
    adv = TrainingAdvisor(
        cfg=cfg, data_root=data_root,
        window=window, report_interval=report_interval,
        watch_prefix=watch_prefix,
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
    cfg=None,
) -> None:
    """
    Вызывать каждую эпоху.

    Передавай cfg чтобы ConfigWatcher детектировал изменения.
    Если в эту эпоху нет валидации — metrics=None.
    """
    if advisor is not None:
        changed = advisor.push(epoch, loss_dict, metrics=metrics, cfg=cfg)
        if changed:
            import logging
            log = logging.getLogger("train")
            for cp in changed:
                log.info(
                    "[Advisor/Watcher] Epoch %d: обнаружено изменение %s: %s → %s",
                    epoch, cp.key, cp.old_value, cp.new_value,
                )


def advisor_val_push(
    advisor: Optional[TrainingAdvisor],
    epoch: int,
    loss_dict: dict,
    score_info: dict,
    cfg=None,
) -> None:
    """
    Вызывать после валидации.
    score_info — дикт из _quality_score(): содержит 'f1', 'precision', 'recall'.
    Передавай cfg чтобы ConfigWatcher зафиксировал состояние и обновил F1.
    """
    if advisor is not None:
        metrics = {"f1": score_info.get("f1", 0.0),
                   "precision": score_info.get("precision", 0.0),
                   "recall": score_info.get("recall", 0.0)}
        advisor.push(epoch, loss_dict, metrics=metrics, cfg=cfg)
