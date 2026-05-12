"""
advisor/experience_db.py — постоянное хранилище опытов обучения

Каждый "опыт" (Experience) — это снапшот состояния:
  - значения гиперпараметров
  - метрики до и после изменения
  - что было изменено (action)
  - результат: delta_f1 (положительный = улучшение)

Данные сохраняются в JSON (один файл на проект) и доступны
между сессиями.

API:
    db = ExperienceDB(path)             # загружает существующий или создаёт
    db.record(exp)                      # добавляет запись
    db.query(action_key) -> list        # возвращает историю по ключу
    db.best_value(action_key) -> Any    # значение с лучшим delta_f1
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    epoch:       int
    action_key:  str           # e.g. "training.learning_rate"
    old_value:   Any
    new_value:   Any
    f1_before:   float
    f1_after:    float         # -1 если ещё не известен (pending)
    delta_f1:    float         # f1_after - f1_before
    loss_trend:  str           # improving | plateau | diverging | noisy
    nan_rate:    float
    extra:       dict = field(default_factory=dict)


class ExperienceDB:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records: list[Experience] = []
        self._load()

    # ------------------------------------------------------------------
    def record(self, exp: Experience) -> None:
        self._records.append(exp)
        self._save()
        logger.debug("[ExperienceDB] recorded: %s %s→%s Δf1=%+.4f",
                     exp.action_key, exp.old_value, exp.new_value, exp.delta_f1)

    def update_last(self, action_key: str, f1_after: float) -> None:
        """Обновляет последний pending-опыт c нужным ключом."""
        for exp in reversed(self._records):
            if exp.action_key == action_key and exp.f1_after < 0:
                exp.f1_after = f1_after
                exp.delta_f1 = f1_after - exp.f1_before
                self._save()
                return

    def query(self, action_key: str) -> list[Experience]:
        return [e for e in self._records if e.action_key == action_key]

    def best_value(self, action_key: str) -> Optional[Any]:
        """Значение из всех опытов с лучшим средним delta_f1."""
        exps = [e for e in self.query(action_key) if e.delta_f1 > -999]
        if not exps:
            return None
        from collections import defaultdict
        buckets: dict[Any, list[float]] = defaultdict(list)
        for e in exps:
            buckets[e.new_value].append(e.delta_f1)
        best_val = max(buckets, key=lambda v: sum(buckets[v]) / len(buckets[v]))
        return best_val

    def summary(self) -> dict:
        """Сводная статистика по всем ключам."""
        from collections import defaultdict
        import numpy as np
        by_key: dict[str, list[float]] = defaultdict(list)
        for e in self._records:
            if e.delta_f1 > -999:
                by_key[e.action_key].append(e.delta_f1)
        return {
            k: {
                "n":       len(v),
                "mean_df1": round(float(np.mean(v)), 4),
                "pos_rate": round(sum(1 for x in v if x > 0) / len(v), 2),
            }
            for k, v in by_key.items()
        }

    # ------------------------------------------------------------------
    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(e) for e in self._records]
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning("ExperienceDB: не удалось сохранить: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._records = [Experience(**d) for d in data]
            logger.info("[ExperienceDB] Загружено %d опытов из %s",
                        len(self._records), self._path)
        except Exception as exc:
            logger.warning("ExperienceDB: не удалось загрузить: %s", exc)
