"""
advisor/learner.py — байесовский learner поверх ExperienceDB

Чему учится:
  1. Для каждого гиперпараметра строит распределение delta_f1
     по значениям → знает, что «обычно работает».
  2. После каждой валидации обновляет ExperienceDB и пересчитывает
     prior-ы для правил.
  3. generate_learned_advices() возвращает советы на основе накопленного
     опыта (дополняет правила из rules.py).

Алгоритм:
  - Для каждого action_key смотрим на историю (old_value, new_value, delta_f1).
  - Используем Upper Confidence Bound (UCB1) по бакетам значений:
      score(v) = mean_delta_f1(v) + C * sqrt(ln(N_total) / n(v))
    UCB1 балансирует exploitation (лучшее известное значение)
    и exploration (мало пробованные значения).
  - Если рекомендованное значение != текущему → выдаём Advice.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from advisor.experience_db import ExperienceDB, Experience
from advisor.loss_tracker  import LossAnalysis
from advisor.rules         import Advice

logger = logging.getLogger(__name__)

# Гиперпараметры UCB1
_UCB_C         = 1.0   # exploration weight
_MIN_SAMPLES   = 3     # минимум опытов для уверенного совета
_MIN_DELTA_F1  = 0.005 # минимальный прирост F1 чтобы рекомендовать

# Гиперпараметры которые мы отслеживаем
_TRACKED_KEYS = [
    "training.learning_rate",
    "training.max_grad_norm",
    "training.stage2_forward_chunk",
    "training.lambda_reg",
    "training.lambda_v_reg",
    "training.lambda_offset",
    "training.lambda_centerness",
    "training.bce_weight",
    "training.dice_weight",
]


class BayesianAdvisorLearner:
    """
    Учится на истории ExperienceDB и генерирует советы.

    Parameters
    ----------
    db          : ExperienceDB
    ucb_c       : exploration weight для UCB1
    min_samples : минимум опытов для выдачи совета
    """

    def __init__(
        self,
        db: ExperienceDB,
        ucb_c: float = _UCB_C,
        min_samples: int = _MIN_SAMPLES,
    ) -> None:
        self._db          = db
        self._ucb_c       = ucb_c
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Запись нового опыта
    # ------------------------------------------------------------------

    def record_action(
        self,
        epoch: int,
        action: dict,
        cfg,
        f1_before: float,
        loss_analysis: LossAnalysis,
    ) -> None:
        """Вызывается когда Advisor решил что-то порекомендовать."""
        for key, new_val in action.items():
            old_val = self._get_cfg_val(cfg, key)
            exp = Experience(
                epoch       = epoch,
                action_key  = key,
                old_value   = old_val,
                new_value   = new_val,
                f1_before   = f1_before,
                f1_after    = -1.0,     # pending — обновится после следующей val
                delta_f1    = -999.0,   # sentinel
                loss_trend  = loss_analysis.trend,
                nan_rate    = loss_analysis.nan_rate,
            )
            self._db.record(exp)

    def update_f1_after(
        self,
        action_key: str,
        f1_after: float,
    ) -> None:
        """После val обновляем последний pending опыт."""
        self._db.update_last(action_key, f1_after)

    # ------------------------------------------------------------------
    # UCB1 рекомендации
    # ------------------------------------------------------------------

    def generate_learned_advices(
        self,
        cfg,
        current_f1: float,
        loss_analysis: LossAnalysis,
    ) -> list[Advice]:
        """
        Возвращает Advice на основе накопленного опыта.
        Советует попробовать значение с наибольшим UCB1-score.
        """
        advices: list[Advice] = []

        for key in _TRACKED_KEYS:
            exps = [
                e for e in self._db.query(key)
                if e.delta_f1 > -999  # только финализированные
            ]
            if len(exps) < self._min_samples:
                continue

            current_val = self._get_cfg_val(cfg, key)
            best_val, best_score, n_total = self._ucb1_best(
                exps, exclude_val=current_val
            )
            if best_val is None:
                continue

            # Проверяем ожидаемый прирост
            bucket_vals = [e.delta_f1 for e in exps if e.new_value == best_val]
            mean_delta  = float(np.mean(bucket_vals)) if bucket_vals else 0.0

            if mean_delta < _MIN_DELTA_F1:
                continue

            n_tries = len(bucket_vals)
            advices.append(Advice(
                level    = "info",
                category = "learned",
                text     = (
                    f"[Learner] По опыту {len(exps)} экспериментов: "
                    f"{key}={best_val} даёт +{mean_delta:.3f} F1 в среднем "
                    f"(n={n_tries}, UCB={best_score:.3f}). "
                    f"Текущее: {current_val}."
                ),
                action={key: best_val},
            ))

        return advices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cfg_val(cfg, dotted_key: str) -> Any:
        """cfg.training.learning_rate → float."""
        parts = dotted_key.split(".")
        obj   = cfg
        for p in parts:
            try:
                obj = getattr(obj, p) if not isinstance(obj, dict) else obj[p]
            except (AttributeError, KeyError):
                try:
                    obj = obj.get(p, None)
                except Exception:
                    return None
        return obj

    def _ucb1_best(
        self,
        exps: list[Experience],
        exclude_val: Any = None,
    ) -> tuple[Optional[Any], float, int]:
        """
        UCB1 по бакетам (уникальным new_value).
        Возвращает (best_value, ucb_score, n_total).
        """
        buckets: dict[Any, list[float]] = defaultdict(list)
        for e in exps:
            if e.new_value != exclude_val:
                buckets[e.new_value].append(e.delta_f1)

        if not buckets:
            return None, 0.0, len(exps)

        N = len(exps)
        best_val, best_score = None, -1e9
        for val, deltas in buckets.items():
            mean  = float(np.mean(deltas))
            n     = len(deltas)
            bonus = self._ucb_c * math.sqrt(math.log(max(N, 2)) / max(n, 1))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_val   = val

        return best_val, best_score, N

    # ------------------------------------------------------------------
    # Post-training analysis
    # ------------------------------------------------------------------

    def post_training_report(self) -> str:
        """Текстовый отчёт после завершения обучения."""
        summary = self._db.summary()
        if not summary:
            return "[Learner] Нет данных для анализа."

        lines = ["\n" + "="*55,
                 "  🧠  LEARNER POST-TRAINING REPORT",
                 "="*55]
        for key, stat in sorted(summary.items(),
                                key=lambda x: -abs(x[1]["mean_df1"])):
            sign = "+" if stat["mean_df1"] >= 0 else ""
            lines.append(
                f"  {key:<40}  n={stat['n']:>3}  "
                f"mean_ΔF1={sign}{stat['mean_df1']:+.4f}  "
                f"pos_rate={stat['pos_rate']:.0%}"
            )
        lines.append("="*55)
        return "\n".join(lines)
