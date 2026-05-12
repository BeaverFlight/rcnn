"""
advisor/config_watcher.py — автодетектор изменений конфига

Логика:
  Каждую эпоху advisor вызывает watcher.snapshot(cfg).
  Watcher сравнивает текущий снапшот с предыдущим:
    - Если что-то изменилось → возвращает список ChangedParam(key, old, new)
    - Эти изменения считаются "применёнными" и записываются как pending в DB
  После следующей валидации advisor финализирует delta_f1.

Также строит историю корреляций cfg-параметров и F1/loss:
  watcher.correlations() → dict{key: pearson_r} для анализа.

Отслеживаемые ключи берутся из WATCHED_PATHS — плоский список dotted-key.
По умолчанию — все ключи из cfg.training (рекурсивно).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class ChangedParam:
    key:       str
    old_value: Any
    new_value: Any


@dataclass
class _EpochRecord:
    epoch:    int
    cfg_flat: dict[str, Any]   # {dotted_key: value}
    f1:       Optional[float]  # None если валидации не было
    loss:     Optional[float]


def _flatten(cfg, prefix: str = "") -> dict[str, Any]:
    """OmegaConf или dict → плоский {dotted.key: value}."""
    if hasattr(cfg, "_metadata"):  # OmegaConf DictConfig
        cfg = OmegaConf.to_container(cfg, resolve=True)
    result: dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return {prefix: cfg} if prefix else {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, full_key))
        else:
            result[full_key] = v
    return result


class ConfigWatcher:
    """
    Следит за изменениями конфига между эпохами.

    Parameters
    ----------
    watch_prefix : str
        Префикс для фильтрации ключей (например, 'training').
        Если None — следим за всем конфигом.
    min_epochs_between : int
        Минимальный интервал между двумя записями одного ключа,
        чтобы избежать шума при частых изменениях.
    """

    def __init__(
        self,
        watch_prefix: str = "training",
        min_epochs_between: int = 1,
    ) -> None:
        self._prefix            = watch_prefix
        self._min_between       = min_epochs_between
        self._prev_flat:  dict[str, Any] = {}
        self._history:    list[_EpochRecord] = []
        self._last_change_epoch: dict[str, int] = {}  # key → epoch когда последний раз менялся

    # ------------------------------------------------------------------
    # Основной API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        cfg,
        epoch: int,
        f1: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> list[ChangedParam]:
        """
        Вызывать каждую эпоху. Возвращает список изменённых параметров.

        Parameters
        ----------
        cfg   : OmegaConf cfg
        epoch : текущая эпоха
        f1    : F1 этой эпохи (None если не было валидации)
        loss  : mean train loss этой эпохи
        """
        flat = _flatten(cfg)
        if self._prefix:
            flat = {k: v for k, v in flat.items() if k.startswith(self._prefix)}

        # Сохраняем в историю
        self._history.append(_EpochRecord(epoch=epoch, cfg_flat=flat, f1=f1, loss=loss))

        if not self._prev_flat:
            self._prev_flat = flat
            return []

        changed: list[ChangedParam] = []
        for key, new_val in flat.items():
            old_val = self._prev_flat.get(key)
            if old_val is None or old_val == new_val:
                continue
            last_ep = self._last_change_epoch.get(key, -999)
            if epoch - last_ep < self._min_between:
                continue  # слишком часто — пропускаем шум
            changed.append(ChangedParam(key=key, old_value=old_val, new_value=new_val))
            self._last_change_epoch[key] = epoch
            logger.info(
                "[ConfigWatcher] Epoch %d: %s  %s → %s",
                epoch, key, old_val, new_val,
            )

        self._prev_flat = flat
        return changed

    def update_last_f1(self, f1: float) -> None:
        """
        Если f1 стал известен уже после snapshot() — обновляем последнюю запись.
        Вызывать после валидации.
        """
        if self._history:
            self._history[-1].f1 = f1

    # ------------------------------------------------------------------
    # Корреляционный анализ
    # ------------------------------------------------------------------

    def correlations(self) -> dict[str, dict]:
        """
        Строит корреляцию между значением каждого cfg-ключа и F1.

        Возвращает:
        {
            'training.learning_rate': {
                'pearson_r':  0.71,
                'n_points':   12,
                'values_f1': [(lr_val, f1), ...],   # для scatter
                'direction':  'positive'|'negative'|'unclear',
            },
            ...
        }
        """
        import math
        from collections import defaultdict

        # Собираем пары (value, f1) по ключам — только эпохи с валидацией
        pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for rec in self._history:
            if rec.f1 is None:
                continue
            for key, val in rec.cfg_flat.items():
                try:
                    pairs[key].append((float(val), rec.f1))
                except (TypeError, ValueError):
                    pass  # строки и None пропускаем

        result: dict[str, dict] = {}
        for key, pts in pairs.items():
            if len(pts) < 3:
                continue  # нужен минимум 3 точки
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            r  = _pearson(xs, ys)
            if r is None:
                continue
            direction = (
                "positive" if r > 0.3 else
                "negative" if r < -0.3 else
                "unclear"
            )
            result[key] = {
                "pearson_r":  round(r, 4),
                "n_points":   len(pts),
                "values_f1":  pts,
                "direction":  direction,
            }
        return result

    def param_history(
        self,
        key: str,
    ) -> list[tuple[int, Any, Optional[float], Optional[float]]]:
        """
        Возвращает историю параметра:
        [(epoch, value, f1, loss), ...]
        """
        return [
            (rec.epoch, rec.cfg_flat.get(key), rec.f1, rec.loss)
            for rec in self._history
            if key in rec.cfg_flat
        ]

    def changed_params_report(self) -> str:
        """
        Текстовый отчёт: какие параметры менялись и как изменился F1 после.
        """
        # Найти все эпохи с изменениями
        changes: list[tuple[int, str, Any, Any]] = []  # (epoch, key, old, new)
        prev: dict[str, Any] = {}
        for rec in self._history:
            for key, val in rec.cfg_flat.items():
                if prev.get(key) is not None and prev[key] != val:
                    changes.append((rec.epoch, key, prev[key], val))
            prev = dict(rec.cfg_flat)

        if not changes:
            return "[ConfigWatcher] Изменений конфига не зафиксировано."

        lines = ["\n" + "=" * 58,
                 "  🔍  CONFIG CHANGES & F1 IMPACT",
                 "=" * 58]
        for epoch, key, old, new in changes:
            # F1 до и после
            f1_before = self._f1_near_epoch(epoch - 1)
            f1_after  = self._f1_near_epoch(epoch + 1)
            if f1_before is not None and f1_after is not None:
                delta = f1_after - f1_before
                sign  = "+" if delta >= 0 else ""
                impact = f"ΔF1={sign}{delta:.4f}  (before={f1_before:.4f} after={f1_after:.4f})"
            else:
                impact = "F1 данных нет"
            lines.append(
                f"  Epoch {epoch:4d}  {key}  {old} → {new}   {impact}"
            )
        lines.append("=" * 58)
        return "\n".join(lines)

    def _f1_near_epoch(self, epoch: int) -> Optional[float]:
        """Ближайший F1 к данной эпохе (в окне ±5)."""
        best_dist = 999
        best_f1   = None
        for rec in self._history:
            if rec.f1 is None:
                continue
            d = abs(rec.epoch - epoch)
            if d < best_dist:
                best_dist = d
                best_f1   = rec.f1
        return best_f1 if best_dist <= 5 else None


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    import math
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)
