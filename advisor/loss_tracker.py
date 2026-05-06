"""
advisor/loss_tracker.py — накапливает историю loss + метрик

API:
    tracker = LossTracker(window=50)
    tracker.push(epoch, loss_dict, metrics_dict | None)
    analysis = tracker.analyse()

Analysis fields:
    trend          — 'improving' | 'plateau' | 'diverging' | 'noisy' | 'too_early'
    loss_slope     — наклон за последние window эпох (отриц.— хорошо)
    loss_cv        — коэффициент вариации (std/mean) — мера шума
    best_f1        — лучший F1 за всё время
    last_f1        — F1 на последней валидации
    epochs_no_improve — число эпох без улучшения F1
    nan_rate       — доля NaN/Inf loss батчей
    loss_components — средние по компонентам
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LossRecord:
    epoch:   int
    loss:    float
    is_nan:  bool
    comps:   dict[str, float]  # loss_stage1_cls, loss_stage2_reg, ...
    f1:      Optional[float]   # None если в эту эпоху не было val


@dataclass
class LossAnalysis:
    trend:            str             # improving | plateau | diverging | noisy | too_early
    loss_slope:       float           # наклон (%/эпоха)
    loss_cv:          float           # std/mean — мера шума
    best_f1:          float
    last_f1:          float
    epochs_no_improve:int
    nan_rate:         float           # 0.0..1.0
    loss_components:  dict[str, float] = field(default_factory=dict)


class LossTracker:
    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._records: deque[LossRecord] = deque()
        self._best_f1 = 0.0
        self._epochs_no_improve = 0
        self._last_f1 = 0.0

    def push(
        self,
        epoch: int,
        loss_dict: dict,
        metrics: Optional[dict] = None,
    ) -> None:
        total = loss_dict.get("total_loss", float("nan"))
        is_nan = not np.isfinite(float(total)) if total is not None else True
        comps = {
            k: float(v)
            for k, v in loss_dict.items()
            if k != "total_loss" and np.isfinite(float(v))
        }
        f1 = float(metrics["f1"]) if metrics and "f1" in metrics else None
        if f1 is not None:
            if f1 > self._best_f1:
                self._best_f1 = f1
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1
            self._last_f1 = f1
        self._records.append(LossRecord(
            epoch=epoch, loss=float(total) if not is_nan else float("nan"),
            is_nan=is_nan, comps=comps, f1=f1,
        ))

    def analyse(self) -> LossAnalysis:
        recs = list(self._records)
        N = len(recs)
        if N < 5:
            return LossAnalysis(
                trend="too_early", loss_slope=0.0, loss_cv=0.0,
                best_f1=self._best_f1, last_f1=self._last_f1,
                epochs_no_improve=self._epochs_no_improve,
                nan_rate=0.0,
            )

        nan_rate = sum(r.is_nan for r in recs) / N
        valid    = [r for r in recs if not r.is_nan]
        win      = valid[-self._window:] if len(valid) >= self._window else valid

        if not win:
            return LossAnalysis(
                trend="diverging", loss_slope=float("nan"), loss_cv=float("nan"),
                best_f1=self._best_f1, last_f1=self._last_f1,
                epochs_no_improve=self._epochs_no_improve, nan_rate=nan_rate,
            )

        losses = np.array([r.loss for r in win])
        epochs = np.arange(len(losses), dtype=float)
        slope  = float(np.polyfit(epochs, losses, 1)[0])   # абсолютный наклон
        mean_l = float(np.mean(losses))
        cv     = float(np.std(losses) / mean_l) if mean_l > 0 else 0.0

        # relative slope (%/эпоха)
        rel_slope = slope / mean_l * 100 if mean_l > 0 else slope

        if nan_rate > 0.1:
            trend = "diverging"
        elif cv > 0.3:
            trend = "noisy"
        elif rel_slope < -0.5:
            trend = "improving"
        elif rel_slope > 1.0:
            trend = "diverging"
        else:
            trend = "plateau"

        # Средние по компонентам
        all_comps: dict[str, list[float]] = {}
        for r in win:
            for k, v in r.comps.items():
                all_comps.setdefault(k, []).append(v)
        avg_comps = {k: float(np.mean(v)) for k, v in all_comps.items()}

        return LossAnalysis(
            trend=trend, loss_slope=rel_slope, loss_cv=cv,
            best_f1=self._best_f1, last_f1=self._last_f1,
            epochs_no_improve=self._epochs_no_improve,
            nan_rate=nan_rate, loss_components=avg_comps,
        )
