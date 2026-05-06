"""
advisor/advisor.py — главный класс

API:
    advisor = TrainingAdvisor(cfg, data_root)
    advisor.report()                      # все советы в stdout

    # внутри тренинг-лупа:
    advisor.push(epoch, loss_dict, metrics_dict | None)
    advice_list = advisor.advise()        # list[Advice]

    # из Jupyter:
    advisor.report_jupyter()              # ричный HTML
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from advisor.system_probe import probe_system, SystemSnapshot
from advisor.data_probe   import probe_dataset, DatasetStats
from advisor.loss_tracker import LossTracker, LossAnalysis
from advisor.rules        import generate_advices, Advice

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    "critical": "🔴",
    "warning":  "🟡",
    "info":     "🔵",
    "ok":       "✅",
}
_LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2, "ok": 3}


class TrainingAdvisor:
    """
    Один инстанс на одн фолд обучения.

    Parameters
    ----------
    cfg        : OmegaConf или любой cfg с .get()
    data_root  : путь к датасету
    window     : длина окна для анализа loss-тренда
    report_interval : частота автоматического логирования в эпохах (0=нет)
    """

    def __init__(
        self,
        cfg,
        data_root: str | Path = ".",
        window: int = 50,
        report_interval: int = 0,
    ) -> None:
        self._cfg       = cfg
        self._data_root = str(data_root)
        self._tracker   = LossTracker(window=window)
        self._report_interval = report_interval
        self._sys:  Optional[SystemSnapshot] = None
        self._data: Optional[DatasetStats]   = None

    # ------------------------------------------------------------------
    # Probing (lazy, cached until explicitly refreshed)
    # ------------------------------------------------------------------

    def refresh_system(self) -> SystemSnapshot:
        self._sys = probe_system(self._data_root)
        return self._sys

    def refresh_data(self) -> DatasetStats:
        self._data = probe_dataset(Path(self._data_root), self._cfg)
        return self._data

    @property
    def system(self) -> SystemSnapshot:
        if self._sys is None:
            self.refresh_system()
        return self._sys  # type: ignore

    @property
    def data(self) -> DatasetStats:
        if self._data is None:
            self.refresh_data()
        return self._data  # type: ignore

    # ------------------------------------------------------------------
    # Training loop integration
    # ------------------------------------------------------------------

    def push(
        self,
        epoch: int,
        loss_dict: dict,
        metrics: Optional[dict] = None,
    ) -> None:
        """
        Вызывается один раз за эпоху в train_fold().
        metrics = {'f1': 0.72, 'precision': 0.8, ...} — если была валидация.
        """
        self._tracker.push(epoch, loss_dict, metrics)
        if (
            self._report_interval > 0
            and epoch > 0
            and epoch % self._report_interval == 0
        ):
            self.refresh_system()   # обновляем VRAM-статистику
            advices = self.advise()
            if advices:
                logger.info("[Advisor] Epoch %d — %d советов:", epoch, len(advices))
                for a in advices:
                    logger.info("  %s [%s/%s] %s",
                                _LEVEL_EMOJI.get(a.level, "?"),
                                a.level.upper(), a.category, a.text)
                    if a.action:
                        logger.info("    → Действие: %s", a.action)

    # ------------------------------------------------------------------
    # Advice generation
    # ------------------------------------------------------------------

    def advise(self) -> list[Advice]:
        la = self._tracker.analyse()
        return generate_advices(
            sys_snap      = self.system,
            data_stats    = self.data,
            loss_analysis = la,
            cfg           = self._cfg,
        )

    # ------------------------------------------------------------------
    # Report: stdout
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Полный отчёт в stdout."""
        self.refresh_system()
        self.refresh_data()
        la = self._tracker.analyse()
        advices = generate_advices(self.system, self.data, la, self._cfg)

        print("\n" + "=" * 60)
        print("  📊  TRAINING ADVISOR REPORT")
        print("=" * 60)

        self._print_system()
        self._print_data()
        self._print_training(la)
        self._print_advices(advices)

        print("=" * 60 + "\n")

    def _print_system(self) -> None:
        s = self.system
        print("\n🖥️  СИСТЕМА")
        print(f"  CPU: {s.cpu_count} ядер")
        if s.cpu_load_pct is not None:
            print(f"  CPU load: {s.cpu_load_pct:.0f}%")
        print(f"  RAM: {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f} GB  "
              f"(свободно {s.ram_free_gb:.1f} GB)")
        print(f"  Disk free: {s.disk_free_gb:.1f} GB")
        if s.gpus:
            for g in s.gpus:
                print(f"  GPU: {g.name}")
                print(f"    VRAM: {g.vram_used_gb:.1f}/{g.vram_total_gb:.1f} GB  "
                      f"({g.vram_used_gb/g.vram_total_gb*100:.0f}%)")
                if g.temp_c is not None:
                    print(f"    Temp: {g.temp_c:.0f}°C")
                if g.utilization_pct is not None:
                    print(f"    Util: {g.utilization_pct:.0f}%")
        else:
            print("  GPU: нет (CPU режим)")
        print(f"  PyTorch: {s.torch_version}")

    def _print_data(self) -> None:
        d = self.data
        print("\n🌳  ДАННЫЕ")
        print(f"  Плотов: {d.n_plots}")
        print(f"  Деревьев всего: {d.n_trees_total} "
              f"(среднее {d.n_trees_total/max(d.n_plots,1):.1f}/плот)")
        if d.pts_per_plot:
            import numpy as np
            print(f"  Точек/плот: min={min(d.pts_per_plot)} max={max(d.pts_per_plot)} "
                  f"сред={np.mean(d.pts_per_plot):.0f}")
        print(f"  Точек/дерево: сред={d.pts_per_tree_mean:.0f} min={d.pts_per_tree_min:.0f}")
        print(f"  Высота деревьев: μ={d.height_mean:.1f}m σ={d.height_std:.1f}m "
              f"[{d.height_min:.1f} .. {d.height_max:.1f}]")
        if d.sparse_plots:
            print(f"  ⚠️  Редкие плоты: {d.sparse_plots}")

    def _print_training(self, la: LossAnalysis) -> None:
        trend_emoji = {
            "improving": "⬇️", "plateau": "➡️", "diverging": "⬆️",
            "noisy": "🎲", "too_early": "⏳",
        }
        print("\n📈  ОБУЧЕНИЕ")
        print(f"  Trend: {trend_emoji.get(la.trend, '?')} {la.trend}  "
              f"(slope={la.loss_slope:+.2f}%/ep, CV={la.loss_cv:.2f})")
        print(f"  Best F1: {la.best_f1:.4f}  Last F1: {la.last_f1:.4f}")
        print(f"  Эпох без улучшения: {la.epochs_no_improve}")
        print(f"  NaN-rate: {la.nan_rate:.1%}")
        if la.loss_components:
            total = sum(la.loss_components.values())
            print("  Loss компоненты (avg):")
            for k, v in sorted(la.loss_components.items(), key=lambda x: -x[1]):
                bar_len = int(v / total * 20) if total > 0 else 0
                print(f"    {k:<30} {v:.4f}  {'|'*bar_len}")

    def _print_advices(self, advices: list[Advice]) -> None:
        if not advices:
            print("\n✅  Советов нет — всё в норме.")
            return
        advices_sorted = sorted(advices, key=lambda a: _LEVEL_ORDER.get(a.level, 9))
        print(f"\n💡  СОВЕТЫ ({len(advices)}):")
        for i, a in enumerate(advices_sorted, 1):
            emoji = _LEVEL_EMOJI.get(a.level, "?")
            print(f"  {i}. {emoji} [{a.level.upper()}/{a.category}] {a.text}")
            if a.action:
                for k, v in a.action.items():
                    print(f"       → {k}: {v}")

    # ------------------------------------------------------------------
    # Report: Jupyter (HTML)
    # ------------------------------------------------------------------

    def report_jupyter(self) -> None:
        """
        Ричный вывод в Jupyter Notebook (IPython.display.HTML).
        Если IPython недоступен, работает как report().
        """
        try:
            from IPython.display import display, HTML
        except ImportError:
            self.report()
            return

        self.refresh_system()
        self.refresh_data()
        la = self._tracker.analyse()
        advices = generate_advices(self.system, self.data, la, self._cfg)
        advices_sorted = sorted(advices, key=lambda a: _LEVEL_ORDER.get(a.level, 9))

        colour = {"critical": "#ff4444", "warning": "#ffaa00",
                  "info": "#4488ff", "ok": "#44bb44"}

        rows = ""
        for a in advices_sorted:
            c   = colour.get(a.level, "#aaa")
            act = ("<br><code style='font-size:11px'>&rarr; "
                   + "; ".join(f"{k}={v}" for k, v in a.action.items())
                   + "</code>") if a.action else ""
            rows += (
                f"<tr><td style='color:{c};font-weight:bold'>{a.level.upper()}</td>"
                f"<td>{a.category}</td>"
                f"<td>{a.text}{act}</td></tr>"
            )

        s = self.system
        g = s.primary_gpu
        gpu_str = (
            f"{g.name} | VRAM {g.vram_used_gb:.1f}/{g.vram_total_gb:.1f} GB"
            if g else "No GPU"
        )
        d = self.data

        html = f"""
<div style="font-family:monospace;border:1px solid #444;padding:12px;border-radius:6px">
<b style="font-size:15px">📊 Training Advisor Report</b>
<hr style="margin:6px 0">
<table style="width:100%;font-size:12px">
<tr><td><b>CPU</b></td><td>{s.cpu_count} cores, load {s.cpu_load_pct or '?'}%</td>
    <td><b>RAM</b></td><td>{s.ram_used_gb:.1f}/{s.ram_total_gb:.1f} GB free={s.ram_free_gb:.1f}</td></tr>
<tr><td><b>GPU</b></td><td colspan=3>{gpu_str}</td></tr>
<tr><td><b>Data</b></td><td>{d.n_plots} plots, {d.n_trees_total} trees</td>
    <td><b>pts/tree</b></td><td>μ={d.pts_per_tree_mean:.0f} min={d.pts_per_tree_min:.0f}</td></tr>
<tr><td><b>Trend</b></td><td>{la.trend} (slope={la.loss_slope:+.2f}%/ep)</td>
    <td><b>F1</b></td><td>best={la.best_f1:.4f} last={la.last_f1:.4f}</td></tr>
</table>
<hr style="margin:6px 0">
<b>💡 Советы ({len(advices)}):</b>
<table style="width:100%;font-size:12px;margin-top:4px">
<tr><th align=left>Level</th><th align=left>Category</th><th align=left>Advice</th></tr>
{rows if rows else "<tr><td colspan=3 style='color:#44bb44'>✅ Всё в норме.</td></tr>"}
</table>
</div>"""
        display(HTML(html))
