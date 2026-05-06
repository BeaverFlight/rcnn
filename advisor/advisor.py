"""
advisor/advisor.py — главный класс (v2: самообучающийся + ConfigWatcher)

API:
    advisor = TrainingAdvisor(cfg, data_root)
    advisor.report()                             # начальный отчёт в stdout
    advisor.report_jupyter()                     # HTML в Jupyter

    # внутри train-цикла:
    advisor.push(epoch, loss_dict, metrics=None) # каждую эпоху

    # после завершения обучения:
    advisor.finalize()                           # постоянный отчёт + сохранение DB

Как теперь работает подтверждение:
  ConfigWatcher каждую эпоху сравнивает конфиг с предыдущей эпохой.
  Если что-то изменилось — это считается "применённым действием" и
  автоматически создаётся pending-запись в ExperienceDB.
  Никакого ручного confirm_applied() не нужно.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from advisor.system_probe  import probe_system, SystemSnapshot
from advisor.data_probe    import probe_dataset, DatasetStats
from advisor.loss_tracker  import LossTracker, LossAnalysis
from advisor.rules         import generate_advices, Advice
from advisor.experience_db import ExperienceDB
from advisor.learner       import BayesianAdvisorLearner
from advisor.config_watcher import ConfigWatcher, ChangedParam

logger = logging.getLogger(__name__)

_LEVEL_EMOJI  = {"critical": "🔴", "warning": "🟡", "info": "🔵", "ok": "✅", "learned": "🧠"}
_LEVEL_ORDER  = {"critical": 0, "warning": 1, "learned": 2, "info": 3, "ok": 4}


class TrainingAdvisor:
    """
    Анализирует ход обучения, систему и данные.
    Самообучается: записывает опыты в ExperienceDB и генерирует
    советы через BayesianAdvisorLearner (UCB1 по гиперпараметрам).

    Изменения конфига детектируются автоматически через ConfigWatcher:
    каждую эпоху push() сравнивает конфиг с предыдущим снапшотом и
    создаёт pending-запись в ExperienceDB без ручного вмешательства.

    Parameters
    ----------
    cfg              : OmegaConf cfg
    data_root        : путь к датасету
    db_path          : путь к JSON-базе опытов (None → <data_root>/advisor_db.json)
    window           : длина окна для trend-анализа loss
    report_interval  : каждые N эпох логировать советы (0 = никогда)
    watch_prefix     : какие ключи конфига отслеживать (default='training')
    """

    def __init__(
        self,
        cfg,
        data_root: str | Path = ".",
        db_path: Optional[str | Path] = None,
        window: int = 50,
        report_interval: int = 0,
        watch_prefix: str = "training",
    ) -> None:
        self._cfg       = cfg
        self._data_root = str(data_root)
        self._tracker   = LossTracker(window=window)
        self._report_interval = report_interval

        if db_path is None:
            db_path = Path(data_root) / "advisor_db.json"
        self._db      = ExperienceDB(db_path)
        self._learner = BayesianAdvisorLearner(self._db)
        self._watcher = ConfigWatcher(watch_prefix=watch_prefix)

        self._sys:  Optional[SystemSnapshot] = None
        self._data: Optional[DatasetStats]   = None

        self._last_f1:    float = 0.0
        self._last_epoch: int   = 0
        # pending_actions: action_key → f1_before
        # Заполняется ТОЛЬКО через ConfigWatcher при обнаружении реальных изменений.
        self._pending_actions: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Probing
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
        cfg=None,
    ) -> list[ChangedParam]:
        """
        Вызывается каждую эпоху.

        Parameters
        ----------
        epoch     : текущая эпоха
        loss_dict : словарь loss-компонентов из model.forward()
        metrics   : {'f1': 0.72, ...} — если была валидация
        cfg       : текущий конфиг (передавай чтобы детектировать изменения).
                    Если None — ConfigWatcher пропускает снапшот.

        Returns
        -------
        list[ChangedParam]
            Список параметров, которые изменились в эту эпоху.
            Пустой список если изменений нет.
        """
        self._tracker.push(epoch, loss_dict, metrics)
        self._last_epoch = epoch

        f1 = metrics.get("f1") if metrics else None
        mean_loss = None
        if loss_dict:
            import torch
            vals = []
            for v in loss_dict.values():
                try:
                    vals.append(float(v.item()) if hasattr(v, 'item') else float(v))
                except (TypeError, ValueError):
                    pass
            mean_loss = sum(vals) / len(vals) if vals else None

        # --- ConfigWatcher: детектируем изменения ---
        changed: list[ChangedParam] = []
        if cfg is not None:
            changed = self._watcher.snapshot(
                cfg, epoch=epoch, f1=f1, loss=mean_loss
            )
            # Нашли реальные изменения → пишем pending в DB
            la = self._tracker.analyse()
            for cp in changed:
                self._learner.record_action(
                    epoch         = epoch,
                    action        = {cp.key: cp.new_value},
                    cfg           = cfg,
                    f1_before     = self._last_f1,
                    loss_analysis = la,
                )
                self._pending_actions[cp.key] = self._last_f1

        if f1 is not None:
            # Обновляем watcher если f1 пришёл после snapshot
            self._watcher.update_last_f1(f1)
            # Финализируем pending
            for key in list(self._pending_actions.keys()):
                self._learner.update_f1_after(key, float(f1))
            self._pending_actions.clear()
            self._last_f1 = float(f1)

        if (
            self._report_interval > 0
            and epoch > 0
            and epoch % self._report_interval == 0
        ):
            self.refresh_system()
            self._log_advices(epoch)

        return changed

    # ------------------------------------------------------------------
    # Advice generation (rules + learned)
    # NOTE: advise() НЕ пишет в ExperienceDB.
    #       Запись происходит автоматически через ConfigWatcher в push().
    # ------------------------------------------------------------------

    def advise(self) -> list[Advice]:
        """
        Генерирует и возвращает список советов.
        НЕ записывает ничего в ExperienceDB.
        """
        la = self._tracker.analyse()
        rule_advices = generate_advices(
            sys_snap      = self.system,
            data_stats    = self.data,
            loss_analysis = la,
            cfg           = self._cfg,
        )
        learned_advices = self._learner.generate_learned_advices(
            cfg           = self._cfg,
            current_f1    = self._last_f1,
            loss_analysis = la,
        )
        return rule_advices + learned_advices

    def _log_advices(self, epoch: int) -> None:
        advices = self.advise()
        if not advices:
            return
        logger.info("[Advisor] Epoch %d — %d советов:", epoch, len(advices))
        for a in sorted(advices, key=lambda x: _LEVEL_ORDER.get(x.level, 9)):
            logger.info("  %s [%s/%s] %s",
                        _LEVEL_EMOJI.get(a.level, "?"),
                        a.level.upper(), a.category, a.text)
            if a.action:
                logger.info("    → %s", a.action)

    # ------------------------------------------------------------------
    # Correlation / param history
    # ------------------------------------------------------------------

    def correlations(self) -> dict[str, dict]:
        """
        Корреляции (Pearson r) между cfg-параметрами и F1.
        Возвращает только ключи с n_points >= 3.
        """
        return self._watcher.correlations()

    def param_history(self, key: str):
        """
        История конкретного параметра:
        [(epoch, value, f1, loss), ...]
        """
        return self._watcher.param_history(key)

    def changes_report(self) -> str:
        """Текстовый отчёт: какие параметры менялись и как изменился F1 после."""
        return self._watcher.changed_params_report()

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Постоянный отчёт + Learner-аналитика + изменения конфига.
        Вызывать в конце train_fold().
        """
        if self._pending_actions:
            logger.warning(
                "[Advisor] finalize: есть %d pending без f1_after: %s",
                len(self._pending_actions), list(self._pending_actions.keys())
            )
        self.refresh_system()
        self.report()
        print(self._watcher.changed_params_report())
        print(self._learner.post_training_report())
        logger.info("[Advisor] ExperienceDB: %d записей, %s",
                    len(self._db._records), self._db._path)

    # ------------------------------------------------------------------
    # Report: stdout
    # ------------------------------------------------------------------

    def report(self) -> None:
        self.refresh_system()
        self.refresh_data()
        la      = self._tracker.analyse()
        advices = generate_advices(self.system, self.data, la, self._cfg)
        learned = self._learner.generate_learned_advices(
            self._cfg, self._last_f1, la
        )
        all_adv = advices + learned

        print("\n" + "=" * 62)
        print("  📊  TRAINING ADVISOR REPORT")
        print("=" * 62)
        self._print_system()
        self._print_data()
        self._print_training(la)
        self._print_advices(all_adv)
        print("=" * 62 + "\n")

    def _print_system(self) -> None:
        s = self.system
        print("\n🖥️  СИСТЕМА")
        print(f"  CPU: {s.cpu_count} ядер", end="")
        if s.cpu_load_pct is not None:
            print(f", загрузка {s.cpu_load_pct:.0f}%", end="")
        print()
        print(f"  RAM: {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f} GB  (свободно {s.ram_free_gb:.1f} GB)")
        print(f"  Disk free: {s.disk_free_gb:.1f} GB")
        if s.gpus:
            for g in s.gpus:
                pct = g.vram_used_gb / g.vram_total_gb * 100
                print(f"  GPU: {g.name}")
                print(f"    VRAM: {g.vram_used_gb:.1f}/{g.vram_total_gb:.1f} GB ({pct:.0f}%)")
                if g.temp_c is not None:       print(f"    Temp: {g.temp_c:.0f}°C")
                if g.utilization_pct is not None: print(f"    Util: {g.utilization_pct:.0f}%")
        else:
            print("  GPU: нет (CPU режим)")
        print(f"  PyTorch: {s.torch_version}")

    def _print_data(self) -> None:
        d = self.data
        print("\n🌳  ДАННЫЕ")
        print(f"  Плотов: {d.n_plots}")
        print(f"  Деревьев: {d.n_trees_total} (среднее {d.n_trees_total/max(d.n_plots,1):.1f}/плот)")
        if d.pts_per_plot:
            import numpy as np
            print(f"  Точек/плот: min={min(d.pts_per_plot)} max={max(d.pts_per_plot)} "
                  f"μ={np.mean(d.pts_per_plot):.0f}")
        print(f"  Точек/дерево: μ={d.pts_per_tree_mean:.0f} min={d.pts_per_tree_min:.0f}")
        print(f"  Высота деревьев: μ={d.height_mean:.1f}m σ={d.height_std:.1f}m "
              f"[{d.height_min:.1f}..{d.height_max:.1f}]")
        if d.sparse_plots:
            print(f"  ⚠️  Редкие плоты: {d.sparse_plots}")

    def _print_training(self, la: LossAnalysis) -> None:
        trend_emoji = {
            "improving": "⬇️", "plateau": "➡️", "diverging": "⬆️",
            "noisy": "🎲", "too_early": "⏳",
        }
        print("\n📈  ОБУЧЕНИЕ")
        print(f"  Trend: {trend_emoji.get(la.trend,'?')} {la.trend}  "
              f"(slope={la.loss_slope:+.2f}%/ep, CV={la.loss_cv:.2f})")
        print(f"  Best F1: {la.best_f1:.4f}  Last F1: {la.last_f1:.4f}")
        print(f"  Эпох без улучшения: {la.epochs_no_improve}")
        print(f"  NaN-rate: {la.nan_rate:.1%}")

        if la.loss_components:
            total = sum(la.loss_components.values())
            print("  Loss компоненты (avg):")
            for k, v in sorted(la.loss_components.items(), key=lambda x: -x[1]):
                bar = int(v / total * 20) if total > 0 else 0
                print(f"    {k:<35} {v:.4f}  {'|'*bar}")

        summary = self._db.summary()
        if summary:
            print("\n🧠  LEARNER (накопленный опыт):")
            for key, stat in sorted(summary.items(),
                                    key=lambda x: -abs(x[1]['mean_df1']))[:5]:
                sign = "+" if stat['mean_df1'] >= 0 else ""
                print(f"    {key:<40} n={stat['n']:>3}  "
                      f"ΔF1={sign}{stat['mean_df1']:+.4f}  "
                      f"pos={stat['pos_rate']:.0%}")

        # Топ корреляций
        corrs = self._watcher.correlations()
        if corrs:
            print("\n📐  КОРРЕЛЯЦИИ (cfg → F1):")
            for key, info in sorted(
                corrs.items(), key=lambda x: -abs(x[1]['pearson_r'])
            )[:5]:
                bar = int(abs(info['pearson_r']) * 20)
                sign = "+" if info['pearson_r'] >= 0 else "-"
                print(f"    {key:<40} r={sign}{abs(info['pearson_r']):.3f}  "
                      f"{'|'*bar}  n={info['n_points']}  [{info['direction']}]")

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
    # Report: Jupyter
    # ------------------------------------------------------------------

    def report_jupyter(self) -> None:
        try:
            from IPython.display import display, HTML
        except ImportError:
            self.report()
            return

        self.refresh_system()
        self.refresh_data()
        la      = self._tracker.analyse()
        advices = generate_advices(self.system, self.data, la, self._cfg)
        learned = self._learner.generate_learned_advices(self._cfg, self._last_f1, la)
        all_adv = sorted(advices + learned,
                         key=lambda a: _LEVEL_ORDER.get(a.level, 9))

        colour = {
            "critical": "#ff4444", "warning": "#ffaa00",
            "info": "#4488ff",    "ok": "#44bb44",
            "learned": "#aa44ff",
        }
        rows = ""
        for a in all_adv:
            c   = colour.get(a.level, "#aaa")
            act = (
                "<br><code style='font-size:11px'>&rarr; "
                + "; ".join(f"{k}={v}" for k, v in a.action.items())
                + "</code>"
            ) if a.action else ""
            rows += (
                f"<tr><td style='color:{c};font-weight:bold'>{a.level.upper()}</td>"
                f"<td>{a.category}</td>"
                f"<td>{a.text}{act}</td></tr>"
            )

        # Корреляции в HTML
        corrs = self._watcher.correlations()
        corr_rows = ""
        for key, info in sorted(corrs.items(), key=lambda x: -abs(x[1]['pearson_r']))[:8]:
            sign = "+" if info['pearson_r'] >= 0 else ""
            col  = "#44bb44" if info['direction'] == 'positive' else (
                   "#ff6644" if info['direction'] == 'negative' else "#aaaaaa"
            )
            corr_rows += (
                f"<tr><td>{key}</td>"
                f"<td style='color:{col};font-weight:bold'>{sign}{info['pearson_r']:.3f}</td>"
                f"<td>{info['n_points']}</td>"
                f"<td>{info['direction']}</td></tr>"
            )

        s = self.system; g = s.primary_gpu; d = self.data
        gpu_str = f"{g.name} | VRAM {g.vram_used_gb:.1f}/{g.vram_total_gb:.1f} GB" if g else "No GPU"
        summary = self._db.summary()
        learner_rows = ""
        for key, stat in sorted(summary.items(),
                                key=lambda x: -abs(x[1]['mean_df1']))[:5]:
            sign = "+" if stat['mean_df1'] >= 0 else ""
            learner_rows += (
                f"<tr><td>{key}</td><td>{stat['n']}</td>"
                f"<td>{sign}{stat['mean_df1']:+.4f}</td>"
                f"<td>{stat['pos_rate']:.0%}</td></tr>"
            )

        html = f"""
<div style="font-family:monospace;border:1px solid #555;padding:12px;border-radius:6px">
<b style="font-size:15px">📊 Training Advisor Report</b>
<hr style="margin:6px 0">
<table style="width:100%;font-size:12px">
<tr><td><b>CPU</b></td><td>{s.cpu_count} cores {s.cpu_load_pct or '?'}%</td>
    <td><b>RAM</b></td><td>{s.ram_used_gb:.1f}/{s.ram_total_gb:.1f} GB</td></tr>
<tr><td><b>GPU</b></td><td colspan=3>{gpu_str}</td></tr>
<tr><td><b>Plots</b></td><td>{d.n_plots}</td>
    <td><b>Trees</b></td><td>{d.n_trees_total}</td></tr>
<tr><td><b>Trend</b></td><td>{la.trend} ({la.loss_slope:+.2f}%/ep)</td>
    <td><b>F1</b></td><td>best={la.best_f1:.4f} last={la.last_f1:.4f}</td></tr>
</table>
<hr style="margin:6px 0">
<b>💡 Советы ({len(all_adv)}):</b>
<table style="width:100%;font-size:12px;margin-top:4px">
<tr><th align=left>Level</th><th align=left>Category</th><th align=left>Advice</th></tr>
{rows if rows else "<tr><td colspan=3 style='color:#44bb44'>✅ Всё в норме.</td></tr>"}
</table>
{f'''
<hr style="margin:6px 0">
<b>📐 Корреляции (cfg → F1):</b>
<table style="width:100%;font-size:12px;margin-top:4px">
<tr><th align=left>Key</th><th>Pearson r</th><th>n</th><th>Direction</th></tr>
{corr_rows}
</table>
''' if corr_rows else ''}
{f'''
<hr style="margin:6px 0">
<b>🧠 Learner (топ-5 гиперпараметров):</b>
<table style="width:100%;font-size:12px;margin-top:4px">
<tr><th align=left>Key</th><th>n</th><th>mean ΔF1</th><th>pos%</th></tr>
{learner_rows}
</table>
''' if learner_rows else ''}
</div>"""
        display(HTML(html))
