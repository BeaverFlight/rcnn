"""
advisor/advisor.py — главный класс (v2: самообучающийся)

API:
    advisor = TrainingAdvisor(cfg, data_root)
    advisor.report()                       # начальный отчёт в stdout
    advisor.report_jupyter()               # HTML в Jupyter

    # внутри train-цикла:
    advisor.push(epoch, loss_dict, metrics=None)

    # после завершения обучения:
    advisor.finalize()                     # постоянный отчёт + сохранение DB
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

logger = logging.getLogger(__name__)

_LEVEL_EMOJI  = {"critical": "🔴", "warning": "🟡", "info": "🔵", "ok": "✅", "learned": "🧠"}
_LEVEL_ORDER  = {"critical": 0, "warning": 1, "learned": 2, "info": 3, "ok": 4}


class TrainingAdvisor:
    """
    Анализирует ход обучения, систему и данные.
    Самообучается: записывает опыты в ExperienceDB и генерирует
    советы через BayesianAdvisorLearner (UCB1 по гиперпараметрам).

    Parameters
    ----------
    cfg              : OmegaConf cfg
    data_root        : путь к датасету
    db_path          : путь к JSON-базе опытов (None → <data_root>/advisor_db.json)
    window           : длина окна для trend-анализа loss
    report_interval  : каждые N эпох логировать советы (0 = никогда)
    """

    def __init__(
        self,
        cfg,
        data_root: str | Path = ".",
        db_path: Optional[str | Path] = None,
        window: int = 50,
        report_interval: int = 0,
    ) -> None:
        self._cfg       = cfg
        self._data_root = str(data_root)
        self._tracker   = LossTracker(window=window)
        self._report_interval = report_interval

        # ExperienceDB — персистентная БД опытов
        if db_path is None:
            db_path = Path(data_root) / "advisor_db.json"
        self._db      = ExperienceDB(db_path)
        self._learner = BayesianAdvisorLearner(self._db)

        self._sys:  Optional[SystemSnapshot] = None
        self._data: Optional[DatasetStats]   = None

        # Отслеживание состояния для записи опытов
        self._last_f1:      float = 0.0
        self._last_epoch:   int   = 0
        self._pending_actions: dict[str, float] = {}  # action_key → f1_before

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
    ) -> None:
        """
        Вызывается каждую эпоху.
        metrics = {'f1': 0.72, ...} — если была валидация.
        """
        self._tracker.push(epoch, loss_dict, metrics)
        self._last_epoch = epoch

        f1 = metrics.get("f1") if metrics else None
        if f1 is not None:
            # Финализируем pending опыты
            for key in list(self._pending_actions.keys()):
                self._learner.update_f1_after(key, float(f1))
            self._pending_actions.clear()
            self._last_f1 = float(f1)

        # Автоматический периодический репорт
        if (
            self._report_interval > 0
            and epoch > 0
            and epoch % self._report_interval == 0
        ):
            self.refresh_system()
            self._log_advices(epoch)

    # ------------------------------------------------------------------
    # Advice generation (rules + learned)
    # ------------------------------------------------------------------

    def advise(self) -> list[Advice]:
        la      = self._tracker.analyse()
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

        # Записываем recommended actions как pending опыты
        all_advices = rule_advices + learned_advices
        for a in all_advices:
            if a.action and a.level in ("critical", "warning"):
                la_now = self._tracker.analyse()
                self._learner.record_action(
                    epoch         = self._last_epoch,
                    action        = a.action,
                    cfg           = self._cfg,
                    f1_before     = self._last_f1,
                    loss_analysis = la_now,
                )
                for key in a.action:
                    self._pending_actions[key] = self._last_f1

        return all_advices

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
    # Finalize (вызывать после окончания обучения)
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Постоянный отчёт + Learner-аналитика.
        Вызывать в конце train_fold().
        """
        self.refresh_system()
        self.report()
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

        # Learner summary
        summary = self._db.summary()
        if summary:
            print("\n🧠  LEARNER (накопленный опыт):")
            for key, stat in sorted(summary.items(),
                                    key=lambda x: -abs(x[1]['mean_df1']))[:5]:
                sign = "+" if stat['mean_df1'] >= 0 else ""
                print(f"    {key:<40} n={stat['n']:>3}  "
                      f"ΔF1={sign}{stat['mean_df1']:+.4f}  "
                      f"pos={stat['pos_rate']:.0%}")

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
<b>🧠 Learner (топ-5 гипперпараметров):</b>
<table style="width:100%;font-size:12px;margin-top:4px">
<tr><th align=left>Key</th><th>n</th><th>mean ΔF1</th><th>pos%</th></tr>
{learner_rows}
</table>
''' if learner_rows else ''}
</div>"""
        display(HTML(html))
