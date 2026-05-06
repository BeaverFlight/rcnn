"""
advisor/rules.py — правила для генерации советов

Каждое правило — чистая функция (system, data, loss_analysis, cfg) -> list[Advice]

Advice содержит:
  level    — 'critical' | 'warning' | 'info' | 'ok'
  category — 'system' | 'data' | 'training' | 'config'
  text     — человечески читаемый совет
  action   — конкретное действие (cfg-ключ: новое значение)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from advisor.system_probe import SystemSnapshot
from advisor.data_probe   import DatasetStats
from advisor.loss_tracker import LossAnalysis


@dataclass
class Advice:
    level:    str                          # critical | warning | info | ok
    category: str                          # system | data | training | config
    text:     str
    action:   Optional[dict] = None        # {cfg_key: new_value}


# ---------------------------------------------------------------------------
# System rules
# ---------------------------------------------------------------------------

def _rules_system(sys: SystemSnapshot, cfg) -> list[Advice]:
    out: list[Advice] = []

    gpu = sys.primary_gpu
    if not sys.cuda_available:
        out.append(Advice(
            level="warning", category="system",
            text="CUDA недоступна. Обучение на CPU будет в 10–20× медленнее.",
        ))
    elif gpu is not None:
        vram_used_pct = gpu.vram_used_gb / gpu.vram_total_gb * 100
        if vram_used_pct > 85:
            chunk = int(cfg.training.get("stage2_forward_chunk", 256))
            out.append(Advice(
                level="critical", category="system",
                text=f"VRAM загружена на {vram_used_pct:.0f}% ({gpu.vram_used_gb:.1f}/{gpu.vram_total_gb:.1f} GB). "
                     f"Риск OOM.",
                action={"training.stage2_forward_chunk": max(64, chunk // 2)},
            ))
        elif vram_used_pct < 40 and gpu.vram_total_gb > 10:
            chunk = int(cfg.training.get("stage2_forward_chunk", 256))
            out.append(Advice(
                level="info", category="system",
                text=f"VRAM загружена только на {vram_used_pct:.0f}%. "
                     f"Можно увеличить stage2_forward_chunk для ускорения.",
                action={"training.stage2_forward_chunk": min(1024, chunk * 2)},
            ))
        if gpu.temp_c is not None and gpu.temp_c > 85:
            out.append(Advice(
                level="warning", category="system",
                text=f"GPU перегревается: {gpu.temp_c:.0f}°C. Проверьте охлаждение.",
            ))

    if sys.ram_free_gb < 4.0 and sys.ram_total_gb > 0:
        out.append(Advice(
            level="warning", category="system",
            text=f"Свободной RAM осталось {sys.ram_free_gb:.1f} GB. "
                 "Уменьшите num_workers или max_points.",
            action={"training.num_workers": 0},
        ))

    if sys.disk_free_gb < 5.0 and sys.disk_free_gb > 0:
        out.append(Advice(
            level="critical", category="system",
            text=f"На диске осталось {sys.disk_free_gb:.1f} GB. Чекпоинты могут перестать сохраняться.",
        ))

    return out


# ---------------------------------------------------------------------------
# Data rules
# ---------------------------------------------------------------------------

def _rules_data(data: DatasetStats, cfg) -> list[Advice]:
    out: list[Advice] = []

    if data.n_plots == 0:
        out.append(Advice(level="critical", category="data",
                          text="Данные не найдены в data_root."))
        return out

    trees_pp = data.n_trees_total / max(data.n_plots, 1)
    if trees_pp < 5:
        out.append(Advice(
            level="warning", category="data",
            text=f"Мало деревьев на плот (среднее {trees_pp:.1f}). "
                 "Возможно, недостаточно аннотаций или плоты слишком малы.",
        ))

    if data.pts_per_tree_mean < 100:
        out.append(Advice(
            level="warning", category="data",
            text=f"Низкая плотность: ~{data.pts_per_tree_mean:.0f} точек/дерево. "
                 "Рекомендуется ≥200 для нормальной работы SA-слоёв.",
        ))

    if data.sparse_plots:
        out.append(Advice(
            level="warning", category="data",
            text=f"Редкие плоты (<50 точек/дерево): {data.sparse_plots}. "
                 "Исключите из обучения или уменьшите MIN_PTS.",
        ))

    if data.height_std / max(data.height_mean, 1) > 1.0:
        out.append(Advice(
            level="info", category="data",
            text=f"Большой разброс высот деревьев "
                 f"(σ={data.height_std:.1f}m, μ={data.height_mean:.1f}m). "
                 "Добавьте больше уровней высот в anchors.height_levels.",
        ))

    return out


# ---------------------------------------------------------------------------
# Training rules
# ---------------------------------------------------------------------------

def _rules_training(la: LossAnalysis, cfg) -> list[Advice]:
    out: list[Advice] = []

    if la.trend == "too_early":
        out.append(Advice(level="info", category="training",
                          text="Слишко мало данных для анализа (нужно ≥ 5 эпох)."))
        return out

    if la.trend == "diverging":
        lr = float(cfg.training.get("learning_rate", 1e-3))
        out.append(Advice(
            level="critical", category="training",
            text=f"Обучение расходится (NaN-rate={la.nan_rate:.0%}, slope={la.loss_slope:+.2f}%/эп). "
                 f"Уменьшите learning_rate.",
            action={"training.learning_rate": round(lr / 5, 7)},
        ))

    if la.trend == "plateau" and la.epochs_no_improve > 200:
        lr = float(cfg.training.get("learning_rate", 1e-3))
        out.append(Advice(
            level="warning", category="training",
            text=f"Плато F1 ({la.epochs_no_improve} эпох без улучшения, best={la.best_f1:.3f}). "
                 "Попробуйте снизить LR или усилить аугментацию.",
            action={"training.learning_rate": round(lr * 0.3, 7)},
        ))

    if la.trend == "noisy":
        mn = float(cfg.training.get("max_grad_norm", 1.0))
        out.append(Advice(
            level="warning", category="training",
            text=f"Шумный loss (CV={la.loss_cv:.2f}). "
                 "Попробуйте уменьшить max_grad_norm или LR.",
            action={"training.max_grad_norm": round(mn * 0.5, 2)},
        ))

    if la.trend == "improving" and la.loss_slope < -5.0:
        out.append(Advice(
            level="ok", category="training",
            text=f"Хорошая сходимость (slope={la.loss_slope:+.2f}%/эп). Обучение идёт хорошо.",
        ))

    # Dominance компоненты
    comps = la.loss_components
    if comps:
        total = sum(comps.values())
        for k, v in comps.items():
            frac = v / total if total > 0 else 0
            if frac > 0.8:
                out.append(Advice(
                    level="warning", category="training",
                    text=f"Компонента {k} доминирует в loss ({frac:.0%}). "
                         "Разбаланс может мешать обучению других голов.",
                ))

    return out


# ---------------------------------------------------------------------------
# Config rules
# ---------------------------------------------------------------------------

def _rules_config(sys: SystemSnapshot, data: DatasetStats, cfg) -> list[Advice]:
    out: list[Advice] = []
    gpu = sys.primary_gpu

    # max_points vs RAM
    max_pts = int(cfg.training.get("max_points", 40_000))
    if gpu and max_pts > 100_000 and gpu.vram_total_gb < 8:
        out.append(Advice(
            level="warning", category="config",
            text=f"max_points={max_pts} может вызвать OOM на GPU с {gpu.vram_total_gb:.0f} GB VRAM.",
            action={"training.max_points": 40_000},
        ))

    # AMP не включен
    if sys.cuda_available and not bool(cfg.training.get("amp", True)):
        out.append(Advice(
            level="info", category="config",
            text="AMP отключён. Включите (amp: true) для ускорения ~1.5×.",
            action={"training.amp": True},
        ))

    # val_interval слишком редко
    val_int = int(cfg.training.get("val_interval", 100))
    epochs  = int(cfg.training.get("epochs", 5000))
    if val_int > epochs // 5:
        out.append(Advice(
            level="info", category="config",
            text=f"val_interval={val_int} при {epochs} эпохах — валидация будет редко. "
                 "Рекомендуется {epochs // 10}.",
            action={"training.val_interval": epochs // 10},
        ))

    # freeze_stage3 слишком мало (только для v2)
    version = str(getattr(cfg, "model_version", "v1")).lower()
    if version == "v2":
        freeze_s3 = int(cfg.training.get("freeze_stage3_epochs", 50))
        if freeze_s3 < 30:
            out.append(Advice(
                level="warning", category="config",
                text=f"freeze_stage3_epochs={freeze_s3} мало. "
                     "Рекомендуется ≥ 50 эпох: сначала Stage-2 должен сходиться.",
                action={"training.freeze_stage3_epochs": 50},
            ))

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_advices(
    sys_snap:     SystemSnapshot,
    data_stats:   DatasetStats,
    loss_analysis: LossAnalysis,
    cfg,
) -> list[Advice]:
    advices: list[Advice] = []
    advices += _rules_system(sys_snap, cfg)
    advices += _rules_data(data_stats, cfg)
    advices += _rules_training(loss_analysis, cfg)
    advices += _rules_config(sys_snap, data_stats, cfg)
    return advices
