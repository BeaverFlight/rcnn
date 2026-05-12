# TreeRCNN v2.0 — Архитектурный план

## Общий пайплайн

```
LAS файл
    ↓
[Тайлинг] → Тайл 40×40м, 32k точек, локальный DEM
    ↓
[Богатый вектор] → (x, y, z, nx, ny, nz, verticality, z_norm, intensity) = 9D
    ↓
[Backbone: Deep PointNet++ + SA-extra] → SA1→SA2→SA3(extra)→SA4(global)
    ↓
[FPN] → p2 (подлесок), p3 (средний масштаб), p4 (крупные деревья)
    ↓
[Stage 1: RPN с FPN] → ~500 Proposals
    ↓
[Stage 2: Refinement] → Offset Head + Center-ness + FC-neck → Точные боксы
    ↓
[Stage 3: Relation Head (Transformer)] → Умная фильтрация с учётом соседей
    ↓
Финальный NMS (страховочный, IoU > 0.8)
    ↓
Детектированные деревья
```

## Новые модули

| Файл | Назначение |
|------|------------|
| `utils/tiling.py` | Нарезка LAS на тайлы (TrainingTiler + InferenceTiler) |
| `utils/rich_features.py` | Предрасчёт нормалей, verticality, z_norm (9D вектор) |
| `models/fpn.py` | Feature Pyramid Network поверх SA-иерархии |
| `models/stage2_head.py` | Stage 2 с Offset Head + Center-ness + FC-neck |
| `models/relation_head.py` | Stage 3: Transformer Relation Head (Learnable NMS) |
| `models/losses.py` | offset_loss, centerness_loss, relation_loss, stage2_loss_v2 |

## Backbone: Deep PointNet++ с SA-extra

```
SA1: npoint=2048, radius=0.5м  → features 128d  (ветви, локальная текстура)
SA2: npoint=512,  radius=1.5м  → features 256d  (части кроны)
SA3: npoint=128,  radius=3.0м  → features 512d  (целые деревья) ← SA-extra
SA4: npoint=None, radius=None  → features 1024d (глобальный контекст тайла)
```

SA3 (`SA-extra`) — новый промежуточный слой с радиусом 3м.
Позволяет сети строить иерархическое представление до глобального пулинга.

## Stage 2: Voting-механизм

Ключевая идея: точки внутри proposal «голосуют» за центр ствола.

1. Point-wise признаки из SA-слоёв Stage 2
2. **Offset Head**: каждая точка предсказывает `(dx, dy, dz)` до центра ствола
3. Смещение точек: `shifted_xyz = xyz + offsets`
4. **Center-ness Head**: каждая точка получает вес уверенности
5. **Взвешенный пулинг**: `global_feat = Σ(centerness × shifted_features)`
6. **FC-neck**: `feat_dim → feat_dim*2 → feat_dim` с LayerNorm + GELU
7. **Финальные головы**: `cls_head` (BCE) и `reg_head` (6D бокс)

## Stage 3: Relation Head

- Принимает до 500 кандидатов из Stage 2
- Позиционное кодирование: `(cx, cy, cz, w, h) → feat_dim`
- TransformerEncoder (2 слоя, 8 голов, Pre-LN)
- Каждое дерево «смотрит» на соседей → подавляет дубликаты
- Финальный score заменяет жёсткий NMS

## Loss-функции

| Loss | Тип | λ (начальный) |
|------|-----|---------------|
| Stage 1 cls | Focal | 1.0 |
| Stage 1 reg | Smooth L1 | 1.0 |
| Stage 2 cls | BCE | 1.0 |
| Stage 2 reg | Smooth L1 | 1.0 |
| Stage 2 offset | Smooth L1 | 0.5 |
| Stage 2 centerness | BCE | 0.5 |
| Stage 3 relation | BCE | 0.5 |

## Порядок размораживания при обучении

- Эпохи 1–20: только Stage 1 (backbone + RPN)
- Эпохи 21–50: Stage 1 + Stage 2 (offset/centerness λ растёт с 0.1 до целевого)
- Эпохи 51+: все три стейджа, Relation Head с LR = 1e-5

## Roadmap

1. **v1 (текущая)**: Доучить до 300 эпох → зафиксировать baseline F1
2. **v1.5**: Тайлинг + Богатый вектор → `batch_size=4`, стабильные градиенты
3. **v2.0**: FPN + SA-extra + Stage2Head v2 → обучить с нуля
4. **v2.1**: Relation Head (Stage 3) → добавить поверх v2.0
