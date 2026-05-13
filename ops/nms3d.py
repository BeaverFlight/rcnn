"""3D Non-Maximum Suppression using IoUv — fully vectorised GPU implementation.

Содержит:
  nms3d      — hard NMS (оригинал)
  soft_nms3d — Soft-NMS с Gaussian-decay (новый)
               вместо подавления боксов снижает их scores пропорционально IoU,
               что сохраняет перекрывающиеся деревья с разумным score.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ops.iou3d import iou_volume

# Максимум боксов для одного матричного IoU-расчёта.
# При 22GB VRAM: 6000×6000 float32 = ~144MB — безопасно.
_NMS_CHUNK = 6000


def nms3d(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    max_output: int | None = None,
    score_threshold: float = 0.0,
) -> Tensor:
    """
    Полностью векторизованный 3D NMS на GPU.

    Алгоритм:
      1. score pre-filter  — отсекаем заведомо слабые боксы до IoU-матрицы
      2. сортировка по score (descending)
      3. IoU-матрица считается чанками (_NMS_CHUNK × _NMS_CHUNK), чтобы
         не аллоцировать N×N тензор целиком (был бы 152k×152k = 88GB)
      4. bitset-suppress через cumulative OR по маске подавления

    Args:
        boxes:            (N, 6) [x, y, z_c, w, l, h]
        scores:           (N,)
        iou_threshold:    подавлять бокс если IoUv > порога с лучшим
        max_output:       максимум боксов на выходе
        score_threshold:  отбросить боксы со score < порога ДО NMS
                          (не влияет на качество при разумном значении)
    Returns:
        keep: (K,) индексы в исходном (несортированном) массиве
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # --- 1. score pre-filter ---
    if score_threshold > 0.0:
        valid = scores >= score_threshold
        if valid.sum() == 0:
            # если ни один не прошёл — берём топ-1 чтобы не вернуть пустой
            valid[scores.argmax()] = True
        orig_idx = torch.where(valid)[0]
        boxes  = boxes[orig_idx]
        scores = scores[orig_idx]
    else:
        orig_idx = torch.arange(len(boxes), device=boxes.device)

    N = len(boxes)
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # --- 2. сортировка ---
    order   = scores.argsort(descending=True)      # (N,)
    boxes_s = boxes[order]                         # (N, 6) — отсортировано

    # --- 3. чанковая IoU-матрица + битовая маска подавления ---
    # suppressed[i] = True  →  бокс i уже подавлён
    suppressed = torch.zeros(N, dtype=torch.bool, device=boxes.device)

    C = _NMS_CHUNK
    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        # Рассматриваем «запросные» боксы [i_start, i_end)
        # Они подавляют всех с индексом > i_end-1 у которых IoU высок
        alive_i = ~suppressed[i_start:i_end]       # (C_i,)
        if not alive_i.any():
            continue

        # IoU между живыми боксами блока и всеми боксами правее
        j_start = i_end  # подавляем только тех что ещё не рассмотрены
        if j_start >= N:
            break

        for j_start_c in range(j_start, N, C):
            j_end_c = min(j_start_c + C, N)

            alive_j = ~suppressed[j_start_c:j_end_c]   # (C_j,)
            if not alive_j.any():
                continue

            # IoU только для живых боксов — экономим память
            qi = torch.where(alive_i)[0] + i_start     # абс. индексы
            qj = torch.where(alive_j)[0] + j_start_c

            iou = iou_volume(boxes_s[qi], boxes_s[qj])  # (|qi|, |qj|)

            # для каждого j: подавить если IoU с любым живым i > порога
            suppress_j_local = (iou > iou_threshold).any(dim=0)  # (|qj|,)
            suppressed[qj[suppress_j_local]] = True

    keep_sorted = torch.where(~suppressed)[0]          # индексы в sorted
    if max_output is not None:
        keep_sorted = keep_sorted[:max_output]

    # маппинг: sorted → pre-filter → original
    return orig_idx[order[keep_sorted]]


def soft_nms3d(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    max_output: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Soft-NMS с Gaussian-decay для 3D боксов (IoUv).

    Вместо жёсткого подавления боксов снижает score конкурирующих боксов
    пропорционально их IoU с выбранным кандидатом:
        score_j ← score_j * exp(-IoU(i,j)² / sigma)

    Это позволяет сохранить перекрывающиеся деревья (типичная ситуация при
    густом лесе), не выбрасывая их полностью, а лишь понижая уверенность.

    Алгоритм O(N²) итеративный (greedy, как оригинальный Soft-NMS):
      1. Сортируем по score
      2. Берём максимальный бокс → добавляем в keep
      3. Для всех оставшихся пересчитываем score через Gaussian
      4. Повторяем пока score_max >= score_threshold

    Args:
        boxes:            (N, 6) [x, y, z_c, w, l, h]
        scores:           (N,) — будут изменены in-place (копия внутри)
        iou_threshold:    NMS не применяется напрямую, но используется
                          как reference для sigma-масштабирования
        sigma:            ширина Gaussian-окна (default 0.5)
        score_threshold:  порог score для остановки (default 0.001)
        max_output:       максимум боксов на выходе

    Returns:
        keep:        (K,) индексы в исходном массиве
        new_scores:  (K,) обновлённые scores после decay
    """
    if boxes.numel() == 0:
        device = boxes.device
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float, device=device),
        )

    device = boxes.device
    N = len(boxes)
    scores_work = scores.clone().float()

    # Маппинг: позиция → исходный индекс
    indices = torch.arange(N, device=device)

    keep_idx: list[int] = []
    keep_sc:  list[float] = []

    for _ in range(N):
        if scores_work.numel() == 0:
            break
        # Выбираем лучший из оставшихся
        best_pos = int(scores_work.argmax().item())
        best_score = float(scores_work[best_pos].item())
        if best_score < score_threshold:
            break

        best_orig = int(indices[best_pos].item())
        keep_idx.append(best_orig)
        keep_sc.append(best_score)

        if max_output is not None and len(keep_idx) >= max_output:
            break

        # Убираем выбранный бокс из рабочего набора
        best_box = boxes[best_orig].unsqueeze(0)  # (1, 6)

        # Все оставшиеся (кроме best_pos)
        rest_mask = torch.ones(len(scores_work), dtype=torch.bool, device=device)
        rest_mask[best_pos] = False

        if not rest_mask.any():
            break

        rest_indices = indices[rest_mask]
        rest_scores  = scores_work[rest_mask]
        rest_boxes   = boxes[rest_indices]  # (M, 6)

        # IoU между best и всеми оставшимися
        iou = iou_volume(best_box, rest_boxes).squeeze(0)  # (M,)

        # Gaussian decay: score_j *= exp(-iou² / sigma)
        decay = torch.exp(-(iou ** 2) / sigma)
        rest_scores = rest_scores * decay

        # Обновляем рабочие массивы
        indices     = rest_indices
        scores_work = rest_scores

    if not keep_idx:
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float, device=device),
        )

    keep_t    = torch.tensor(keep_idx, dtype=torch.long,  device=device)
    keep_sc_t = torch.tensor(keep_sc,  dtype=torch.float, device=device)
    return keep_t, keep_sc_t
