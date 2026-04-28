"""3D Non-Maximum Suppression using IoUv — fully vectorised GPU implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from ops.iou3d import iou_volume

# Максимум боксов для одного матричного IoU-расчёта.
# При 22GB VRAM: 6000×6000 float32 = ~144MB — безопасно.
# Увеличь если VRAM > 40GB; уменьши до 3000 при OOM.
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
