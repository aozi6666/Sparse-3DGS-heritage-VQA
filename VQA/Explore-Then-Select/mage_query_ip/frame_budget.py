"""按帧相关分 q_t 分配固定总 patch 预算。

【主路径不用】仅服务失败的 queryip_clip（E4）。proxy queryip 不走本模块。
"""

from __future__ import annotations

import numpy as np


def allocate_budgets(
    q_t: np.ndarray | list[float],
    total_k: int,
    *,
    min_per_frame: int = 1,
    temperature: float = 0.07,
) -> np.ndarray:
    """Softmax-weighted integer budgets; ``sum(budgets) == total_k``.

    Parameters
    ----------
    q_t:
        Per-frame relevance (higher = more budget). Length = n_frames.
    total_k:
        Global patch budget (already merge-aligned by caller if needed).
    min_per_frame:
        Floor per frame for temporal coverage (reduced if ``min * n > total_k``).
    temperature:
        Softmax temperature; smaller → peakier allocation.
    """
    q = np.asarray(q_t, dtype=np.float64).reshape(-1)
    n = int(q.size)
    if n <= 0:
        raise ValueError("q_t must be non-empty")
    total_k = int(total_k)
    if total_k < 1:
        raise ValueError(f"total_k must be >= 1, got {total_k}")

    base = int(max(0, min_per_frame))
    if base * n > total_k:
        base = total_k // n
    remaining = total_k - base * n

    # Softmax on finite scores
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = q - float(q.max())
    temp = float(max(temperature, 1e-6))
    w = np.exp(q / temp)
    w_sum = float(w.sum())
    if w_sum < 1e-12 or not np.isfinite(w_sum):
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = w / w_sum

    raw = w * float(remaining)
    floors = np.floor(raw).astype(np.int64)
    budgets = floors + base
    leftover = int(remaining - int(floors.sum()))
    if leftover > 0:
        frac = raw - floors.astype(np.float64)
        order = list(np.argsort(-frac))
        for i in range(leftover):
            budgets[order[i % n]] += 1
    elif leftover < 0:
        # Should not happen; peel from largest
        order = list(np.argsort(-budgets))
        need = -leftover
        for i in order:
            if need <= 0:
                break
            take = min(need, max(0, int(budgets[i]) - base))
            budgets[i] -= take
            need -= take

    budgets = budgets.astype(np.int64)
    if int(budgets.sum()) != total_k:
        # Final repair: adjust largest frame
        diff = total_k - int(budgets.sum())
        j = int(np.argmax(budgets))
        budgets[j] = max(0, int(budgets[j]) + diff)
    if int(budgets.sum()) != total_k:
        raise RuntimeError(
            f"budget alloc failed: sum={int(budgets.sum())} total_k={total_k}"
        )
    return budgets
