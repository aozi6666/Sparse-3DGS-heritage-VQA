"""Image text seats: OCR tile indices + texture fill (non-CLIP)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .ocr_candidates import merge_ocr_indices
from .text_seat import need_text_evidence, text_seat_count


def select_image_text_seat_kept(
    *,
    texture_score: np.ndarray,
    budget_k: int,
    need_text: bool,
    ocr_tile_indices: Sequence[int] | None = None,
    seat_frac: float = 0.10,
    sms: int = 2,
) -> tuple[list[int], dict[str, int | bool | float]]:
    """Reserve OCR/text seats when ``need_text``, fill rest by texture."""
    score = np.asarray(texture_score, dtype=np.float64).reshape(-1)
    total = int(score.shape[0])
    k = min(int(budget_k), total)
    if total <= 0 or k <= 0:
        raise ValueError("empty pool or budget")

    ocr = merge_ocr_indices(ocr_tile_indices, n_tiles=total)
    m_seats = text_seat_count(k, sms=sms, seat_frac=seat_frac) if need_text else 0

    seats: list[int] = []
    if m_seats > 0:
        ocr_ranked = sorted(ocr, key=lambda i: (-float(score[i]), i))
        for i in ocr_ranked:
            if len(seats) >= m_seats:
                break
            seats.append(i)
        if len(seats) < m_seats:
            used = set(seats)
            for i in np.argsort(-score).tolist():
                if i in used:
                    continue
                seats.append(i)
                used.add(i)
                if len(seats) >= m_seats:
                    break

    used = set(seats)
    fill_n = k - len(seats)
    order = np.argsort(-score).tolist()
    fill = [i for i in order if i not in used][:fill_n]
    kept = sorted(set(seats) | set(fill))
    if len(kept) > k:
        seat_set = set(seats)
        extras = sorted(
            [i for i in kept if i not in seat_set],
            key=lambda i: (-float(score[i]), i),
        )
        kept = sorted(list(seat_set) + extras[: max(0, k - len(seat_set))])
    elif len(kept) < k:
        used = set(kept)
        kept = sorted(kept + [i for i in order if i not in used][: k - len(kept)])

    meta: dict[str, int | bool | float] = {
        "need_text": bool(need_text),
        "n_text_seats": int(len(seats)),
        "m_target": int(m_seats),
        "n_ocr_prefer": int(len(ocr)),
        "n_wide": int(total),
        "seat_frac": float(seat_frac),
    }
    return list(kept), meta


# Re-export for eval convenience
__all__ = [
    "need_text_evidence",
    "select_image_text_seat_kept",
    "text_seat_count",
]
