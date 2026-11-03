"""Question-triggered static / text seats (non-CLIP QueryIP branch).

``need_text`` from keywords → reserve ``m≈0.10·B`` seats for low-motion +
high-residual tiles (OCR indices first when provided). Remaining seats use
proxy ``queryip`` ranking. No CLIP / cosine similarity.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .ocr_candidates import merge_ocr_indices, propose_ocr_patch_indices

# Heritage / reading cues (EN + ZH). Keep narrow to avoid over-trigger on Video-MME.
_TEXT_PATTERNS: list[str] = [
    r"\bread\b",
    r"\btext\b",
    r"\bsign\b",
    r"\blogo\b",
    r"\bword\b",
    r"\bwords\b",
    r"\bcaption\b",
    r"\bwritten\b",
    r"\binscription\b",
    r"\bbanner\b",
    r"\bscoreboard\b",
    r"\bsubtitle\b",
    r"\blabel\b",
    r"字",
    r"匾",
    r"写",
    r"文字",
    r"题记",
    r"碑文",
    r"铭文",
    r"招牌",
    r"字幕",
]


def need_text_evidence(question: str) -> bool:
    """True when the question likely needs readable / static text evidence."""
    text = (question or "").strip()
    if not text:
        return False
    return any(re.search(p, text, flags=re.I) for p in _TEXT_PATTERNS)


def text_seat_count(budget_k: int, *, sms: int = 2, seat_frac: float = 0.10) -> int:
    """``m = align(max(1, floor(seat_frac·B)))``; 0 when budget too small."""
    k = int(budget_k)
    if k <= 0:
        return 0
    step = int(sms) * int(sms)
    raw = max(1, int(k * float(seat_frac)))  # floor via int trunc for positive
    m = (raw // step) * step
    if m < step:
        m = step if k >= 2 * step else max(1, min(k // 4, k - 1)) if k > 1 else 1
    # Leave room for proxy fill.
    if k > step:
        m = min(m, k - step)
    return int(max(0, min(m, k)))


def static_text_prior(
    motion: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Higher = more plaque-like: low motion, high residual / texture."""
    m = np.asarray(motion, dtype=np.float64).reshape(-1)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if m.size != r.size:
        raise ValueError(f"motion/residual length mismatch {m.size} vs {r.size}")
    return (1.0 - m) * r


def select_text_seat_kept(
    *,
    proxy_score: np.ndarray,
    motion: np.ndarray,
    residual: np.ndarray,
    budget_k: int,
    need_text: bool,
    ocr_patch_indices: Sequence[int] | None = None,
    always_keep_first_patch: bool = True,
    sms: int = 2,
    seat_frac: float = 0.10,
) -> tuple[list[int], dict[str, int | bool | float]]:
    """Return kept indices and meta for ``queryip_textseat``."""
    score = np.asarray(proxy_score, dtype=np.float64).reshape(-1)
    total = int(score.shape[0])
    k = int(budget_k)
    if total <= 0 or k <= 0:
        raise ValueError("empty pool or budget")
    k = min(k, total)

    prior = static_text_prior(motion, residual)
    ocr = merge_ocr_indices(ocr_patch_indices, n_tiles=total)
    if not ocr:
        # Hook stays callable; still empty in this phase.
        ocr = merge_ocr_indices(propose_ocr_patch_indices(), n_tiles=total)

    m_seats = text_seat_count(k, sms=sms, seat_frac=seat_frac) if need_text else 0
    seats: list[int] = []
    if m_seats > 0:
        # Prefer OCR indices by prior, then fill from global prior ranking.
        ocr_ranked = sorted(ocr, key=lambda i: (-float(prior[i]), -float(score[i]), i))
        for i in ocr_ranked:
            if len(seats) >= m_seats:
                break
            seats.append(i)
        if len(seats) < m_seats:
            order_prior = np.argsort(-prior).tolist()
            used = set(seats)
            for i in order_prior:
                if i in used:
                    continue
                seats.append(i)
                used.add(i)
                if len(seats) >= m_seats:
                    break

    used = set(seats)
    fill_n = k - len(seats)
    order_proxy = np.argsort(-score).tolist()
    fill = [i for i in order_proxy if i not in used][:fill_n]
    kept = sorted(set(seats) | set(fill))

    if len(kept) > k:
        # Prefer seats; drop lowest-proxy among fill extras.
        seat_set = set(seats)
        extras = [i for i in kept if i not in seat_set]
        extras = sorted(extras, key=lambda i: (-float(score[i]), i))
        kept = sorted(list(seat_set) + extras[: max(0, k - len(seat_set))])
    elif len(kept) < k:
        used = set(kept)
        kept = sorted(kept + [i for i in order_proxy if i not in used][: k - len(kept)])

    if always_keep_first_patch and 0 not in kept and k >= 1:
        drop_pool = [i for i in kept if i not in seats and i != 0] or [
            i for i in kept if i != 0
        ]
        if drop_pool:
            worst = min(drop_pool, key=lambda i: (float(score[i]), i))
            kept = sorted([0] + [i for i in kept if i != worst and i != 0])
            kept = kept[:k]
            if 0 not in seats and m_seats > 0:
                # index 0 stole a seat slot visually; meta still reports m_seats.
                pass

    meta: dict[str, int | bool | float] = {
        "need_text": bool(need_text),
        "n_text_seats": int(len(seats)),
        "m_target": int(m_seats),
        "n_ocr_prefer": int(len(ocr)),
        "n_wide": int(total),
        "seat_frac": float(seat_frac),
    }
    return list(kept), meta
