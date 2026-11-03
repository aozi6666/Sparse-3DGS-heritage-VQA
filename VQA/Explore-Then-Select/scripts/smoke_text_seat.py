#!/usr/bin/env python
"""CPU smoke for textseat (no Mage / no GPU). Agent-safe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ETS = Path(__file__).resolve().parents[1]
if str(_ETS) not in sys.path:
    sys.path.insert(0, str(_ETS))

from mage_query_ip.ocr_candidates import propose_ocr_patch_indices
from mage_query_ip.text_seat import (
    need_text_evidence,
    select_text_seat_kept,
    text_seat_count,
)


def main() -> int:
    assert need_text_evidence("What text is on the sign?")
    assert need_text_evidence("匾额上写了什么？")
    assert not need_text_evidence("Who scored the goal?")
    assert propose_ocr_patch_indices() == []

    n, b = 64, 16
    rng = np.random.default_rng(0)
    motion = rng.random(n).astype(np.float64)
    residual = rng.random(n).astype(np.float64)
    proxy = 0.85 * motion + 0.15 * residual
    m = text_seat_count(b)
    assert m > 0 and m < b

    kept_off, meta_off = select_text_seat_kept(
        proxy_score=proxy,
        motion=motion,
        residual=residual,
        budget_k=b,
        need_text=False,
    )
    assert len(kept_off) == b
    assert meta_off["need_text"] is False
    assert meta_off["n_text_seats"] == 0

    kept_on, meta_on = select_text_seat_kept(
        proxy_score=proxy,
        motion=motion,
        residual=residual,
        budget_k=b,
        need_text=True,
        ocr_patch_indices=[3, 7, 999],
    )
    assert len(kept_on) == b
    assert meta_on["need_text"] is True
    assert int(meta_on["n_text_seats"]) == int(meta_on["m_target"])
    assert int(meta_on["n_ocr_prefer"]) == 2  # 999 clipped
    assert 0 in kept_on  # always_keep_first

    print("[ok] smoke_text_seat")
    print(f"  m_target={meta_on['m_target']} n_text_seats={meta_on['n_text_seats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
