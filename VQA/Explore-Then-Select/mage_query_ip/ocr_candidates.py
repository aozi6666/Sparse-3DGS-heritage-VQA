"""OCR patch-candidate hook.

Video path: default empty (省盘).
Image OCRBench path: use ``image_ocr.ImageOCRIndexer`` via ``eval_ocrbench_v2_samek --use-ocr``.
This module stays importable without EasyOCR installed.
"""

from __future__ import annotations

from typing import Any, Sequence


def propose_ocr_patch_indices(
    *args: Any,
    **kwargs: Any,
) -> list[int]:
    """Return flat tile indices that should prefer text-seat slots.

    Default no-op for video QueryIP. Image eval passes indices explicitly.
    """
    _ = args, kwargs
    return []


def merge_ocr_indices(
    preferred: Sequence[int] | None,
    *,
    n_tiles: int,
) -> list[int]:
    """Dedupe / clip an optional OCR index list into ``[0, n_tiles)``."""
    if not preferred:
        return []
    out: list[int] = []
    seen: set[int] = set()
    n = int(n_tiles)
    for x in preferred:
        i = int(x)
        if i < 0 or i >= n or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out
