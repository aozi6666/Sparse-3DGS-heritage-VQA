"""EasyOCR → image tile indices (optional; 省盘时可关). Does not touch video paths."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from PIL import Image

from .image_patch_select import extract_image_tiles

logger = logging.getLogger(__name__)


def _bbox_to_xyxy(bbox: Sequence) -> tuple[int, int, int, int]:
    pts = np.asarray(bbox, dtype=np.float32).reshape(-1, 2)
    return (
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
    )


class ImageOCRIndexer:
    """Map EasyOCR boxes to flat tile indices overlapping each box."""

    def __init__(self, langs: list[str] | None = None, gpu: bool | None = None):
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "EasyOCR required for --use-ocr. Install easyocr or run without --use-ocr."
            ) from e
        import torch

        use_gpu = bool(torch.cuda.is_available()) if gpu is None else bool(gpu)
        self.reader = easyocr.Reader(langs or ["en", "ch_sim"], gpu=use_gpu)
        logger.info("EasyOCR ready gpu=%s langs=%s", use_gpu, langs or ["en", "ch_sim"])

    def propose_tile_indices(
        self,
        image: Image.Image,
        *,
        patch: int = 32,
        min_conf: float = 0.3,
    ) -> list[int]:
        tiles, gh, gw, ph, pw = extract_image_tiles(image, patch=patch)
        _ = tiles
        arr = np.asarray(image.convert("RGB"))
        try:
            results = self.reader.readtext(arr)
        except Exception as e:
            logger.warning("EasyOCR failed: %s", e)
            return []

        hit: set[int] = set()
        for bbox, _text, conf in results:
            if float(conf) < float(min_conf):
                continue
            x0, y0, x1, y1 = _bbox_to_xyxy(bbox)
            r0 = max(0, y0 // ph)
            r1 = min(gh - 1, max(0, (y1 - 1) // ph))
            c0 = max(0, x0 // pw)
            c1 = min(gw - 1, max(0, (x1 - 1) // pw))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    hit.add(r * gw + c)
        return sorted(hit)
