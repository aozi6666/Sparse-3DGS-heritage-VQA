#!/usr/bin/env python
"""CPU smoke for image textseat / pack (no Mage / no EasyOCR)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ETS = Path(__file__).resolve().parents[1]
if str(_ETS) not in sys.path:
    sys.path.insert(0, str(_ETS))

from mage_query_ip.image_patch_select import (
    extract_image_tiles,
    pack_tiles_collage,
    select_image_tiles,
    tile_texture_scores,
)
from mage_query_ip.image_textseat import need_text_evidence, select_image_text_seat_kept


def main() -> int:
    assert need_text_evidence("What text is written on the sign?")
    assert not need_text_evidence("What color is the car?")

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    tiles, gh, gw, _, _ = extract_image_tiles(img, patch=32)
    assert len(tiles) == gh * gw
    tex = tile_texture_scores(tiles)
    kept, meta = select_image_text_seat_kept(
        texture_score=tex,
        budget_k=16,
        need_text=True,
        ocr_tile_indices=[0, 1, 2],
    )
    assert len(kept) == 16
    assert meta["need_text"] is True
    collage = pack_tiles_collage(tiles, kept, cell=32)
    assert collage.size[0] > 0

    r_full = select_image_tiles(img, mode="full_image")
    assert r_full.mode == "full_image"
    r_rand = select_image_tiles(img, mode="random_tiles", budget_k=16, seed=0)
    assert len(r_rand.kept) == 16
    r_seat = select_image_tiles(
        img,
        mode="queryip_image_textseat",
        budget_k=16,
        need_text=True,
        ocr_tile_indices=[0, 1],
    )
    assert len(r_seat.kept) == 16

    print("[ok] smoke_image_textseat")
    print(f"  tiles={len(tiles)} seats={meta['n_text_seats']} collage={collage.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
