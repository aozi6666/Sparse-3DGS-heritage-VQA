"""Image grid tiles + pack collage (OCRBench / image QueryIP). No CLIP. No video edits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


def _align_budget(k: int, total: int, step: int = 4) -> int:
    k = int(max(1, min(int(k), int(total))))
    k = (k // step) * step
    if k < step:
        k = step if total >= step else total
    return int(min(k, total))


def extract_image_tiles(
    image: Image.Image,
    *,
    patch: int = 32,
) -> tuple[list[np.ndarray], int, int, int, int]:
    """Row-major tiles. Returns (tiles, gh, gw, ph, pw)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    ps = max(8, int(patch))
    gh = max(1, h // ps)
    gw = max(1, w // ps)
    ph = max(1, h // gh)
    pw = max(1, w // gw)
    tiles: list[np.ndarray] = []
    for r in range(gh):
        for c in range(gw):
            tile = rgb[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            if tile.size == 0:
                tile = np.zeros((ph, pw, 3), dtype=np.uint8)
            tiles.append(np.ascontiguousarray(tile))
    return tiles, gh, gw, ph, pw


def tile_texture_scores(tiles: list[np.ndarray]) -> np.ndarray:
    """Higher = more textured (residual / variance proxy)."""
    out = np.zeros(len(tiles), dtype=np.float64)
    for i, t in enumerate(tiles):
        g = t.astype(np.float32).mean(axis=2)
        out[i] = float(g.var() + 1e-6)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-8:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo)


def pack_tiles_collage(
    tiles: list[np.ndarray],
    kept: list[int],
    *,
    cell: int = 32,
) -> Image.Image:
    """Pack kept tiles into a near-square RGB collage (single Mage image)."""
    if not kept:
        raise ValueError("no tiles to pack")
    n = len(kept)
    gw = int(np.ceil(np.sqrt(n)))
    gh = int(np.ceil(n / gw))
    cs = int(cell)
    canvas = np.zeros((gh * cs, gw * cs, 3), dtype=np.uint8)
    for i, idx in enumerate(kept):
        r, c = divmod(i, gw)
        im = Image.fromarray(tiles[idx], mode="RGB").resize((cs, cs), Image.BILINEAR)
        canvas[r * cs : (r + 1) * cs, c * cs : (c + 1) * cs] = np.asarray(im, dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


@dataclass
class ImageSelectResult:
    image: Image.Image
    kept: list[int]
    budget_k: int
    n_tiles: int
    mode: str
    knobs: dict


def select_image_tiles(
    image: Image.Image,
    *,
    mode: str,
    keep_ratio: float = 0.75,
    budget_k: int | None = None,
    seed: int = 0,
    patch: int = 32,
    cell: int = 32,
    ocr_tile_indices: list[int] | None = None,
    need_text: bool = True,
    seat_frac: float = 0.10,
) -> ImageSelectResult:
    """Select tiles and pack. ``full_image`` returns the original image."""
    if mode == "full_image":
        return ImageSelectResult(
            image=image.convert("RGB"),
            kept=[],
            budget_k=0,
            n_tiles=0,
            mode=mode,
            knobs={"score_source": "full_image"},
        )

    tiles, gh, gw, _, _ = extract_image_tiles(image, patch=patch)
    n = len(tiles)
    texture = tile_texture_scores(tiles)
    if budget_k is None:
        budget_k = max(1, int(round(n * float(keep_ratio))))
    budget_k = _align_budget(int(budget_k), n, step=4)

    from .image_textseat import select_image_text_seat_kept

    if mode == "random_tiles":
        rng = np.random.default_rng(int(seed))
        kept = sorted(rng.choice(n, size=budget_k, replace=False).tolist())
        knobs = {"score_source": "random_tiles", "n_tiles": n, "gh": gh, "gw": gw}
    elif mode in ("queryip_image_textseat", "queryip_image_textseat_shuf"):
        kept, meta = select_image_text_seat_kept(
            texture_score=texture,
            budget_k=budget_k,
            need_text=bool(need_text),
            ocr_tile_indices=ocr_tile_indices,
            seat_frac=float(seat_frac),
        )
        knobs = {"score_source": "image_textseat", "gh": gh, "gw": gw, **meta}
    elif mode == "queryip_image_texture":
        # Texture-only top-K (no OCR seats); useful ablation.
        order = np.argsort(-texture).tolist()
        kept = sorted(order[:budget_k])
        knobs = {"score_source": "image_texture", "n_tiles": n, "gh": gh, "gw": gw}
    else:
        raise ValueError(f"unknown image select mode: {mode}")

    collage = pack_tiles_collage(tiles, kept, cell=cell)
    return ImageSelectResult(
        image=collage,
        kept=list(kept),
        budget_k=budget_k,
        n_tiles=n,
        mode=mode,
        knobs=knobs,
    )
