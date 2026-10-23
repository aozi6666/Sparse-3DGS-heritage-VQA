"""Codec tile 上的区域–问题 CLIP 分（2×2 / 4×4）。

【主路径不用】§9 v1 / edge 均失败（伤底 + shuffle 反向）。停 QueryIP 语义选区。
保留供归档复现；不要再接到主推叙事。复用本地 CLIP-L。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from .clip_frame_score import ClipFrameScorer


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _montage_block(cells: list[np.ndarray], *, out_cell: int = 112) -> Image.Image:
    """Stitch a list of RGB tiles into a square-ish grid (row-major)."""
    if not cells:
        raise ValueError("empty region")
    n = len(cells)
    # Prefer 2-wide for block=2 (2×2); else near-square.
    gw = int(np.ceil(np.sqrt(n)))
    gh = int(np.ceil(n / gw))
    canvas = np.zeros((gh * out_cell, gw * out_cell, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, gw)
        im = Image.fromarray(np.asarray(cell, dtype=np.uint8), mode="RGB").resize(
            (out_cell, out_cell), Image.BILINEAR
        )
        arr = np.asarray(im, dtype=np.uint8)
        canvas[r * out_cell : (r + 1) * out_cell, c * out_cell : (c + 1) * out_cell] = arr
    return Image.fromarray(canvas, mode="RGB")


def _canvas_grid_from_positions(
    positions: np.ndarray, start: int, ppc: int
) -> tuple[int, int]:
    """Infer (gw, gh) from (t,h,w) rows for one canvas."""
    pos = positions[start : start + ppc]
    if pos.size == 0:
        return 1, 1
    gh = int(pos[:, 1].max()) + 1
    gw = int(pos[:, 2].max()) + 1
    if gh * gw < ppc:
        # Fallback near-square
        from .codec_reweight import _factor_rect

        return _factor_rect(ppc)
    return gw, gh


class RegionClipScorer(ClipFrameScorer):
    """CLIP scorer with region montage API (inherits encode_text / encode_images)."""

    def score_flat_tiles(
        self,
        tiles: Sequence[np.ndarray],
        *,
        n_canvas: int,
        ppc: int,
        positions: np.ndarray,
        question: str,
        block: int = 2,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return per-tile region–query cosine, shape (n_tiles,).

        Non-overlapping ``block×block`` regions within each canvas; leftover
        edge strips form smaller regions. Scores broadcast to member tiles.
        """
        n = len(tiles)
        if n != n_canvas * ppc:
            raise ValueError(f"tiles {n} != n_canvas*ppc {n_canvas * ppc}")
        block = max(1, int(block))
        out = np.zeros(n, dtype=np.float32)
        region_imgs: list[Image.Image] = []
        region_members: list[list[int]] = []

        for ci in range(n_canvas):
            base = ci * ppc
            gw, gh = _canvas_grid_from_positions(positions, base, ppc)
            # Map (r,c) -> flat index within canvas (row-major as extract).
            def idx(r: int, c: int) -> int:
                return base + r * gw + c

            r0 = 0
            while r0 < gh:
                r1 = min(gh, r0 + block)
                c0 = 0
                while c0 < gw:
                    c1 = min(gw, c0 + block)
                    members: list[int] = []
                    cells: list[np.ndarray] = []
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            j = idx(r, c)
                            if j - base >= ppc:
                                continue
                            members.append(j)
                            cells.append(np.asarray(tiles[j], dtype=np.uint8))
                    if members and cells:
                        region_imgs.append(_montage_block(cells))
                        region_members.append(members)
                    c0 = c1
                r0 = r1

        if not region_imgs:
            return out

        text_f = self.encode_text(question)
        sims: list[float] = []
        for i in range(0, len(region_imgs), int(batch_size)):
            batch = region_imgs[i : i + int(batch_size)]
            img_f = self.encode_images(batch)
            sim = (img_f * text_f).sum(dim=-1)
            sims.extend(sim.detach().float().cpu().tolist())

        for members, s in zip(region_members, sims):
            for j in members:
                # Max if overlapping (shouldn't with non-overlap); else assign.
                out[j] = max(float(out[j]), float(s))
        return out


def fuse_proxy_region(
    proxy: np.ndarray,
    region: np.ndarray,
    *,
    lam: float = 0.4,
) -> np.ndarray:
    """S = norm(proxy) + λ · norm(region)."""
    return _minmax(proxy) + float(lam) * _minmax(region)
