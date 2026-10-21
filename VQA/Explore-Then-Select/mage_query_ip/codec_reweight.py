"""Query reweight on Mage codec outputs (proxy path + shared packers).

``queryip`` scores patches on the decoded canvas (frame-diff / variance).
``queryip_bitcost`` lives in ``mvres_score`` (official block bitcost) and is
wired through ``MageQueryIPRunner.run_select_mode``. Both share the same
top-K + dense pack so Mage's image_grid_thw ↔ src_positions contract stays valid.

Also keeps the older canvas-level helper for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from .question_type import QuestionType, weights_for

SelectMode = Literal["queryip", "queryip_bitcost", "prefix", "random"]


@dataclass
class ReweightResult:
    images: list[Image.Image]
    src_positions: np.ndarray
    qtype: QuestionType
    alpha: float
    beta: float
    budget_k: int
    n_before: int
    n_after: int
    kept_canvas_indices: list[int]
    canvas_motion: list[float]
    canvas_residual: list[float]
    canvas_score: list[float]
    mean_motion_kept: float
    mean_motion_all: float
    knobs: dict[str, Any] = field(default_factory=dict)
    # Route B extras
    mode: str = "canvas"
    kept_patch_indices: list[int] = field(default_factory=list)


def codec_knobs_for(qtype: QuestionType, base: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(base or {})
    if qtype == QuestionType.STATIC:
        cfg.setdefault("note", "static: prefer low-motion patches in post-select")
    else:
        cfg.setdefault("note", "dynamic: prefer high-motion patches in post-select")
    return cfg


def _to_gray(im: Image.Image, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    g = im.convert("L").resize(size, Image.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_canvases(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """Return (motion, residual_proxy) per canvas in [0, 1]-ish range."""
    if not images:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    grays = [_to_gray(im) for im in images]
    motion = np.zeros(len(grays), dtype=np.float32)
    residual = np.zeros(len(grays), dtype=np.float32)
    for i, g in enumerate(grays):
        residual[i] = float(g.var() + 1e-6)
        if i == 0:
            motion[i] = 0.0
        else:
            motion[i] = float(np.mean(np.abs(g - grays[i - 1])) + 1e-6)
    return _norm(motion), _norm(residual)


def reweight_codec_canvases(
    images: list[Image.Image],
    src_positions: np.ndarray,
    qtype: QuestionType,
    budget_k: int | None = None,
    keep_ratio: float = 0.75,
    always_keep_first: bool = True,
) -> ReweightResult:
    """Legacy canvas-level select (route B archive / old smoke)."""
    n_canvas = len(images)
    if n_canvas == 0:
        raise ValueError("no canvases to reweight")
    total = int(src_positions.shape[0])
    if total % n_canvas != 0:
        raise ValueError(
            f"src_positions length {total} not divisible by canvas count {n_canvas}"
        )
    ppc = total // n_canvas
    motion, residual = score_canvases(images)
    w = weights_for(qtype)
    alpha, beta = float(w["alpha"]), float(w["beta"])
    if qtype == QuestionType.STATIC:
        score = alpha * (1.0 - motion) + beta * residual
    else:
        score = alpha * motion + beta * residual

    if budget_k is None:
        budget_k = max(ppc, int(round(total * float(keep_ratio))))
    budget_k = int(max(ppc, min(budget_k, total)))
    n_keep = max(1, min(n_canvas, (budget_k + ppc - 1) // ppc))

    order = list(np.argsort(-score))
    if always_keep_first and 0 not in order[:n_keep]:
        order = [0] + [i for i in order if i != 0]

    kept = sorted(order[:n_keep])
    while len(kept) * ppc > budget_k and len(kept) > 1:
        drop_candidates = [i for i in kept if i != 0] or kept[1:]
        drop_i = min(drop_candidates, key=lambda i: score[i])
        kept.remove(drop_i)

    pos = src_positions.reshape(n_canvas, ppc, 3)
    kept_images = [images[i] for i in kept]
    kept_pos = pos[np.array(kept, dtype=np.int64)].reshape(-1, 3)

    mean_all = float(motion.mean()) if len(motion) else 0.0
    mean_kept = float(motion[np.array(kept)].mean()) if kept else 0.0

    return ReweightResult(
        images=kept_images,
        src_positions=kept_pos,
        qtype=qtype,
        alpha=alpha,
        beta=beta,
        budget_k=budget_k,
        n_before=total,
        n_after=int(kept_pos.shape[0]),
        kept_canvas_indices=list(kept),
        canvas_motion=motion.tolist(),
        canvas_residual=residual.tolist(),
        canvas_score=score.tolist(),
        mean_motion_kept=mean_kept,
        mean_motion_all=mean_all,
        knobs=codec_knobs_for(qtype),
        mode="canvas",
        kept_patch_indices=[],
    )


def _canvas_grid(width: int, height: int, ppc: int, patch: int = 16) -> tuple[int, int]:
    gw = max(1, width // int(patch))
    gh = max(1, height // int(patch))
    if gw * gh != ppc:
        # Fall back to a factor pair of ppc closest to aspect ratio.
        aspect = width / max(height, 1)
        best = (gw, gh)
        best_err = abs((gw / max(gh, 1)) - aspect) + 1e3 * abs(gw * gh - ppc)
        for h in range(1, ppc + 1):
            if ppc % h:
                continue
            w = ppc // h
            err = abs((w / h) - aspect)
            if err < best_err:
                best_err = err
                best = (w, h)
        gw, gh = best
    return int(gw), int(gh)


def _factor_rect(n: int) -> tuple[int, int]:
    """Return (gw, gh) with gw*gh == n, aspect near square."""
    best = (n, 1)
    best_diff = abs(n - 1)
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            diff = abs(w - h)
            if diff < best_diff:
                best_diff = diff
                best = (w, h)
    return best


def _factor_rect_mergeable(n: int, sms: int = 2) -> tuple[int, int]:
    """Like _factor_rect but both sides divisible by spatial_merge_size (Mage MRoPE)."""
    best = None
    best_diff = None
    for h in range(sms, n + 1, sms):
        if n % h:
            continue
        w = n // h
        if w % sms:
            continue
        diff = abs(w - h)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (w, h)
    if best is None:
        # Should not happen if n % (sms*sms) == 0; fall back.
        return _factor_rect(n)
    return best


def _align_budget(k: int, total: int, sms: int = 2) -> int:
    """Budget must be >0, <=total, and divisible by sms^2 for token/feature match."""
    step = int(sms) * int(sms)
    k = int(max(step, min(k, total)))
    k = (k // step) * step
    if k < step:
        k = step if total >= step else total
    return int(min(k, total))


def _extract_tiles(
    images: list[Image.Image],
    src_positions: np.ndarray,
    patch: int = 16,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Return flat tiles (u8 HxWx3), positions, ppc. Tile order = row-major on canvas."""
    n_canvas = len(images)
    total = int(src_positions.shape[0])
    if n_canvas == 0 or total % n_canvas != 0:
        raise ValueError("bad canvas/position shapes for tile extract")
    ppc = total // n_canvas
    tiles: list[np.ndarray] = []
    for im in images:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]
        gw = max(1, w // int(patch))
        gh = max(1, h // int(patch))
        if gw * gh != ppc:
            gw, gh = _canvas_grid(w, h, ppc, patch=patch)
        ph, pw = max(1, h // gh), max(1, w // gw)
        canvas_tiles: list[np.ndarray] = []
        for r in range(gh):
            for c in range(gw):
                tile = arr[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
                if tile.size == 0:
                    tile = np.zeros((max(ph, 1), max(pw, 1), 3), dtype=np.uint8)
                canvas_tiles.append(np.ascontiguousarray(tile))
        if len(canvas_tiles) > ppc:
            canvas_tiles = canvas_tiles[:ppc]
        while len(canvas_tiles) < ppc:
            canvas_tiles.append(np.zeros((ph, pw, 3), dtype=np.uint8))
        tiles.extend(canvas_tiles)
    return tiles, src_positions.copy(), ppc


def score_patches(
    tiles: list[np.ndarray],
    n_canvas: int,
    ppc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch motion (vs previous canvas same slot) and residual (variance)."""
    n = len(tiles)
    motion = np.zeros(n, dtype=np.float32)
    residual = np.zeros(n, dtype=np.float32)
    grays = []
    for t in tiles:
        g = t.astype(np.float32).mean(axis=2) / 255.0
        if g.shape[0] >= 8 and g.shape[1] >= 8:
            ys, xs = g.shape[0] // 8, g.shape[1] // 8
            g = g[: ys * 8, : xs * 8].reshape(8, ys, 8, xs).mean(axis=(1, 3))
        grays.append(g)
        residual[len(grays) - 1] = float(g.var() + 1e-6)

    for ci in range(n_canvas):
        for j in range(ppc):
            idx = ci * ppc + j
            if ci == 0:
                motion[idx] = 0.0
            else:
                prev = (ci - 1) * ppc + j
                a, b = grays[idx], grays[prev]
                hh = min(a.shape[0], b.shape[0])
                ww = min(a.shape[1], b.shape[1])
                motion[idx] = float(np.mean(np.abs(a[:hh, :ww] - b[:hh, :ww])) + 1e-6)
    return _norm(motion), _norm(residual)


def _pack_tiles(
    tiles: list[np.ndarray],
    positions: np.ndarray,
    kept: list[int],
    patch: int = 16,
    max_patches_per_canvas: int = 576,
    spatial_merge_size: int = 2,
) -> tuple[list[Image.Image], np.ndarray]:
    """Pack kept tiles as patch×patch cells with dense (t,h,w) per canvas.

    Original codec timestamps must NOT be reused across a packed canvas: mixed
    ``t`` values break ``rewrite_text_with_codec_positions`` (token count //
    merge^2 drops remainders) → tokens/features mismatch at generate().
    """
    if not kept:
        raise ValueError("no patches kept")
    ps = int(patch)
    sms = int(spatial_merge_size)
    # Resize every tile to exactly patch×patch (matches Qwen2VL patch grid).
    kept_tiles = []
    for i in kept:
        im = Image.fromarray(tiles[i], mode="RGB").resize((ps, ps), Image.BILINEAR)
        kept_tiles.append(np.asarray(im, dtype=np.uint8))

    # Chunk size must be divisible by sms^2 and admit even gw,gh.
    chunk = int(max_patches_per_canvas)
    step = sms * sms
    chunk = max(step, (chunk // step) * step)

    images: list[Image.Image] = []
    out_pos: list[np.ndarray] = []
    for canvas_i, start in enumerate(range(0, len(kept_tiles), chunk)):
        chunk_tiles = kept_tiles[start : start + chunk]
        n = len(chunk_tiles)
        # Pad chunk to mergeable length if needed (duplicate last tile).
        if n % step != 0:
            pad_n = step - (n % step)
            chunk_tiles = chunk_tiles + [chunk_tiles[-1]] * pad_n
            n = len(chunk_tiles)
        gw, gh = _factor_rect_mergeable(n, sms=sms)
        canvas = np.zeros((gh * ps, gw * ps, 3), dtype=np.uint8)
        # Dense grid coords; single synthetic t so timestamp runs stay merge-aligned.
        t_id = int(canvas_i)  # 0,1,2,... — fine for RoPE grouping
        pos2 = np.zeros((n, 3), dtype=np.int64)
        for i, tile in enumerate(chunk_tiles):
            r, c = divmod(i, gw)
            canvas[r * ps : (r + 1) * ps, c * ps : (c + 1) * ps] = tile
            pos2[i, 0] = t_id
            pos2[i, 1] = r
            pos2[i, 2] = c
        images.append(Image.fromarray(canvas, mode="RGB"))
        out_pos.append(pos2)

    return images, np.concatenate(out_pos, axis=0)


def select_patches_samek(
    images: list[Image.Image],
    src_positions: np.ndarray,
    *,
    mode: SelectMode,
    qtype: QuestionType,
    budget_k: int | None = None,
    keep_ratio: float = 0.75,
    seed: int = 0,
    patch: int = 16,
    always_keep_first_patch: bool = True,
) -> ReweightResult:
    """Same-K patch select: queryip / prefix / random, then pack into canvases.

    ``queryip_bitcost`` is handled by ``mvres_score.select_patches_bitcost`` via
    ``MageQueryIPRunner.run_select_mode`` (needs the source video path).
    """
    if mode == "queryip_bitcost":
        raise ValueError(
            "mode=queryip_bitcost requires video bitcost scoring; "
            "use MageQueryIPRunner.run_select_mode (not select_patches_samek alone)"
        )
    tiles, positions, ppc = _extract_tiles(images, src_positions, patch=patch)
    n_canvas = len(images)
    total = len(tiles)
    motion, residual = score_patches(tiles, n_canvas=n_canvas, ppc=ppc)
    w = weights_for(qtype)
    alpha, beta = float(w["alpha"]), float(w["beta"])
    if qtype == QuestionType.STATIC:
        score = alpha * (1.0 - motion) + beta * residual
    else:
        score = alpha * motion + beta * residual

    if budget_k is None:
        budget_k = max(1, int(round(total * float(keep_ratio))))
    budget_k = _align_budget(int(budget_k), total, sms=2)

    if mode == "prefix":
        kept = list(range(budget_k))
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        kept = sorted(rng.choice(total, size=budget_k, replace=False).tolist())
    elif mode == "queryip":
        order = list(np.argsort(-score))
        kept = order[:budget_k]
        if always_keep_first_patch and 0 not in kept:
            kept = [0] + [i for i in kept if i != 0][: budget_k - 1]
        kept = sorted(kept)
    else:
        raise ValueError(f"unknown mode: {mode}")

    new_images, new_pos = _pack_tiles(
        tiles,
        positions,
        kept,
        patch=patch,
        max_patches_per_canvas=ppc,
        spatial_merge_size=2,
    )
    # Drop fully-negative-t canvases if any (should not happen with dup pad)
    mean_all = float(motion.mean()) if len(motion) else 0.0
    mean_kept = float(motion[np.array(kept)].mean()) if kept else 0.0
    # canvas indices that contributed at least one kept patch
    contrib = sorted({i // ppc for i in kept})

    return ReweightResult(
        images=new_images,
        src_positions=new_pos,
        qtype=qtype,
        alpha=alpha,
        beta=beta,
        budget_k=budget_k,
        n_before=total,
        n_after=int(new_pos.shape[0]),
        kept_canvas_indices=contrib,
        canvas_motion=motion.tolist(),
        canvas_residual=residual.tolist(),
        canvas_score=score.tolist(),
        mean_motion_kept=mean_kept,
        mean_motion_all=mean_all,
        knobs=codec_knobs_for(qtype),
        mode=str(mode),
        kept_patch_indices=list(kept),
    )


def jaccard(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))
