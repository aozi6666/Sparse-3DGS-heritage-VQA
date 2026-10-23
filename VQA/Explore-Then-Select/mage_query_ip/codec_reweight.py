"""Mage codec 输出上的选区 / 打包。

【主推】``queryip``：解码 canvas 上运动差 + 残差 proxy → 同 K top-K → ``_pack_tiles``。
【归档·不作贡献】clip / region / edge / bitcost / dyn·sta — 见各模块中文头注释。
bitcost 在 ``mvres_score``；与 proxy 共用 dense pack（image_grid_thw 契约）。
另保留旧版 canvas 级 helper（行 B 兼容）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from .question_type import QuestionType, weights_for

SelectMode = Literal[
    "queryip",
    "queryip_dyn",  # force DYNAMIC α/β (ignore classify)
    "queryip_sta",  # force STATIC α/β (ignore classify)
    "queryip_bitcost",
    "queryip_clip",
    "queryip_region",  # wide pool C + region CLIP fuse (v1 · archived fail)
    "queryip_region_shuf",  # same as region; shuf Q supplied by eval
    "queryip_edge",  # freeze proxy top-(K-m); fill m from band by region
    "queryip_edge_shuf",  # same as edge; shuf Q supplied by eval
    "prefix",
    "random",
]


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


def _select_kept_by_frame_budget(
    score: np.ndarray,
    *,
    n_canvas: int,
    ppc: int,
    budget_k: int,
    frame_scores: np.ndarray,
    always_keep_first_patch: bool,
    temperature: float = 0.07,
    min_per_frame: int = 1,
) -> tuple[list[int], np.ndarray]:
    """CLIP frame budgets + within-canvas codec top-K; total kept == budget_k."""
    from .frame_budget import allocate_budgets

    q = np.asarray(frame_scores, dtype=np.float64).reshape(-1)
    if q.size != n_canvas:
        raise ValueError(
            f"frame_scores length {q.size} != n_canvas {n_canvas}"
        )
    # Cap per-frame by ppc
    budgets = allocate_budgets(
        q, budget_k, min_per_frame=min(min_per_frame, ppc), temperature=temperature
    )
    budgets = np.minimum(budgets.astype(np.int64), int(ppc))
    # Repair sum after per-frame cap (may need multiple passes).
    guard = 0
    while int(budgets.sum()) != int(budget_k) and guard < budget_k + n_canvas + 8:
        guard += 1
        diff = int(budget_k) - int(budgets.sum())
        if diff > 0:
            order = list(np.argsort(-q))
            progressed = False
            for i in order:
                if int(budgets[i]) < ppc:
                    budgets[i] += 1
                    progressed = True
                    break
            if not progressed:
                break
        else:
            order = list(np.argsort(q))  # peel from least relevant
            progressed = False
            for i in order:
                if int(budgets[i]) > 0:
                    budgets[i] -= 1
                    progressed = True
                    break
            if not progressed:
                break
    if int(budgets.sum()) != int(budget_k):
        raise RuntimeError(
            f"frame budgets sum {int(budgets.sum())} != budget_k {budget_k} "
            f"(ppc={ppc}, budgets={budgets.tolist()})"
        )
    kept: list[int] = []
    for ci in range(n_canvas):
        k_i = int(budgets[ci])
        if k_i <= 0:
            continue
        idxs = np.arange(ci * ppc, (ci + 1) * ppc)
        order = idxs[np.argsort(-score[idxs])]
        kept.extend(order[:k_i].tolist())
    if always_keep_first_patch and 0 not in kept and budget_k >= 1:
        if len(kept) >= budget_k:
            worst = min(kept, key=lambda i: float(score[i]))
            kept = [0] + [i for i in kept if i != worst and i != 0]
            kept = kept[:budget_k]
        else:
            kept = [0] + kept
    kept = sorted(set(kept))
    if len(kept) > budget_k:
        order = sorted(kept, key=lambda i: -float(score[i]))
        if always_keep_first_patch and 0 in order:
            order = [0] + [i for i in order if i != 0]
        kept = sorted(order[:budget_k])
    elif len(kept) < budget_k:
        remaining = [i for i in np.argsort(-score).tolist() if i not in kept]
        kept = sorted(kept + remaining[: budget_k - len(kept)])
    return kept, budgets


def _wide_pool_indices(proxy: np.ndarray, budget_k: int) -> set[int]:
    """Pool C: all tiles, or proxy top-M with M=min(n, 2K)."""
    total = int(proxy.shape[0])
    m = min(total, max(int(budget_k) * 2, int(budget_k)))
    if total <= m:
        return set(range(total))
    return set(np.argsort(-proxy)[:m].tolist())


def _edge_m(budget_k: int, *, sms: int = 2) -> int:
    """Boundary width m=⌊0.05K⌋, aligned to sms², clipped to [step, K/4]."""
    k = int(budget_k)
    step = int(sms) * int(sms)
    if k <= 1:
        return 1
    m = k // 20  # floor(0.05K)
    m = (m // step) * step
    if m < step:
        m = step if k >= 2 * step else max(1, k // 4)
    cap = max(step, k // 4)
    m = min(m, cap)
    # Leave at least ``step`` frozen seats when possible.
    if k > step:
        m = min(m, k - step)
    return int(max(1, m))


def _select_kept_edge(
    proxy: np.ndarray,
    region_scores: np.ndarray,
    *,
    budget_k: int,
    edge_m: int,
    always_keep_first_patch: bool,
) -> tuple[list[int], dict[str, int]]:
    """Freeze proxy top-(K-m); fill m seats from band [K-m, K+m) by region score."""
    total = int(proxy.shape[0])
    k = int(budget_k)
    m = int(edge_m)
    m = max(1, min(m, k - 1 if k > 1 else 1))
    order = np.argsort(-np.asarray(proxy, dtype=np.float64)).tolist()
    freeze_n = k - m
    freeze = order[:freeze_n]
    band_end = min(total, k + m)
    band = order[freeze_n:band_end]
    rs = np.asarray(region_scores, dtype=np.float64).reshape(-1)
    band_ranked = sorted(band, key=lambda i: (-float(rs[i]), -float(proxy[i]), i))
    fill = band_ranked[:m]
    if len(fill) < m:
        # Short band: pad by remaining proxy order (still outside freeze).
        need = m - len(fill)
        used = set(freeze) | set(fill)
        pad = [i for i in order if i not in used][:need]
        fill = fill + pad
    kept = sorted(set(freeze) | set(fill))
    if len(kept) > k:
        # Prefer freeze; drop lowest-region among fill extras.
        extra = [i for i in kept if i not in freeze]
        extra = sorted(extra, key=lambda i: (-float(rs[i]), -float(proxy[i]), i))
        kept = sorted(list(freeze) + extra[:m])
    elif len(kept) < k:
        used = set(kept)
        kept = sorted(kept + [i for i in order if i not in used][: k - len(kept)])
    if always_keep_first_patch and 0 not in kept and k >= 1:
        # Drop the worst fill member (or worst overall if fill empty).
        drop_pool = [i for i in kept if i not in freeze] or [i for i in kept if i != 0]
        if drop_pool:
            worst = min(drop_pool, key=lambda i: (float(rs[i]), float(proxy[i])))
            kept = sorted([0] + [i for i in kept if i != worst and i != 0])
            kept = kept[:k]
    meta = {
        "edge_m": m,
        "freeze_n": freeze_n,
        "band_size": len(band),
        "band_lo": freeze_n,
        "band_hi": band_end,
    }
    return kept, meta


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
    frame_scores: np.ndarray | list[float] | None = None,
    clip_temperature: float = 0.07,
    region_scores: np.ndarray | list[float] | None = None,
    region_lambda: float = 0.4,
    edge_m: int | None = None,
) -> ReweightResult:
    """Same-K patch select: queryip / region / edge / clip / prefix / random, then pack.

    ``queryip_bitcost`` is handled by ``mvres_score.select_patches_bitcost`` via
    ``MageQueryIPRunner.run_select_mode`` (needs the source video path).

    ``queryip_clip`` requires ``frame_scores`` (one relevance score per canvas /
    frame); within each canvas patches are ranked by proxy codec score.

    ``queryip_region`` / ``queryip_region_shuf``: wide pool C + fuse
    ``norm(proxy)+λ·norm(region)`` (default λ=0.4); shuf text is the caller's job.

    ``queryip_edge`` / ``queryip_edge_shuf``: freeze proxy top-(K-m); fill m from
    proxy band [K-m, K+m) by region score (default m=⌊0.05K⌋ aligned).

    ``queryip_dyn`` / ``queryip_sta``: same top-K as ``queryip`` but force
    DYNAMIC / STATIC weights (α/β ablation; ignore caller qtype).
    """
    if mode == "queryip_bitcost":
        raise ValueError(
            "mode=queryip_bitcost requires video bitcost scoring; "
            "use MageQueryIPRunner.run_select_mode (not select_patches_samek alone)"
        )
    if mode == "queryip_dyn":
        qtype = QuestionType.DYNAMIC
    elif mode == "queryip_sta":
        qtype = QuestionType.STATIC

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

    frame_budgets: np.ndarray | None = None
    pool_size: int | None = None
    fused_score: np.ndarray | None = None
    edge_meta: dict[str, int] | None = None
    select_like_queryip = mode in ("queryip", "queryip_dyn", "queryip_sta")
    region_modes = ("queryip_region", "queryip_region_shuf")
    edge_modes = ("queryip_edge", "queryip_edge_shuf")
    if mode == "prefix":
        kept = list(range(budget_k))
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        kept = sorted(rng.choice(total, size=budget_k, replace=False).tolist())
    elif select_like_queryip:
        order = list(np.argsort(-score))
        kept = order[:budget_k]
        if always_keep_first_patch and 0 not in kept:
            kept = [0] + [i for i in kept if i != 0][: budget_k - 1]
        kept = sorted(kept)
    elif mode == "queryip_clip":
        if frame_scores is None:
            raise ValueError("mode=queryip_clip requires frame_scores")
        kept, frame_budgets = _select_kept_by_frame_budget(
            score,
            n_canvas=n_canvas,
            ppc=ppc,
            budget_k=budget_k,
            frame_scores=np.asarray(frame_scores, dtype=np.float64),
            always_keep_first_patch=always_keep_first_patch,
            temperature=float(clip_temperature),
            min_per_frame=1,
        )
    elif mode in region_modes:
        if region_scores is None:
            raise ValueError(f"mode={mode} requires region_scores")
        from .region_clip_score import fuse_proxy_region

        rs = np.asarray(region_scores, dtype=np.float64).reshape(-1)
        if rs.size != total:
            raise ValueError(
                f"region_scores length {rs.size} != n_tiles {total}"
            )
        pool = _wide_pool_indices(score, budget_k)
        pool_size = len(pool)
        fused = fuse_proxy_region(score, rs, lam=float(region_lambda))
        fused_score = fused
        order = [i for i in np.argsort(-fused).tolist() if i in pool]
        kept = order[:budget_k]
        if always_keep_first_patch and 0 not in kept and budget_k >= 1:
            if 0 in pool:
                kept = [0] + [i for i in kept if i != 0][: budget_k - 1]
            else:
                kept = [0] + kept[: budget_k - 1]
        kept = sorted(kept)
    elif mode in edge_modes:
        if region_scores is None:
            raise ValueError(f"mode={mode} requires region_scores")
        rs = np.asarray(region_scores, dtype=np.float64).reshape(-1)
        if rs.size != total:
            raise ValueError(
                f"region_scores length {rs.size} != n_tiles {total}"
            )
        m_use = int(edge_m) if edge_m is not None else _edge_m(budget_k, sms=2)
        kept, edge_meta = _select_kept_edge(
            score,
            rs,
            budget_k=budget_k,
            edge_m=m_use,
            always_keep_first_patch=always_keep_first_patch,
        )
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
    mean_all = float(motion.mean()) if len(motion) else 0.0
    mean_kept = float(motion[np.array(kept)].mean()) if kept else 0.0
    contrib = sorted({i // ppc for i in kept})
    knobs = codec_knobs_for(qtype)
    if mode in ("queryip_dyn", "queryip_sta"):
        knobs = dict(knobs)
        knobs["force_qtype"] = qtype.value
        knobs["score_source"] = "proxy_forced_ab"
    if frame_budgets is not None:
        knobs = dict(knobs)
        knobs["frame_budgets"] = [int(x) for x in frame_budgets.tolist()]
        knobs["frame_scores"] = [float(x) for x in np.asarray(frame_scores).tolist()]
        knobs["score_source"] = "proxy+clip_frame"
    if mode in region_modes:
        knobs = dict(knobs)
        knobs["score_source"] = "proxy+region_clip_v1"
        knobs["region_lambda"] = float(region_lambda)
        knobs["pool_size"] = int(pool_size or 0)
        knobs["pool_cap"] = int(min(total, max(budget_k * 2, budget_k)))
        if region_scores is not None:
            knobs["region_score_mean"] = float(np.mean(region_scores))
        if fused_score is not None:
            knobs["fused_score_mean"] = float(np.mean(fused_score))
    if mode in edge_modes:
        knobs = dict(knobs)
        knobs["score_source"] = "proxy+region_edge"
        if edge_meta:
            knobs["edge_m"] = int(edge_meta["edge_m"])
            knobs["freeze_n"] = int(edge_meta["freeze_n"])
            knobs["band_size"] = int(edge_meta["band_size"])
            knobs["band_lo"] = int(edge_meta["band_lo"])
            knobs["band_hi"] = int(edge_meta["band_hi"])
        if region_scores is not None:
            knobs["region_score_mean"] = float(np.mean(region_scores))

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
        canvas_score=(fused_score if fused_score is not None else score).tolist(),
        mean_motion_kept=mean_kept,
        mean_motion_all=mean_all,
        knobs=knobs,
        mode=str(mode),
        kept_patch_indices=list(kept),
    )


def jaccard(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))
