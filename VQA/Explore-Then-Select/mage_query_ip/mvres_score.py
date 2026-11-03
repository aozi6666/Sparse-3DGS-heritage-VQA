"""官方 codec block bitcost 打分（已在 Mage canvas 上的 patch）。

【主路径不用】queryip_bitcost 失败消融（≤ random）。开放栈无分开的真 MV+残差。
cv_reader_fast 仅有 bitcost；cv_reader_fetch_mvres 不可用。保留供对照复现。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .codec_reweight import (
    ReweightResult,
    _align_budget,
    _extract_tiles,
    _pack_tiles,
    codec_knobs_for,
    jaccard,
    score_patches,
    select_patches_samek,
)
from .question_type import QuestionType, weights_for


def _bitcost_pixels(item: dict[str, Any]) -> np.ndarray | None:
    """Block bitcost resized to the decoded frame. H.264 uses MB, HEVC uses CTU.

    Returns None when the bitcost map is missing/empty (caller scores those patches 0).
    """
    import cv2

    bc = item.get("bitcost") or {}
    if not bc:
        return None
    codec = str(item.get("codec_name") or "").lower()
    hevc = any(tok in codec for tok in ("265", "hevc", "hvc1"))
    if hevc and "ctu_bit_cost" in bc:
        key, hk, wk = "ctu_bit_cost", "ctu_height", "ctu_width"
    else:
        key, hk, wk = "mb_bit_cost", "mb_height", "mb_width"
    if key not in bc:
        return None
    arr = np.asarray(bc[key], dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.maximum(arr, 0.0)
    if hk in bc and wk in bc:
        arr = arr[: int(bc[hk]), : int(bc[wk])]
    if arr.size == 0:
        return None
    height, width = int(item["height"]), int(item["width"])
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)


def _fetch_bitcost(video: str, frame_ids: list[int]) -> dict[int, dict[str, Any]]:
    from codec_video_prep.cv_reader_fast import read_video_fast_selected

    rows = read_video_fast_selected(
        str(video), [int(x) for x in frame_ids], thread_count=1, export_bitcost=1
    )
    wanted = {int(x) for x in frame_ids}
    got: dict[int, dict[str, Any]] = {}
    for item in rows:
        fid = int(item.get("frame_idx", -1))
        if fid in wanted:
            got[fid] = item
    return got


def _on_processed(
    native: np.ndarray,
    resize_hw: tuple[int, int],
    processed_hw: tuple[int, int],
) -> np.ndarray:
    import cv2

    rh, rw = int(resize_hw[0]), int(resize_hw[1])
    ph, pw = int(processed_hw[0]), int(processed_hw[1])
    resized = cv2.resize(native, (rw, rh), interpolation=cv2.INTER_NEAREST)
    if (ph, pw) == (rh, rw):
        return resized.astype(np.float32)
    out = np.zeros((ph, pw), dtype=np.float32)
    out[: min(rh, ph), : min(rw, pw)] = resized[: min(rh, ph), : min(rw, pw)]
    return out


def _geometry(meta: dict[str, Any] | None, images: list[Image.Image]) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """Return (resize_hw, processed_hw, from_meta)."""
    meta = meta or {}
    processed = meta.get("processed_hw")
    resize = meta.get("resize_hw")
    if (
        isinstance(processed, (list, tuple))
        and len(processed) == 2
        and isinstance(resize, (list, tuple))
        and len(resize) == 2
    ):
        return (int(resize[0]), int(resize[1])), (int(processed[0]), int(processed[1])), True
    if images:
        w, h = images[0].size
        hw = (int(h), int(w))
        return hw, hw, False
    raise ValueError("no meta geometry and no canvases")


def _pct_norm(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = raw[valid]
    scale = float(np.percentile(vals, 95))
    scale = max(scale, 1e-6)
    out[valid] = np.clip(vals / scale, 0.0, 1.0)
    return out


def score_patches_bitcost(
    video: str,
    images: list[Image.Image],
    src_positions: np.ndarray,
    meta: dict[str, Any] | None = None,
    patch: int = 16,
) -> dict[str, Any]:
    """Per-canvas-patch official bitcost. ``s_res`` is zero (no residual map)."""
    pos = np.asarray(src_positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"src_positions must be (N,3), got {pos.shape}")
    n = int(pos.shape[0])
    valid = pos[:, 0] >= 0
    fids = sorted({int(t) for t in pos[valid, 0].tolist()})
    if not fids:
        raise ValueError("no non-padding source frames in src_positions")

    got = _fetch_bitcost(str(video), fids)
    resize_hw, processed_hw, from_meta = _geometry(meta, images)
    placed: dict[int, np.ndarray] = {}
    empty_fids: list[int] = []
    fail_n = 0
    for fid, item in got.items():
        try:
            pix = _bitcost_pixels(item)
        except Exception:  # noqa: BLE001 — tolerate bad frames
            fail_n += 1
            empty_fids.append(int(fid))
            continue
        if pix is None:
            empty_fids.append(int(fid))
            continue
        placed[fid] = _on_processed(pix, resize_hw, processed_hw)

    missing = [fid for fid in fids if fid not in got]
    empty_fids = empty_fids + missing
    n_req = len(fids)
    n_hit = len(placed)
    n_miss = n_req - n_hit
    # One summary line per video — no per-frame spam.
    if n_miss > 0:
        warnings.warn(
            f"bitcost miss {n_miss}/{n_req} frames "
            f"(hit={n_hit}, empty_or_fail={len(empty_fids)}, parse_fail={fail_n}); "
            f"miss patches use proxy fallback at select",
            stacklevel=2,
        )

    ps = int(patch)
    ph_max, pw_max = int(processed_hw[0]) // ps, int(processed_hw[1]) // ps
    raw_mv = np.zeros(n, dtype=np.float32)
    raw_res = np.zeros(n, dtype=np.float32)
    indexed = np.zeros(n, dtype=bool)
    frame_ok = np.zeros(n, dtype=bool)

    for i in range(n):
        if not valid[i]:
            continue
        fid, pr, pc = int(pos[i, 0]), int(pos[i, 1]), int(pos[i, 2])
        frame_ok[i] = fid in placed
        if fid not in placed:
            continue
        if pr < 0 or pc < 0 or pr >= ph_max or pc >= pw_max:
            continue
        mag = placed[fid]
        y0, x0 = pr * ps, pc * ps
        raw_mv[i] = float(mag[y0 : y0 + ps, x0 : x0 + ps].mean())
        indexed[i] = True

    usable = valid & indexed
    return {
        "s_mv": _pct_norm(raw_mv, usable),
        "s_bit": _pct_norm(raw_mv, usable),
        "s_res": _pct_norm(raw_res, usable),
        "raw_mv": raw_mv,
        "raw_res": raw_res,
        "valid": valid,
        "indexed": indexed,
        "frame_ok": frame_ok,
        "n_requested_frames": n_req,
        "n_hit_frames": n_hit,
        "n_empty_frames": n_miss,
        "empty_frame_ids": empty_fids,
        "geometry_from_meta": from_meta,
        "processed_hw": list(processed_hw),
        "resize_hw": list(resize_hw),
        "score_source": "bitcost",
    }


# Backward-compatible alias (route-A narrow experiment naming).
score_patches_mvres = score_patches_bitcost


def save_patch_scores(path: str | Path, scored: dict[str, Any], src_positions: np.ndarray) -> Path:
    """Write per-patch scores only (not full-frame maps)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        s_mv=np.asarray(scored["s_mv"], dtype=np.float32),
        s_res=np.asarray(scored["s_res"], dtype=np.float32),
        raw_mv=np.asarray(scored["raw_mv"], dtype=np.float32),
        raw_res=np.asarray(scored["raw_res"], dtype=np.float32),
        src_positions=np.asarray(src_positions, dtype=np.int32),
        valid=np.asarray(scored["valid"], dtype=np.bool_),
        indexed=np.asarray(scored["indexed"], dtype=np.bool_),
    )
    return out


def _combine(s_mv: np.ndarray, s_res: np.ndarray, qtype: QuestionType) -> tuple[np.ndarray, float, float]:
    w = weights_for(qtype)
    alpha, beta = float(w["alpha"]), float(w["beta"])
    if qtype == QuestionType.STATIC:
        score = alpha * (1.0 - s_mv) + beta * s_res
    else:
        score = alpha * s_mv + beta * s_res
    return score.astype(np.float32), alpha, beta


def select_patches_bitcost(
    images: list[Image.Image],
    src_positions: np.ndarray,
    scored: dict[str, Any],
    *,
    qtype: QuestionType,
    budget_k: int | None = None,
    keep_ratio: float = 0.75,
    patch: int = 16,
    always_keep_first_patch: bool = True,
    proxy_fallback: bool = True,
) -> ReweightResult:
    """Top-K by α·bitcost (+ β·0); miss frames filled with proxy (hybrid).

    When ``proxy_fallback`` is True (default), patches without a bitcost map
    use the same motion/residual proxies as ``queryip`` so ranking stays
    defined. Papers should call this ``bitcost_hybrid``, not pure bitcost.
    """
    tiles, positions, ppc = _extract_tiles(images, src_positions, patch=patch)
    total = len(tiles)
    n_canvas = len(images)
    s_bit = np.asarray(scored.get("s_bit", scored["s_mv"]), dtype=np.float32).copy()
    s_res = np.asarray(scored["s_res"], dtype=np.float32).copy()
    if s_bit.shape[0] != total:
        raise ValueError(f"score length {s_bit.shape[0]} != tiles {total}")

    indexed = np.asarray(scored.get("indexed"), dtype=bool)
    if indexed.shape[0] != total:
        indexed = np.ones(total, dtype=bool)
    n_fallback = 0
    if proxy_fallback and (not bool(np.all(indexed))):
        motion, residual = score_patches(tiles, n_canvas=n_canvas, ppc=ppc)
        miss = ~indexed
        n_fallback = int(miss.sum())
        s_bit[miss] = motion[miss]
        s_res[miss] = residual[miss]

    score, alpha, beta = _combine(s_bit, s_res, qtype)
    if budget_k is None:
        budget_k = max(1, int(round(total * float(keep_ratio))))
    budget_k = _align_budget(int(budget_k), total, sms=2)

    order = list(np.argsort(-score))
    kept = order[:budget_k]
    if always_keep_first_patch and 0 not in kept:
        kept = [0] + [i for i in kept if i != 0][: budget_k - 1]
    kept = sorted(int(i) for i in kept)

    new_images, new_pos = _pack_tiles(
        tiles,
        positions,
        kept,
        patch=patch,
        max_patches_per_canvas=ppc,
        spatial_merge_size=2,
    )
    contrib = sorted({i // ppc for i in kept})
    knobs = codec_knobs_for(qtype)
    src = "bitcost_hybrid" if n_fallback > 0 else str(scored.get("score_source") or "bitcost")
    knobs["score_source"] = src
    knobs["n_proxy_fallback_patches"] = n_fallback
    knobs["proxy_fallback"] = bool(proxy_fallback)
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
        canvas_motion=s_bit.tolist(),
        canvas_residual=s_res.tolist(),
        canvas_score=score.tolist(),
        mean_motion_kept=float(s_bit[np.array(kept)].mean()) if kept else 0.0,
        mean_motion_all=float(s_bit.mean()) if len(s_bit) else 0.0,
        knobs=knobs,
        mode="queryip_bitcost",
        kept_patch_indices=list(kept),
    )


# Backward-compatible alias.
select_patches_mvres = select_patches_bitcost


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def alignment_report(
    images: list[Image.Image],
    src_positions: np.ndarray,
    scored: dict[str, Any],
    *,
    keep_ratio: float = 0.75,
    seed: int = 0,
    patch: int = 16,
) -> dict[str, Any]:
    """Compare MV/residual top-K with the pixel-proxy queryip top-K. No generate."""
    tiles, _, ppc = _extract_tiles(images, src_positions, patch=patch)
    motion, _ = score_patches(tiles, n_canvas=len(images), ppc=ppc)
    usable = np.asarray(scored["indexed"], dtype=bool)
    corr = _corr(np.asarray(scored["s_mv"])[usable], motion[usable])

    total = len(tiles)
    budget_k = _align_budget(max(1, int(round(total * float(keep_ratio)))), total, sms=2)
    jacc: dict[str, float] = {}
    n_after: dict[str, int] = {}
    for qtype in (QuestionType.STATIC, QuestionType.DYNAMIC):
        mv = select_patches_bitcost(
            images, src_positions, scored, qtype=qtype, budget_k=budget_k, patch=patch
        )
        proxy = select_patches_samek(
            images,
            src_positions,
            mode="queryip",
            qtype=qtype,
            budget_k=budget_k,
            seed=seed,
            patch=patch,
        )
        jacc[qtype.value] = jaccard(mv.kept_patch_indices, proxy.kept_patch_indices)
        n_after[qtype.value] = int(mv.n_after)
        n_after[f"proxy_{qtype.value}"] = int(proxy.n_after)

    n_valid = int(np.asarray(scored["valid"]).sum())
    frame_hit = float(np.asarray(scored["frame_ok"])[scored["valid"]].mean()) if n_valid else 0.0
    grid_hit = float(np.asarray(scored["indexed"])[scored["valid"]].mean()) if n_valid else 0.0
    same_k = len(set(n_after.values())) == 1
    aligned = frame_hit >= 0.99 and grid_hit >= 0.99 and same_k
    redundant = (
        aligned
        and corr is not None
        and corr > 0.9
        and jacc["static"] >= 0.98
        and jacc["dynamic"] >= 0.98
    )
    if not aligned:
        verdict = "ALIGN_FAIL"
    elif redundant:
        verdict = "ALIGN_REDUNDANT"
    else:
        verdict = "ALIGN_OK"
    return {
        "verdict": verdict,
        "frame_hit": frame_hit,
        "grid_hit": grid_hit,
        "corr_s_mv_proxy": corr,
        "jaccard_mvres_proxy": jacc,
        "n_after": n_after,
        "budget_k": int(budget_k),
        "n_patches": int(total),
        "n_valid": n_valid,
        "n_hit_frames": int(scored["n_hit_frames"]),
        "n_requested_frames": int(scored["n_requested_frames"]),
        "geometry_from_meta": bool(scored["geometry_from_meta"]),
        "same_k": same_k,
        "score_source": str(scored.get("score_source") or "bitcost"),
    }
