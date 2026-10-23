#!/usr/bin/env python3
"""区域语义 v1 机制冒烟（无 Acc）。

【主路径不用】n=200 Acc 已失败归档；仅复现选区机制（SMOKE_OK）。
用法：python .../smoke_region_clip_v1.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ETS = REPO / "VQA" / "Explore-Then-Select"
MAGE_MODELS = REPO / "VQA" / "mage_vl" / "models"
for p in (str(ETS), str(MAGE_MODELS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    default_video = MAGE_MODELS / "examples" / "soccer-broadcast.mp4"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=MAGE_MODELS)
    ap.add_argument("--video", type=Path, default=default_video)
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--engine", default="hevc", choices=("hevc", "dcvc-rt"))
    ap.add_argument("--device", default=None, help="CLIP device (cuda|cpu)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.4)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument(
        "--real-q",
        default="Which player is moving with the ball and what action is happening?",
    )
    ap.add_argument(
        "--wrong-q",
        default="What text or logo is visible on the scoreboard or banner?",
    )
    args = ap.parse_args()

    if not args.model.is_dir() or not args.video.is_file():
        print(f"[error] model or video missing: {args.model} / {args.video}", file=sys.stderr)
        return 2

    from mage_query_ip.codec_reweight import (
        _extract_tiles,
        _wide_pool_indices,
        jaccard,
        select_patches_samek,
        score_patches,
    )
    from mage_query_ip.mage_forward import MageQueryIPRunner
    from mage_query_ip.question_type import QuestionType, classify_question
    from mage_query_ip.region_clip_score import RegionClipScorer

    print(f"[region_v1] video={args.video}", flush=True)
    print(f"[region_v1] λ={args.lam} block={args.block}", flush=True)

    runner = MageQueryIPRunner(args.model, local_files_only=True)
    enc = runner.encode_codec(
        str(args.video),
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        engine=args.engine,
    )
    total = int(enc["n_patches"])
    budget_k = max(1, int(round(total * float(args.keep_ratio))))
    # Align like select_patches_samek
    from mage_query_ip.codec_reweight import _align_budget

    budget_k = _align_budget(budget_k, total, sms=2)
    print(
        f"[region_v1] n_patches={total} n_canvas={enc['n_canvas']} K={budget_k}",
        flush=True,
    )

    tiles, positions, ppc = _extract_tiles(enc["images"], enc["src_positions"])
    motion, residual = score_patches(tiles, n_canvas=len(enc["images"]), ppc=ppc)
    qtype = classify_question(args.real_q)
    from mage_query_ip.question_type import weights_for

    w = weights_for(qtype)
    if qtype == QuestionType.STATIC:
        proxy = w["alpha"] * (1.0 - motion) + w["beta"] * residual
    else:
        proxy = w["alpha"] * motion + w["beta"] * residual
    pool = _wide_pool_indices(proxy, budget_k)
    print(f"[region_v1] |C|={len(pool)} (cap=min(n,2K)={min(total, budget_k * 2)})", flush=True)

    scorer = RegionClipScorer(device=args.device, local_files_only=True)
    rs_real = scorer.score_flat_tiles(
        tiles,
        n_canvas=len(enc["images"]),
        ppc=ppc,
        positions=positions,
        question=args.real_q,
        block=int(args.block),
    )
    rs_wrong = scorer.score_flat_tiles(
        tiles,
        n_canvas=len(enc["images"]),
        ppc=ppc,
        positions=positions,
        question=args.wrong_q,
        block=int(args.block),
    )
    mean_real = float(rs_real.mean())
    mean_wrong = float(rs_wrong.mean())
    gap = mean_real - mean_wrong
    print(
        f"[region_v1] region_mean real={mean_real:.4f} wrong={mean_wrong:.4f} gap={gap:.4f}",
        flush=True,
    )

    kept: dict[str, list[int]] = {}
    specs = (
        ("queryip", None),
        ("queryip_region", rs_real),
        ("queryip_region_shuf", rs_wrong),
    )
    for mode, rs in specs:
        kw: dict = {}
        if rs is not None:
            kw["region_scores"] = rs
            kw["region_lambda"] = float(args.lam)
        rw = select_patches_samek(
            enc["images"],
            enc["src_positions"],
            mode=mode,
            qtype=qtype,
            budget_k=budget_k,
            **kw,
        )
        kept[mode] = list(rw.kept_patch_indices)
        print(
            f"[region_v1] {mode}: |kept|={len(kept[mode])} "
            f"n_after={rw.n_after} pool={(rw.knobs or {}).get('pool_size')}",
            flush=True,
        )

    j_ab = jaccard(kept["queryip"], kept["queryip_region"])
    j_bc = jaccard(kept["queryip_region"], kept["queryip_region_shuf"])
    j_ac = jaccard(kept["queryip"], kept["queryip_region_shuf"])
    print(f"[region_v1] Jaccard A∩B={j_ab:.3f} B∩C={j_bc:.3f} A∩C={j_ac:.3f}", flush=True)

    ok_len = all(len(kept[m]) == budget_k for m in kept)
    ok_pool = len(pool) >= budget_k
    # Mechanism: question text must move the region-score distribution (either sign).
    ok_gap = abs(gap) > 1e-5
    # Mechanism: fuse must change kept set vs codec-only (not a no-op).
    ok_select = j_ab < 1.0 - 1e-9

    if ok_len and ok_pool and ok_gap and ok_select:
        print("SMOKE_OK", flush=True)
        return 0

    print(
        f"[fail] ok_len={ok_len} ok_pool={ok_pool} ok_gap={ok_gap} "
        f"ok_select={ok_select} (|kept|==K, |C|>=K, |gap|>0, Jaccard(A,B)<1)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
