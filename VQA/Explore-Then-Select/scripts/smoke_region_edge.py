#!/usr/bin/env python3
"""区域边界 edge 机制冒烟（无 Acc）。

【主路径不用】n=200 Acc 已失败；语义选区已停。仅复现冻结+band 机制。
用法：python .../smoke_region_edge.py --device cuda
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
    ap.add_argument("--device", default=None)
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
        _align_budget,
        _edge_m,
        _extract_tiles,
        jaccard,
        select_patches_samek,
        score_patches,
    )
    from mage_query_ip.mage_forward import MageQueryIPRunner
    from mage_query_ip.question_type import QuestionType, classify_question, weights_for
    from mage_query_ip.region_clip_score import RegionClipScorer

    print(f"[edge] video={args.video}", flush=True)

    runner = MageQueryIPRunner(args.model, local_files_only=True)
    enc = runner.encode_codec(
        str(args.video),
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        engine=args.engine,
    )
    total = int(enc["n_patches"])
    budget_k = _align_budget(
        max(1, int(round(total * float(args.keep_ratio)))), total, sms=2
    )
    m = _edge_m(budget_k, sms=2)
    print(
        f"[edge] n_patches={total} K={budget_k} m={m} freeze_n={budget_k - m}",
        flush=True,
    )

    tiles, positions, ppc = _extract_tiles(enc["images"], enc["src_positions"])
    motion, residual = score_patches(tiles, n_canvas=len(enc["images"]), ppc=ppc)
    qtype = classify_question(args.real_q)
    w = weights_for(qtype)
    if qtype == QuestionType.STATIC:
        proxy = w["alpha"] * (1.0 - motion) + w["beta"] * residual
    else:
        proxy = w["alpha"] * motion + w["beta"] * residual

    import numpy as np

    order = np.argsort(-proxy).tolist()
    freeze = set(order[: budget_k - m])

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

    kept: dict[str, list[int]] = {}
    for mode, rs in (
        ("queryip", None),
        ("queryip_edge", rs_real),
        ("queryip_edge_shuf", rs_wrong),
    ):
        kw: dict = {}
        if rs is not None:
            kw["region_scores"] = rs
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
            f"[edge] {mode}: |kept|={len(kept[mode])} "
            f"m={(rw.knobs or {}).get('edge_m')} "
            f"band={(rw.knobs or {}).get('band_size')} "
            f"freeze_n={(rw.knobs or {}).get('freeze_n')}",
            flush=True,
        )

    j_ab = jaccard(kept["queryip"], kept["queryip_edge"])
    j_bc = jaccard(kept["queryip_edge"], kept["queryip_edge_shuf"])
    print(f"[edge] Jaccard A∩B={j_ab:.3f} B∩C={j_bc:.3f}", flush=True)

    edge_set = set(kept["queryip_edge"])
    ok_len = all(len(kept[x]) == budget_k for x in kept)
    ok_freeze = freeze.issubset(edge_set)
    ok_select = j_ab < 1.0 - 1e-9
    gap = float(rs_real.mean() - rs_wrong.mean())
    ok_gap = abs(gap) > 1e-5
    print(f"[edge] freeze⊂kept={ok_freeze} |gap|={abs(gap):.4f}", flush=True)

    if ok_len and ok_freeze and ok_select and ok_gap:
        print("SMOKE_OK", flush=True)
        return 0

    print(
        f"[fail] ok_len={ok_len} ok_freeze={ok_freeze} "
        f"ok_select={ok_select} ok_gap={ok_gap}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
