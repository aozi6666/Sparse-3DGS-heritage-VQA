#!/usr/bin/env python3
"""CLIP 帧预算脚手架冒烟（无 Mage Acc）。

【主路径不用】E4 已失败；仅复现/回归。默认测 allocate_budgets；
``--with-wire`` / ``--with-clip`` 见下。

Usage:
  python VQA/Explore-Then-Select/scripts/smoke_clip_frame_budget.py
  python .../smoke_clip_frame_budget.py --with-wire
  python .../smoke_clip_frame_budget.py --with-clip --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ETS = REPO / "VQA" / "Explore-Then-Select"
sys.path.insert(0, str(ETS))


def _self_test_budget() -> None:
    import numpy as np
    from mage_query_ip.frame_budget import allocate_budgets

    b = allocate_budgets([0.5, 0.5, 0.5, 0.5], 40, min_per_frame=2)
    assert int(b.sum()) == 40 and int(b.min()) >= 2
    b2 = allocate_budgets([0.0, 0.0, 0.0, 5.0], 40, min_per_frame=1, temperature=0.05)
    assert int(b2.sum()) == 40 and int(b2[-1]) == int(b2.max())
    # Cap-style peak then redistribute manually (mirrors select repair)
    q = np.array([0.1, 0.1, 0.9, 0.2], dtype=np.float64)
    b3 = allocate_budgets(q, 32, min_per_frame=1, temperature=0.05)
    ppc = 16
    b3 = np.minimum(b3, ppc)
    while int(b3.sum()) < 32:
        for i in np.argsort(-q):
            if int(b3.sum()) >= 32:
                break
            if int(b3[i]) < ppc:
                b3[i] += 1
    assert int(b3.sum()) == 32 and int(b3[2]) == int(b3.max())
    print("[ok] allocate_budgets self-test", flush=True)


def _self_test_wire() -> None:
    import numpy as np
    from PIL import Image
    from mage_query_ip.codec_reweight import select_patches_samek
    from mage_query_ip.question_type import QuestionType

    patch, n_canvas, ppc = 16, 4, 16
    total = n_canvas * ppc
    budget_k = 32
    images = [
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).convert("RGB")
        for _ in range(n_canvas)
    ]
    pos = np.zeros((total, 3), dtype=np.int64)
    for i in range(total):
        ci, j = divmod(i, ppc)
        pos[i, 0] = ci
        pos[i, 1] = j // 4
        pos[i, 2] = j % 4
    frame_scores = np.array([0.1, 0.1, 0.9, 0.2], dtype=np.float32)
    rw = select_patches_samek(
        images,
        pos,
        mode="queryip_clip",
        qtype=QuestionType.DYNAMIC,
        budget_k=budget_k,
        frame_scores=frame_scores,
    )
    fb = (rw.knobs or {}).get("frame_budgets")
    assert fb is not None and sum(fb) == budget_k
    assert len(rw.kept_patch_indices) == budget_k
    assert fb[2] == max(fb), f"expected peak on frame 2, got {fb}"
    print(
        f"[ok] queryip_clip wire: kept={len(rw.kept_patch_indices)} frame_budgets={fb}",
        flush=True,
    )


def _try_clip(args: argparse.Namespace) -> int:
    import numpy as np
    from mage_query_ip.clip_frame_score import ClipFrameScorer, sample_video_frames
    from mage_query_ip.frame_budget import allocate_budgets

    if not args.clip.is_dir():
        print(f"[error] CLIP missing: {args.clip}", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"[error] video missing: {args.video}", file=sys.stderr)
        return 2
    print(f"[clip] loading {args.clip} device={args.device or 'auto'}", flush=True)
    frames = sample_video_frames(args.video, num_frames=args.num_frames)
    scorer = ClipFrameScorer(args.clip, device=args.device, local_files_only=True)
    q_t = scorer.score_frames(frames, args.question)
    budgets = allocate_budgets(q_t, int(args.total_k), min_per_frame=1)
    print("[clip] q_t:", np.round(q_t, 4).tolist(), flush=True)
    print("[clip] budgets:", budgets.tolist(), f"sum={int(budgets.sum())}", flush=True)
    assert int(budgets.sum()) == int(args.total_k)
    print("[ok] CLIP frame-budget smoke passed", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CLIP frame-budget CPU smoke")
    ap.add_argument(
        "--clip",
        type=Path,
        default=REPO / "VQA" / "models" / "CLIP-ViT-Large-patch14",
    )
    ap.add_argument(
        "--video",
        type=Path,
        default=REPO
        / "VQA"
        / "mage_vl"
        / "models"
        / "examples"
        / "soccer-broadcast.mp4",
    )
    ap.add_argument(
        "--question",
        default="What color is the jersey of the player with the ball?",
    )
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--total-k", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--with-wire",
        action="store_true",
        help="Import select_patches_samek wiring test (more RAM)",
    )
    ap.add_argument(
        "--with-clip",
        action="store_true",
        help="Load real CLIP (needs GPU or >>2Gi RAM)",
    )
    args = ap.parse_args()

    _self_test_budget()
    if args.with_wire:
        _self_test_wire()
    else:
        print("[skip] --with-wire not set (select_patches_samek)", flush=True)
    if args.with_clip:
        return _try_clip(args)
    print(
        "[skip] --with-clip not set (use on GPU / larger-RAM instance).",
        flush=True,
    )
    print("[ok] scaffold smoke passed (allocate_budgets)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
