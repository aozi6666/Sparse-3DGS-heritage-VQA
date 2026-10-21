#!/usr/bin/env python
"""Small smoke test: soccer-broadcast.mp4 with static vs dynamic questions.

Compares Mage default codec vs query-IP reweight under a shared patch budget.
Does not modify mage_vl sources. See ../COMMANDS.md for how to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../VQA/Explore-Then-Select/mage_query_ip/smoke_soccer.py
    return Path(__file__).resolve().parents[3]


def _print_stats(tag: str, s) -> None:
    print(f"\n===== {tag} =====")
    print(f"mode={s.mode} qtype={s.qtype}")
    print(f"question={s.question!r}")
    print(f"n_patches={s.n_patches} n_vis_tokens≈{s.n_vis_tokens} wall_s={s.wall_s:.2f}")
    if s.extra:
        print(
            f"extra={json.dumps(s.extra, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))}"
        )
    if s.reweight is not None:
        rw = s.reweight
        print(
            f"reweight: before={rw.n_before} after={rw.n_after} "
            f"kept_canvases={rw.kept_canvas_indices} "
            f"mean_motion all={rw.mean_motion_all:.4f} kept={rw.mean_motion_kept:.4f}"
        )
    print(f"answer={s.answer[:500]}")


def main() -> int:
    root = _repo_root()
    default_model = root / "VQA" / "mage_vl" / "models"
    default_video = default_model / "examples" / "soccer-broadcast.mp4"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--video", type=Path, default=default_video)
    ap.add_argument("--budget-mode", choices=("match_default",), default="match_default")
    ap.add_argument(
        "--keep-ratio",
        type=float,
        default=0.75,
        help="When reweighting, keep this fraction of default patch count (same K for both Qs).",
    )
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--engine", default="hevc", choices=("hevc", "dcvc-rt"))
    ap.add_argument(
        "--static-q",
        default="What text or logo is visible on the scoreboard or banner?",
    )
    ap.add_argument(
        "--dynamic-q",
        default="Which player is moving with the ball and what action is happening?",
    )
    args = ap.parse_args()

    if not args.model.is_dir():
        print(f"[error] model dir missing: {args.model}", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"[error] video missing: {args.video}", file=sys.stderr)
        return 2

    # Ensure local imports resolve when run as a script.
    ets = str(root / "VQA" / "Explore-Then-Select")
    models = str(root / "VQA" / "mage_vl" / "models")
    for p in (ets, models):
        if p not in sys.path:
            sys.path.insert(0, p)

    from mage_query_ip.mage_forward import MageQueryIPRunner
    from mage_query_ip.question_type import classify_question

    print(f"[smoke] model={args.model}")
    print(f"[smoke] video={args.video}")
    print(f"[smoke] static_q classify -> {classify_question(args.static_q).value}")
    print(f"[smoke] dynamic_q classify -> {classify_question(args.dynamic_q).value}")

    runner = MageQueryIPRunner(args.model, local_files_only=True)

    # One default pass to establish N and K.
    base = runner.run_default(
        video=str(args.video),
        question=args.dynamic_q,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        engine=args.engine,
    )
    _print_stats("DEFAULT (codec, no reweight)", base)

    if args.budget_mode == "match_default":
        budget_k = max(1, int(round(base.n_patches * float(args.keep_ratio))))
    else:
        budget_k = max(1, int(round(base.n_patches * float(args.keep_ratio))))
    print(f"\n[smoke] shared budget_k={budget_k} (keep_ratio={args.keep_ratio})")

    st = runner.run_reweight(
        video=str(args.video),
        question=args.static_q,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        engine=args.engine,
        budget_k=budget_k,
        keep_ratio=args.keep_ratio,
    )
    _print_stats("REWEIGHT static-q", st)

    dy = runner.run_reweight(
        video=str(args.video),
        question=args.dynamic_q,
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        engine=args.engine,
        budget_k=budget_k,
        keep_ratio=args.keep_ratio,
    )
    _print_stats("REWEIGHT dynamic-q", dy)

    # Effect summary
    print("\n===== EFFECT CHECK =====")
    same_k = st.n_patches == dy.n_patches
    diff_set = set(st.extra.get("kept_canvas_indices", [])) != set(
        dy.extra.get("kept_canvas_indices", [])
    )
    motion_gap = abs(
        float(st.extra.get("mean_motion_kept", 0)) - float(dy.extra.get("mean_motion_kept", 0))
    )
    print(f"same_patch_budget={same_k} (static={st.n_patches}, dynamic={dy.n_patches})")
    print(f"different_kept_canvases={diff_set}")
    print(f"|Δ mean_motion_kept|={motion_gap:.4f}")
    print(f"both_answered={bool(st.answer) and bool(dy.answer) and bool(base.answer)}")
    if same_k and (diff_set or motion_gap > 1e-4) and st.answer and dy.answer:
        print("SMOKE_OK: selection differs by question type under fixed K; answers produced.")
        return 0
    print(
        "SMOKE_SOFT: ran end-to-end; selection difference weak or empty answer — check logs.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
