"""Soccer CPU alignment for codec MV/residual vs pixel proxy.

Writes a compact per-patch score file. Does not generate unless ``--generate``.
Agent does not run this — see COMMANDS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_codec(video: Path, num_frames: int, max_pixels: int, patch: int):
    from codec_video_processing_mage_vl import (
        CodecConfig,
        drop_padding_canvases,
        process_codec_video,
    )

    cfg = CodecConfig(
        engine="hevc",
        target_canvas=int(num_frames),
        patch=int(patch),
        max_pixels=int(max_pixels),
    )
    payload = process_codec_video(str(video), cfg)
    images, src_positions, dropped = drop_padding_canvases(
        payload["images"], payload["src_positions"]
    )
    return images, src_positions, payload.get("meta") or {}, float(payload["fps"]), int(dropped)


def _maybe_generate(args, images, src_positions, scored, fps: float) -> None:
    from mage_query_ip.mage_forward import MageQueryIPRunner
    from mage_query_ip.mvres_score import select_patches_mvres
    from mage_query_ip.question_type import QuestionType

    print("[mvres] loading Mage weights for one question")
    runner = MageQueryIPRunner(args.model, local_files_only=True)
    rw = select_patches_mvres(
        images,
        src_positions,
        scored,
        qtype=QuestionType.STATIC,
        keep_ratio=float(args.keep_ratio),
        patch=int(args.patch),
    )
    inputs, n_vis = runner._build_inputs_from_codec_payload(
        args.question, rw.images, rw.src_positions, fps, int(args.max_pixels)
    )
    answer = runner._generate(inputs, int(args.max_new_tokens))
    print(
        f"[mvres] generate n_patches={rw.n_after} n_vis≈{n_vis} "
        f"answer={answer[:200]!r}"
    )


def main() -> int:
    root = _repo_root()
    default_video = root / "VQA" / "mage_vl" / "models" / "examples" / "soccer-broadcast.mp4"
    default_scores = root / "VQA" / "eval_results" / "soccer_mvres_scores.npz"
    default_report = root / "VQA" / "eval_results" / "soccer_mvres_align.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=default_video)
    ap.add_argument("--model", type=Path, default=root / "VQA" / "mage_vl" / "models")
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores", type=Path, default=default_scores)
    ap.add_argument("--report", type=Path, default=default_report)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--question",
        default="What text or logo is visible on the scoreboard or banner?",
    )
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"[error] video missing: {args.video}", file=sys.stderr)
        return 2

    ets = str(root / "VQA" / "Explore-Then-Select")
    models = str(root / "VQA" / "mage_vl" / "models")
    for p in (ets, models):
        if p not in sys.path:
            sys.path.insert(0, p)

    from mage_query_ip.mvres_score import alignment_report, save_patch_scores, score_patches_mvres

    print(f"[mvres] video={args.video}")
    images, src_positions, meta, fps, dropped = _load_codec(
        args.video, args.num_frames, args.max_pixels, args.patch
    )
    print(
        f"[mvres] canvases={len(images)} patches={src_positions.shape[0]} "
        f"dropped_pad_canvas={dropped}"
    )
    scored = score_patches_mvres(
        str(args.video), images, src_positions, meta=meta, patch=int(args.patch)
    )
    score_path = save_patch_scores(args.scores, scored, src_positions)
    report = alignment_report(
        images,
        src_positions,
        scored,
        keep_ratio=float(args.keep_ratio),
        seed=int(args.seed),
        patch=int(args.patch),
    )
    report["scores"] = str(score_path)
    report["fps"] = fps
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    corr = report["corr_s_mv_proxy"]
    corr_s = "None" if corr is None else f"{corr:.3f}"
    print(f"[mvres] score_source={report['score_source']} (no motion_vector/residual in cv_reader_fast)")
    print(f"[mvres] scores={score_path}")
    print(f"[mvres] report={args.report}")
    print(
        f"[mvres] frame_hit={report['frame_hit']:.3f} grid_hit={report['grid_hit']:.3f} "
        f"frames={report['n_hit_frames']}/{report['n_requested_frames']} "
        f"geometry_from_meta={report['geometry_from_meta']}"
    )
    print(f"[mvres] corr(s_mv, frame-diff)={corr_s}")
    print(f"[mvres] jaccard mvres∩proxy={report['jaccard_mvres_proxy']}")
    print(f"[mvres] n_after={report['n_after']} budget_k={report['budget_k']} same_k={report['same_k']}")
    print(f"[mvres] {report['verdict']}")
    if report["verdict"] != "ALIGN_OK":
        print("[mvres] stop: do not run short-60")
    else:
        print("[mvres] gate open; short-60 is not part of this command")

    if args.generate:
        _maybe_generate(args, images, src_positions, scored, fps)
    return 0 if report["verdict"] != "ALIGN_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
