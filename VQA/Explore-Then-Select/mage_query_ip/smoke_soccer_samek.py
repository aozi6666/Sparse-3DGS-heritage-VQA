#!/usr/bin/env python
"""Soccer same-K ablation: prefix / random / queryip (patch-level proxy).

Success (open-ended, no Acc): equal n_vis_tokens; QueryIP kept-set differs from
prefix and random; answers non-empty. Agent does not run this — see COMMANDS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> int:
    root = _repo_root()
    default_model = root / "VQA" / "mage_vl" / "models"
    default_video = default_model / "examples" / "soccer-broadcast.mp4"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--video", type=Path, default=default_video)
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--engine", default="hevc", choices=("hevc", "dcvc-rt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--static-q",
        default="What text or logo is visible on the scoreboard or banner?",
    )
    ap.add_argument(
        "--dynamic-q",
        default="Which player is moving with the ball and what action is happening?",
    )
    args = ap.parse_args()

    if not args.model.is_dir() or not args.video.is_file():
        print("[error] model or video missing", file=sys.stderr)
        return 2

    ets = str(root / "VQA" / "Explore-Then-Select")
    models = str(root / "VQA" / "mage_vl" / "models")
    for p in (ets, models):
        if p not in sys.path:
            sys.path.insert(0, p)

    from mage_query_ip.mage_forward import MageQueryIPRunner
    from mage_query_ip.question_type import classify_question

    print(f"[samek] model={args.model}")
    print(f"[samek] video={args.video}")
    print(f"[samek] keep_ratio={args.keep_ratio} seed={args.seed}")

    runner = MageQueryIPRunner(args.model, local_files_only=True)
    enc = runner.encode_codec(
        str(args.video),
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        engine=args.engine,
    )
    budget_k = max(1, int(round(enc["n_patches"] * float(args.keep_ratio))))
    print(
        f"[samek] codec n_patches={enc['n_patches']} n_canvas={enc['n_canvas']} "
        f"budget_k={budget_k}"
    )

    questions = [
        ("static", args.static_q),
        ("dynamic", args.dynamic_q),
    ]
    modes = ("prefix", "random", "queryip")
    ok_all = True

    for tag, question in questions:
        print(f"\n===== Q={tag} classify={classify_question(question).value} =====")
        print(f"question={question!r}")
        results = {}
        for mode in modes:
            st = runner.run_select_mode(
                question,
                enc,
                mode=mode,
                budget_k=budget_k,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                keep_ratio=args.keep_ratio,
                seed=args.seed,
            )
            results[mode] = st
            kept = (st.extra or {}).get("kept_patch_indices", [])
            print(
                f"  [{mode}] n_patches={st.n_patches} n_vis≈{st.n_vis_tokens} "
                f"wall_s={st.wall_s:.2f} kept_n={len(kept)}"
            )
            print(f"         answer={st.answer[:200]!r}")

        toks = [results[m].n_vis_tokens for m in modes]
        patches = [results[m].n_patches for m in modes]
        same_tok = len(set(toks)) == 1
        same_pat = len(set(patches)) == 1
        kp = {
            m: (results[m].extra or {}).get("kept_patch_indices", []) for m in modes
        }
        j_qr = _jaccard(kp["queryip"], kp["random"])
        j_qp = _jaccard(kp["queryip"], kp["prefix"])
        j_rp = _jaccard(kp["random"], kp["prefix"])
        answered = all(bool(results[m].answer) for m in modes)
        differ = j_qr < 0.95 and j_qp < 0.95
        print(
            f"  check same_tokens={same_tok} ({toks}) same_patches={same_pat} ({patches})"
        )
        print(f"  jaccard queryip∩random={j_qr:.3f} queryip∩prefix={j_qp:.3f} random∩prefix={j_rp:.3f}")
        print(f"  answered={answered} differ_from_baselines={differ}")
        if not (same_tok and same_pat and answered and differ):
            ok_all = False

    print("\n===== SAMEK EFFECT CHECK =====")
    if ok_all:
        print(
            "SAMEK_OK: equal budget; QueryIP kept-set differs from prefix/random; answers OK."
        )
        return 0
    print(
        "SAMEK_SOFT: ran end-to-end but success criteria weak — inspect overlaps/answers.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
