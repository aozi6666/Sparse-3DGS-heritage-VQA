#!/usr/bin/env python
"""Soccer same-K ablation: prefix / random / queryip / queryip_bitcost.

Success (open-ended, no Acc): equal n_vis_tokens; queryip_bitcost kept-set differs
from queryip (Jaccard < 1); answers non-empty.
Writes ``VQA/eval_results/Mage-VL-QueryIP_soccer_samek4.json``.
Agent does not run this — see COMMANDS.md.
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
    out_default = root / "VQA" / "eval_results" / "Mage-VL-QueryIP_soccer_samek4.json"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--video", type=Path, default=default_video)
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--engine", default="hevc", choices=("hevc", "dcvc-rt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=out_default.parent)
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

    print(f"[samek4] model={args.model}")
    print(f"[samek4] video={args.video}")
    print(f"[samek4] keep_ratio={args.keep_ratio} seed={args.seed}")

    runner = MageQueryIPRunner(args.model, local_files_only=True)
    enc = runner.encode_codec(
        str(args.video),
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        engine=args.engine,
    )
    budget_k = max(1, int(round(enc["n_patches"] * float(args.keep_ratio))))
    print(
        f"[samek4] codec n_patches={enc['n_patches']} n_canvas={enc['n_canvas']} "
        f"budget_k={budget_k}"
    )

    questions = [
        ("static", args.static_q),
        ("dynamic", args.dynamic_q),
    ]
    modes = ("prefix", "random", "queryip", "queryip_bitcost")
    report: dict = {
        "video": str(args.video),
        "keep_ratio": float(args.keep_ratio),
        "budget_k": int(budget_k),
        "n_patches_codec": int(enc["n_patches"]),
        "seed": int(args.seed),
        "modes": list(modes),
        "questions": {},
    }
    ok_all = True
    redundant_any = False
    fail_any = False

    for tag, question in questions:
        qtype = classify_question(question)
        print(f"\n===== Q={tag} classify={qtype.value} =====")
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
            src = (st.extra or {}).get("score_source", "")
            print(
                f"  [{mode}] n_patches={st.n_patches} n_vis≈{st.n_vis_tokens} "
                f"score_source={src} wall_s={st.wall_s:.2f} kept_n={len(kept)}"
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
        j_br = _jaccard(kp["queryip_bitcost"], kp["random"])
        j_bp = _jaccard(kp["queryip_bitcost"], kp["prefix"])
        j_bq = _jaccard(kp["queryip_bitcost"], kp["queryip"])
        answered = all(bool(results[m].answer) for m in modes)
        differ_base = j_qr < 0.95 and j_qp < 0.95
        bitcost_differs = j_bq < 1.0
        print(
            f"  check same_tokens={same_tok} ({toks}) same_patches={same_pat} ({patches})"
        )
        print(
            f"  jaccard queryip∩random={j_qr:.3f} queryip∩prefix={j_qp:.3f} "
            f"bitcost∩queryip={j_bq:.3f} bitcost∩random={j_br:.3f}"
        )
        print(
            f"  answered={answered} differ_baselines={differ_base} "
            f"bitcost_differs_proxy={bitcost_differs}"
        )

        q_rec = {
            "tag": tag,
            "qtype": qtype.value,
            "question": question,
            "n_vis": {m: results[m].n_vis_tokens for m in modes},
            "n_patches": {m: results[m].n_patches for m in modes},
            "score_source": {
                m: (results[m].extra or {}).get("score_source") for m in modes
            },
            "mean_score_kept": {
                m: (results[m].extra or {}).get("mean_motion_kept") for m in modes
            },
            "jaccard": {
                "queryip_random": j_qr,
                "queryip_prefix": j_qp,
                "bitcost_queryip": j_bq,
                "bitcost_random": j_br,
                "bitcost_prefix": j_bp,
            },
            "answers": {m: results[m].answer[:200] for m in modes},
            "same_tokens": same_tok,
            "same_patches": same_pat,
            "answered": answered,
            "bitcost_differs_proxy": bitcost_differs,
        }
        report["questions"][tag] = q_rec

        if not (same_tok and same_pat and answered):
            fail_any = True
            ok_all = False
        elif not bitcost_differs:
            redundant_any = True
            ok_all = False
        elif not differ_base:
            ok_all = False

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "Mage-VL-QueryIP_soccer_samek4.json"
    if fail_any:
        verdict = "SAMEK4_FAIL"
    elif redundant_any:
        verdict = "SAMEK4_REDUNDANT"
    elif ok_all:
        verdict = "SAMEK4_OK"
    else:
        verdict = "SAMEK4_SOFT"
    report["verdict"] = verdict
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[samek4] wrote {out_path}")

    print("\n===== SAMEK4 EFFECT CHECK =====")
    if verdict == "SAMEK4_OK":
        print(
            "SAMEK4_OK: equal budget; bitcost kept differs from proxy; "
            "answers OK."
        )
        return 0
    if verdict == "SAMEK4_REDUNDANT":
        print(
            "SAMEK4_REDUNDANT: bitcost kept-set == proxy (no new signal).",
            file=sys.stderr,
        )
        return 0
    if verdict == "SAMEK4_FAIL":
        print(
            "SAMEK4_FAIL: token mismatch or empty answers — inspect log/json.",
            file=sys.stderr,
        )
        return 1
    print(
        "SAMEK4_SOFT: ran end-to-end but baselines weak — inspect overlaps/answers.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
