#!/usr/bin/env python
"""Video-MME short first-60 same-K Acc (four modes).

Modes: prefix / random / queryip (proxy) / queryip_bitcost.
Run only AFTER soccer SAMEK4_OK. Agent does not execute — see COMMANDS.md.

Writes ``VQA/eval_results/Mage-VL-QueryIP_short60_samek4.jsonl`` (and a
legacy-compatible ``..._short60_samek.jsonl`` symlink-style copy of the same
rows for older notes).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _letter(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    if t[:1] in "ABCD":
        return t[:1]
    return ""


def main() -> int:
    root = _repo_root()
    default_tsv = root / "VQA" / "mage_vl" / "data" / "VideoMME" / "Video-MME.tsv"
    default_model = root / "VQA" / "mage_vl" / "models"
    out_dir = root / "VQA" / "eval_results"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", type=Path, default=default_tsv)
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--n", type=int, default=60, help="first N short questions")
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="hevc")
    ap.add_argument("--outdir", type=Path, default=out_dir)
    ap.add_argument(
        "--modes",
        default="prefix,random,queryip,queryip_bitcost",
        help="comma-separated select modes (proxy sweep: random,queryip)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="optional suffix for output names, e.g. n200_kr050",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    short = df[df["duration"] == "short"].head(int(args.n)).reset_index(drop=True)
    if len(short) == 0:
        print("[error] no short rows", file=sys.stderr)
        return 2
    video_root = args.tsv.parent
    print(f"[short60] n={len(short)} videos={short['video'].nunique()}")

    ets = str(root / "VQA" / "Explore-Then-Select")
    models = str(root / "VQA" / "mage_vl" / "models")
    for p in (ets, models):
        if p not in sys.path:
            sys.path.insert(0, p)

    from mage_query_ip.mage_forward import MageQueryIPRunner

    runner = MageQueryIPRunner(args.model, local_files_only=True)
    allowed = {"prefix", "random", "queryip", "queryip_bitcost"}
    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    if not modes or any(m not in allowed for m in modes):
        print(f"[error] bad --modes={args.modes!r}; allowed={sorted(allowed)}", file=sys.stderr)
        return 2
    print(f"[short60] modes={modes} keep_ratio={args.keep_ratio}")
    correct = {m: 0 for m in modes}
    total = 0
    rows_out = []
    cache_enc: dict[str, dict] = {}

    for _, row in short.iterrows():
        vid = str(row["video"])
        vpath = video_root / "video" / f"{vid}.mp4"
        if not vpath.is_file():
            alt = video_root / str(row.get("video_path", "")).lstrip("./")
            vpath = alt if alt.is_file() else vpath
        if not vpath.is_file():
            print(f"[warn] missing video {vid}", file=sys.stderr)
            continue
        key = str(vpath)
        if key not in cache_enc:
            cache_enc[key] = runner.encode_codec(
                key,
                num_frames=args.num_frames,
                max_pixels=args.max_pixels,
                engine=args.engine,
            )
        enc = cache_enc[key]
        budget_k = max(1, int(round(enc["n_patches"] * float(args.keep_ratio))))
        gt = str(row["answer"]).strip().upper()[:1]
        q = str(row["question"])
        cands = row["candidates"]
        if isinstance(cands, str):
            try:
                cands = eval(cands, {})
            except Exception:
                cands = [cands]
        opt_txt = "\n".join(str(c) for c in cands)
        prompt = f"{q}\nOptions:\n{opt_txt}\nAnswer with the letter only."

        total += 1
        rec = {
            "video": vid,
            "gt": gt,
            "question": q,
            "budget_k": budget_k,
            "modes": list(modes),
        }
        for mode in modes:
            st = runner.run_select_mode(
                prompt,
                enc,
                mode=mode,
                budget_k=budget_k,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                keep_ratio=args.keep_ratio,
                seed=args.seed,
            )
            pred = _letter(st.answer)
            hit = int(pred == gt and gt in "ABCD")
            correct[mode] += hit
            rec[f"{mode}_pred"] = pred
            rec[f"{mode}_n_vis"] = st.n_vis_tokens
            rec[f"{mode}_score_source"] = (st.extra or {}).get("score_source")
            rec[f"{mode}_answer"] = st.answer[:200]
        rows_out.append(rec)
        print(
            f"[{total}/{len(short)}] {vid} gt={gt} "
            + " ".join(f"{m}={rec[f'{m}_pred']}" for m in modes)
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if str(args.tag).strip() else ""
    out_jsonl = args.outdir / f"Mage-VL-QueryIP_short60_samek4{suffix}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also write a compact summary for PLAN §5.5 fill-in.
    summary = {
        "n": total,
        "keep_ratio": float(args.keep_ratio),
        "seed": int(args.seed),
        "modes": list(modes),
        "tag": str(args.tag) or None,
        "acc": {m: correct[m] / max(1, total) for m in modes},
        "correct": correct,
        "jsonl": str(out_jsonl),
    }
    summary_path = args.outdir / f"Mage-VL-QueryIP_short60_samek4_summary{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SHORT60 SAMEK4 ACC =====")
    print(f"n={total} jsonl={out_jsonl}")
    print(f"summary={summary_path}")
    for m in modes:
        acc = correct[m] / max(1, total)
        print(f"  {m}: {correct[m]}/{total} = {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
