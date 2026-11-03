#!/usr/bin/env python
"""Video-MME short first-N same-K Acc (modes include queryip_textseat).

Incremental jsonl + resume by TSV ``index`` (disk-light: no codec dump).
Mid-wide pool: raise ``--max-pixels`` (e.g. 300000) so N_wide grows; all modes
cut to the same B. Agent does not execute Acc — see COMMANDS.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable


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


def _load_done(jsonl: Path, modes: tuple[str, ...]) -> tuple[dict[int, dict], dict[str, int]]:
    """Return {index: row} and correct counts; skip bad / mode-mismatched lines."""
    done: dict[int, dict] = {}
    correct = {m: 0 for m in modes}
    if not jsonl.is_file():
        return done, correct
    with jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skip bad jsonl L{line_no}", file=sys.stderr)
                continue
            if "index" not in rec:
                continue
            try:
                idx = int(rec["index"])
            except (TypeError, ValueError):
                continue
            # Require all mode preds present (partial row → re-run).
            if any(f"{m}_pred" not in rec for m in modes):
                print(f"[warn] incomplete index={idx}, will re-run", file=sys.stderr)
                continue
            done[idx] = rec
            gt = str(rec.get("gt", "")).strip().upper()[:1]
            for m in modes:
                pred = str(rec.get(f"{m}_pred", "")).strip().upper()[:1]
                correct[m] += int(pred == gt and gt in "ABCD")
    return done, correct


def _write_summary(
    path: Path,
    *,
    total: int,
    keep_ratio: float,
    seed: int,
    modes: tuple[str, ...],
    tag: str,
    correct: dict[str, int],
    jsonl: Path,
    max_pixels: int | None = None,
    num_frames: int | None = None,
) -> None:
    summary = {
        "n": total,
        "keep_ratio": float(keep_ratio),
        "seed": int(seed),
        "modes": list(modes),
        "tag": str(tag) or None,
        "max_pixels": int(max_pixels) if max_pixels is not None else None,
        "num_frames": int(num_frames) if num_frames is not None else None,
        "acc": {m: correct[m] / max(1, total) for m in modes},
        "correct": correct,
        "jsonl": str(jsonl),
        "partial": True,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(path)


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
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=150000,
        help="codec pixel budget; mid-wide textseat smoke uses 300000",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="hevc")
    ap.add_argument("--outdir", type=Path, default=out_dir)
    ap.add_argument(
        "--modes",
        default="prefix,random,queryip,queryip_bitcost",
        help="comma-separated select modes (textseat: random,queryip,queryip_textseat,queryip_textseat_shuf)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="optional suffix for output names, e.g. n200_kr050",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from existing jsonl by question index (default: on)",
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
    allowed = {
        "prefix",
        "random",
        "queryip",
        "queryip_dyn",
        "queryip_sta",
        "queryip_bitcost",
        "queryip_clip",
        "queryip_region",
        "queryip_region_shuf",
        "queryip_edge",
        "queryip_edge_shuf",
        "queryip_textseat",
        "queryip_textseat_shuf",
    }
    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    if not modes or any(m not in allowed for m in modes):
        print(f"[error] bad --modes={args.modes!r}; allowed={sorted(allowed)}", file=sys.stderr)
        return 2
    print(
        f"[short60] modes={modes} keep_ratio={args.keep_ratio} "
        f"max_pixels={args.max_pixels} resume={args.resume}"
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if str(args.tag).strip() else ""
    out_jsonl = args.outdir / f"Mage-VL-QueryIP_short60_samek4{suffix}.jsonl"
    summary_path = args.outdir / f"Mage-VL-QueryIP_short60_samek4_summary{suffix}.json"

    done: dict[int, dict] = {}
    correct = {m: 0 for m in modes}
    if args.resume:
        done, correct = _load_done(out_jsonl, modes)
        if done:
            print(f"[resume] loaded {len(done)} done rows from {out_jsonl}")

    # RAM only; never dump codec to disk (省盘). Keep at most one video.
    cache_enc: dict[str, dict] = {}
    last_key: str | None = None

    # Pre-build prompts for batch circular shuffle (题 i ← 题 i+1).
    rows = list(short.iterrows())
    prompts: list[str] = []
    for _, row in rows:
        q = str(row["question"])
        cands = row["candidates"]
        if isinstance(cands, str):
            try:
                cands = eval(cands, {})
            except Exception:
                cands = [cands]
        opt_txt = "\n".join(str(c) for c in cands)
        prompts.append(f"{q}\nOptions:\n{opt_txt}\nAnswer with the letter only.")
    n_batch = len(prompts)
    use_shuf = any(
        m in modes
        for m in (
            "queryip_region_shuf",
            "queryip_edge_shuf",
            "queryip_textseat_shuf",
        )
    )
    if use_shuf and n_batch > 0:
        print(f"[short60] shuf modes: circular shift over n_batch={n_batch}")

    # Append mode if resuming with existing file; else create fresh.
    write_mode = "a" if (args.resume and out_jsonl.is_file() and done) else "w"
    if write_mode == "w" and out_jsonl.is_file() and not args.resume:
        print(f"[warn] --no-resume: overwriting {out_jsonl}", file=sys.stderr)

    n_target = len(short)
    n_skip0 = sum(1 for _, r in short.iterrows() if int(r["index"]) in done)
    total = n_skip0  # resumed rows already in Acc via _load_done
    pbar = tqdm(
        total=n_target,
        initial=n_skip0,
        desc="samek",
        unit="q",
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stderr,
    )

    with out_jsonl.open(write_mode, encoding="utf-8") as fout:
        for pos_i, (_, row) in enumerate(rows):
            try:
                q_index = int(row["index"])
            except (KeyError, TypeError, ValueError):
                print("[error] TSV missing usable index", file=sys.stderr)
                return 2

            if q_index in done:
                continue

            vid = str(row["video"])
            vpath = video_root / "video" / f"{vid}.mp4"
            if not vpath.is_file():
                alt = video_root / str(row.get("video_path", "")).lstrip("./")
                vpath = alt if alt.is_file() else vpath
            if not vpath.is_file():
                print(f"[warn] missing video {vid}", file=sys.stderr)
                pbar.update(1)
                continue
            key = str(vpath)
            if last_key is not None and key != last_key:
                cache_enc.pop(last_key, None)
            if key not in cache_enc:
                cache_enc[key] = runner.encode_codec(
                    key,
                    num_frames=args.num_frames,
                    max_pixels=args.max_pixels,
                    engine=args.engine,
                )
            last_key = key
            enc = cache_enc[key]
            n_wide = int(enc["n_patches"])
            budget_k = max(1, int(round(n_wide * float(args.keep_ratio))))
            gt = str(row["answer"]).strip().upper()[:1]
            q = str(row["question"])
            prompt = prompts[pos_i]
            shuf_prompt = prompts[(pos_i + 1) % n_batch] if n_batch else prompt

            rec = {
                "index": q_index,
                "video": vid,
                "gt": gt,
                "question": q,
                "budget_k": budget_k,
                "n_wide": n_wide,
                "max_pixels": int(args.max_pixels),
                "modes": list(modes),
            }
            for mode in modes:
                sel_kw: dict = {}
                if mode in (
                    "queryip_region_shuf",
                    "queryip_edge_shuf",
                    "queryip_textseat_shuf",
                ):
                    sel_kw["select_question"] = shuf_prompt
                st = runner.run_select_mode(
                    prompt,
                    enc,
                    mode=mode,
                    budget_k=budget_k,
                    max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                    keep_ratio=args.keep_ratio,
                    seed=args.seed,
                    **sel_kw,
                )
                pred = _letter(st.answer)
                hit = int(pred == gt and gt in "ABCD")
                correct[mode] += hit
                rec[f"{mode}_pred"] = pred
                rec[f"{mode}_n_vis"] = st.n_vis_tokens
                rec[f"{mode}_score_source"] = (st.extra or {}).get("score_source")
                rec[f"{mode}_answer"] = st.answer[:200]
                if mode in ("queryip_textseat", "queryip_textseat_shuf"):
                    ex = st.extra or {}
                    rec[f"{mode}_need_text"] = bool(ex.get("need_text"))
                    rec[f"{mode}_n_text_seats"] = ex.get("n_text_seats")
                    rec[f"{mode}_n_ocr_seats"] = ex.get("n_ocr_seats")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[q_index] = rec
            total += 1
            pred_s = " ".join(f"{m}={rec[f'{m}_pred']}" for m in modes)
            tqdm.write(
                f"[{total}/{n_target}] idx={q_index} {vid} "
                f"n_wide={n_wide} B={budget_k} gt={gt} {pred_s}"
            )
            pbar.set_postfix_str(
                " ".join(f"{m}={correct[m]/max(1,total):.0%}" for m in modes),
                refresh=False,
            )
            pbar.update(1)
            # Lightweight running summary (overwrite tiny json only).
            _write_summary(
                summary_path,
                total=total,
                keep_ratio=args.keep_ratio,
                seed=args.seed,
                modes=modes,
                tag=str(args.tag),
                correct=correct,
                jsonl=out_jsonl,
                max_pixels=int(args.max_pixels),
                num_frames=int(args.num_frames),
            )
    pbar.close()

    # Final summary: mark complete if all short rows present in done.
    expected = {int(x) for x in short["index"].tolist()}
    complete = expected.issubset(set(done.keys()))
    summary = {
        "n": total,
        "keep_ratio": float(args.keep_ratio),
        "seed": int(args.seed),
        "modes": list(modes),
        "tag": str(args.tag) or None,
        "max_pixels": int(args.max_pixels),
        "num_frames": int(args.num_frames),
        "acc": {m: correct[m] / max(1, total) for m in modes},
        "correct": correct,
        "jsonl": str(out_jsonl),
        "partial": not complete,
        "done": len(done),
        "target": len(short),
    }
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(summary_path)

    print("\n===== SHORT60 SAMEK4 ACC =====")
    print(f"n={total} done={len(done)}/{len(short)} jsonl={out_jsonl}")
    print(f"summary={summary_path} partial={not complete}")
    for m in modes:
        acc = correct[m] / max(1, total)
        print(f"  {m}: {correct[m]}/{total} = {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
