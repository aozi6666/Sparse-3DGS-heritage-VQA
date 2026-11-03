#!/usr/bin/env python
"""OCRBench_v2 image same-B Acc (full / random / image textseat).

Incremental jsonl + resume by ``index``. Agent does not execute Acc — see COMMANDS.md §10.
Images: prefer $LMUData/images/OCRBench_v2/{index}.png; else decode TSV base64 (慢).
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _lmu_root() -> Path:
    env = os.environ.get("LMUData")
    if env:
        return Path(env)
    return Path.home() / "LMUData"


def _load_done(jsonl: Path, modes: tuple[str, ...]) -> tuple[dict[int, dict], dict[str, int]]:
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
            if any(f"{m}_pred" not in rec for m in modes):
                print(f"[warn] incomplete index={idx}, will re-run", file=sys.stderr)
                continue
            done[idx] = rec
            for m in modes:
                correct[m] += int(bool(rec.get(f"{m}_hit")))
    return done, correct


def _parse_answers(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if pd.isna(raw):
        return []
    s = str(raw)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    except Exception:
        return [s]


def _hit(pred: str, answers: list[str]) -> int:
    p = (pred or "").strip().lower()
    if not p:
        return 0
    for a in answers:
        a_n = str(a).strip().lower()
        if not a_n:
            continue
        if a_n == p or a_n in p or p in a_n:
            return 1
    return 0


def _try_open(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _load_image(
    index: int,
    image_field,
    img_root: Path,
    image_path=None,
) -> Image.Image | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        im = _try_open(img_root / f"{index}{ext}")
        if im is not None:
            return im

    for cand_raw in (image_path, image_field):
        if not isinstance(cand_raw, str) or not cand_raw.strip():
            continue
        s = cand_raw.strip()
        # absolute / relative filesystem path (not jpeg base64)
        if not s.startswith("/9j") and len(s) < 4096:
            p = Path(s)
            im = _try_open(p if p.is_absolute() else img_root / s)
            if im is not None:
                return im

    # TSV embedded base64
    if isinstance(image_field, str) and len(image_field) > 64:
        try:
            raw = base64.b64decode(image_field)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    return None


def _write_summary(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(path)


def main() -> int:
    root = _repo_root()
    default_tsv = root / "VQA" / "datas" / "ORCBench_v2" / "OCRBench_v2.tsv"
    default_model = root / "VQA" / "mage_vl" / "models"
    out_dir = root / "VQA" / "eval_results" / "queryip_samek" / "ocrbench_v2_image_textseat"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", type=Path, default=None, help="explicit TSV wins; else LMU local/default")
    ap.add_argument("--model", type=Path, default=default_model)
    ap.add_argument("--n", type=int, default=200, help="first N rows; 0 = all")
    ap.add_argument("--keep-ratio", type=float, default=0.75)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=150000)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--outdir", type=Path, default=out_dir)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--modes",
        default="full_image,random_tiles,queryip_image_textseat,queryip_image_textseat_shuf",
    )
    ap.add_argument("--use-ocr", action="store_true", help="EasyOCR tile seats")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--img-root",
        type=Path,
        default=None,
        help="image dir; default: LMU images/OCRBench or OCRBench_v2 by TSV name",
    )
    args = ap.parse_args()

    lmu = _lmu_root()
    # Explicit --tsv always wins (do NOT silently swap to v2_local).
    if args.tsv is not None:
        tsv_path = args.tsv
    else:
        local_tsv = lmu / "OCRBench_v2_local.tsv"
        tsv_path = local_tsv if local_tsv.is_file() else default_tsv
        if not tsv_path.is_file():
            alt = lmu / "OCRBench_v2.tsv"
            if alt.is_file():
                tsv_path = alt
    if not tsv_path.is_file():
        print(f"[error] missing tsv {tsv_path}", file=sys.stderr)
        return 2

    name = tsv_path.name.lower()
    if args.img_root is not None:
        img_root = args.img_root
    elif "ocrbench_v2" in name:
        img_root = lmu / "images" / "OCRBench_v2"
    else:
        # OCRBench.tsv (v1) → images/OCRBench/{i}.png ; else still try v2 then v1
        v1 = lmu / "images" / "OCRBench"
        img_root = v1 if v1.is_dir() else (lmu / "images" / "OCRBench_v2")
    print(f"[ocrbench] tsv={tsv_path} img_root={img_root}")

    # Load without forcing all base64 into memory if localized.
    df = pd.read_csv(tsv_path, sep="\t")
    if int(args.n) > 0:
        df = df.head(int(args.n)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if len(df) == 0:
        print("[error] empty dataframe", file=sys.stderr)
        return 2

    ets = str(root / "VQA" / "Explore-Then-Select")
    models = str(root / "VQA" / "mage_vl" / "models")
    for p in (ets, models):
        if p not in sys.path:
            sys.path.insert(0, p)

    from mage_query_ip.image_mage_forward import ImageMageRunner
    from mage_query_ip.image_patch_select import select_image_tiles
    from mage_query_ip.image_textseat import need_text_evidence

    allowed = {
        "full_image",
        "random_tiles",
        "queryip_image_textseat",
        "queryip_image_textseat_shuf",
        "queryip_image_texture",
    }
    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    if not modes or any(m not in allowed for m in modes):
        print(f"[error] bad --modes={args.modes!r}; allowed={sorted(allowed)}", file=sys.stderr)
        return 2

    ocr_indexer = None
    if args.use_ocr:
        from mage_query_ip.image_ocr import ImageOCRIndexer

        ocr_indexer = ImageOCRIndexer()

    runner = ImageMageRunner(args.model, local_files_only=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if str(args.tag).strip() else ""
    out_jsonl = args.outdir / f"Mage-VL-ImageQueryIP_ocrbench_v2{suffix}.jsonl"
    summary_path = args.outdir / f"Mage-VL-ImageQueryIP_ocrbench_v2_summary{suffix}.json"

    done, correct = ({}, {m: 0 for m in modes})
    if args.resume:
        done, correct = _load_done(out_jsonl, modes)
        if done:
            print(f"[resume] loaded {len(done)} rows from {out_jsonl}")

    questions = [str(x) for x in df["question"].tolist()]
    n_batch = len(questions)
    write_mode = "a" if (args.resume and out_jsonl.is_file() and done) else "w"
    n_target = len(df)
    n_skip0 = sum(1 for _, r in df.iterrows() if int(r["index"]) in done)
    total = n_skip0
    pbar = tqdm(
        total=n_target,
        initial=n_skip0,
        desc="ocrbench_samek",
        unit="q",
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stderr,
    )

    with out_jsonl.open(write_mode, encoding="utf-8") as fout:
        for pos_i, (_, row) in enumerate(df.iterrows()):
            q_index = int(row["index"])
            if q_index in done:
                continue

            img = _load_image(
                q_index,
                row.get("image"),
                img_root,
                image_path=row.get("image_path"),
            )
            if img is None:
                print(f"[warn] missing image index={q_index}", file=sys.stderr)
                pbar.update(1)
                continue

            q = questions[pos_i]
            shuf_q = questions[(pos_i + 1) % n_batch] if n_batch else q
            answers = _parse_answers(row.get("answer"))
            category = str(row.get("category", ""))

            ocr_idx: list[int] | None = None
            if ocr_indexer is not None:
                ocr_idx = ocr_indexer.propose_tile_indices(img, patch=int(args.patch))

            rec = {
                "index": q_index,
                "question": q,
                "answers": answers,
                "category": category,
                "modes": list(modes),
                "keep_ratio": float(args.keep_ratio),
                "patch": int(args.patch),
                "use_ocr": bool(args.use_ocr),
                "n_ocr_tiles": int(len(ocr_idx or [])),
            }

            for mode in modes:
                need_text = True
                seat_q = q
                if mode == "queryip_image_textseat":
                    need_text = need_text_evidence(q)
                    seat_q = q
                elif mode == "queryip_image_textseat_shuf":
                    need_text = need_text_evidence(shuf_q)
                    seat_q = shuf_q

                # OCR seats only when need_text (shuf gate); detection still image-only.
                ocr_for_mode = ocr_idx if need_text else []

                sel = select_image_tiles(
                    img,
                    mode=mode,
                    keep_ratio=float(args.keep_ratio),
                    seed=int(args.seed) + q_index,
                    patch=int(args.patch),
                    cell=int(args.cell),
                    ocr_tile_indices=ocr_for_mode,
                    need_text=need_text,
                )
                st = runner.generate_image(
                    sel.image,
                    q,  # always answer the true question
                    mode=mode,
                    max_new_tokens=int(args.max_new_tokens),
                    max_pixels=int(args.max_pixels),
                    extra={
                        "budget_k": sel.budget_k,
                        "n_tiles": sel.n_tiles,
                        "need_text": need_text,
                        "seat_q": seat_q[:120],
                        **(sel.knobs or {}),
                    },
                )
                hit = _hit(st.answer, answers)
                correct[mode] += hit
                rec[f"{mode}_pred"] = st.answer[:500]
                rec[f"{mode}_hit"] = hit
                rec[f"{mode}_n_vis"] = st.n_vis_tokens
                rec[f"{mode}_budget_k"] = sel.budget_k
                rec[f"{mode}_n_tiles"] = sel.n_tiles
                if mode.startswith("queryip_image_textseat"):
                    rec[f"{mode}_need_text"] = need_text
                    rec[f"{mode}_n_text_seats"] = (sel.knobs or {}).get("n_text_seats")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[q_index] = rec
            total += 1
            pred_s = " ".join(f"{m}={rec[f'{m}_hit']}" for m in modes)
            tqdm.write(f"[{total}/{n_target}] idx={q_index} {pred_s}")
            pbar.set_postfix_str(
                " ".join(f"{m}={correct[m]/max(1,total):.0%}" for m in modes),
                refresh=False,
            )
            pbar.update(1)
            _write_summary(
                summary_path,
                {
                    "n": total,
                    "keep_ratio": float(args.keep_ratio),
                    "seed": int(args.seed),
                    "modes": list(modes),
                    "tag": str(args.tag) or None,
                    "use_ocr": bool(args.use_ocr),
                    "acc": {m: correct[m] / max(1, total) for m in modes},
                    "correct": correct,
                    "jsonl": str(out_jsonl),
                    "partial": True,
                    "metric": "substring_or_exact_vs_answer_list",
                },
            )

    pbar.close()
    expected = {int(x) for x in df["index"].tolist()}
    complete = expected.issubset(set(done.keys()))
    _write_summary(
        summary_path,
        {
            "n": total,
            "keep_ratio": float(args.keep_ratio),
            "seed": int(args.seed),
            "modes": list(modes),
            "tag": str(args.tag) or None,
            "use_ocr": bool(args.use_ocr),
            "acc": {m: correct[m] / max(1, total) for m in modes},
            "correct": correct,
            "jsonl": str(out_jsonl),
            "partial": not complete,
            "done": len(done),
            "target": len(df),
            "metric": "substring_or_exact_vs_answer_list",
        },
    )
    print("\n===== OCRBENCH_V2 IMAGE SAMEK =====")
    print(f"n={total} done={len(done)}/{len(df)} jsonl={out_jsonl}")
    for m in modes:
        print(f"  {m}: {correct[m]}/{total} = {correct[m]/max(1,total):.3f}")
    print(
        "Note: running Acc is a proxy match; official EN/CN Overall uses "
        "VLMEvalKit OCRBench_v2.evaluate on Mage-VL baseline preds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
