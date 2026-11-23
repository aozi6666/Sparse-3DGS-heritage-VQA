"""Offline VideoMME inference using native Mage HEVC canvases and AKS evidence."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
import sqlite3
import tempfile
import time
from tqdm import tqdm
from fusion import fuse
from itertools import groupby
from pathlib import Path
from unittest.mock import patch

from prepare import DATA, HERE, letter

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL = HERE.parent / "mage_vl/models"
PREFIX = ("These are the frames of a video. Select the best answer to the following multiple-choice "
          "question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.")


def inputs_for(proc, codec, payload, video, q, cfg):
    prompt = PREFIX + "\n\nQuestion: " + q["question"] + "\n" + "\n".join(q["options"]) + "\nAnswer:"
    messages = [{"role":"user","content":[{"type":"video"},{"type":"text","text":prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Inject only this question's in-memory payload into the original processor.
    with patch.object(codec, "process_codec_video", return_value=payload) as source:
        result = proc(text=[text], videos=[str(video)], video_backend="codec", codec_config=cfg,
                    max_pixels=150000, padding=True, return_tensors="pt")
    source.assert_called_once()
    n_patches = int(result["image_grid_thw"].prod(dim=1).sum())
    n_slots = int((result["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    if n_slots * 4 != n_patches or result["patch_positions"].shape[0] != n_patches:
        raise ValueError(f"Visual token alignment failed for {q['id']}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="CPU preprocess one video and its questions")
    p.add_argument("--video", action="append", help="Evaluate only this video ID, reusing the full-run checkpoint")
    args = p.parse_args()
    questions = json.loads((HERE/"questions.json").read_text())
    out = HERE/"runs"
    out.mkdir(exist_ok=True)
    lock = (out/"run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = hashlib.sha256()
    for name in ("evaluate.py", "fusion.py", "prepare.py", "questions.json"):
        digest.update((HERE / name).read_bytes())
    signature = digest.hexdigest()[:16]
    db = sqlite3.connect(":memory:" if args.check else out/f"{signature}.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY, record TEXT)")
    done = {r[0] for r in db.execute("SELECT id FROM results")}
    selected = questions
    if args.video:
        wanted = set(args.video)
        unknown = wanted.difference(q["video"] for q in questions)
        if unknown:
            p.error(f"Unknown video IDs: {sorted(unknown)}")
        selected = [q for q in questions if q["video"] in wanted]
    completed = sum(q["id"] in done for q in selected)
    if not args.check:
        print(f"Checkpoint {out / (signature + '.sqlite')} | saved {len(done)}/{len(questions)} | selected {completed}/{len(selected)}", flush=True)
    if args.check or completed < len(selected):
        proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        codec = importlib.import_module(".codec_video_processing_mage_vl", type(proc).__module__.rsplit(".",1)[0])
        model = None
        if not args.check:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU required for inference. Use --check for CPU verification.")
            model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True,
                                                         torch_dtype="auto", device_map="auto").eval()
    progress = tqdm(total=len(selected), initial=completed, desc="VideoMME", unit="q", dynamic_ncols=True, disable=args.check)
    for vid, group in groupby(selected,key=lambda q:q["video"]):
        pending = [q for q in group if q["id"] not in done]
        if not pending:
            continue
        started = time.monotonic()
        progress.set_postfix_str(f"{pending[0]['duration']} {vid}")
        video = DATA/"video"/(vid+".mp4")
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            cfg = dict(engine="hevc",target_canvas=32,patch=16,cache_root=Path(tmp))
            settings = {**proc._codec_config_defaults, **cfg, "max_pixels":150000}
            native = codec.process_codec_video(str(video),codec.CodecConfig(**settings))
            cap = cv2.VideoCapture(str(video))
            try:
                for q in pending:
                    payload, anchors, pairs = fuse(native,cap,q["anchors"])
                    assert [v["aks"] for v in pairs] == q["expected"], q["id"]
                    inputs = inputs_for(proc,codec,payload,video,q,cfg)
                    if args.check:
                        original = inputs_for(proc,codec,native,video,q,cfg)
                        assert torch.equal(inputs["image_grid_thw"],original["image_grid_thw"])
                        assert inputs["pixel_values"].shape == original["pixel_values"].shape
                        assert not anchors or not torch.equal(inputs["pixel_values"],original["pixel_values"])
                        print(q["id"], anchors, list(inputs["pixel_values"].shape),flush=True)
                        continue
                    inputs = inputs.to(model.device)
                    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
                    with torch.inference_mode():
                        tokens = model.generate(**inputs,max_new_tokens=256,do_sample=False)
                    raw = proc.tokenizer.decode(tokens[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                    pred = letter(raw)
                    record = dict(id=q["id"],raw=raw,pred=pred,answer=q["answer"],correct=pred==q["answer"],
                                  baseline=q["baseline"],duration=q["duration"],anchors=anchors,pairs=pairs,aks_correct=q["aks_correct"])
                    with db:
                        db.execute("INSERT INTO results VALUES (?,?)",(q["id"],json.dumps(record)))
                    done.add(q["id"])
                    progress.update(1)
            finally:
                cap.release()
        if args.check:
            return
        tqdm.write(f"{vid} {pending[0]['duration']}: {len(pending)} questions in {time.monotonic()-started:.1f}s")
    progress.close()
    if len(done) < len(questions):
        db.close()
        return
    records = [json.loads(v) for v, in db.execute("SELECT record FROM results ORDER BY id")]
    assert len(records) == len(questions)
    (out/f"{signature}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    summary = {}
    for name in ["long"]:
        rows = [r for r in records if name == "overall" or r["duration"] == name]
        summary[name] = dict(aks_correct=sum(r["aks_correct"] for r in rows), midpoint_only=sum(r["correct"] and not r["aks_correct"] for r in rows), aks_only=sum(r["aks_correct"] and not r["correct"] for r in rows), questions=len(rows), correct=sum(r["correct"] for r in rows),
                             accuracy=sum(r["correct"] for r in rows)/len(rows),
                             gains=sum(r["correct"] and not r["baseline"] for r in rows),
                             losses=sum(not r["correct"] and r["baseline"] for r in rows),
                             unparsed=sum(r["pred"] is None for r in rows))
    report = json.dumps(summary, indent=2)
    (out/f"{signature}_summary.json").write_text(report+"\n")
    print(report, flush=True)
    db.close()

if __name__ == "__main__":
    main()
