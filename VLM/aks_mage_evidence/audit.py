"""Reconstruct the replaced canvases from the frozen offline VideoMME experiment."""
import argparse
import base64
from collections import defaultdict
import fcntl
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "aks_mage"
MODEL = HERE.parent / "mage_vl/models"
sys.path.extend([str(SOURCE), str(MODEL)])
from fusion import fuse
from codec_video_processing_mage_vl import CodecConfig, process_codec_video

SIGNATURE = "cd294b5855bcae9c"


def atomic(path, content):
    tmp = path.with_suffix(".part")
    tmp.write_text(content)
    os.replace(tmp, path)


def bootstrap(rows, rng):
    by_video = defaultdict(list)
    for q, r in rows:
        by_video[q["video"]].append(int(r["correct"]) - r["baseline"])
    samples = list(by_video.values())
    assert all(len(x) == 3 for x in samples)
    deltas = np.mean(samples, axis=1)
    estimates = rng.choice(deltas, (10000, len(deltas)), replace=True).mean(axis=1)
    return dict(videos=len(deltas), questions=3*len(deltas), delta_pp=100*deltas.mean(),
                percentile_95_pp=(100*np.quantile(estimates, [.025,.975])).tolist())


def image_tag(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=92)
    return '<img src="data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def card(q, r, native, cap):
    merged, used = fuse(native, cap, q["anchors"])
    if used != r["anchors"]:
        raise ValueError(f"Anchor mismatch {q['id']}: {used} vs {r['anchors']}")
    orig = native["images"]
    src = native["src_positions"].reshape(len(orig), -1, 3)
    dst = merged["src_positions"].reshape(src.shape)
    fps = native["fps"]
    blocks = []
    frames = []
    for i, (a,b) in enumerate(zip(orig,merged["images"])):
        if a is b:
            continue
        frame = int(dst[i,0,0])
        frames.append(frame)
        ts = np.unique(src[i,:,0])
        if ts.min()<0 or not np.all(dst[i,:,0]==frame):
            raise ValueError("Invalid canvas coordinates")
        blocks.append(f"<p>画布 {i+1} · 原版源帧范围 {ts.min()/fps:.1f}–{ts.max()/fps:.1f} 秒（{len(ts)} 个不同帧号） · AKS 第 {frame} 帧，{frame/fps:.1f} 秒</p>"
                      + "<div>"+image_tag(a)+image_tag(b)+"</div>")
    assert sorted(frames)==sorted(used)
    options = "".join("<li>"+escape(opt)+"</li>" for opt in q["options"])
    return (f'<section data-id="{q["id"]}"><h2>{q["id"]} · {escape(q["video"])}</h2>'
            f'<p>{escape(q["question"])}</p><ul>{options}</ul>'+ "".join(blocks)
            +'<label>左侧证据支持 <select></select></label> <label>右侧证据支持 <select></select></label> '
            +'<input placeholder="依据、缺失的上下文"><button class="reveal">记录后核对答案</button>'
            +f'<p class="key" hidden>原版 {escape(str(q["old_raw"]))} · 融合 {escape(r["pred"])} · 正确 {escape(q["answer"])}</p></section>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="Generate one audit question for a smoke test")
    args = p.parse_args()
    digest = hashlib.sha256()
    for name in ("evaluate.py","fusion.py","prepare.py","questions.json"):
        digest.update((SOURCE/name).read_bytes())
    if digest.hexdigest()[:16] != SIGNATURE:
        raise RuntimeError("Frozen experiment signature changed")
    qs = {q["id"]:q for q in json.loads((SOURCE/"questions.json").read_text())}
    predictions = (SOURCE/"runs"/(SIGNATURE+".jsonl")).read_bytes()
    rs = [json.loads(line) for line in predictions.splitlines()]
    assert len(qs)==len(rs)==714 and {r["id"] for r in rs}==set(qs)
    rows = [(qs[r["id"]],r) for r in rs]
    assert all(q["answer"]==r["answer"] and q["baseline"]==r["baseline"] and
               (r["pred"]==q["answer"])==r["correct"] for q,r in rows)
    long = [(q,r) for q,r in rows if q["duration"]=="long"]
    cases = [(q,r) for q,r in long if r["correct"]!=r["baseline"]]
    assert len(cases)==23 and sum(r["correct"] for q,r in cases)==17
    out = HERE/"results"/SIGNATURE
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/"audit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    rng=np.random.default_rng(20261122)
    atomic(out/"bootstrap.json",json.dumps({k:bootstrap(x,rng) for k,x in [("overall",rows),("long",long)]},indent=2))
    selected = [(q,r) for q,r in cases if args.only is None or q["id"]==args.only]
    if not selected:
        p.error("No matching long-video flip")
    with sqlite3.connect(out/"cards.sqlite") as db:
        db.execute("create table if not exists cases(id text primary key, card text)")
        db.execute("create table if not exists provenance(hash text primary key)")
        fingerprint = hashlib.sha256(predictions+Path(__file__).read_bytes()+(MODEL/"codec_video_processing_mage_vl.py").read_bytes()+(MODEL/"preprocessor_config.json").read_bytes()).hexdigest()
        prior=db.execute("select hash from provenance").fetchone()
        if prior and prior[0]!=fingerprint:
            raise RuntimeError("Audit inputs changed; archive the isolated results directory before restarting")
        with db:
            db.execute("insert or ignore into provenance values (?)",(fingerprint,))
        done={v[0] for v in db.execute("select id from cases")}
        pending=defaultdict(list)
        for q,r in selected:
            if q["id"] not in done:
                pending[q["video"]].append((q,r))
        cfg=json.loads((MODEL/"preprocessor_config.json").read_text())["codec"]
        with tqdm(total=len(selected),initial=len(selected)-sum(map(len,pending.values())),unit="题",desc="证据复核") as bar:
            for vid, pairs in pending.items():
                video=HERE.parent/"mage_vl/data/VideoMME/video"/(vid+".mp4")
                with tempfile.TemporaryDirectory(dir=out) as tmp:
                    native=process_codec_video(str(video),CodecConfig(**{**cfg,"engine":"hevc","patch":16,"max_pixels":150000,"target_canvas":32,"cache_root":Path(tmp)}))
                    cap=cv2.VideoCapture(str(video))
                    try:
                        for q,r in pairs:
                            html=card(q,r,native,cap)
                            with db:
                                db.execute("insert into cases values (?,?)",(q["id"],html))
                            bar.update()
                    finally:
                        cap.release()
        cards=[v[0] for v in db.execute("select card from cases order by id")]
        template=(HERE/"viewer.html").read_text()
        atomic(out/"review.html",template.replace("<!-- CARDS -->","".join(cards)))
    print(out/"review.html",f"({len(cards)}/23)")

if __name__=="__main__":
    main()
