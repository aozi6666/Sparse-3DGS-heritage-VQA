"""Offline nested canvas replacement budgets; a single durable transaction per prediction."""
import collections
import fcntl
import hashlib
import importlib
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from candidates import HERE, OLD, VIDEO
from sampling import plans
from render import render, pure_frames
from figures import plot_results
from tqdm import tqdm
import cv2

BUDGETS=(16,20,24,28,32,"pure_aks")
def write_json(path, data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    tmp.replace(path)

def run():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true',help='Import or verify historical results without loading the GPU model')
    args=parser.parse_args()
    sys.path.insert(0,str(OLD))
    from prepare import letter
    manifest=OLD/"questions.json"
    history=OLD/"runs/cd294b5855bcae9c.jsonl"
    questions=json.loads(manifest.read_text())
    old={r["id"]:r for r in map(json.loads,history.read_text().splitlines())}
    ranked=json.loads((HERE/"candidates.json").read_text())
    assert len(questions)==714 and set(old)==set(ranked)=={q["id"] for q in questions}
    previous=HERE.parent/"aks_mage_budget/results/0a998b8b6c2088cf/predictions.jsonl"
    fixed={r["id"]:r for r in map(json.loads,previous.read_text().splitlines()) if r["budget"]==16}
    assert set(fixed)==set(old) and sum(r["correct"] for r in fixed.values())==475
    orig=hashlib.sha256()
    for n in ["evaluate.py","fusion.py","prepare.py","questions.json"]:
        orig.update((OLD/n).read_bytes())
    assert orig.hexdigest()[:16]=="cd294b5855bcae9c"
    digest=orig.copy()
    for file in [previous,history,HERE/"candidates.json",HERE/"sampling.py",HERE/"render.py",Path(__file__)]:
        digest.update(file.read_bytes())
    out=HERE/"results"/digest.hexdigest()[:16]
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with closing(sqlite3.connect(out/"checkpoint.sqlite")) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS predictions(q TEXT,k INTEGER,data TEXT,PRIMARY KEY(q,k))")
        def save(q,k,raw,layout,source):
            parsed=letter(raw)
            record=dict(id=q["id"],video=q["video"],duration=q["duration"],budget=k,
                        raw=raw,pred=parsed,answer=q["answer"],correct=parsed==q["answer"],
                        frames=list(layout.values()),slots=list(layout),source=source)
            db.execute("INSERT INTO predictions VALUES(?,?,?)",(q["id"],k,json.dumps(record)))
        done=set(db.execute("SELECT q,k FROM predictions"))
        groups=collections.defaultdict(list)
        with db:
            for q in questions:
                key=q["id"]
                r=old[key]
                assert int(letter(q["old_raw"])==q["answer"])==q["baseline"]==r["baseline"]
                assert (letter(r["raw"])==q["answer"])==r["correct"] and r["answer"]==q["answer"]
                rec=fixed[key]
                assert rec["answer"]==q["answer"] and rec["video"]==q["video"]
                if (key,16) not in done:
                    db.execute("INSERT INTO predictions VALUES(?,?,?)",(key,16,json.dumps(rec)))
                    done.add((key,16))
                groups[q["video"]].append(q)
        print("Results:",out,"| saved:",len(done),"/ 4284",flush=True)
        if args.prepare_only:
            print("SQLite integrity:",db.execute("PRAGMA integrity_check").fetchone()[0])
            return
        if len(done)<4284:
            import evaluate as base
            torch=base.torch
            if not torch.cuda.is_available():
                raise SystemExit("No CUDA GPU. Historical predictions have been saved; rerun on the GPU instance.")
            proc=base.AutoProcessor.from_pretrained(base.MODEL,trust_remote_code=True,local_files_only=True)
            codec=importlib.import_module(".codec_video_processing_mage_vl",type(proc).__module__.rsplit(".",1)[0])
            model=base.AutoModelForCausalLM.from_pretrained(base.MODEL,trust_remote_code=True,
                        local_files_only=True,torch_dtype="auto",device_map="auto").eval()
            scratch=out/"scratch"
            # Only this experiment writes this directory, under the experiment lock.
            if scratch.exists():
                shutil.rmtree(scratch)
            scratch.mkdir()
            with tqdm(total=4284,initial=len(done),unit="题档") as bar:
                for vid,group in groups.items():
                    if all((q["id"],k) in done for q in group for k in BUDGETS):
                        continue
                    video=VIDEO/(vid+".mp4")
                    bar.set_postfix_str(vid)
                    with TemporaryDirectory(dir=scratch) as tmp:
                        cfg=dict(engine="hevc",target_canvas=32,patch=16,cache_root=Path(tmp))
                        settings={**proc._codec_config_defaults,**cfg,"max_pixels":150000}
                        native=codec.process_codec_video(str(video),codec.CodecConfig(**settings))
                        cap=cv2.VideoCapture(str(video))
                        try:
                            for q in group:
                                key=q["id"]
                                layout=plans(native,ranked[key],old[key]["anchors"])
                                frozen=fixed[key]
                                assert list(layout[16])==frozen["slots"] and list(layout[16].values())==frozen["frames"], key
                                assert layout[16].items()<=layout[20].items()<=layout[24].items()
                                pure=sorted(f for quarter in ranked[key] for f,_ in quarter)
                                assert 0<len(pure)<=32 and len(pure)==len(set(pure))
                                layout["pure_aks"]=dict(enumerate(pure))
                                def signature(k):
                                    return (k=="pure_aks",tuple(sorted(layout[k].items())))
                                # Reuse an answer only for exactly identical question and visual input.
                                known={}
                                for k,s in db.execute("SELECT k,data FROM predictions WHERE q=? ORDER BY k",(key,)):
                                    known[signature(k)]=(k,json.loads(s)["raw"])
                                for k in BUDGETS[1:]:
                                    if (key,k) in done:
                                        continue
                                    tag=signature(k)
                                    if tag in known:
                                        source_k,raw=known[tag]
                                        source=f"identical_input_{source_k}"
                                    else:
                                        payload=pure_frames(native,cap,pure) if k=="pure_aks" else render(native,cap,layout[k])
                                        inputs=base.inputs_for(proc,codec,payload,video,q,cfg).to(model.device)
                                        inputs["pixel_values"]=inputs["pixel_values"].to(model.dtype)
                                        with torch.inference_mode():
                                            tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False)
                                        raw=proc.tokenizer.decode(tokens[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                                        del inputs,tokens,payload
                                        source="inference"
                                    with db:
                                        save(q,k,raw,layout[k],source)
                                    known[tag]=(k,raw)
                                    done.add((key,k))
                                    bar.update()
                        finally:
                            cap.release()
        rows=[json.loads(s) for s, in db.execute("SELECT data FROM predictions ORDER BY q,k")]
        assert len(rows)==4284
        tmp=out/"predictions.tmp"
        tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
        tmp.replace(out/"predictions.jsonl")
        summary={}
        for k in BUDGETS:
            summary[k]={}
            for d in ("short","medium","long","overall"):
                rr=[r for r in rows if r["budget"]==k and (d=="overall" or r["duration"]==d)]
                n=len(rr)
                summary[k][d]=dict(n=n,correct=sum(r["correct"] for r in rr),accuracy=sum(r["correct"] for r in rr)/n,
                    actual_replacements=dict(collections.Counter(len(r["frames"]) for r in rr)),
                    gains_vs16=sum(r["correct"] and not fixed[r["id"]]["correct"] for r in rr),
                    losses_vs16=sum(not r["correct"] and fixed[r["id"]]["correct"] for r in rr),
                    unparsed=sum(r["pred"] is None for r in rr))
        write_json(out/"summary.json",summary)
        plot_results(out, rows, previous)
        for k,s in summary.items():
            print(k,{d:round(x["accuracy"]*100,2) for d,x in s.items()})
if __name__=="__main__":
    run()
