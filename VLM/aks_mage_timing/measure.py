"""Experiment 4. Synchronized offline latency with precomputed AKS scores."""
import argparse, collections, csv, fcntl, hashlib, importlib, itertools, json, os, platform, random, sqlite3, sys, tempfile
from pathlib import Path
from time import perf_counter as clock
import cv2
import numpy as np
from tqdm import tqdm
from selection import plan
from timed_render import render

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"aks_mage"
FINAL=HERE.parent/"aks_mage_budget_final"
SEED=20261123
ARMS=("codec","fixed20","short4_else20")

def read(path):
    return json.loads(path.read_text())

def write(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)

def select():
    groups=collections.defaultdict(dict)
    for q in read(OLD/"questions.json"):
        groups[q["duration"]].setdefault(q["video"],[]).append(q)
    rng=random.Random(SEED)
    selected=[]
    orders=list(itertools.permutations(ARMS))
    for duration in ("short","medium","long"):
        g=groups[duration]
        vids=rng.sample(sorted(g),12)
        for i,vid in enumerate(vids):
            assert len(g[vid])==3
            selected.append((vid,g[vid],orders[i%6]))
    rng.shuffle(selected)
    return selected

def summary(db,out):
    data=[json.loads(s) for s, in db.execute("SELECT record FROM videos ORDER BY video,arm")]
    assert len(data)==108
    write(out/"video_records.json",data)
    rows=[]
    for arm,d in itertools.product(ARMS,("short","medium","long","balanced_all")):
        videos=[v for v in data if v["arm"]==arm and (d=="balanced_all" or v["duration"]==d)]
        qs=[q for v in videos for q in v["questions"]]
        a=np.array([q["question_s"] for q in qs])
        row=dict(arm=arm,duration=d,videos=len(videos),questions=len(qs),
                 mean_video_three_q_s=np.mean([v["video_total_s"] for v in videos]),
                 p50_question_s=np.median(a),p90_question_s=np.percentile(a,90),
                 mean_codec_s=np.mean([v["codec_s"] for v in videos]))
        for field in ("tokens","replaced","lookup_s","plan_s","rgb_decode_s","rebuild_s","processor_transfer_s","generate_s","output_tokens"):
            row["mean_"+field]=np.mean([q[field] for q in qs])
        rows.append(row)
    tmp=out/"summary.tmp"
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    tmp.replace(out/"summary.csv")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    selected=select()
    csv_path=FINAL/"candidates.json"
    historical=OLD/"runs/cd294b5855bcae9c.jsonl"
    deps=[historical,Path(__file__),HERE/"selection.py",HERE/"timed_render.py",HERE/"run.sh",OLD/"evaluate.py",OLD/"questions.json",csv_path]
    h=hashlib.sha256()
    for p in deps:h.update(p.read_bytes())
    out=HERE/"results"/h.hexdigest()[:16];out.mkdir(parents=True,exist_ok=True)
    lock=(out/"run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(out/"checkpoint.sqlite")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS videos(video TEXT,arm TEXT,record TEXT,PRIMARY KEY(video,arm))")
    done=set(db.execute("SELECT video,arm FROM videos"))
    write(out/"protocol.json",dict(seed=SEED,selection="12 videos per length; three original questions per video",
          timing="Precomputed BLIP rankings. One native codec time charged once per video per arm. Question latency excludes shared codec and video-open costs. Three-question total includes both. GPU synchronized. No cross-arm prediction reuse.",
          videos=[dict(video=v,duration=g[0]["duration"],ids=[q["id"] for q in g],order=o) for v,g,o in selected]))
    print(out, len(done),"/108",flush=True)
    if args.prepare:return
    if len(done)==108:
        summary(db,out);return
    sys.path.insert(0,str(OLD))
    import evaluate as base
    torch=base.torch
    if not torch.cuda.is_available():raise SystemExit("GPU required")
    sync=torch.cuda.synchronize
    gpu=torch.cuda.get_device_name()
    import transformers
    env=dict(gpu=gpu,torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,opencv=cv2.__version__,python=platform.python_version())
    envfile=out/"environment.json"
    if envfile.exists() and read(envfile)!=env:raise RuntimeError("GPU or software changed; use a matching environment for comparable timing.")
    write(envfile,env)
    t=clock();ranks=read(csv_path);load_s=clock()-t
    used4={r["id"]:r["anchors"] for r in map(json.loads,historical.read_text().splitlines())}
    proc=base.AutoProcessor.from_pretrained(base.MODEL,trust_remote_code=True,local_files_only=True)
    codec=importlib.import_module(".codec_video_processing_mage_vl",type(proc).__module__.rsplit(".",1)[0])
    model=base.AutoModelForCausalLM.from_pretrained(base.MODEL,trust_remote_code=True,local_files_only=True,torch_dtype="auto",device_map="auto").eval()
    def question(q,arm,native,cap,video,cfg):
        sync();start=clock()
        k=0 if arm=="codec" else (4 if arm=="short4_else20" and q["duration"]=="short" else 20)
        lookup=planning=decode=rebuild=0.;layout={};payload=native
        if k:
            t=clock();rank=ranks[q["id"]];lookup=clock()-t
            t=clock();layout=plan(native,rank,used4[q["id"]],k);planning=clock()-t
            payload,decode,rebuild=render(native,cap,layout)
        t=clock();inputs=base.inputs_for(proc,codec,payload,video,q,cfg).to(model.device)
        inputs["pixel_values"]=inputs["pixel_values"].to(model.dtype)
        sync();prep=clock()-t
        t=clock()
        with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False)
        sync();gen=clock()-t
        response=tokens[0,inputs["input_ids"].shape[1]:]
        raw=proc.tokenizer.decode(response,skip_special_tokens=True).strip()
        visual=int((inputs["input_ids"]==proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum().item())
        return dict(id=q["id"],raw=raw,answer=q["answer"],cap=k,tokens=visual,replaced=len(layout),lookup_s=lookup,
                    plan_s=planning,rgb_decode_s=decode,rebuild_s=rebuild,processor_transfer_s=prep,generate_s=gen,
                    output_tokens=len(response),question_s=clock()-start)
    warmed=False
    with tqdm(total=108,initial=len(done),unit="video-arm") as bar:
        for vid,group,order in selected:
            if all((vid,a) in done for a in order):continue
            video=base.DATA/"video"/(vid+".mp4")
            with tempfile.TemporaryDirectory(dir=out) as td:
                cfg=dict(engine="hevc",target_canvas=32,patch=16,cache_root=Path(td))
                opts={**proc._codec_config_defaults,**cfg,"max_pixels":150000}
                t=clock();native=codec.process_codec_video(str(video),codec.CodecConfig(**opts));codec_s=clock()-t
                prior=db.execute("SELECT record FROM videos WHERE video=? LIMIT 1",(vid,)).fetchone()
                if prior:
                    codec_s=json.loads(prior[0])["codec_s"]
                if not warmed:
                    cap=cv2.VideoCapture(str(video))
                    try:
                        for a in ARMS:question(group[0],a,native,cap,video,cfg)
                    finally:cap.release()
                    warmed=True
                for arm in order:
                    if (vid,arm) in done:continue
                    sync();start=clock()
                    cap=cv2.VideoCapture(str(video)) if arm!="codec" else None
                    try:qs=[question(q,arm,native,cap,video,cfg) for q in group]
                    finally:
                        if cap is not None:cap.release()
                    total=codec_s+(clock()-start)
                    rec=dict(video=vid,arm=arm,duration=group[0]["duration"],questions=qs,codec_s=codec_s,video_total_s=total,session_rank_file_load_s=load_s)
                    with db:db.execute("INSERT INTO videos VALUES(?,?,?)",(vid,arm,json.dumps(rec)))
                    bar.set_postfix_str(vid+" "+arm);bar.update()
    summary(db,out)
if __name__=="__main__":main()
