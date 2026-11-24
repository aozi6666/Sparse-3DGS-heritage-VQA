#!/usr/bin/env python3
"""Localhost-only interactive silhouette editor; no training endpoint."""
import argparse
import base64
import fcntl
import io
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sys
import shutil
import threading
from urllib.parse import urlsplit,parse_qs

import numpy as np
from PIL import Image
from sparse_scene_init import GO,sha,write_json
from prepare_trex_silhouettes import setup_sam,default_prompts,mask_manifest


def png_uri(mask):
    b=io.BytesIO();Image.fromarray(mask.astype(np.uint8)*255).save(b,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()


class Editor:
    def __init__(self,cache,reference,device):
        self.cache=cache.resolve();self.reference=reference.resolve();self.device=device
        self.split=json.loads((self.reference/"v9/split.json").read_text());self.cache.mkdir(parents=True,exist_ok=True)
        self.ids=list(self.split["train_ids"])+list(self.split["test_ids"])
        self.prompts=self.cache/"prompts.json"
        if not self.prompts.exists():write_json(self.prompts,default_prompts(self.split))
        self.lock=threading.RLock();self.predictor=None;self.current=None;self.source=None
    def group(self,idx):
        if idx not in self.ids:raise ValueError("Unknown frame")
        return "train" if idx in self.split["train_ids"] else "eval"
    def image(self,idx):
        self.group(idx);return self.reference/"v9/images_4"/self.split["names"][idx]
    def path(self,idx):
        return self.cache/f"masks_{self.group(idx)}"/(Path(self.split["names"][idx]).stem+".png")
    def info(self):
        frames=[]
        for idx in self.ids:
            p=self.path(idx);meta=p.with_suffix(".json")
            reviewed=False
            if p.exists() and meta.exists():
                data=json.loads(meta.read_text());reviewed=bool(data.get("manual_reviewed")) and data.get("mask_sha256")==sha(p) and data.get("signature",{}).get("image_sha256")==sha(self.image(idx))
            frames.append({"idx":idx,"group":self.group(idx),"reviewed":reviewed})
        return {"frames":frames}
    def frame(self,idx):
        p=self.path(idx);mask=None
        if p.exists():mask=png_uri(np.asarray(Image.open(p).convert("L"))>127)
        prompt=json.loads(self.prompts.read_text())["frames"][str(idx)]
        return {"idx":idx,"name":self.split["names"][idx],"group":self.group(idx),"prompt":prompt,"mask":mask}
    def prompt(self,request):
        positive=request.get("positive",[]);negative=request.get("negative",[]);box=request.get("box")
        for points in (positive,negative):
            if len(points)>150:raise ValueError("Too many prompt points")
            if any(len(p)!=2 or not np.isfinite(p).all() or not all(0<=x<=1 for x in p) for p in points):raise ValueError("Invalid prompt coordinates")
        if box is not None and (len(box)!=4 or not np.isfinite(box).all() or not all(0<=x<=1 for x in box) or box[2]<=box[0] or box[3]<=box[1]):raise ValueError("Invalid box")
        return {"positive":positive,"negative":negative,"box":box}
    def predict(self,request):
        import torch
        idx=int(request["idx"]);self.group(idx);prompt=self.prompt(request)
        if not prompt["positive"] and prompt["box"] is None:raise ValueError("先点选恐龙，或拖框指定范围。")
        if self.predictor is None:
            sam,weights,self.source=setup_sam()
            model=sam.sam_model_registry["vit_b"](checkpoint=str(weights)).requires_grad_(False).eval().to(self.device)
            self.predictor=sam.SamPredictor(model)
        rgb=np.asarray(Image.open(self.image(idx)).convert("RGB"));h,w=rgb.shape[:2]
        points=prompt["positive"]+prompt["negative"]
        coords=np.asarray(points,dtype=np.float32)*[w,h] if points else None
        labels=np.asarray([1]*len(prompt["positive"])+[0]*len(prompt["negative"])) if points else None
        box=np.asarray(prompt["box"])*[w-1,h-1,w-1,h-1] if prompt["box"] is not None else None
        with torch.inference_mode():
            if self.current!=idx:self.predictor.set_image(rgb);self.current=idx
            masks,scores,_=self.predictor.predict(point_coords=coords,point_labels=labels,box=box,multimask_output=True)
        selected=min(range(len(masks)),key=lambda i:float(masks[i].mean()))
        return {"selected":selected,"candidates":[{"score":float(s),"area":float(m.mean()),"mask":png_uri(m)} for m,s in zip(masks,scores)]}
    def save(self,request):
        idx=int(request["idx"]);prompt=self.prompt(request);image=self.image(idx)
        data=request["mask"]
        if not data.startswith("data:image/png;base64,"):raise ValueError("PNG required")
        raw=base64.b64decode(data.split(",",1)[1],validate=True)
        im=Image.open(io.BytesIO(raw)).convert("L")
        if im.size!=Image.open(image).size:raise ValueError("轮廓尺寸与照片不一致。")
        mask=np.asarray(im)>127;area=float(mask.mean())
        if not .01<area<.55:raise ValueError(f"轮廓面积{area:.1%}异常，请检查是否覆盖了大量背景或漏选恐龙。")
        data_root=self.cache.parent
        run=GO/"output"/data_root.name if data_root.is_relative_to(GO/"data/LLFF") else data_root/"test_run"
        run.mkdir(parents=True,exist_ok=True)
        # Never mutate masks while the hull builder/trainer is running, or
        # after training has begun. Earlier generated hulls can be rebuilt.
        with (run/".runner.lock").open("w") as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError("几何构建或训练正在运行，请先停止后再修改轮廓。")
            if list(run.glob("v*/*/logs/train.log")) or list(run.glob("v*/logs/train.log")):
                raise ValueError("已有训练记录。请另开实验版本修正轮廓，避免改变既有实验输入。")
            if (data_root/"hulls").exists():shutil.rmtree(data_root/"hulls")
            metadata=run/"run_config.json"
            if metadata.exists() and json.loads(metadata.read_text()).get("variant")=="SynGS_foreground_hull_scene_extension_not_exact_publication":
                metadata.unlink()
            for n in (4,6,9):
                for name in ("h","v","hv"):
                    for suffix in ("ply","json"):(data_root/f"v{n}"/f"init_{name}.{suffix}").unlink(missing_ok=True)
            return self._save(idx,prompt,image,mask,area,request)
    def _save(self,idx,prompt,image,mask,area,request):
        p=self.path(idx);p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(p)
        prompts=json.loads(self.prompts.read_text());prompts["frames"][str(idx)]=prompt;write_json(self.prompts,prompts)
        write_json(p.with_suffix(".json"),{"manual_reviewed":True,"mask_sha256":sha(p),"signature":{"image_sha256":sha(image),"prompt":prompt},
                   "source":"interactive SAM candidate selection and/or manual brush", "candidate":request.get("candidate"),"area":area})
        return {"saved":True,"area":area}
    def finalize(self):
        missing=[f["idx"] for f in self.info()["frames"] if not f["reviewed"]]
        if missing:raise ValueError("还有未保存检查的照片："+", ".join(map(str,missing)))
        source=self.source
        if source is None:
            _,_,source=setup_sam()
        write_json(self.cache/"mask_manifest.json",{"source":source,"prompts_sha256":sha(self.prompts),
                   "masks":mask_manifest(self.cache,self.split),"train_ids":self.split["train_ids"],"test_ids":self.split["test_ids"],
                   "manually_reviewed_ids":sorted(self.ids),"review":"all frames explicitly saved in interactive editor"})
        return {"message":"16张轮廓已全部保存检查。下一步先检查Visual Hull投影，再运行消融训练。"}


def handler(editor):
    class Handler(BaseHTTPRequestHandler):
        def json(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                route=urlsplit(self.path);query=parse_qs(route.query)
                if route.path=="/":
                    data=(GO/"tools/trex_mask_editor.html").read_bytes();kind="text/html; charset=utf-8"
                elif route.path=="/image":data=editor.image(int(query["idx"][0])).read_bytes();kind="image/jpeg"
                elif route.path=="/api/info":
                    with editor.lock:return self.json(200,editor.info())
                elif route.path=="/api/frame":
                    with editor.lock:return self.json(200,editor.frame(int(query["idx"][0])))
                else:return self.json(404,{"error":"Not found"})
                self.send_response(200);self.send_header("Content-Type",kind);self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
            except Exception as e:self.json(400,{"error":str(e)})
        def do_POST(self):
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<8000000:raise ValueError("Invalid request size")
                request=json.loads(self.rfile.read(size))
                with editor.lock:
                    if self.path=="/api/predict":value=editor.predict(request)
                    elif self.path=="/api/save":value=editor.save(request)
                    elif self.path=="/api/finalize":value=editor.finalize()
                    else:return self.json(404,{"error":"Not found"})
                self.json(200,value)
            except Exception as e:self.json(400,{"error":str(e)})
    return Handler


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--port",type=int,default=8765);p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--cache",type=Path,default=GO/"data/LLFF/trex_syngs_scene_ablation_v1/silhouettes")
    p.add_argument("--reference-data",type=Path,default=GO/"data/LLFF/trex_sparse469_v1")
    a=p.parse_args();editor=Editor(a.cache,a.reference_data,a.device)
    print(f"Open http://127.0.0.1:{a.port} via SSH port forwarding. No training is started.",flush=True)
    ThreadingHTTPServer(("127.0.0.1",a.port),handler(editor)).serve_forever()
