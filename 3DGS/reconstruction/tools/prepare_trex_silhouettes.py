#!/usr/bin/env python3
"""Frozen SAM silhouettes for foreground hull only. Never alters training RGB."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sparse_scene_init import GO, sha, write_json

SAM_REV = "6fdee8f2727f4506cfbbe553e23b895e27956588"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_BYTES = 375042383


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".part")
    print(f"[download] {url}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "15",
                    "--max-time", "600", "--output", str(tmp), url], check=True)
    tmp.replace(path)


def setup_sam():
    # Vendor just the official package, without pip changes to torch/the venv.
    vendor = GO/"tools/vendor"/f"sam_{SAM_REV}"
    if not (vendor/"segment_anything/__init__.py").is_file():
        archive = vendor.parent/f"sam_{SAM_REV}.zip"
        download(f"https://codeload.github.com/facebookresearch/segment-anything/zip/{SAM_REV}", archive)
        vendor.mkdir(parents=True, exist_ok=True)
        prefix = f"segment-anything-{SAM_REV}/"
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.startswith(prefix):
                    continue
                relative = Path(name[len(prefix):])
                if ".." in relative.parts or relative.is_absolute():
                    raise RuntimeError("Invalid official archive member")
                if (str(relative).startswith("segment_anything/") and relative.suffix == ".py") or str(relative)=="LICENSE":
                    out = vendor/relative
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
        write_json(vendor/"source.json", {"official_commit": SAM_REV, "archive_sha256": sha(archive)})
        archive.unlink()
    weights = GO/"models/sam_vit_b_01ec64.pth"
    if not weights.exists():
        download(SAM_URL, weights)
    if weights.stat().st_size != SAM_BYTES:
        raise RuntimeError(f"Wrong SAM checkpoint size: {weights}; expected {SAM_BYTES}")
    sys.path.insert(0, str(vendor))
    module = importlib.import_module("segment_anything")
    return module, weights, {"official_commit": SAM_REV, "weight_sha256": sha(weights), "weight_url": SAM_URL}


def default_prompts(split):
    # Image-coordinate prompts from the unmodified training photographs, not GT geometry.
    heads = {1:[.25,.36],7:[.25,.35],14:[.25,.34],21:[.25,.29],28:[.27,.35],
             34:[.27,.35],41:[.25,.37],47:[.25,.31],54:[.25,.29]}
    hips = {1:[.69,.55],7:[.71,.52],14:[.71,.52],21:[.70,.49],28:[.73,.58],
            34:[.73,.61],41:[.70,.54],47:[.71,.53],54:[.71,.52]}
    prompts = {}
    for idx in sorted(set(split["train_ids"]) | set(split["test_ids"])):
        head, hip = heads.get(idx,[.25,.34]), hips.get(idx,[.71,.54])
        prompts[str(idx)] = {"box": [.10,.15,1.,1.], "positive": [head, hip],
                             "negative": [[.50,.07],[.06,.75],[.94,.18],[.40,.28],
                                          [.83,.35],[.17,.83],[.40,.86],[.95,.42]]}
    return {"coordinates": "normalized xy in original rectified images_4", "frames": prompts}


def mask_manifest(cache, split):
    result = {"train": {}, "eval": {}}
    for group, ids in (("train",split["train_ids"]),("eval",split["test_ids"])):
        for idx in ids:
            name = Path(split["names"][idx]).stem+".png"
            path = cache/f"masks_{group}"/name
            with Image.open(path) as im:
                a = np.asarray(im)
            if a.ndim != 2 or not .01 < (a>127).mean() < .70:
                raise RuntimeError(f"Suspicious mask area: {path}")
            result[group][str(idx)] = {"file": str(path), "sha256": sha(path), "area": float((a>127).mean())}
    return result


def prepare(cache, reference_data, device="cuda"):
    import torch
    cache=cache.resolve();reference_data=reference_data.resolve()
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable GPU for SAM masks")
    split = json.loads((reference_data/"v9/split.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    prompt_file = cache/"prompts.json"
    if not prompt_file.exists():
        write_json(prompt_file, default_prompts(split))
    prompts = json.loads(prompt_file.read_text())
    sam, weights, source = setup_sam()
    model = None
    sheets = []
    for group, ids in (("train",split["train_ids"]),("eval",split["test_ids"])):
        sheet = Image.new("RGB", (3*504, ((len(ids)+2)//3)*410), "white")
        draw = ImageDraw.Draw(sheet)
        for i, idx in enumerate(ids):
            filename = split["names"][idx]
            image = reference_data/"v9/images_4"/filename
            rgb = np.asarray(Image.open(image).convert("RGB"))
            h,w = rgb.shape[:2]
            prompt = prompts["frames"][str(idx)]
            coords = np.asarray(prompt["positive"]+prompt["negative"],dtype=np.float32).reshape(-1,2)*[w,h]
            labels = np.asarray([1]*len(prompt["positive"])+[0]*len(prompt["negative"]))
            box = np.asarray(prompt["box"])*[w,h,w-1,h-1] if prompt.get("box") is not None else None
            output = cache/f"masks_{group}"/(Path(filename).stem+".png")
            info = output.with_suffix(".json")
            signature = {**source,"image_sha256":sha(image),"prompt":prompt,"version":2,
                         "selection":"smallest >=0.6 score candidate respecting all point labels; explicit candidate override allowed"}
            previous=json.loads(info.read_text()) if info.exists() else {}
            if output.exists() and previous.get("manual_reviewed") and previous.get("signature",{}).get("image_sha256")==sha(image):
                mask = np.asarray(Image.open(output))>127
            elif output.exists() and previous.get("signature")==signature:
                mask = np.asarray(Image.open(output))>127
            else:
                if model is None:
                    model = sam.sam_model_registry["vit_b"](checkpoint=str(weights))
                    model.requires_grad_(False).eval().to(device)
                    predictor = sam.SamPredictor(model)
                with torch.inference_mode():
                    predictor.set_image(rgb)
                    candidates,scores,_ = predictor.predict(point_coords=coords if len(coords) else None,point_labels=labels if len(coords) else None,box=box,multimask_output=True)
                # The highest-scoring broad candidate can include the museum.
                # Choose the tightest prompt-consistent object candidate, with
                # no use of rendering metrics; retain every candidate for review.
                candidate_dir=cache/"candidates"/Path(filename).stem
                candidate_dir.mkdir(parents=True,exist_ok=True)
                for ci,candidate in enumerate(candidates):
                    Image.fromarray(candidate.astype(np.uint8)*255).save(candidate_dir/f"{ci}.png")
                if "candidate" in prompt:
                    chosen=int(prompt["candidate"])
                    if chosen not in (0,1,2):raise ValueError("candidate override must be 0, 1 or 2")
                else:
                    u=np.clip(np.rint(coords[:,0]).astype(int),0,w-1)
                    v=np.clip(np.rint(coords[:,1]).astype(int),0,h-1)
                    valid=[ci for ci,m in enumerate(candidates) if scores[ci]>=.6 and
                           np.all(m[v[labels==1],u[labels==1]]) and not np.any(m[v[labels==0],u[labels==0]])]
                    if not valid:
                        raise RuntimeError(f"No prompt-consistent SAM candidate for idx {idx}; review {candidate_dir} and edit {prompt_file}")
                    chosen=min(valid,key=lambda ci:float(candidates[ci].mean()))
                mask = candidates[chosen]
                output.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask.astype(np.uint8)*255).save(output)
                write_json(info,{"signature":signature,"sam_score":float(scores[chosen]),"candidate":chosen,
                                 "foreground_area":float(mask.mean()),"requires_visual_review":True})
            overlay = rgb.copy()
            overlay[mask] = (.6*overlay[mask]+.4*np.array([30,240,80])).astype(np.uint8)
            tile=Image.fromarray(overlay);d=ImageDraw.Draw(tile)
            for (x,y),label in zip(coords,labels):
                d.ellipse((x-5,y-5,x+5,y+5),fill="white" if label else "red")
            tile=ImageOps.contain(tile,(504,378));x,y=(i%3)*504,(i//3)*410
            sheet.paste(tile,(x,y+30));draw.text((x+6,y+8),f"{group} idx={idx}; foreground={mask.mean():.1%}",fill="black")
            print(f"[mask] {group} {idx} {filename} area={mask.mean():.3f}",flush=True)
        preview=cache/f"preview_{group}.jpg";sheet.save(preview,quality=92);sheets.append(str(preview))
    # Previews are always saved before area checks, so failures remain reviewable.
    manifest=mask_manifest(cache,split)
    write_json(cache/"mask_manifest.json",{"source":source,"prompts_sha256":sha(prompt_file),"masks":manifest,
               "train_ids":split["train_ids"],"test_ids":split["test_ids"],"previews":sheets})
    print("MASKS READY. Inspect green foreground, then explicitly run the ablation with --accept-masks.",flush=True)
    print("\n".join(sheets),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--reference-data",type=Path,required=True)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    a=p.parse_args();prepare(a.cache,a.reference_data,a.device)
