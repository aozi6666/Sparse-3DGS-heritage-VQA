#!/usr/bin/env python3
"""Full-frame metrics plus separately labelled foreground PNG PSNR."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from sparse_scene_init import write_json

VARIANTS={"B":(0,0,0),"D":(0,0,1),"V":(0,1,0),"VD":(0,1,1),
          "H":(1,0,0),"HD":(1,0,1),"HV":(1,1,0),"HVD":(1,1,1)}


def summarize(run,index,mask_cache,iterations):
    rows=[]
    for key,model_string in sorted(index.items()):
        views,variant=key.split("/");n=int(views[1:]);model=Path(model_string)
        output=model/f"test/ours_{iterations}"
        metrics=output/"results.json"
        if not metrics.exists():continue
        values=json.loads(metrics.read_text())
        if not all(math.isfinite(values[k]) for k in ("PSNR","SSIM","LPIPS")):
            raise RuntimeError(f"Invalid metrics: {metrics}")
        mapping=json.loads((output/"camera_names.json").read_text())
        foreground=[]
        for filename,name in mapping.items():
            mask=np.asarray(Image.open(mask_cache/"masks_eval"/(Path(name).stem+".png")).convert("L"))>127
            pred=np.asarray(Image.open(output/"renders"/filename).convert("RGB"),dtype=np.float64)/255
            gt=np.asarray(Image.open(output/"gt"/filename).convert("RGB"),dtype=np.float64)/255
            if mask.shape!=gt.shape[:2] or mask.sum()<100:raise RuntimeError("Invalid eval silhouette")
            mse=float(np.mean((pred[mask]-gt[mask])**2))
            foreground.append(-10*math.log10(max(mse,1e-12)))
        h,v,d=VARIANTS[variant]
        rows.append({"views":n,"variant":variant,"H":h,"V":v,"D":d,
                     **{k:values[k] for k in ("PSNR","SSIM","LPIPS")},
                     "foreground_PNG_PSNR":float(np.mean(foreground)),"source":str(model)})
    run.mkdir(parents=True,exist_ok=True)
    if not rows:return
    rows.sort(key=lambda r:(r["views"],list(VARIANTS).index(r["variant"])))
    with (run/"ablation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lookup={(r["views"],r["variant"]):r for r in rows}
    text=["# SynGS scene-extension engineering ablation", "",
          "H = foreground Visual Hull additions; V = frozen single-frame VGGT additions; D = compensated DropGaussian.",
          "Full RGB scene and identical 7 held-out views. B is the matched project 3DGS baseline.",
          "This is NOT an exact reproduction of published SynGS (no adapted multi-view depth branch or dual-branch DVR).", "",
          "Foreground PNG PSNR uses SAM evaluation-only silhouettes and saved/clamped 8-bit PNGs; full-frame metrics use the original floating-render evaluation.","",
          "| Views | Variant | H | V | D | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FG PNG PSNR ↑ |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['views']} | {r['variant']} | {r['H']} | {r['V']} | {r['D']} | {r['PSNR']:.4f} | {r['SSIM']:.6f} | {r['LPIPS']:.6f} | {r['foreground_PNG_PSNR']:.4f} |")
    text += ["","## Component differences (first minus second; negative LPIPS is better)","",
             "| Views | Comparison | ΔPSNR | ΔSSIM | ΔLPIPS | ΔFG PNG PSNR |",
             "|---:|---|---:|---:|---:|---:|"]
    for n in (4,6,9):
        for a,b in (("H","B"),("HV","H"),("HVD","HV"),("HVD","VD"),("HVD","HD"),("HVD","B"),
                    ("D","B"),("V","B"),("VD","V")):
            if (n,a) not in lookup or (n,b) not in lookup:continue
            diff={k:lookup[n,a][k]-lookup[n,b][k] for k in ("PSNR","SSIM","LPIPS","foreground_PNG_PSNR")}
            text.append(f"| {n} | {a} - {b} | {diff['PSNR']:+.4f} | {diff['SSIM']:+.6f} | {diff['LPIPS']:+.6f} | {diff['foreground_PNG_PSNR']:+.4f} |")
    text += ["","Single scene, single seed. Report improvements and regressions as measured; module gains may depend on other enabled components."]
    (run/"ablation.md").write_text("\n".join(text)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key in zip(axes,("PSNR","SSIM","LPIPS")):
        for name in VARIANTS:
            subset=[r for r in rows if r["variant"]==name]
            if subset:ax.plot([r["views"] for r in subset],[r[key] for r in subset],"o-",label=name)
        ax.set_title(key+(" (lower better)" if key=="LPIPS" else " (higher better)"));ax.set_xticks([4,6,9]);ax.set_xlabel("Training views");ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(run/"metrics_curves.png",dpi=180);plt.close(fig)
    for n in (4,6,9):
        selected=[x for x in ("B","H","HV","HVD","VD") if (n,x) in lookup]
        if not selected:continue
        first=Path(lookup[n,selected[0]]["source"])/f"test/ours_{iterations}"
        mapping=json.loads((first/"camera_names.json").read_text());w,h,label=336,252,28
        sheet=Image.new("RGB",((len(selected)+1)*w,len(mapping)*(h+label)),"white");draw=ImageDraw.Draw(sheet)
        for i,filename in enumerate(sorted(mapping)):
            sources=[("GT",first/"gt"/filename)]+[(name,Path(lookup[n,name]["source"])/f"test/ours_{iterations}/renders"/filename) for name in selected]
            for col,(name,path) in enumerate(sources):
                with Image.open(path) as im:tile=ImageOps.contain(im.convert("RGB"),(w,h))
                x,y=col*w,i*(h+label);sheet.paste(tile,(x+(w-tile.width)//2,y+label+(h-tile.height)//2));draw.text((x+5,y+7),f"{n} views | {name} | test {i}",fill="black")
        sheet.save(run/f"stages_v{n}.png")
    write_json(run/"model_index.json",index)
    print(f"[summary] {len(rows)}/24 results: {run/'ablation.md'}",flush=True)
