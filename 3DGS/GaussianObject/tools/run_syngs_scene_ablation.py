#!/usr/bin/env python3
"""8-way foreground-Hull / frozen-VGGT / Drop ablation with real backgrounds."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

from run_sparse_469 import GO, execute, stage
from run_sparse_469_baseline import check_reference, baseline_command, link_input, read_json, set_option
from sparse_scene_init import sha, write_json
from foreground_hull import build_hull, build_union
from summarize_syngs_scene import VARIANTS,summarize


def prepare_data(original,target,n,resolution):
    target.mkdir(parents=True,exist_ok=True)
    if (target/"zoe_depth").exists():
        raise RuntimeError("This ablation uses VGGT initialization only; do not attach depth-loss inputs")
    for name in ("images",f"images_{resolution}","split.json","full_scene_manifest.json",f"sparse_{n}.txt",
                 "sparse_test.txt","sparse/0/cameras.bin","sparse/0/images.bin","sparse/0/points3D.ply"):
        link_input(original/name,target/name)


def validate_masks(cache,reference_data):
    from prepare_trex_silhouettes import mask_manifest
    if not (cache/"mask_manifest.json").exists():
        raise RuntimeError("Foreground masks are incomplete/unreviewed. Correct dinosaur silhouettes first; no training will start.")
    manifest=read_json(cache/"mask_manifest.json")
    split=read_json(reference_data/"v9/split.json")
    if manifest.get("manually_reviewed_ids")!=sorted(set(split["train_ids"])|set(split["test_ids"])):
        raise RuntimeError("Finish all 16 silhouettes in the interactive editor before running the ablation")
    if manifest["masks"]!=mask_manifest(cache,split):
        raise RuntimeError("Masks changed since preview creation; rerun --stage masks and review again")
    if manifest["prompts_sha256"]!=sha(cache/"prompts.json"):
        raise RuntimeError("Mask prompts changed; regenerate previews first")
    assert manifest["train_ids"]==split["train_ids"] and manifest["test_ids"]==split["test_ids"]
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag",default="trex_syngs_scene_ablation_v1")
    p.add_argument("--stage",choices=["check","masks","hulls","run","summary"],default="check")
    p.add_argument("--accept-masks",action="store_true",help="Confirm visual review of previews before training")
    p.add_argument("--grid",type=int,default=144)
    p.add_argument("--mask-dilation",type=int,default=3)
    p.add_argument("--max-hull-points",type=int,default=60000)
    a=p.parse_args()
    if Path(a.run_tag).name!=a.run_tag or a.run_tag in ("",".",".."):
        raise ValueError("run-tag must be one directory name")
    if not 64<=a.grid<=256 or not 0<=a.mask_dilation<=10 or not 100<=a.max_hull_points<=100000:
        raise ValueError("Invalid hull settings")
    combo=GO/"output/trex_sparse469_v1";baseline=GO/"output/trex_sparse469_baseline_v1"
    config,reference_hashes=check_reference(combo)
    if config["resolution"]!=4:raise RuntimeError("This trex silhouette entry expects resolution=4")
    if not (baseline/"COMPLETE.json").exists():raise RuntimeError("Baseline must be complete")
    baseline_config=read_json(baseline/"run_config.json")
    for name,expected in baseline_config["reference_hashes"].items():
        if sha(Path(name))!=expected:raise RuntimeError(f"Reference baseline input changed: {name}")
    run=GO/"output"/a.run_tag;data_root=GO/"data/LLFF"/a.run_tag;cache=data_root/"silhouettes"
    reference_data=Path(config["data"]);source=Path(config["source"])
    index={}
    for n in (4,6,9):
        for name in VARIANTS:
            model=baseline/f"v{n}" if name=="B" else combo/f"v{n}" if name=="VD" else run/f"v{n}"/name
            index[f"v{n}/{name}"]=str(model)
    if a.stage=="check":
        print(json.dumps({"output":str(run),"mask_cache":str(cache),"variants":VARIANTS,
              "new_training_runs":18,"reused_runs":6,"steps_each":config["iterations"],
              "shared_test_ids":config["test_ids"],"background":"unchanged full RGB",
              "module_H":"foreground-only silhouette surface added to existing scene geometry",
              "module_V":"reuse frozen single-frame VGGT additions, no finetuning",
              "module_D":"existing compensated DropGaussian, not paper dual-branch DVR",
              "exact_published_SynGS":False,"free_disk_GiB":round(shutil.disk_usage(GO).free/1024**3,2)},indent=2))
        return
    run.mkdir(parents=True,exist_ok=True)
    with (run/".runner.lock").open("w") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("Another ablation process is active")
        if a.stage=="masks":
            execute([sys.executable,GO/"tools/prepare_trex_silhouettes.py","--cache",cache,
                     "--reference-data",reference_data],run/"logs/masks.log")
            return
        manifest=validate_masks(cache,reference_data)
        if a.stage=="summary":
            summarize(run,index,cache,config["iterations"]);return
        if not a.accept_masks:
            raise RuntimeError(f"Review {cache}/preview_train.jpg and preview_eval.jpg, then use --accept-masks")
        if a.stage=="hulls":
            for n in (4,6,9):
                original=reference_data/f"v{n}";data=data_root/f"v{n}"
                prepare_data(original,data,n,config["resolution"])
                hull=build_hull(source,original,data_root/"hulls",n,cache,a.grid,a.mask_dilation,a.max_hull_points)
                for label,h,v in (("h",True,False),("v",False,True),("hv",True,True)):
                    counts=build_union(original,hull,data/f"init_{label}.ply",h,v);write_json(data/f"init_{label}.json",counts)
            print("CPU HULLS READY: inspect data/LLFF/<run-tag>/hulls/v*/hull_projection_first_train.jpg before training",flush=True)
            return
        import torch
        if not torch.cuda.is_available():raise RuntimeError("Enable GPU before training")
        if shutil.disk_usage(GO).free<5*1024**3:raise RuntimeError("Need at least 5 GiB free disk")
        for n in (4,6,9):
            for model in (baseline/f"v{n}",combo/f"v{n}"):
                file=model/f"test/ours_{config['iterations']}/results.json"
                reference_hashes[str(file)]=sha(file)
        immutable={"variant":"SynGS_foreground_hull_scene_extension_not_exact_publication", "settings":{
                   "grid":a.grid,"mask_dilation":a.mask_dilation,"max_hull_points":a.max_hull_points},
                   "masks_manifest_sha256":sha(cache/"mask_manifest.json"),"reference_hashes":reference_hashes,
                   "index":index,"train_ids":config["train_ids"],"test_ids":config["test_ids"],
                   "code_hashes":{f:sha(GO/f) for f in ("tools/run_syngs_scene_ablation.py",
                   "tools/foreground_hull.py","tools/prepare_trex_silhouettes.py","tools/summarize_syngs_scene.py")}}
        saved=run/"run_config.json"
        if saved.exists() and read_json(saved)!=immutable:
            raise RuntimeError("Code/settings/masks/references changed; choose a new run tag")
        write_json(saved,immutable)
        write_json(run/"mask_review.json",{"accepted_by":"explicit --accept-masks flag", "manifest_sha256":sha(cache/"mask_manifest.json")})
        for n in (4,6,9):
            original=reference_data/f"v{n}";data=data_root/f"v{n}"
            prepare_data(original,data,n,config["resolution"])
            hull=build_hull(source,original,data_root/"hulls",n,cache,a.grid,a.mask_dilation,a.max_hull_points)
            for label,h,v in (("h",True,False),("v",False,True),("hv",True,True)):
                init=data/f"init_{label}.ply"
                report=data/f"init_{label}.json"
                if not init.exists() or not report.exists():
                    counts=build_union(original,hull,init,h,v);write_json(report,counts)
            train_reference=read_json(combo/f"v{n}/.train.done.json")["command"]
            eval_reference=read_json(combo/f"v{n}/.evaluate.done.json")["command"]
            # First obtain cumulative stages; then complete the factorial controls.
            for variant in ("H","HV","HVD","D","V","HD"):
                h,v,d=VARIANTS[variant];out=Path(index[f"v{n}/{variant}"])
                out.mkdir(parents=True,exist_ok=True)
                init="init_hv" if h and v else "init_h" if h else "init_v" if v else "origin"
                train=baseline_command(train_reference,data,out)
                train=set_option(train,"--init_pcd_name",init)
                if d:
                    train += ["--use_drop","--drop_rate",str(config["drop_rate"]),"--drop_schedule","linear"]
                final=out/f"point_cloud/iteration_{config['iterations']}/point_cloud.ply"
                stage(out,"train",train,[final,out/"cfg_args"])
                (out/"input.ply").unlink(missing_ok=True)
                evaluation=baseline_command(eval_reference,data,out)
                evaluation=set_option(evaluation,"--init_pcd_name",init)
                images=out/f"test/ours_{config['iterations']}"
                artifacts=[images/"results.json"]+[images/f"{sub}/{i:05d}.png" for sub in ("renders","gt") for i in range(len(config["test_ids"]))]
                stage(out,"evaluate",evaluation,artifacts)
                write_json(images/"camera_names.json",read_json(combo/f"v{n}/test/ours_{config['iterations']}/camera_names.json"))
                write_json(out/"variant.json",{"H":h,"V":v,"D":d,"foreground_hull_added_only":True,
                           "background_preserved":True,"scene_init":init})
            summarize(run,index,cache,config["iterations"])
        write_json(run/"COMPLETE.json",{"groups":24,"new_training_runs":18,"reused":6,
                   "finished_at":time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"\nCOMPLETE: {run}",flush=True)


if __name__=="__main__":main()
