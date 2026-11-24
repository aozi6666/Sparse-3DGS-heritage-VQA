#!/usr/bin/env python3
"""Resume-safe all-view full-RGB reconstruction runner; legacy defaults unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

GO=Path(__file__).resolve().parents[1]
PY=sys.executable


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    os.replace(tmp,path)


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def cpu_memory_limit():
    limits=[]
    for name in ("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        path=Path(name)
        if path.exists():
            value=path.read_text().strip()
            if value.isdigit():limits.append(int(value))
    limits.append(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"))
    return min(limits)


def execute(command,log):
    print("\n[run]",shlex.join([str(x) for x in command]),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as f:
        f.write("\nCOMMAND "+shlex.join([str(x) for x in command])+"\n")
        f.flush()
        process=subprocess.Popen([str(x) for x in command],cwd=GO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end="",flush=True);f.write(line);f.flush()
        code=process.wait()
    if code: raise RuntimeError(f"Command failed ({code}); see {log}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=GO/"data/LLFF/trex")
    p.add_argument("--prepared",type=Path,default=GO/"data/LLFF/trex_fullscene")
    p.add_argument("--run-tag",default="trex_fullscene_v1")
    p.add_argument("--resolution",type=int,choices=[2,4,8],default=4)
    p.add_argument("--iterations",type=int,default=30000)
    p.add_argument("--loo-iterations",type=int,default=10000)
    p.add_argument("--lora-steps",type=int,default=1800)
    p.add_argument("--repair-steps",type=int,default=4000)
    p.add_argument("--max-gaussians",type=int,default=1000000)
    p.add_argument("--loo-max-gaussians",type=int,default=300000)
    p.add_argument("--drop-rate",type=float,default=.2)
    p.add_argument("--sd-ckpt",type=Path,default=GO/"models/absolutereality_v16.safetensors")
    p.add_argument("--vggt-weights",type=Path,default=GO/"third_party"/"vggt/models/model.pt")
    p.add_argument("--camera-name",default=None)
    p.add_argument("--prompt",default="museum dinosaur skeleton and exhibition hall")
    p.add_argument("--start",choices=["depth","train","loo","lora","repair","export"],default="depth")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--plan",action="store_true",help="Read-only plan; no preprocessing or CUDA")
    p.add_argument("--check",action="store_true",help="Read-only environment/data/GPU report")
    a=p.parse_args()
    if Path(a.run_tag).name!=a.run_tag or a.run_tag in ("",".",".."): raise ValueError("run-tag must be one directory name")
    if a.iterations<100 or a.loo_iterations<=6000 or a.lora_steps<1 or a.repair_steps<1: raise ValueError("LOO must finish after checkpoint 6000")
    if not 0<=a.drop_rate<1: raise ValueError("drop-rate must be in [0,1)")
    a.source=a.source.resolve();a.prepared=a.prepared.resolve()
    run=GO/"output/full_scene"/a.run_tag
    manifest=a.prepared/"full_scene_manifest.json"
    if manifest.exists():
        meta=json.loads(manifest.read_text())
    else:
        import pycolmap
        rec=pycolmap.Reconstruction(str(a.source/"sparse/0"))
        names=sorted(im.name for im in rec.images.values())
        meta={"train_count":len(names),"names":names,"display_camera":Path(names[len(names)//2]).stem}
    n=meta["train_count"]
    camera=Path(a.camera_name).stem if a.camera_name else meta["display_camera"]
    if camera not in {Path(name).stem for name in meta["names"]}: raise ValueError(f"Unknown display camera: {camera}")
    saves=sorted({max(1,round(a.iterations*f)) for f in (1/60,1/30,.1,7/30,.5,1)})
    if len(saves)!=6: raise ValueError("Need six distinct coarse checkpoints")
    stages=["depth","train","loo","lora","repair","export"]
    coarse=run/"coarse";loo=run/"loo";lora_rel=f"full_scene/{a.run_tag}/controlnet"
    final=run/"refinement/final/save/last.ply"
    summary={"source":str(a.source),"prepared":str(a.prepared),"run":str(run),
             "train_views":n,"test_views":0,"mode":"full_scene","init":"SfM_background_plus_VGGT_not_VisualHull",
             "camera":camera,"coarse_checkpoints":saves,"final_ply":str(final),
             "stages":stages[stages.index(a.start):],"seven_renderings":str(run/"display")}
    if a.plan:
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    if a.check:
        import importlib.util
        import torch
        summary.update(cuda_available=torch.cuda.is_available(),gpu_count=torch.cuda.device_count(),
                       pycolmap_available=importlib.util.find_spec("pycolmap") is not None,
                       cpu_memory_limit_gib=round(cpu_memory_limit()/1024**3,2),
                       weights_present={str(x):x.is_file() for x in (a.vggt_weights,a.sd_ckpt,GO/"models/control_v11f1e_sd15_tile.pth")},
                       free_disk_gib=round(shutil.disk_usage(GO).free/1024**3,2))
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    from prepare_full_scene import prepare
    prepare(a.source,a.prepared)
    meta=json.loads(manifest.read_text())
    if a.prepare_only: return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. CPU preparation is complete; enable an AutoDL GPU and rerun the same command.")
    if cpu_memory_limit()<12*1024**3 and stages.index(a.start)<=4:
        raise RuntimeError("Full VGGT/LoRA/repair requires at least 12 GiB CPU memory for checkpoint loading. Increase the AutoDL instance memory and rerun.")
    for weight in (a.vggt_weights,a.sd_ckpt,GO/"models/control_v11f1e_sd15_tile.pth"):
        if not weight.is_file(): raise FileNotFoundError(weight)
    if shutil.disk_usage(GO).free<8*1024**3: raise RuntimeError("Need at least 8 GiB free disk for staged LOO")
    config={**vars(a),"prepared_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "weight_hashes":{str(path):file_sha256(path) for path in
                            (a.vggt_weights,a.sd_ckpt,GO/"models/control_v11f1e_sd15_tile.pth")},
            "display_camera":camera,"coarse_saves":saves,"train_views":n,
            "code_hashes":{name:hashlib.sha256((GO/name).read_bytes()).hexdigest() for name in (
                "train_gs.py","leave_one_out_stage1.py","leave_one_out_stage2.py","train_lora.py",
                "dataset_lora.py","render.py","utils/full_scene.py","tools/full_scene_depth_init.py",
                "threestudio/data/loo.py","threestudio/systems/gaussian_object_system.py")}}
    for key in ("start","prepare_only","plan","check"): config.pop(key,None)
    config=json.loads(json.dumps(config,default=str))
    cfg=run/"run_config.json"
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise RuntimeError(f"Run configuration changed. Use a new --run-tag; existing run preserved: {run}")
    write_json(cfg,config)
    logs=run/"logs"
    common=["-s",a.prepared,"-r",a.resolution,"--sparse_view_num",n,"--sh_degree",2,
            "--init_pcd_name","scene_sfm_vggt","--full_scene"]

    def stage(name,command,required):
        stamp=run/"stages"/f"{name}.json"
        if stamp.exists() and all(Path(x).exists() for x in required):
            print("[reuse]",name,flush=True);return
        started=time.time()
        execute(command,logs/f"{name}.log")
        if not all(Path(x).exists() for x in required): raise RuntimeError(f"Stage {name} missing expected files: {required}")
        write_json(stamp,{"stage":name,"seconds":time.time()-started,"outputs":[str(x) for x in required]})

    start=stages.index(a.start)
    if start<=0:
        stage("depth",[PY,"tools/full_scene_depth_init.py","--data",a.prepared,"--weights",a.vggt_weights,
              "--resolution",a.resolution],[a.prepared/"scene_sfm_vggt.ply",a.prepared/"scene_init_report.json"])
    if not (a.prepared/"scene_sfm_vggt.ply").exists(): raise FileNotFoundError("Run the depth stage first")
    if start<=1:
        cmd=[PY,"train_gs.py",*common,"-m",coarse,"--iterations",a.iterations,
             "--position_lr_max_steps",a.iterations,"--densify_until_iter",int(a.iterations*.5),
             "--max_num_splats",a.max_gaussians,"--save_iterations",*saves,"--test_iterations",a.iterations,
             "--checkpoint_iterations",a.iterations]
        if a.drop_rate:cmd+=["--use_drop","--drop_rate",a.drop_rate,"--drop_schedule","linear"]
        stage("train",cmd,[coarse/"point_cloud"/f"iteration_{i}"/"point_cloud.ply" for i in saves])
    if start<=2:
        for index in range(n):
            leaf=loo/f"leave_{index}"
            done=leaf/"full_scene_complete.json"
            if done.exists() and (leaf/"diffs.pkl").exists() and any((leaf/"left_image").glob("*.png")):
                print(f"[reuse] LOO {index+1}/{n}",flush=True);continue
            shared=[*common,"-m",loo,"--iterations",a.loo_iterations,"--max_num_splats",a.loo_max_gaussians,
                    "--save_iterations",a.loo_iterations,"--test_iterations",a.loo_iterations,"--leave_out_ids",index]
            execute([PY,"leave_one_out_stage1.py",*shared,"--checkpoint_iterations",6000],logs/f"loo_{index:03d}_1.log")
            execute([PY,"leave_one_out_stage2.py",*shared],logs/f"loo_{index:03d}_2.log")
            if not (leaf/"diffs.pkl").exists() or not any((leaf/"left_image").glob("*.png")):
                raise RuntimeError(f"Incomplete LOO pair for {index}")
            # Clean only this new run's temporary LOO models, after paired samples exist.
            target=leaf/"point_cloud"
            if target.is_dir():shutil.rmtree(target)
            for target in leaf.glob("*.pth"):target.unlink()
            for target in leaf.glob("events.out.tfevents.*"):target.unlink()
            target=leaf/"input.ply"
            if target.exists():target.unlink()
            write_json(done,{"image_index":index,"frame":meta["names"][index],"samples_retained":True,
                            "transient_checkpoint_removed_after_stage2":True})
    if start<=3:
        stage("lora",[PY,"train_lora.py","--data_dir",a.prepared,"--gs_dir",coarse,"--loo_dir",loo,
              "--exp_name",lora_rel,"--sd_ckpt",a.sd_ckpt,"--prompt",a.prompt,"--seed",0,
              "--sh_degree",2,"--resolution",a.resolution,"--sparse_num",n,"--image_size",512,
              "--batch_size",1,"--lora_rank",32,"--max_steps",a.lora_steps,
              "--callbacks_every_n_train_steps",min(600,a.lora_steps),"--cache_max_iter",100,
              "--full_scene","--sd_locked","--train_lora","--add_diffusion_lora","--add_control_lora"],
              [run/"controlnet/ckpts-lora"/f"lora-step={a.lora_steps-1}.ckpt"])
    if start<=4:
        stage("repair",[PY,"train_repair.py","--config","configs/gaussian-object.yaml","--train",
              "--gpu",os.environ.get("CUDA_VISIBLE_DEVICES","0"),
              f"exp_root_dir={run}","name=refinement","tag=final",
              f"system.init_dreamer={coarse}",f"system.exp_name={run/'controlnet'}",
              f"system.sd_ckpt={a.sd_ckpt}",f"system.lora_name=lora-step={a.lora_steps-1}.ckpt",
              "system.lora_rank=32","system.add_clip_lora=False","system.full_scene=True","data.full_scene=True",
              "data.bg_white=False",f"data.data_dir={a.prepared}",f"data.resolution={a.resolution}",
              f"data.sparse_num={n}",f"data.prompt={json.dumps(a.prompt)}",
              f"system.scene_extent={meta['scene_extent']}","system.use_conf_weight=False",
              "system.refresh_size=8","data.refresh_size=8","system.save_controlnet_preview=True",
              "system.gaussian_opt_params.random_background=False",
              f"trainer.max_steps={a.repair_steps}",f"system.gaussian_opt_params.iterations={a.repair_steps}",
              f"data.length={a.repair_steps}",f"checkpoint.every_n_train_steps={a.repair_steps}"],[final])
    if start<=5:
        export=[]
        for iteration in saves:
            execute([PY,"render.py",*common,"-m",coarse,"--iteration",iteration,"--camera_name",camera,
                     "--skip_test","--skip_all","--not_generate_video"],logs/f"render_{iteration}.log")
            folder=coarse/"train"/f"ours_{iteration}"
            export.append((f"coarse_{iteration}",folder/"renders/00000.png",folder/"gt/00000.png"))
        execute([PY,"render.py",*common,"-m",coarse,"--load_ply",final,"--camera_name",camera,
                 "--skip_test","--skip_all","--not_generate_video"],logs/"render_final.log")
        export.append(("repair_final",coarse/"train/ours_None/renders/00000.png",coarse/"train/ours_None/gt/00000.png"))
        display=run/"display";display.mkdir(exist_ok=True)
        scored=[]
        for label,image,reference in export:
            rgb=np.asarray(Image.open(image).convert("RGB"),dtype=np.float64)/255
            gt=np.asarray(Image.open(reference).convert("RGB"),dtype=np.float64)/255
            if rgb.shape!=gt.shape:raise ValueError("Display camera dimensions differ")
            mse=float(np.mean((rgb-gt)**2));score=float(-10*np.log10(max(mse,1e-12)))
            scored.append({"stage":label,"source":str(image),"training_view_psnr":score})
        scored.sort(key=lambda x:x["training_view_psnr"],reverse=True)
        grid=Image.new("RGB",(448*7,372),"white");draw=ImageDraw.Draw(grid)
        for i,item in enumerate(scored,1):
            dest=display/f"{i:02d}.png";shutil.copyfile(item["source"],dest)
            im=Image.open(dest).convert("RGB").resize((448,336),Image.Resampling.LANCZOS)
            grid.paste(im,((i-1)*448,36));draw.text(((i-1)*448+218,10),str(i),fill="black")
            item.update(rank=i,file=dest.name)
        grid.save(display/"comparison_1_to_7.png")
        shutil.copyfile(export[0][2],display/"reference.png")
        write_json(display/"render_manifest.json",{"camera":camera,"sort":"training_view_PSNR_descending",
                   "all_source_views_used_for_training":True,"not_held_out_evaluation":True,
                   "entries":scored,"final_model_rank":next(x["rank"] for x in scored if x["stage"]=="repair_final")})
        print("[complete]",display/"comparison_1_to_7.png",flush=True)


if __name__=="__main__":
    main()
