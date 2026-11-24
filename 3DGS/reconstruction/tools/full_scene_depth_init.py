#!/usr/bin/env python3
"""Existing VGGT weights -> COLMAP-aligned float depth -> full-scene initial PLY.

All SfM background points are retained. This is an explicit full-scene variant,
not the silhouette-based published SynGS initialization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image

GO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(GO))
sys.path.insert(0,str(GO/"third_party"/"vggt"))
from colmap_sparse_depth import qvec2rotmat  # tools directory is on sys.path


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def calibration(camera,width,height):
    if camera.model_name=="PINHOLE": fx,fy,cx,cy=camera.params
    elif camera.model_name=="SIMPLE_PINHOLE":
        fx,cx,cy=camera.params;fy=fx
    else: raise ValueError("Run prepare_full_scene.py first")
    sx,sy=width/camera.width,height/camera.height
    return np.array([fx*sx,fy*sy,cx*sx,cy*sy])


def predict(data,weights,resolution):
    import pycolmap
    import torch
    import torch.nn.functional as F
    from vggt.models.vggt import VGGT
    from vggt_depthft_ckpt import load_vggt_with_optional_depth_head
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for VGGT depth inference")
    rec=pycolmap.Reconstruction(str(data/"sparse/0"))
    depth_dir=data/"zoe_depth"; conf_dir=data/"vggt_conf"
    depth_dir.mkdir(exist_ok=True);conf_dir.mkdir(exist_ok=True)
    weights_hash=sha(weights)
    model=None
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for im in sorted(rec.images.values(),key=lambda x:x.name):
        stem=Path(im.name).stem
        image=data/f"images_{resolution}"/im.name if resolution!=1 else data/"images"/im.name
        signature={"weights_sha256":weights_hash,"image_sha256":sha(image),"backend":"VGGT_single_frame_colmap_median_scale","version":1}
        info=depth_dir/f"{stem}.json"
        if info.exists() and (depth_dir/f"{stem}.npy").exists() and (conf_dir/f"{stem}.npy").exists():
            previous=json.loads(info.read_text())
            if previous.get("signature")!=signature: raise RuntimeError(f"Stale depth; use a new prepared directory: {stem}")
            dep=np.load(depth_dir/f"{stem}.npy",allow_pickle=False)
            if np.isfinite(dep).all() and (dep>0).all():
                print("[reuse] depth",stem,flush=True);continue
        if model is None:
            model=VGGT()
            load_vggt_with_optional_depth_head(model,str(weights))
            model=model.eval().cuda()
        rgb=np.asarray(Image.open(image).convert("RGB"))
        height,width=rgb.shape[:2]
        ratio=min(1.0,518/max(height,width))
        nh=max(14,round(height*ratio/14)*14); nw=max(14,round(width*ratio/14)*14)
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
        tensor=F.interpolate(tensor[None],size=(nh,nw),mode="bilinear",align_corners=False)
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=dtype):
            result=model(tensor[None])
        dep=result["depth"].float().squeeze().detach()
        conf=result["depth_conf"].float().squeeze().detach()
        dep=F.interpolate(dep[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        conf=F.interpolate(conf[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0].cpu().numpy()
        camera=rec.cameras[im.camera_id]
        R=np.asarray(im.cam_from_world().rotation.matrix());t=np.asarray(im.cam_from_world().translation)
        ratios=[]
        for point in im.points2D:
            if not point.has_point3D() or point.point3D_id not in rec.points3D: continue
            z=(R@np.asarray(rec.points3D[point.point3D_id].xyz)+t)[2]
            u,v=np.asarray(point.xy)*[width/camera.width,height/camera.height]
            u,v=int(round(u)),int(round(v))
            if z>0 and 0<=u<width and 0<=v<height and np.isfinite(dep[v,u]) and dep[v,u]>0:
                ratios.append(z/dep[v,u])
        if len(ratios)<30: raise RuntimeError(f"Insufficient scale anchors for {stem}: {len(ratios)}")
        scale=float(np.median(ratios));dep=(dep*scale).astype(np.float32)
        if not np.isfinite(dep).all() or (dep<=0).any() or not np.isfinite(conf).all():
            raise ValueError(f"Nonfinite/nonpositive VGGT output for {stem}")
        np.save(depth_dir/f"{stem}.npy",dep,allow_pickle=False)
        np.save(conf_dir/f"{stem}.npy",conf.astype(np.float32),allow_pickle=False)
        info.write_text(json.dumps({"signature":signature,"scale":scale,"n_scale_anchors":len(ratios),"shape":[height,width],"unit":"COLMAP_scene_coordinates","not_ground_truth":True},indent=2)+"\n")
        print(f"[depth] {stem} scale={scale:.5g} anchors={len(ratios)}",flush=True)
        del result,tensor
    if model is not None:
        del model
        torch.cuda.empty_cache()


def build_init(data,resolution,max_new_points=100000):
    import pycolmap
    from plyfile import PlyData,PlyElement
    rec=pycolmap.Reconstruction(str(data/"sparse/0"))
    frames=sorted(rec.images.values(),key=lambda x:x.name)
    base=np.array([p.xyz for p in rec.points3D.values()],dtype=np.float32)
    colors=np.array([p.color for p in rec.points3D.values()],dtype=np.uint8)
    if not len(base): raise ValueError("No SfM points")
    manifest=json.loads((data/"full_scene_manifest.json").read_text())
    radius=manifest["scene_extent"]
    lo,hi=np.quantile(base,[.005,.995],axis=0); pad=(hi-lo)*.1
    cache=[]
    for im in frames:
        stem=Path(im.name).stem
        depth=np.load(data/"zoe_depth"/f"{stem}.npy",allow_pickle=False)
        conf=np.load(data/"vggt_conf"/f"{stem}.npy",allow_pickle=False)
        rgb=np.asarray(Image.open(data/f"images_{resolution}"/im.name if resolution!=1 else data/"images"/im.name).convert("RGB"))
        h,w=depth.shape
        if rgb.shape[:2]!=(h,w): raise ValueError("Depth/RGB mismatch")
        R=np.asarray(im.cam_from_world().rotation.matrix());t=np.asarray(im.cam_from_world().translation)
        cache.append((depth,conf,rgb,R,t,calibration(rec.cameras[im.camera_id],w,h),-R.T@t))
    points=[base];rgbs=[colors];records=[]
    per_frame=max(1,max_new_points//len(frames))
    for index,(depth,conf,rgb,R,t,k,C) in enumerate(cache):
        y,x=np.mgrid[0:depth.shape[0]:8,0:depth.shape[1]:8]
        y,x=y.ravel(),x.ravel();z=depth[y,x]
        good=np.isfinite(z)&(z>0)&(conf[y,x]>=np.quantile(conf,.5))
        x,y,z=x[good],y[good],z[good]
        xyzc=np.stack([(x-k[2])/k[0]*z,(y-k[3])/k[1]*z,z],axis=1)
        world=(xyzc-t)@R
        good=((world>=lo-pad)&(world<=hi+pad)).all(axis=1)
        support=np.zeros(len(world),dtype=np.int32)
        neighbours=np.argsort([np.linalg.norm(item[6]-C) for item in cache])
        for ni in [int(n) for n in neighbours if n!=index][:3]:
            other,_,_,r2,t2,k2,_=cache[ni]
            proj=world@r2.T+t2;zz=proj[:,2]
            u=np.rint(proj[:,0]/np.maximum(zz,1e-6)*k2[0]+k2[2]).astype(int)
            v=np.rint(proj[:,1]/np.maximum(zz,1e-6)*k2[1]+k2[3]).astype(int)
            inside=(zz>0)&(u>=0)&(v>=0)&(u<other.shape[1])&(v<other.shape[0])
            idx=np.flatnonzero(inside)
            support[idx]+=np.abs(other[v[idx],u[idx]]-zz[idx])<=np.maximum(.05*zz[idx],.01*radius)
        selected=np.flatnonzero(good&(support>=2))
        if len(selected)>per_frame: selected=selected[np.linspace(0,len(selected)-1,per_frame,dtype=int)]
        points.append(world[selected].astype(np.float32));rgbs.append(rgb[y[selected],x[selected]])
        records.append({"frame":frames[index].name,"candidates":len(world),"accepted":len(selected)})
    xyz=np.concatenate(points);rgb=np.concatenate(rgbs)
    # Preserve every original SfM point; voxel-filter supplementary points only.
    if len(xyz)>len(base):
        vox=np.floor(xyz[len(base):]/max(radius/500,1e-5)).astype(np.int64)
        _,unique=np.unique(vox,axis=0,return_index=True)
        keep=np.r_[np.arange(len(base)),len(base)+np.sort(unique)]
        xyz,rgb=xyz[keep],rgb[keep]
    fields=[("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4"),("red","u1"),("green","u1"),("blue","u1")]
    values=np.empty(len(xyz),dtype=fields)
    for i,name in enumerate(("x","y","z")): values[name]=xyz[:,i]
    for name in ("nx","ny","nz"): values[name]=0
    for i,name in enumerate(("red","green","blue")): values[name]=rgb[:,i]
    out=data/"scene_sfm_vggt.ply"
    PlyData([PlyElement.describe(values,"vertex")]).write(str(out))
    report={"variant":"full_scene_sfm_plus_vggt","visual_hull_used":False,"base_points":len(base),"new_points":len(xyz)-len(base),"total_points":len(xyz),"frames":records,"evaluation_scope":"all_view_demonstration"}
    (data/"scene_init_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"[init] {out}: {len(base)} SfM + {len(xyz)-len(base)} supplemental points",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--weights",type=Path,default=GO/"third_party"/"vggt/models/model.pt")
    p.add_argument("--resolution",type=int,choices=[1,2,4,8],default=4)
    p.add_argument("--max_new_points",type=int,default=100000)
    a=p.parse_args()
    predict(a.data.resolve(),a.weights.resolve(),a.resolution)
    build_init(a.data.resolve(),a.resolution,a.max_new_points)
