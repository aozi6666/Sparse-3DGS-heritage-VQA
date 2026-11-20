#!/usr/bin/env python3
"""Foreground-only silhouette intersection; retains the independent SfM background."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
from sparse_scene_init import frames_for, calibration, write_ply, write_json, sha


def project(points, frame):
    _, mask, _, R, t, k = frame
    cam = points @ R.T + t
    z = cam[:, 2]
    u = np.rint(cam[:,0]/np.maximum(z,1e-8)*k[0]+k[2]).astype(np.int64)
    v = np.rint(cam[:,1]/np.maximum(z,1e-8)*k[1]+k[3]).astype(np.int64)
    visible = (z>0)&(u>=0)&(v>=0)&(u<mask.shape[1])&(v<mask.shape[0])
    hit = np.zeros(len(points),dtype=bool)
    ii = np.flatnonzero(visible)
    hit[ii] = mask[v[ii],u[ii]]
    return visible, hit, u, v


def silhouette_membership(points, frames):
    visible = np.zeros(len(points),dtype=np.int16)
    contradicted = np.zeros(len(points),dtype=bool)
    for frame in frames:
        seen, hit, _, _ = project(points,frame)
        visible += seen
        contradicted |= seen & ~hit
    # Out-of-frame silhouette regions are unknown (cropped tail/feet).
    # All in-frame masks must agree, and >=2 views must constrain the voxel.
    return (~contradicted)&(visible>=2)


def build_hull(source, reference_data, cache, n, mask_cache, grid=144, dilation=3, max_points=60000):
    split = json.loads((reference_data/"split.json").read_text())
    ids = split["train_ids"]
    rec, images = frames_for(source,ids)
    geom = np.load(reference_data/"training_geometry.npz",allow_pickle=False)
    base = geom["xyz"]
    out = cache/f"v{n}"
    out.mkdir(parents=True,exist_ok=True)
    frames, masks_sha = [], {}
    for im in images:
        path = mask_cache/"masks_train"/(Path(im.name).stem+".png")
        # This function never reads masks_eval or test RGB/depth.
        raw = np.asarray(Image.open(path).convert("L"))>127
        rgb = np.asarray(Image.open(reference_data/"images_4"/im.name).convert("RGB"))
        if raw.shape != rgb.shape[:2]:
            raise RuntimeError(f"Mask and RGB dimensions differ: {path}")
        mask = binary_dilation(raw,iterations=dilation) if dilation else raw
        h,w = raw.shape
        R=np.asarray(im.cam_from_world().rotation.matrix());t=np.asarray(im.cam_from_world().translation)
        k=calibration(rec.cameras[im.camera_id],w,h)
        frames.append((im.name,mask,rgb,R,t,k));masks_sha[im.name]=sha(path)
    signature={"train_ids":ids,"geometry_sha256":sha(reference_data/"training_geometry.npz"),
               "train_masks":masks_sha,"grid":grid,"dilation_pixels":dilation,"max_points":max_points,
               "code_sha256":sha(Path(__file__)),"version":1}
    info=out/"hull_report.json";ply=out/"foreground_hull.ply"
    if info.exists():
        if json.loads(info.read_text())["signature"]!=signature:
            raise RuntimeError("Hull settings/masks changed; choose a new run tag")
        if ply.exists():
            print(f"[reuse] foreground hull v{n}",flush=True);return ply
    # Bound the silhouette volume using >=2 foreground observations of the
    # training-only triangulated points, without using held-out geometry.
    votes=np.zeros(len(base),dtype=int)
    for frame in frames:
        votes += project(base,frame)[1]
    seeds=base[votes>=2]
    if len(seeds)<100:
        raise RuntimeError(f"Only {len(seeds)} foreground geometry anchors for {n} views; review SAM masks")
    lo,hi=np.quantile(seeds,[.005,.995],axis=0)
    padding=np.maximum((hi-lo)*.12,np.linalg.norm(hi-lo)*.01)
    lo,hi=lo-padding,hi+padding
    axes=[np.linspace(lo[i],hi[i],grid) for i in range(3)]
    total=grid**3;occupied=np.zeros(total,dtype=bool)
    for start in range(0,total,65536):
        flat=np.arange(start,min(start+65536,total))
        ix,iy,iz=np.unravel_index(flat,(grid,grid,grid))
        points=np.column_stack([axes[0][ix],axes[1][iy],axes[2][iz]])
        occupied[flat]=silhouette_membership(points,frames)
    volume=occupied.reshape(grid,grid,grid)
    surface=volume & ~binary_erosion(volume)
    cells=np.column_stack(np.where(surface))
    if len(cells)<100:
        raise RuntimeError(f"Silhouette intersection left only {len(cells)} surface samples in v{n}; review masks; no silent fallback")
    if len(cells)>max_points:
        cells=cells[np.linspace(0,len(cells)-1,max_points,dtype=int)]
    xyz=np.column_stack([axes[i][cells[:,i]] for i in range(3)]).astype(np.float32)
    # Fixed VH addition is identical in H, HD, HV and HVD. Preserve base scene
    # points; do not remove background or change VGGT's existing point set.
    colors=np.zeros((len(xyz),3),dtype=np.float64);counts=np.zeros(len(xyz))
    for frame in frames:
        _,hit,u,v=project(xyz,frame)
        colors[hit]+=frame[2][v[hit],u[hit]];counts[hit]+=1
    assert (counts>=2).all()
    colors=np.rint(colors/counts[:,None]).astype(np.uint8)
    write_ply(ply,xyz,colors)
    write_json(info,{"signature":signature,"foreground_anchors":len(seeds),"bbox":[lo.tolist(),hi.tolist()],
               "occupied_voxels":int(volume.sum()),"surface_voxels":int(surface.sum()),"retained_points":len(xyz),
               "boundary_surface_fraction":float(((cells==0)|(cells==grid-1)).any(1).mean()),
               "rule":"all in-frame dilated training silhouettes agree; >=2 in-frame views; training-SfM foreground AABB",
               "cropped_silhouettes":"outside image is unknown, not background",
               "background_preserved":True,"test_masks_used":False,"published_exact_SynGS":False})
    rgb=frames[0][2].copy();_,hit,u,v=project(xyz,frames[0]);rgb[v[hit],u[hit]]=[255,50,20]
    Image.fromarray(rgb).save(out/"hull_projection_first_train.jpg")
    print(f"[hull v{n}] {len(seeds)} training foreground anchors -> {len(xyz)} new hull surface points",flush=True)
    return ply


def read_ply(path):
    from plyfile import PlyData
    v=PlyData.read(str(path))["vertex"]
    return np.column_stack([v[k] for k in ("x","y","z")]),np.column_stack([v[k] for k in ("red","green","blue")])


def build_union(reference_data, hull_path, target, use_hull, use_vggt):
    base,colors=read_ply(reference_data/"sparse/0/points3D.ply")
    points,rgbs=[base],[colors]
    if use_vggt:
        v,c=read_ply(reference_data/"scene_sparse_vggt.ply")
        if not np.array_equal(v[:len(base)],base) or not np.array_equal(c[:len(base)],colors):
            raise RuntimeError("Original VGGT cloud no longer has the same base geometry")
        points.append(v[len(base):]);rgbs.append(c[len(base):])
    if use_hull:
        h,c=read_ply(hull_path);points.append(h);rgbs.append(c)
    write_ply(target,np.concatenate(points),np.concatenate(rgbs))
    return {"base":len(base),"vggt":len(points[1]) if use_vggt else 0,
            "hull":len(points[-1]) if use_hull else 0,"total":sum(map(len,points))}
