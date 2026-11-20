#!/usr/bin/env python3
"""Isolated, known-pose nine-view SfM versus SfM+frozen-VGGT ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(GO), str(GO / 'tools')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_ply(path, xyz, rgb):
    import numpy as np
    from plyfile import PlyData, PlyElement
    v = np.zeros(len(xyz), dtype=[(n, 'f4') for n in ['x','y','z','nx','ny','nz']] + [(n,'u1') for n in ['red','green','blue']])
    for i, n in enumerate(['x','y','z']): v[n] = xyz[:, i]
    for i, n in enumerate(['red','green','blue']): v[n] = rgb[:, i]
    PlyData([PlyElement.describe(v, 'vertex')], text=False).write(str(path))


def load_ply(path):
    import numpy as np
    from plyfile import PlyData
    v = PlyData.read(str(path))['vertex']
    return np.column_stack([v[n] for n in ['x','y','z']]), np.column_stack([v[n] for n in ['red','green','blue']])


def check(a):
    import pycolmap
    from PIL import Image
    rec = pycolmap.Reconstruction(str(a.source / 'sparse/0'))
    if len(rec.images) < 12: raise ValueError('Need at least 12 registered images')
    for im in rec.images.values():
        with Image.open(a.source / 'images' / im.name) as rgb:
            cam = rec.cameras[im.camera_id]
            if rgb.size != (cam.width, cam.height): raise ValueError(im.name)
    for p in [a.weights, Path(os.environ.get('TORCH_HOME', str(Path.home()/'.cache/torch'))) / 'hub/checkpoints/vgg16-397923af.pth']:
        if not p.is_file(): raise FileNotFoundError(p)
    free = shutil.disk_usage(GO).free / 1024**3
    if free < 5: raise RuntimeError('Need at least 5 GiB free')
    print(json.dumps(dict(images=len(rec.images), source_points=len(rec.points3D),
        cameras=[c.model_name for c in rec.cameras.values()], free_GiB=free,
        cuda_device_present=Path('/dev/nvidia0').exists(), output=str(a.out)), indent=2), flush=True)


def prepare(a):
    """Rectify once, retain only 1/4 images; re-triangulate train-only observations."""
    import numpy as np
    import pycolmap
    from PIL import Image
    from prepare_full_scene import serial_undistort
    prepared = a.out / 'data'
    if prepared.exists():
        raise RuntimeError(f'Incomplete preparation preserved at {prepared}; use a new --out')
    rec = pycolmap.Reconstruction(str(a.source / 'sparse/0'))
    prepared.mkdir(parents=True)
    serial_undistort(rec, a.source, prepared)
    # Update cameras AND observations when shrinking the newly rectified RGB.
    for cid, cam in rec.cameras.items():
        oldw, oldh = cam.width, cam.height
        w, h = round(oldw/4), round(oldh/4)
        sx, sy = w/oldw, h/oldh
        cam.params = np.asarray(cam.params) * [sx, sy, sx, sy]
        cam.width, cam.height = w, h
        for im in rec.images.values():
            if im.camera_id != cid: continue
            path = prepared/'images'/im.name
            with Image.open(path) as image:
                small = image.convert('RGB').resize((w,h), Image.Resampling.LANCZOS)
            small.save(path, quality=95)
            for p in im.points2D: p.xy = np.asarray(p.xy) * [sx,sy]
    ordered = sorted(rec.images.values(), key=lambda im: im.name)
    # Every eighth image is held out; the nine training views uniformly span the remainder.
    test = ordered[::8]
    pool = [im for i,im in enumerate(ordered) if i % 8]
    train = [pool[int(i)] for i in np.linspace(0,len(pool)-1,9).round()]
    train_ids = {im.image_id for im in train}
    test_ids = {im.image_id for im in test}
    # Reuse established feature correspondences, but never original XYZ or test observations.
    tracks = [[(e.image_id,e.point2D_idx) for e in p.track.elements if e.image_id in train_ids]
              for p in rec.points3D.values()]
    for pid in list(rec.points3D): rec.delete_point3D(pid)
    rgb_cache = {im.image_id: np.asarray(Image.open(prepared/'images'/im.name).convert('RGB')) for im in train}
    accepted = 0
    for obs in tracks:
        if len(obs) < 2: continue
        matrices, pixels, centers = [], [], []
        for iid, idx in obs:
            im=rec.images[iid]; cam=rec.cameras[im.camera_id]
            R=np.asarray(im.cam_from_world().rotation.matrix()); t=np.asarray(im.cam_from_world().translation)
            fx,fy,cx,cy=cam.params
            xy=np.asarray(im.points2D[idx].xy)
            K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            matrices.append(K@np.column_stack([R,t])); pixels.append(xy); centers.append(-R.T@t)
        A=np.vstack([row for P,xy in zip(matrices,pixels) for row in [xy[0]*P[2]-P[0],xy[1]*P[2]-P[1]]])
        _,_,vh=np.linalg.svd(A)
        if abs(vh[-1,3]) < 1e-10: continue
        xyz=vh[-1,:3]/vh[-1,3]
        proj=np.array([P@np.r_[xyz,1.] for P in matrices])
        if not np.isfinite(xyz).all() or (proj[:,2]<=0).any(): continue
        err=np.linalg.norm(proj[:,:2]/proj[:,2,None]-pixels,axis=1)
        rays=xyz-np.array(centers); rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        angle=np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1)))
        if err.max()>2.0 or angle<1.0: continue
        colors=[]
        for (iid,idx),xy in zip(obs,pixels):
            rgb=rgb_cache[iid]; x,y=np.rint(xy).astype(int)
            if 0<=x<rgb.shape[1] and 0<=y<rgb.shape[0]: colors.append(rgb[y,x])
        if not colors: continue
        track=pycolmap.Track()
        for iid,idx in obs: track.add_element(iid,idx)
        rec.add_point3D(xyz,track,np.mean(colors,axis=0).round().astype(np.uint8)); accepted+=1
    if accepted < 100: raise RuntimeError(f'Too few nine-view triangulated points: {accepted}')
    rec.write(str(prepared/'sparse/0'))
    xyz=np.array([p.xyz for p in rec.points3D.values()]); rgb=np.array([p.color for p in rec.points3D.values()])
    save_ply(prepared/'base.ply',xyz,rgb)
    (prepared/'sparse_9.txt').write_text(''.join(f'{i}\n' for i,im in enumerate(ordered) if im.image_id in train_ids))
    (prepared/'sparse_test.txt').write_text(''.join(f'{i}\n' for i,im in enumerate(ordered) if im.image_id in test_ids))
    # Depth helper sees a separate COLMAP reconstruction containing ONLY training images.
    depthdata=a.out/'train_depth'; (depthdata/'sparse/0').mkdir(parents=True)
    (depthdata/'images').symlink_to(prepared/'images',target_is_directory=True)
    text=a.out/'colmap_text'; text.mkdir(); rec.write_text(str(text))
    shutil.copyfile(text/'cameras.txt',depthdata/'sparse/0/cameras.txt')
    shutil.copyfile(text/'points3D.txt',depthdata/'sparse/0/points3D.txt')
    lines=(text/'images.txt').read_text().splitlines(); filtered=[]; i=0
    while i<len(lines):
        line=lines[i]
        if not line or line.startswith('#'): i+=1; continue
        second=lines[i+1] if i+1<len(lines) else ''
        if int(line.split()[0]) in train_ids: filtered.extend([line,second])
        i+=2
    (depthdata/'sparse/0/images.txt').write_text('\n'.join(filtered)+'\n')
    subset=pycolmap.Reconstruction(str(depthdata/'sparse/0'))
    if {im.image_id for im in subset.images.values()}!=train_ids: raise RuntimeError('Depth train-only isolation failed')
    subset.write(str(depthdata/'sparse/0'))
    anchors={im.name:sum(pt.has_point3D() for pt in im.points2D) for im in subset.images.values()}
    if min(anchors.values())<30: raise RuntimeError(f'Insufficient train-only scale anchors: {anchors}')
    write_json(a.out/'train_scale_anchors.json', anchors)
    centers=np.array([im.cam_from_world().inverse().translation for im in subset.images.values()])
    extent=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    write_json(depthdata/'full_scene_manifest.json',dict(scene_extent=extent))
    save_ply(a.out/'A_init_sfm.ply',xyz,rgb)
    write_json(a.out/'split.json',dict(train=[im.name for im in train], test=[im.name for im in test],
        display_camera=Path(test[len(test)//2].name).stem, base_points=accepted,
        triangulation='DLT from nine train views only; max reprojection error 2 px at quarter resolution; min angle 1 degree',
        calibration_and_correspondences='Existing full-sequence COLMAP camera poses and feature tracks. No original 3D XYZ, test observations, or test RGB used in triangulation or depth alignment. Not raw-nine-image calibration.'))


def depth(a):
    from full_scene_depth_init import predict, build_init
    data=a.out/'train_depth'
    predict(data,a.weights,1)
    build_init(data,1,100000)
    shutil.copyfile(data/'scene_sfm_vggt.ply',a.out/'B_init_sfm_vggt.ply')
    shutil.copyfile(data/'scene_sfm_vggt.ply',a.out/'data/augmented.ply')
    report=json.loads((data/'scene_init_report.json').read_text())
    if report['new_points'] == 0: raise RuntimeError('VGGT accepted zero points; no valid supplementation treatment')
    report['evaluation_scope']='nine train frames only; fixed camera poses; independent held-out evaluation'
    write_json(a.out/'supplementation.json',report)


def params(a, variant):
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.full_scene import full_scene_options
    p=argparse.ArgumentParser(); mp=ModelParams(p); op=OptimizationParams(p); pp=PipelineParams(p)
    args=p.parse_args([])
    args.source_path=str(a.out/'data'); args.model_path=str(a.out/variant)
    args.images='images'; args.resolution=1; args.sh_degree=2; args.eval=False
    args.full_scene=True; args.sparse_view_num=9
    args.init_pcd_name='base.ply' if variant=='A' else 'augmented.ply'
    args.iterations=a.steps; args.position_lr_max_steps=a.steps
    args.densify_until_iter=a.steps//2; args.max_num_splats=300000
    args.use_dust3r=False; args.transform_the_world=False; args.use_drop=False
    args.drop_rate=0.; args.drop_schedule='linear'; args.mono_depth_weight=0.
    args.mono_loss_type='mid'; args.lambda_t_norm=0.
    full_scene_options(args)
    return args,mp,op,pp


def train(a,variant):
    import train_gs
    from utils.general_utils import safe_state
    args,mp,op,pp=params(a,variant)
    def logger(ds):
        Path(ds.model_path).mkdir(exist_ok=True)
        (Path(ds.model_path)/'cfg_args').write_text(str(argparse.Namespace(**vars(ds))))
        return None
    train_gs.prepare_output_and_logger=logger
    safe_state(False)
    train_gs.training(args,mp.extract(args),op.extract(args),pp.extract(args),[],[a.steps],[],None,-1)


def evaluate(a):
    import csv
    import numpy as np
    import torch
    import lpips
    from PIL import Image, ImageDraw
    from scene import Scene
    from gaussian_renderer import GaussianModel, render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    display=json.loads((a.out/'split.json').read_text())['display_camera']
    gallery=a.out/'visualization'; gallery.mkdir(exist_ok=True)
    metric=lpips.LPIPS(net='vgg').cuda().eval(); rows=[]; summaries={}
    for variant in ['A','B']:
        args,mp,op,pp=params(a,variant)
        g=GaussianModel(2); scene=Scene(mp.extract(args),g,load_iteration=a.steps,shuffle=False,extra_opts=args)
        views=scene.getTestCameras()
        with torch.no_grad():
            for view in views:
                image=render(view,g,pp.extract(args),torch.zeros(3,device='cuda'))['render'].clamp(0,1)
                image=torch.round(image*255)/255; gt=torch.round(view.original_image[:3].cuda().clamp(0,1)*255)/255
                row=dict(variant=variant,camera=view.image_name,PSNR=psnr(image,gt).mean().item(),SSIM=ssim(image,gt).mean().item(),LPIPS=metric(image[None],gt[None],normalize=True).item())
                row['LPIPS_x100']=row['LPIPS']*100; rows.append(row)
                if view.image_name==display:
                    for file,tensor in [(f'{variant}.png',image),('GT.png',gt)]:
                        Image.fromarray((tensor.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(gallery/file)
        summaries[variant]={k:float(np.mean([r[k] for r in rows if r['variant']==variant])) for k in ['PSNR','SSIM','LPIPS','LPIPS_x100']}
        summaries[variant]['final_gaussians']=len(g.get_xyz)
        summaries[variant]['initial_points']=len(load_ply(a.out/('A_init_sfm.ply' if variant=='A' else 'B_init_sfm_vggt.ply'))[0])
        del scene,g; torch.cuda.empty_cache()
    with (a.out/'per_view.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summaries['difference_B_minus_A']={k:summaries['B'][k]-summaries['A'][k] for k in ['PSNR','SSIM','LPIPS']}
    write_json(a.out/'test_mean_metrics.json',summaries)
    with (a.out/'comparison_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['variant','initial_points','final_gaussians','PSNR','SSIM','LPIPS','LPIPS_x100'])
        w.writeheader()
        for variant in ['A','B']: w.writerow(dict(variant=variant,**summaries[variant]))
    for variant in ['A','B']:
        r=next(r for r in rows if r['variant']==variant and r['camera']==display)
        image=Image.open(gallery/f'{variant}.png'); labeled=Image.new('RGB',(image.width,image.height+65),'white'); labeled.paste(image,(0,0))
        ImageDraw.Draw(labeled).text((12,image.height+12),f"{variant}: PSNR {r['PSNR']:.3f}  SSIM {r['SSIM']:.4f}  LPIPS {r['LPIPS']:.4f}",fill='black')
        labeled.save(gallery/f'{variant}_labeled.jpg',quality=95)
    images=[Image.open(gallery/f) for f in ['A_labeled.jpg','B_labeled.jpg','GT.png']]
    sheet=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),'white'); x=0
    for im in images: sheet.paste(im,(x,0)); x+=im.width
    sheet.save(gallery/'comparison.jpg',quality=95)
    point_plots(a)
    write_json(a.out/'complete.json',dict(completed=True,display_camera=display,scope='Single-scene known-pose nine-view ablation; means over fixed held-out views, not multi-scene means.'))


def point_plots(a):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz0,rgb0=load_ply(a.out/'A_init_sfm.ply'); xyz1,rgb1=load_ply(a.out/'B_init_sfm_vggt.ply')
    lo,hi=xyz1.min(0),xyz1.max(0); center=(lo+hi)/2; radius=max(hi-lo)/2
    for label,xyz,rgb in [('A',xyz0,rgb0),('B',xyz1,rgb1)]:
        idx=np.linspace(0,len(xyz)-1,min(60000,len(xyz))).round().astype(int)
        fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,projection='3d')
        ax.scatter(*xyz[idx].T,c=rgb[idx]/255,s=.5,depthshade=False)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=15,azim=-65)
        ax.set_title(f'{label} initialization: {len(xyz):,} points (display <=60k)')
        fig.savefig(a.out/'visualization'/f'{label}_points.png',dpi=180); plt.close(fig)
    write_json(a.out/'point_counts.json',dict(A_initial=len(xyz0),B_initial=len(xyz1),added=len(xyz1)-len(xyz0)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=GO/'data/LLFF/horns')
    p.add_argument('--out',type=Path,default=GO/'output/horns_vggt9_v1')
    p.add_argument('--weights',type=Path,default=GO.parent/'vggt/models/model.pt')
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--stage',choices=['check','prepare','depth','A','B','evaluate','run'],default='check')
    a=p.parse_args();
    if a.steps < 1000: p.error('--steps must be >=1000')
    a.source=a.source.resolve(); a.out=a.out.resolve(); a.weights=a.weights.resolve()
    os.chdir(GO)
    if a.stage=='check': check(a); return
    if a.stage=='run':
        import fcntl
        limit=Path('/sys/fs/cgroup/memory.max')
        if limit.exists() and limit.read_text().strip() != 'max' and int(limit.read_text()) < 12*1024**3:
            raise RuntimeError('VGGT checkpoint loading requires >=12 GiB instance RAM; current CPU instance is for check/prepare only')
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('Enable GPU before --stage run')
        check(a); a.out.mkdir(parents=True,exist_ok=True)
        lock=(a.out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature=dict(source=str(a.source),steps=a.steps,weight_sha256=sha(a.weights),script_sha256=sha(Path(__file__)),
            helper_sha256={str(f.relative_to(GO)):sha(f) for f in [GO/'train_gs.py',GO/'tools/full_scene_depth_init.py',GO/'tools/prepare_full_scene.py',GO/'scene/gaussian_model.py',GO/'scene/dataset_readers.py']},
            source_files={str(f.relative_to(a.source)):sha(f) for f in sorted((a.source/'sparse/0').glob('*.bin'))},
            rgb={im.name:sha(im) for im in sorted((a.source/'images').iterdir()) if im.suffix.lower() in ['.jpg','.png','.jpeg'] and not im.name.startswith('._')})
        cfg=a.out/'config.json'
        if cfg.exists() and json.loads(cfg.read_text())!=signature: raise RuntimeError('Configuration changed; use new --out')
        write_json(cfg,signature)
        (a.out/'logs').mkdir(exist_ok=True)
        for stage in ['prepare','depth','A','B','evaluate']:
            stamp=a.out/f'{stage}.done.json'
            if stamp.exists(): print('[reuse]',stage,flush=True); continue
            if shutil.disk_usage(GO).free<3*1024**3: raise RuntimeError('Free disk below 3 GiB')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--stage',stage,'--source',str(a.source),'--out',str(a.out),'--weights',str(a.weights),'--steps',str(a.steps)]
            with (a.out/'logs'/f'{stage}.log').open('a') as log:
                print('[start]',stage,'log:',log.name,flush=True)
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            write_json(stamp,dict(stage=stage,done=True))
        return
    if a.stage=='prepare': prepare(a); return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; --stage check/prepare are CPU compatible')
    if a.stage=='depth': depth(a)
    elif a.stage in ['A','B']: train(a,a.stage)
    elif a.stage=='evaluate': evaluate(a)


if __name__=='__main__': main()
