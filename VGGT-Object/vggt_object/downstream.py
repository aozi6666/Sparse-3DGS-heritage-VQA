"""Known-pose 3DGS adapter, depth inference uses input RGB only, no GT depth alignment."""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image
import torch
from .common import REPO,json_dump,seed_all
from .data import read_object,to_device
from .model import predict
from .geometry import backproject


def camera_scale(pred_E,known_E):
    """Scale from camera-center baselines; never fit predicted depths to GT depths."""
    def centers(E):
        return -(E[:,:3,:3].transpose(-1,-2)@E[:,:3,3,None])[...,0]
    p,g=centers(pred_E),centers(known_E)
    x=p-p.mean(0);y=g-g.mean(0)
    if x.square().sum()<1e-8 or y.square().sum()<1e-8:
        raise ValueError('Degenerate camera-center baseline; cannot align scale without depth GT')
    u,s,vh=torch.linalg.svd(y.T@x)
    correction=torch.eye(3,device=x.device);correction[-1,-1]=torch.det(u@vh)
    scale=(s*correction.diag()).sum()/x.square().sum()
    rotation=u@correction@vh
    rmse=((scale*x@rotation.T-y).square().sum(-1).mean()).sqrt()
    if not torch.isfinite(scale) or scale<=0:
        raise ValueError('Invalid pose-baseline scale')
    return scale,float(rmse)


def write_ply(path,points,colors):
    from plyfile import PlyData,PlyElement
    v=np.zeros(len(points),dtype=[(n,'f4') for n in ('x','y','z','nx','ny','nz')]+[(n,'u1') for n in ('red','green','blue')])
    for i,n in enumerate(('x','y','z')):v[n]=points[:,i]
    for i,n in enumerate(('red','green','blue')):v[n]=colors[:,i]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(path))


def write_colmap(folder,frames,side):
    """Text bridge supports both older and rig-based pycolmap APIs."""
    import pycolmap
    from scipy.spatial.transform import Rotation
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cameras=[];images=[]
    for i,(name,K,E) in enumerate(frames,1):
        cameras.append(f'{i} PINHOLE {side} {side} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}')
        q=Rotation.from_matrix(E[:3,:3]).as_quat()  # xyzw -> COLMAP wxyz
        values=[q[3],q[0],q[1],q[2],*E[:3,3]]
        images.extend([f'{i} '+ ' '.join(str(float(v)) for v in values)+f' {i} {name}',''])
    (folder/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (folder/'images.txt').write_text('\n'.join(images)+'\n')
    (folder/'points3D.txt').write_text('')
    rec=pycolmap.Reconstruction(str(folder));rec.write(str(folder))
    return rec


def nested_camera_split(frames,views):
    """One held-out set for every budget; 4-view inputs are a subset of 6 and 9."""
    n=len(frames);test=list(range(0,n,8));pool=[i for i in range(n) if i not in test]
    if len(pool)<views:raise ValueError('Not enough non-test frames')
    centers=[]
    for f in frames:
        e=np.asarray(f['E']);centers.append(-e[:3,:3].T@e[:3,3])
    centers=np.asarray(centers);chosen=[pool[0]]
    while len(chosen)<min(9,len(pool)):
        remaining=[i for i in pool if i not in chosen]
        distances=np.linalg.norm(centers[remaining,None]-centers[np.array(chosen)][None],axis=-1).min(axis=1)
        chosen.append(remaining[int(np.argmax(distances))])
    return sorted(chosen[:views]),test


@torch.no_grad()
def export_scene(model,o,cfg,out,views=4,confidence_quantile=.5,max_points=100000):
    import pycolmap
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    out=Path(out).resolve()
    if out.exists():
        raise RuntimeError(f'Existing export preserved: {out}; use another --output')
    meta=json.loads((Path(o['path'])/'cameras.json').read_text());n=len(meta['frames'])
    train,test=nested_camera_split(meta['frames'],views)
    # read_object returns GT depth only for evaluation metadata; prediction below receives only images.
    b=to_device(read_object(o,views,cfg['resolution'],random.Random(42),train),'cuda')
    model.eval();p=predict(model,b['images'],pose=True)
    E,_=pose_encoding_to_extri_intri(p['pose_enc'],b['images'].shape[-2:])
    scale,pose_rmse=camera_scale(E[0],b['native_E'][0])
    xyz=[];rgb=[]
    for i in range(views):
        depth=p['depth'][0,i]*scale;conf=p['confidence'][0,i]
        mask=b['valid'][0,i]  # Provided input-view foreground mask; no test-view masks used for initialization.
        mask=mask&torch.isfinite(depth)&(depth>1e-6)
        if confidence_quantile>0:
            mask=mask&(conf>=torch.quantile(conf[mask],confidence_quantile))
        points=backproject(depth,b['K'][0,i],b['native_E'][0,i])
        xyz.append(points[mask].cpu().numpy());rgb.append((b['images'][0,i].permute(1,2,0)[mask]*255).round().byte().cpu().numpy())
    xyz=np.concatenate(xyz);rgb=np.concatenate(rgb)
    if len(xyz)<100:
        raise ValueError('Too few valid initialized points')
    # Deterministic cap; same budget for baseline and adapted models.
    if len(xyz)>max_points:
        ids=np.random.default_rng(42).choice(len(xyz),max_points,replace=False);xyz,rgb=xyz[ids],rgb[ids]
    (out/'images').mkdir(parents=True);(out/'sparse/0').mkdir(parents=True)
    side=cfg['resolution'];camera_frames=[]
    # All fixed known poses are exported; only the explicit train indices optimize 3DGS.
    for i in range(n):
        one=read_object(o,1,side,random.Random(42),[i])
        k=one['K'][0,0].numpy();e=one['native_E'][0,0].numpy()
        name=f'{i:05d}.png';camera_frames.append((name,k,e))
        Image.fromarray((one['images'][0,0].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(out/'images'/name)
    write_colmap(out/'sparse/0',camera_frames,side)
    write_ply(out/'init.ply',xyz,rgb)
    # Loader needs an origin PLY even with init_pcd_name overridden; supply the same actual initializer.
    (out/'sparse/0/points3D.ply').symlink_to(out/'init.ply')
    (out/f'sparse_{views}.txt').write_text(''.join(f'{i}\n' for i in train))
    (out/'sparse_test.txt').write_text(''.join(f'{i}\n' for i in test))
    json_dump(out/'protocol.json',dict(object=o['id'],train_indices=train,test_indices=test,
              known_poses=True,depth_gt_used_for_initialization=False,input_masks_provided=True,
              depth_scale_source='Sim3 camera-center alignment from input-view predicted and known cameras only',
              scale=float(scale),pose_alignment_rmse=pose_rmse,points=len(xyz),confidence_quantile=confidence_quantile,
              resolution=side,seed=42))
    print(f'Exported {len(xyz)} points; {views} train views / {len(test)} test views: {out}',flush=True)


def train_gaussians(data,out,views,steps):
    """Run existing renderer/coarse optimizer, no repair. Export held-out RGB metrics."""
    import os
    GO=REPO/'3DGS/GaussianObject';sys.path.insert(0,str(GO));os.chdir(GO)
    import train_gs
    import scene.dataset_readers as readers
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from utils.full_scene import full_scene_options
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    import lpips
    if out.exists():
        raise RuntimeError(f'Existing 3DGS run preserved: {out}')
    out.mkdir(parents=True)
    parser=argparse.ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([])
    args.source_path=str(data.resolve());args.model_path=str(out.resolve());args.images='images'
    args.full_scene=True;args.eval=False;args.resolution=1;args.sh_degree=2;args.sparse_view_num=views
    args.init_pcd_name='init.ply';args.use_dust3r=False;args.transform_the_world=False
    args.use_drop=False;args.drop_rate=0.;args.drop_schedule='linear';args.mono_depth_weight=0.
    args.mono_loss_type='mid';args.lambda_t_norm=0.;args.iterations=steps;args.position_lr_max_steps=steps
    args.densify_until_iter=steps//2;args.max_num_splats=200000;full_scene_options(args)
    readers.generate_ellipse_path_from_camera_infos=lambda *x,**kw:[]
    def logger(ds):
        (out/'cfg_args').write_text(str(argparse.Namespace(**vars(ds))))
        return None
    vgg_weights=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    if not vgg_weights.is_file():
        raise FileNotFoundError(f'Existing LPIPS VGG cache required; automatic download disabled: {vgg_weights}')
    metric=lpips.LPIPS(net='vgg').cuda().eval()
    def report(tb,step,*rest):
        # training_report(tb, step, Ll1,loss,l1_loss,elapsed,testing,scene,render,render_args)
        if step!=steps:
            return
        scene,render,render_args=rest[5:8]
        rows=[]
        with torch.no_grad():
            for i,cam in enumerate(scene.getTestCameras()):
                pred=render(cam,scene.gaussians,*render_args)['render'].clamp(0,1)
                pred=torch.round(pred*255)/255;gt=torch.round(cam.original_image[:3].clamp(0,1)*255)/255
                rows.append(dict(camera=cam.image_name,PSNR=float(psnr(pred,gt).mean()),SSIM=float(ssim(pred,gt)),
                                 LPIPS=float(metric(pred[None],gt[None],normalize=True))))
                if i<3:
                    for name,tensor in [('render',pred),('GT',gt)]:
                        Image.fromarray((tensor.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(out/f'{cam.image_name}_{name}.png')
        json_dump(out/'heldout_metrics.json',dict(rows=rows,mean={k:sum(r[k] for r in rows)/len(rows) for k in ('PSNR','SSIM','LPIPS')},
                                                protocol='Known-pose, disjoint held-out views. Full-frame RGB; background retained; no repair.'))
    train_gs.prepare_output_and_logger=logger;train_gs.training_report=report
    safe_state(False)
    # Save no optimized Gaussian snapshots by default; images/metrics are the downstream deliverables.
    train_gs.training(args,mp.extract(args),op.extract(args),pp.extract(args),[],[],[],None,-1)
