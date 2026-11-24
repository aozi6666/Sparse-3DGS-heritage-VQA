import json
from pathlib import Path
import random
import time
import hashlib
import cv2
import numpy as np
from PIL import Image
import torch
from .common import json_dump
from .data import read_object,to_device
from .geometry import multiview
from .metrics import depth_metrics,boundary_f1,pose_metrics
from .model import predict


def validation_cases(o,views,seed,count):
    """Fixed, object-specific view subsets shared by baseline and every treatment."""
    n=o['frames']
    first=np.linspace(0,n-1,views).round().astype(int).tolist()
    selected=[first]
    rng=random.Random(seed+int(hashlib.sha256(o['id'].encode()).hexdigest()[:8],16))
    for _ in range(count*30):
        if len(selected)>=count:break
        ids=sorted(rng.sample(range(n),views))
        if ids not in selected:selected.append(ids)
    return selected


@torch.no_grad()
def evaluate(model,pool,cfg,output=None,limit=None):
    was_training=model.training;model.eval();rows=[]
    if limit:
        pool=pool[:limit]
    work=[(index,o,case,indices) for index,o in enumerate(pool) if o['frames']>=cfg['views']
          for case,indices in enumerate(validation_cases(o,cfg['views'],cfg['seed'],cfg.get('validation_cases',6)))]
    for index,o,case,indices in work:
        b=to_device(read_object(o,cfg['views'],cfg['resolution'],random.Random(cfg['seed']),indices),'cuda')
        torch.cuda.synchronize();start=time.monotonic()
        p=predict(model,b['images'],pose=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        mask=b['valid'];pred=p['depth'];gt=b['depth']
        # Scale alignment uses ONE median ratio shared by all views, not per-image fitting.
        scale=(gt[mask]/pred[mask].clamp_min(1e-6)).median()
        mv,n=multiview(pred[0],gt[0],b['K'][0],b['E'][0],mask[0],cfg['mv_stride'],cfg['mv_occlusion_relative'])
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        E,_=pose_encoding_to_extri_intri(p['pose_enc'],b['images'].shape[-2:])
        row=dict(id=o['id'],case=case,indices=indices,dataset=o['dataset'],split=o['split'],
                 canonical_unaligned=depth_metrics(pred,gt,mask),
                 scene_scale_aligned=depth_metrics(pred*scale,gt,mask),
                 scale_alignment=float(scale),boundary_f1=boundary_f1(pred*scale,gt,mask),
                 multiview_relative_error=float(mv),multiview_valid_pixels=n,
                 pose=pose_metrics(E[0],b['E'][0]),latency_seconds=elapsed)
        rows.append(row)
        if output and index<5 and case==0:
            size=cfg['resolution'];sheet=Image.new('RGB',(size*4,size*cfg['views']))
            for v in range(cfg['views']):
                m=mask[0,v].cpu().numpy();g=gt[0,v].cpu().numpy();d=(pred[0,v]*scale).cpu().numpy()
                lo,hi=np.percentile(g[m],[2,98])
                def color(a):
                    im=cv2.applyColorMap((np.clip((a-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1].copy();im[~m]=0
                    return im
                rgb=(b['images'][0,v].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
                conf=p['confidence'][0,v].cpu().numpy();c=conf/(np.percentile(conf[m],99)+1e-6)
                c=np.repeat((np.clip(c,0,1)*255).astype(np.uint8)[...,None],3,-1)
                for col,im in enumerate((rgb,color(g),color(d),c)):
                    sheet.paste(Image.fromarray(im),(col*size,v*size))
            Path(output).mkdir(parents=True,exist_ok=True);sheet.save(Path(output)/f'{index:03d}_rgb_gt_pred_conf.jpg',quality=90)
    if not rows:
        raise ValueError('No evaluable objects for view count')
    identities=list(dict.fromkeys(r['id'] for r in rows))
    # First average fixed cases within each object, then macro-average independent objects.
    summary={mode:{k:sum(sum(r[mode][k] for r in rows if r['id']==identifier)/sum(r['id']==identifier for r in rows)
                          for identifier in identities)/len(identities)
                   for k in ('AbsRel','RMSE','RMSE_Log','delta1','delta2','delta3')}
             for mode in ('canonical_unaligned','scene_scale_aligned')}
    result=dict(objects=len(identities),cases=len(rows),summary=summary,rows=rows,
                protocol='Object macro-average after averaging fixed view subsets per object. Repeated view subsets do NOT increase independent object count. Canonical scale uses GT scene normalization, not metre-scale prediction. Foreground only. Scale-aligned scores use one shared scene median. No per-frame alignment. MV uses GT-visible fixed correspondences and relative target-Z residual.',
                peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3)
    if output:
        json_dump(Path(output)/'metrics.json',result)
    if was_training:
        from .model import train_mode
        train_mode(model)
    return result
