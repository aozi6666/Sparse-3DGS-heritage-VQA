import math
import torch
import torch.nn.functional as F
from .geometry import multiview


def depth_metrics(pred,gt,mask):
    m=mask&torch.isfinite(pred)&(pred>1e-6)&torch.isfinite(gt)&(gt>1e-6)
    if not m.any():
        raise ValueError('No valid pixels for depth metrics')
    p=pred[m].double();g=gt[m].double()
    ratio=torch.maximum(p/g,g/p)
    return dict(AbsRel=float(((p-g).abs()/g).mean()),RMSE=float((p-g).square().mean().sqrt()),
                RMSE_Log=float((p.log()-g.log()).square().mean().sqrt()),
                delta1=float((ratio<1.25).double().mean()),delta2=float((ratio<1.25**2).double().mean()),
                delta3=float((ratio<1.25**3).double().mean()),pixels=int(m.sum()))


def boundary_f1(pred,gt,mask,threshold=.05,radius=2):
    # Relative log-depth discontinuities; fixed threshold and matching tolerance in pixels.
    def edge(d):
        log=d.clamp_min(1e-6).log();e=torch.zeros_like(mask)
        e[...,1:]|=((log[...,1:]-log[...,:-1]).abs()>threshold)&mask[...,1:]&mask[...,:-1]
        e[...,1:,:]|=((log[...,1:,:]-log[...,:-1,:]).abs()>threshold)&mask[...,1:,:]&mask[...,:-1,:]
        return e
    a,b=edge(pred),edge(gt)
    if not a.any() or not b.any():
        return None  # No invented perfect boundary score on featureless depth maps.
    def dilate(x):
        return F.max_pool2d(x.float().reshape(-1,1,*x.shape[-2:]),2*radius+1,1,radius).reshape(x.shape)>.5
    precision=(a&dilate(b)).sum()/a.sum();recall=(b&dilate(a)).sum()/b.sum()
    return float(2*precision*recall/(precision+recall).clamp_min(1e-8))


def pose_metrics(pred,gt):
    errors=[];rots=[];trans=[]
    def hom(e):
        if e.shape[-2]==4:
            return e
        row=e.new_tensor([0,0,0,1]).expand(*e.shape[:-2],1,4)
        return torch.cat([e,row],-2)
    pred=hom(pred).float();gt=hom(gt).float()
    for i in range(len(gt)):
        for j in range(i+1,len(gt)):
            p=pred[j]@torch.linalg.inv(pred[i]);g=gt[j]@torch.linalg.inv(gt[i])
            c=((torch.trace(p[:3,:3]@g[:3,:3].T)-1)/2).clamp(-1,1)
            r=float(torch.rad2deg(torch.acos(c)))
            if p[:3,3].norm()<1e-8 or g[:3,3].norm()<1e-8:
                continue
            c=F.cosine_similarity(p[:3,3],g[:3,3],dim=0).clamp(-1,1)
            # Standard relative translation-direction metric, sign ambiguity folded.
            t=float(torch.rad2deg(torch.acos(c)));t=min(t,180-t)
            rots.append(r);trans.append(t);errors.append(max(r,t))
    if not errors:
        return {'pairs':0,'AUC30':None}
    e=sorted(errors);xs=[0.]+[x for x in e if x<30]+[30.]
    ys=[0.]+[(i+1)/len(e) for i,x in enumerate(e) if x<30]
    ys.append(ys[-1])
    auc=sum((xs[i+1]-xs[i])*(ys[i]+ys[i+1])/2 for i in range(len(xs)-1))/30
    return dict(pairs=len(errors),rotation_mean_deg=sum(rots)/len(rots),translation_mean_deg=sum(trans)/len(trans),
                rotation_accuracy_5=sum(x<5 for x in rots)/len(rots),translation_accuracy_5=sum(x<5 for x in trans)/len(trans),AUC30=auc)


def geometry_metrics(pred_path,gt_path,threshold=.01):
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    def points(path):
        obj=trimesh.load(path,force='mesh',process=False)
        if hasattr(obj,'faces') and len(obj.faces):
            np.random.seed(42);p,_=trimesh.sample.sample_surface(obj,100000)
        else:
            p=np.asarray(obj.vertices)
            p=p[np.random.default_rng(42).choice(len(p),min(len(p),100000),replace=False)]
        return p
    p,g=points(pred_path),points(gt_path)
    dp=cKDTree(g).query(p)[0];dg=cKDTree(p).query(g)[0]
    precision=float((dp<threshold).mean());recall=float((dg<threshold).mean())
    return dict(chamfer_l1=float((dp.mean()+dg.mean())/2),precision=precision,completeness=recall,
                fscore=2*precision*recall/max(precision+recall,1e-12),threshold=threshold,
                note='Inputs must already share units and coordinates; no hidden ICP/alignment.',points=[len(p),len(g)])
