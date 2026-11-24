import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .geometry import normalize_scene


def objects(manifest,split):
    all_objects=json.loads(Path(manifest).read_text())['objects']
    chosen=[o for o in all_objects if o['split']==split]
    if not chosen:
        raise ValueError(f'No {split} objects; populate and validate the dataset first')
    return chosen


def resize_frame(image,depth,mask,K,size):
    """Letterbox, preserve aspect ratio; synchronize RGB/depth/mask and K."""
    h,w=depth.shape
    scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    left,top=(size-nw)//2,(size-nh)//2
    rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
    rgb=F.interpolate(rgb[None],(nh,nw),mode='bilinear',align_corners=False)[0]
    dep=torch.from_numpy(depth.copy()).float()[None,None]
    dep=F.interpolate(dep,(nh,nw),mode='nearest')[0,0]
    valid=F.interpolate(torch.from_numpy(mask.copy()).float()[None,None],(nh,nw),mode='nearest')[0,0]>.5
    K=K.clone().float();K[0,:]*=nw/w;K[1,:]*=nh/h;K[0,2]+=left;K[1,2]+=top
    pad=(left,size-nw-left,top,size-nh-top)
    return F.pad(rgb,pad),F.pad(dep,pad),F.pad(valid,pad),K


def read_object(o,views,size,rng,indices=None):
    root=Path(o['path']);meta=json.loads((root/'cameras.json').read_text())
    frames=meta['frames']
    if len(frames)<views:
        raise ValueError(f'{o["id"]} only has {len(frames)} frames, need {views}')
    indices=sorted(rng.sample(range(len(frames)),views)) if indices is None else list(indices)
    rgb,depth,mask,K,E=[],[],[],[],[]
    for i in indices:
        f=frames[i]
        encoding=f.get('depth_encoding','float32_npy')
        if encoding=='float32_npy':
            dep=np.load(root/f['depth'],allow_pickle=False).astype(np.float32)
        elif encoding=='co3d_float16_png':
            dep=np.asarray(Image.open(root/f['depth'])).astype(np.uint16).view(np.float16).astype(np.float32)
        else:
            raise ValueError(f'Unsupported explicit depth encoding: {encoding}')
        dep=dep*float(f.get('depth_scale',1.))
        if dep.ndim!=2:
            raise ValueError('Depth must be 2D float Z-depth')
        m=np.asarray(Image.open(root/f['mask']).convert('L'))>127
        with Image.open(root/f['image']) as im:
            if im.size!=dep.shape[::-1] and f.get('depth_resize_to_rgb',False):
                import cv2
                dep=cv2.resize(dep,im.size,interpolation=cv2.INTER_NEAREST)
            if im.size != dep.shape[::-1] or m.shape != dep.shape:
                raise ValueError('RGB/depth/mask size mismatch')
            if f.get('depth_type','camera_z_depth')=='ray_distance':
                h,w=dep.shape;yy,xx=np.mgrid[:h,:w]
                rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(np.asarray(f['K'])).T
                dep=dep/np.linalg.norm(rays,axis=-1)
            elif f.get('depth_type','camera_z_depth')!='camera_z_depth':
                raise ValueError('Unknown depth type')
            valid=m&np.isfinite(dep)&(dep>1e-6)
            dep=np.where(valid,dep,0).astype(np.float32)
            r,d,v,k=resize_frame(im,dep,valid,torch.tensor(f['K']),size)
        rgb.append(r);depth.append(d);mask.append(v);K.append(k);E.append(torch.tensor(f['E'],dtype=torch.float32))
    depth=torch.stack(depth);mask=torch.stack(mask);K=torch.stack(K);E=torch.stack(E)
    # Never normalize each frame independently. One shared selected-scene scale.
    native=depth.clone();native_E=E.clone()
    depth,E,scale=normalize_scene(depth,K,E,mask)
    return dict(images=torch.stack(rgb)[None],depth=depth[None],valid=mask[None],K=K[None],E=E[None],
                native_depth=native[None],native_E=native_E[None],scale=float(scale),indices=indices,
                object_id=o['id'],dataset=o['dataset'],supervision_weight=1.0)


class SceneSampler:
    def __init__(self,manifest,cfg):
        self.cfg=cfg;self.rng=random.Random(cfg['seed']);self.groups={}
        pool=objects(manifest,'train')
        for o in pool:
            if cfg['mix'].get(o['dataset'],0)<=0:
                continue
            if o['dataset']=='co3dv2' and not o.get('depth_reviewed',False):
                raise ValueError('CO3D depth not reviewed; do not mix unverified pseudo-depth into training')
            if o['frames']>=cfg['views']:
                self.groups.setdefault(o['dataset'],[]).append(o)
        if not self.groups:
            raise ValueError('No usable training objects for requested mixture/view count')
        # Missing datasets are explicitly reported; no silent claim of requested ratio.
        print('Active datasets:',{k:len(v) for k,v in self.groups.items()},flush=True)
        missing=set(cfg['mix'])-set(self.groups)
        if missing:
            if cfg.get('strict_mix',False):
                raise ValueError(f'Requested mixture contains missing datasets {sorted(missing)}; refuse to label a different mixture as the requested experiment')
            print('Absent datasets (weights renormalized over present datasets):',sorted(missing),flush=True)
        if cfg['overfit_objects']:
            for key in self.groups:
                self.groups[key]=self.groups[key][:cfg['overfit_objects']]
        self.keys=list(self.groups);self.weights=[cfg['mix'][k] for k in self.keys]
        self.queues={k:[] for k in self.keys}

    def next(self):
        key=self.rng.choices(self.keys,weights=self.weights)[0]
        if not self.queues[key]:
            self.queues[key]=list(self.groups[key]);self.rng.shuffle(self.queues[key])
        o=self.queues[key].pop()
        indices=np.linspace(0,o['frames']-1,self.cfg['views']).round().astype(int) if self.cfg['fixed_views'] else None
        if indices is None and self.cfg.get('view_sampling')=='overlap_mix' and self.rng.random()<.7:
            # Choose an anchor plus nearby camera centres for overlap; retain a random
            # global view for coverage. No test-view RGB/depth used for selection.
            meta=json.loads((Path(o['path'])/'cameras.json').read_text())
            frames=meta['frames'];anchor=self.rng.randrange(len(frames))
            centers=[]
            for f in frames:
                e=np.asarray(f['E']);centers.append(-e[:3,:3].T@e[:3,3])
            distances=np.linalg.norm(np.array(centers)-centers[anchor],axis=1)
            near=np.argsort(distances).tolist()
            indices=near[:max(1,self.cfg['views']-1)]
            remaining=[i for i in range(len(frames)) if i not in indices]
            indices+=self.rng.sample(remaining,self.cfg['views']-len(indices))
            indices=sorted(indices)
        b=read_object(o,self.cfg['views'],self.cfg['resolution'],self.rng,indices)
        b['supervision_weight']=self.cfg['co3d_supervision_weight'] if key=='co3dv2' else 1.
        if self.cfg.get('rgb_augmentation',False):
            # Shared scene photometric jitter; no spatial transform, K/E/depth unchanged.
            brightness=self.rng.uniform(.9,1.1);contrast=self.rng.uniform(.9,1.1)
            rgb=(b['images']-.5)*contrast+.5
            rgb=rgb*brightness+torch.randn_like(rgb)*.002
            b['images']=rgb.clamp(0,1)
        return b

    def state_dict(self):
        return {'rng':self.rng.getstate(),'queues':self.queues}

    def load_state_dict(self,state):
        self.rng.setstate(state['rng']);self.queues=state['queues']


def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
