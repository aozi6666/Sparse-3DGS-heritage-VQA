#!/usr/bin/env python3
"""Isolated Omni test preparation/evaluation. check is CPU-only and read-only."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse,json,sys,hashlib,csv,random,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from vggt_object.omni_files import checked,exact_depth,valid_depth

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def read_view(root,frame,fov):
    import cv2
    from PIL import Image
    ip=checked(root/'images'/(Path(frame['file_path']).stem+'.png'));dp=exact_depth(root,ip.stem)
    a=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED) if dp.suffix=='.exr' else np.load(dp,allow_pickle=False)
    if a is None or a.dtype.kind!='f':raise ValueError(f'Not floating depth: {dp}')
    if a.ndim==3:
        if not all(np.array_equal(a[...,0],a[...,c],equal_nan=True) for c in range(1,a.shape[2])):raise ValueError('Depth channels differ')
        a=a[...,0]
    with Image.open(ip) as im:
        if im.mode!='RGBA' or a.shape!=im.size[::-1]:raise ValueError('RGB/depth resolution or alpha mismatch')
        rgb=im.convert('RGB').copy();m=(np.asarray(im.getchannel('A'))>127)&valid_depth(a)
    if m.sum()<100:raise ValueError(f'Insufficient valid foreground pixels: {ip}')
    h,w=a.shape;fx=.5*w/np.tan(fov/2);K=np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1.]])
    return rgb,np.where(m,a,0).astype(np.float32),m,K,ip,dp

def audit(raw,views,pixels=False):
    valid=[];issues=[];stats={}
    for cat in sorted(p for p in raw.iterdir() if p.is_dir() and not p.name.startswith('.')):
        n=0;ok=0
        for obj in sorted(p for p in cat.iterdir() if p.is_dir() and not p.name.startswith('.')):
            n+=1
            try:
                tf=checked(obj/'render/transforms.json');meta=json.loads(tf.read_text());frames=meta['frames'];fov=float(meta['camera_angle_x'])
                if len(frames)<views or not 0<fov<math.pi:raise ValueError('Camera/view count')
                names=[Path(f['file_path']).stem for f in frames]
                if len(set(names))!=len(names):raise ValueError('Duplicate frame IDs')
                # Entire object's declared frames must be uploaded; do not silently admit a partial object.
                for f in frames:
                    root=obj/'render';ip=checked(root/'images'/(Path(f['file_path']).stem+'.png'));dp=exact_depth(root,ip.stem)
                    with ip.open('rb') as z:head=z.read(8)
                    if head!=b'\x89PNG\r\n\x1a\n':raise ValueError('Invalid PNG header')
                    if dp.suffix=='.exr':
                        with dp.open('rb') as z:head=z.read(4)
                        if head!=b'v/1\x01':raise ValueError('Invalid EXR header')
                    M=np.array(f['transform_matrix'],dtype=float)
                    if M.shape!=(4,4) or not np.isfinite(M).all() or not np.allclose(M[3],[0,0,0,1],atol=1e-5) or not np.allclose(M[:3,:3].T@M[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(M[:3,:3]),1,atol=1e-4):raise ValueError('Invalid camera pose')
                ids=np.linspace(0,len(frames)-1,views).round().astype(int).tolist()
                if pixels:
                    for i in ids:read_view(obj/'render',frames[i],fov)
                valid.append(dict(id=obj.name,category=cat.name,root=str(obj.resolve()),frame_ids=ids));ok+=1
            except Exception as e:issues.append(dict(id=obj.name,category=cat.name,reason=str(e)))
        stats[cat.name]=dict(objects=n,complete=ok)
    return valid,issues,stats

def prepare(a):
    if a.plan.exists():raise RuntimeError('Plan exists; preserved. Choose a new --plan.')
    valid,bad,stats=audit(a.raw,a.views,pixels=True)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    if bad:raise RuntimeError(f'{len(bad)} objects incomplete/invalid; finish upload first. Example: {bad[0]}')
    if len(valid)<a.objects:raise RuntimeError(f'Need {a.objects}, only {len(valid)} valid objects')
    groups={}
    for o in valid:groups.setdefault(o['category'],[]).append(o)
    rng=random.Random(42)
    for rows in groups.values():rng.shuffle(rows)
    chosen=[]
    while len(chosen)<a.objects:
        for c in sorted(groups):
            if groups[c] and len(chosen)<a.objects:chosen.append(groups[c].pop())
    old=P/'data/manifests/objects_review_v2.json'
    prior=json.loads(old.read_text())['objects'] if old.exists() else []
    training={o['id'] for o in prior if o['split'] in ('train','val')}
    if training & {o['id'] for o in chosen}:raise ValueError('Object overlap with current train/validation manifest')
    for o in chosen:
        root=Path(o['root'])/'render';meta=json.loads((root/'transforms.json').read_text());files=[root/'transforms.json']
        for i in o['frame_ids']:
            stem=Path(meta['frames'][i]['file_path']).stem;files.extend([root/'images'/(stem+'.png'),exact_depth(root,stem)])
        o['files']={str(p):sha(p) for p in files}
    if len({o['id'] for o in chosen})!=len(chosen):raise ValueError('Duplicate object IDs across categories')
    code_paths=[Path(__file__).resolve(),P/'vggt_object/omni_files.py',P/'vggt_object/data.py',P/'vggt_object/geometry.py',P/'vggt_object/metrics.py',P/'vggt_object/model.py']
    plan=dict(code_sha256={str(p):sha(p) for p in code_paths},protocol='omni_depth_test_v1',objects=chosen,views=a.views,input_views=a.input_views,resolution=a.resolution,depth_type=a.depth_type,units='source scene units; NOT certified metres',seed=42,mask='RGBA alpha>127 & finite native depth>1e-6 & <65504; filter before conversion',scale='GT-defined first-camera canonical normalization; one median ratio per inference group for aligned scores',boundary=dict(log_jump=.05,radius_pixels=2,scope='internal foreground only; excludes silhouette',no_gt_edges=None,no_pred_edges_with_gt=0.),groups='24 selected views, consecutive groups of 4 by default; NOT joint 24-view inference',training_manifest_sha256=sha(old) if old.exists() else None)
    write(a.plan,plan);print('Frozen test plan created; no RGB/depth copies, no GPU:',a.plan)

def run(a):
    import torch
    from vggt_object.common import config,seed_all,require_gpu
    from vggt_object.model import create,predict
    from vggt_object.checkpoint import load_adapter
    from vggt_object.data import resize_frame
    from vggt_object.geometry import normalize_scene
    from vggt_object.metrics import depth_metrics
    import torch.nn.functional as F
    plan=json.loads(a.plan.read_text());planhash=sha(a.plan)
    if plan.get('protocol')!='omni_depth_test_v1' or plan.get('depth_type') not in ('camera_z_depth','ray_distance'):raise ValueError('Invalid plan protocol/depth semantics')
    for name,h in plan['code_sha256'].items():
        if sha(name)!=h:raise ValueError(f'Evaluation code changed after freeze: {name}')
    if plan['views']%plan['input_views']:raise ValueError('View grouping not divisible')
    if a.out.exists():raise RuntimeError('Output exists; preserved. Use a NEW --out.')
    for o in plan['objects']:
        for name,h in o['files'].items():
            if sha(name)!=h:raise ValueError(f'Input changed after plan: {name}')
    require_gpu();seed_all(42)
    cfg=config(experiment='E0');cfg['rank']=0;cfg['adapter_kind']='depth_conv_lora';cfg['aggregator_lora']={'enabled':False}
    if a.adapter:
        cfg=json.loads((a.adapter/'model_info.json').read_text())['config']
        cfg=dict(cfg)
    model,params=create(cfg)
    if a.adapter:load_adapter(a.adapter,model,cfg)
    model.eval();model.requires_grad_(False);a.out.mkdir(parents=True)
    write(a.out/'protocol.json',dict(plan_sha256=planhash,plan=plan,adapter=str(a.adapter) if a.adapter else None,weights=str(cfg['base_checkpoint']),parameters=params,metrics_note='No model-specific pixel filtering; nonfinite/nonpositive predictions abort. Boundary F1 lacks outer silhouette; GT-edge-free groups are null with coverage.'))
    def bf1(p,g,m):
        def edge(z):
            z=z.clamp_min(1e-6).log();e=torch.zeros_like(m)
            e[...,1:]|=(z[...,1:]-z[...,:-1]).abs().gt(.05)&m[...,1:]&m[...,:-1]
            e[...,1:,:]|=(z[...,1:,:]-z[...,:-1,:]).abs().gt(.05)&m[...,1:,:]&m[...,:-1,:]
            return e
        x,y=edge(p),edge(g)
        if not y.any():return None
        if not x.any():return 0.
        dil=lambda t:F.max_pool2d(t.float().reshape(-1,1,*t.shape[-2:]),5,1,2).reshape(t.shape)>0.5
        precision=(x&dil(y)).sum()/x.sum();recall=(y&dil(x)).sum()/y.sum()
        return float(2*precision*recall/(precision+recall).clamp_min(1e-8))
    results=[]
    with torch.inference_mode():
        for o in plan['objects']:
            rows=[];root=Path(o['root'])/'render';meta=json.loads((root/'transforms.json').read_text())
            for start in range(0,plan['views'],plan['input_views']):
                rgb=[];dep=[];mask=[];ks=[];es=[]
                ids=o['frame_ids'][start:start+plan['input_views']]
                for i in ids:
                    fr=meta['frames'][i];im,d,m,k,_,_=read_view(root,fr,meta['camera_angle_x'])
                    if plan['depth_type']=='ray_distance':
                        yy,xx=np.mgrid[:d.shape[0],:d.shape[1]];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T;d=d/np.linalg.norm(rays,axis=-1)
                    rr,dd,mm,kk=resize_frame(im,d.astype('float32'),m,torch.tensor(k),plan['resolution'])
                    rgb.append(rr);dep.append(dd);mask.append(mm);ks.append(kk);es.append(torch.tensor(np.diag([1.,-1.,-1.,1.])@np.linalg.inv(np.array(fr['transform_matrix'])),dtype=torch.float32))
                gt=torch.stack(dep).cuda();m=torch.stack(mask).cuda();K=torch.stack(ks).cuda();E=torch.stack(es).cuda();gt,_,scale=normalize_scene(gt,K,E,m)
                pred=predict(model,torch.stack(rgb)[None].cuda())['depth'][0]
                if not torch.isfinite(pred).all() or (pred<=1e-6).any():raise RuntimeError('Invalid prediction; no dropping pixels to improve scores')
                ratio=(gt[m]/pred[m]).median();aligned=pred*ratio
                rows.append(dict(id=o['id'],category=o['category'],frames=ids,canonical_unaligned=depth_metrics(pred,gt,m),scene_scale_aligned=depth_metrics(aligned,gt,m),boundary_f1=bf1(aligned,gt,m),alignment=float(ratio),canonical_scale=float(scale)))
            results.extend(rows);write(a.out/(o['id']+'.json'),rows);print('Evaluated',o['id'],flush=True)
    objects=[]
    for o in plan['objects']:
        rows=[r for r in results if r['id']==o['id']];f=[r['boundary_f1'] for r in rows if r['boundary_f1'] is not None]
        objects.append(dict(id=o['id'],category=o['category'],**{mode:{key:float(np.mean([r[mode][key] for r in rows])) for key in ['AbsRel','RMSE','RMSE_Log','delta1','delta2','delta3']} for mode in ['canonical_unaligned','scene_scale_aligned']},boundary_f1=float(np.mean(f)) if f else None,boundary_valid_groups=len(f),groups=len(rows)))
    overall={mode:{k:float(np.mean([o[mode][k] for o in objects])) for k in objects[0][mode]} for mode in ['canonical_unaligned','scene_scale_aligned']}
    f=[o['boundary_f1'] for o in objects if o['boundary_f1'] is not None];overall.update(boundary_f1=float(np.mean(f)) if f else None,boundary_valid_objects=len(f),objects=len(objects))
    flat=[]
    for o in objects:
        flat.append(dict(id=o['id'],category=o['category'],boundary_f1=o['boundary_f1'],boundary_valid_groups=o['boundary_valid_groups'],groups=o['groups'],**{mode+'_'+key:value for mode in ['canonical_unaligned','scene_scale_aligned'] for key,value in o[mode].items()}))
    with (a.out/'per_object_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    write(a.out/'summary.json',dict(overall=overall,per_object=objects,plan_sha256=planhash));write(a.out/'complete.json',dict(status='complete',plan_sha256=planhash))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','prepare','run']);p.add_argument('--raw',type=Path,default=P/'data/omni3d_depth');p.add_argument('--objects',type=int,default=100);p.add_argument('--views',type=int,default=24);p.add_argument('--input-views',type=int,default=4);p.add_argument('--resolution',type=int,choices=[392,518],default=392);p.add_argument('--depth-type',choices=['camera_z_depth','ray_distance']);p.add_argument('--plan',type=Path,default=P/'results/omni_depth_test_v1/plan.json');p.add_argument('--out',type=Path,default=P/'results/omni_depth_test_v1/original');p.add_argument('--adapter',type=Path);a=p.parse_args()
    if a.objects<1 or a.views<4 or a.input_views<1 or a.views%a.input_views:p.error('Positive counts; views divisible by input-views required')
    if a.stage in ('prepare','run'):
        for target in [a.plan,a.out]:
            if target.resolve()==a.raw.resolve() or a.raw.resolve() in target.resolve().parents:p.error('Outputs must be outside the raw dataset directory')
    if a.stage=='check':
        _,bad,stats=audit(a.raw,a.views);print(json.dumps(dict(categories=stats,incomplete_objects=len(bad),examples=bad[:5]),ensure_ascii=False,indent=2));print('CPU only, no files written, no GPU initialization.')
        if bad or sum(v['complete'] for v in stats.values())<a.objects:raise SystemExit(2)
    elif a.stage=='prepare':
        if a.depth_type is None:p.error('--depth-type must explicitly declare verified native depth semantics; not guessed')
        prepare(a)
    else:run(a)
if __name__=='__main__':main()
