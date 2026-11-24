"""Nine-view foreground training-stage illustration. Independent of named-method tables."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(GO), str(GO/'tools')]


from materialize_input import materialize

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_out(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def inventory(name):
    import numpy as np
    import pycolmap
    from PIL import Image
    source=GO/'data/mip360'/name
    rec=pycolmap.Reconstruction(str(source/'sparse/0'))
    ims=sorted(rec.images.values(),key=lambda im:Path(im.name).stem)
    tr=np.atleast_1d(np.loadtxt(source/'sparse_9.txt',dtype=int)).tolist()
    te=np.atleast_1d(np.loadtxt(source/'sparse_test.txt',dtype=int)).tolist()
    if len(tr)!=9 or len(set(tr))!=9 or not te or set(tr)&set(te): raise ValueError('Invalid train/test split')
    if min(tr+te)<0 or max(tr+te)>=len(ims): raise ValueError('Split index out of range')
    for cam in rec.cameras.values():
        if cam.model_name!='PINHOLE': raise ValueError('Expected PINHOLE')
        if abs(cam.params[2]-cam.width/2)>1 or abs(cam.params[3]-cam.height/2)>1: raise ValueError('Off-axis camera unsupported')
    for idx in tr+te:
        im=ims[idx]; path=source/'masks'/(Path(im.name).stem+'.png')
        with Image.open(path) as mask:
            m=np.array(mask.convert('L'))>127
            if not m.any() or m.all(): raise ValueError(f'Empty/full mask: {path}')
        with Image.open(source/'images'/im.name) as rgb:
            cam=rec.cameras[im.camera_id]
            if rgb.size!=(cam.width,cam.height): raise ValueError('Image/camera size mismatch')
    # Pick the largest object among held-out images using masks only, before rendering.
    def area(i):
        m=np.asarray(Image.open(source/'masks'/(Path(ims[i].name).stem+'.png')).convert('L'))
        return float((m>127).mean())
    display=max(te,key=area)
    return source,rec,ims,tr,te,display


def check(a):
    import importlib.util
    for mod in ['numpy','pycolmap','PIL','torch','lpips','diff_gaussian_rasterization','plyfile']:
        if importlib.util.find_spec(mod) is None: raise ImportError(mod)
    weight=Path(os.environ.get('TORCH_HOME',str(Path.home()/'.cache/torch')))/'hub/checkpoints/vgg16-397923af.pth'
    if not weight.is_file(): raise FileNotFoundError(weight)
    for name in a.scenes:
        _,_,ims,tr,te,di=inventory(name)
        print(name,'images',len(ims),'train',len(tr),'test',len(te),'camera',ims[di].name,flush=True)
    if shutil.disk_usage(GO).free<4*1024**3: raise RuntimeError('Need >=4 GiB free')
    print('CPU preflight passed; no training started',flush=True)


def prepare(name,leaf):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData,PlyElement
    source,rec,ims,tr,te,di=inventory(name)
    data=leaf/'data'; (data/'sparse/0').mkdir(parents=True)
    for sub in ['images','masks']: materialize(source/sub, data/sub)
    for f in (source/'sparse/0').glob('*.bin'): shutil.copyfile(f,data/'sparse/0'/f.name)
    for f in ['sparse_9.txt','sparse_test.txt']: shutil.copyfile(source/f,data/f)
    points=list(rec.points3D.values()); xyz=np.array([p.xyz for p in points]); rgb=np.array([p.color for p in points])
    visible=np.zeros(len(points),dtype=int); support=visible.copy()
    for idx in tr:
        im=ims[idx]; cam=rec.cameras[im.camera_id]
        m=np.asarray(Image.open(source/'masks'/(Path(im.name).stem+'.png')).convert('L'))
        R=np.asarray(im.cam_from_world().rotation.matrix()); t=np.asarray(im.cam_from_world().translation)
        local=xyz@R.T+t; z=local[:,2]; fx,fy,cx,cy=cam.params
        u=np.rint((local[:,0]/np.maximum(z,1e-8)*fx+cx)*m.shape[1]/cam.width).astype(int)
        v=np.rint((local[:,1]/np.maximum(z,1e-8)*fy+cy)*m.shape[0]/cam.height).astype(int)
        inside=(z>0)&(u>=0)&(v>=0)&(u<m.shape[1])&(v<m.shape[0]); ii=np.flatnonzero(inside)
        visible[ii]+=1; support[ii]+=(m[v[ii],u[ii]]>127)
    keep=(support>=2)&(support>=.8*np.maximum(visible,1))
    xyz,rgb=xyz[keep],rgb[keep]
    if len(xyz)<100: raise RuntimeError(f'{name}: too few foreground initialization points ({len(xyz)})')
    values=np.zeros(len(xyz),dtype=[(n,'f4') for n in ['x','y','z','nx','ny','nz']]+[(n,'u1') for n in ['red','green','blue']])
    for i,n in enumerate(['x','y','z']): values[n]=xyz[:,i]
    for i,n in enumerate(['red','green','blue']): values[n]=rgb[:,i]
    PlyData([PlyElement.describe(values,'vertex')]).write(str(data/'foreground_init.ply'))
    meta=dict(scene=name,display_camera=Path(ims[di].name).stem,train=[ims[i].name for i in tr],test=[ims[i].name for i in te],initial_points=len(xyz),
        initialization='Full-sequence SfM points filtered using nine train silhouettes only; known calibration. Not raw-nine-image reconstruction.',
        display_selection='Largest foreground fraction among test masks, before training')
    json_out(leaf/'protocol.json',meta)
    return meta


def choose(records):
    """Prefer a <=15.3 dB spread ending at the best actual candidate; no target synthesis."""
    ordered=sorted(records,key=lambda r:r['PSNR'])
    window=[r for r in ordered if ordered[-1]['PSNR']-r['PSNR']<=15.3]
    if len(window)<9: window=ordered
    selected=[window[0],window[-1]]
    # Greedy max-min distance across measured PSNR/SSIM/LPIPS; unique real stages.
    keys=['PSNR','SSIM','LPIPS']
    ranges={k:max(max(r[k] for r in window)-min(r[k] for r in window),1e-8) for k in keys}
    while len(selected)<9:
        remaining=[r for r in window if r['step'] not in {s['step'] for s in selected}]
        selected.append(max(remaining,key=lambda r:min(sum(((r[k]-s[k])/ranges[k])**2 for k in keys) for s in selected)))
    return sorted(selected,key=lambda r:r['PSNR'],reverse=True)


def train_scene(a,name):
    import numpy as np
    import torch
    import lpips
    from PIL import Image,ImageDraw
    import train_gs
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    if not torch.cuda.is_available(): raise RuntimeError('Enable CUDA first')
    leaf=a.out/name
    if (leaf/'complete.json').exists(): print('[reuse]',name,flush=True); return
    if leaf.exists(): raise RuntimeError(f'Partial run preserved at {leaf}; use a new --out')
    leaf.mkdir(parents=True); meta=prepare(name,leaf)
    p=argparse.ArgumentParser(); mp=ModelParams(p); op=OptimizationParams(p); pp=PipelineParams(p); args=p.parse_args([])
    args.source_path=str(leaf/'data'); args.model_path=str(leaf/'model'); args.resolution=4
    args.eval=False; args.full_scene=False; args.white_background=True; args.use_mask=True; args.sparse_view_num=9
    args.sh_degree=2; args.init_pcd_name='foreground_init.ply'; args.iterations=a.steps; args.position_lr_max_steps=a.steps
    args.max_num_splats=200000; args.densify_until_iter=a.steps//2
    args.use_dust3r=False; args.transform_the_world=False; args.use_drop=False
    args.drop_rate=0.; args.drop_schedule='linear'; args.mono_depth_weight=0.; args.mono_loss_type='mid'; args.lambda_t_norm=0.
    metric=lpips.LPIPS(net='vgg').cuda().eval(); records=[]; buffers={}
    def logger(ds):
        Path(ds.model_path).mkdir(); (Path(ds.model_path)/'cfg_args').write_text(str(argparse.Namespace(**vars(ds))))
        return None
    @torch.no_grad()
    def capture(tb,step,l1,loss,lf,elapsed,testing,scene,render,render_args):
        if step not in [10,25,50] and step%100: return
        if shutil.disk_usage(GO).free<2*1024**3: raise RuntimeError('Free disk <2 GiB')
        view=next(v for v in scene.getTestCameras() if v.image_name==meta['display_camera'])
        image=torch.round(render(view,scene.gaussians,*render_args)['render'].clamp(0,1)*255)/255
        gt=torch.round(view.original_image[:3].cuda().clamp(0,1)*255)/255
        mask=view.mask[0]>0.5; ys,xs=torch.where(mask)
        x0,x1=max(0,int(xs.min())-12),min(image.shape[2],int(xs.max())+13)
        y0,y1=max(0,int(ys.min())-12),min(image.shape[1],int(ys.max())+13)
        crop=image[:,y0:y1,x0:x1]; ref=gt[:,y0:y1,x0:x1]
        if min(crop.shape[1:])<32: raise RuntimeError('Object crop too small for LPIPS')
        rec=dict(step=step,camera=view.image_name,PSNR=psnr(image,gt).mean().item(),SSIM=ssim(image,gt).mean().item(),
            LPIPS=metric(image[None],gt[None],normalize=True).item(),
            foreground_PSNR=float(-10*torch.log10(((image-gt)**2)[:,mask].mean().clamp_min(1e-12))),
            crop_SSIM=ssim(crop,ref).mean().item(),crop_LPIPS=metric(crop[None],ref[None],normalize=True).item())
        rec['LPIPS_x100']=100*rec['LPIPS']; records.append(rec)
        b=io.BytesIO(); Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(b,format='PNG'); buffers[step]=b.getvalue()
        if not (leaf/'GT.png').exists():
            Image.fromarray((gt.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(leaf/'GT.png')
            json_out(leaf/'evaluation_region.json',dict(crop_xyxy=[x0,y0,x1,y1],foreground_fraction=float(mask.float().mean()),
                full_frame='Raw model render on white; prediction is NOT composited with test mask',crop='Fixed GT-mask bounding box plus 12px margin',foreground='GT foreground pixels only'))
        print(f'{name}: step={step} PSNR={rec["PSNR"]:.3f} SSIM={rec["SSIM"]:.4f} LPIPS={rec["LPIPS"]:.4f}',flush=True)
    # Only real train/test cameras are needed; generated orbit frames contain ndarray placeholders.
    import scene.dataset_readers as readers
    readers.generate_ellipse_path_from_camera_infos = lambda cameras: []
    train_gs.prepare_output_and_logger=logger; train_gs.training_report=capture
    safe_state(False)
    train_gs.training(args,mp.extract(args),op.extract(args),pp.extract(args),[],[a.steps],[],None,-1)
    if a.smoke:
        if len(records)!=1: raise RuntimeError('Smoke capture did not execute')
        json_out(leaf/'smoke.json',records[0]); print('GPU SMOKE PASSED',name,flush=True); return
    selected=choose(records)
    for rank,r in enumerate(selected,1):
        r['file']=f'{rank:02d}_iter{r["step"]:05d}.png'; (leaf/r['file']).write_bytes(buffers[r['step']])
    with (leaf/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)
    json_out(leaf/'candidates.json',records)
    sheet=Image.new('RGB',(2100,760),'white'); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected+[dict(file='GT.png',step='GT')]):
        im=Image.open(leaf/r['file']); im.thumbnail((408,280)); x,y=i%5*420,i//5*380; sheet.paste(im,(x+(420-im.width)//2,y))
        text='GT' if i==9 else f'{i+1:02d} step {r["step"]}\nPSNR {r["PSNR"]:.3f} SSIM {r["SSIM"]:.4f}\nLPIPS {r["LPIPS"]:.4f}'
        draw.multiline_text((x+8,y+286),text,fill='black',spacing=6)
    sheet.save(leaf/'overview.jpg',quality=95)
    spans={k:max(r[k] for r in selected)-min(r[k] for r in selected) for k in ['PSNR','SSIM','LPIPS']}
    json_out(leaf/'complete.json',dict(count=9,spans=spans,reference_ranges=dict(PSNR=[11.72,15.29],SSIM=[.092,.117],LPIPS=[.1574,.1719]),
        reference_range_met={k:lo<=spans[k]<=hi for k,(lo,hi) in dict(PSNR=[11.72,15.29],SSIM=[.092,.117],LPIPS=[.1574,.1719]).items()},
        scope='Single-view training-stage illustration, not method comparisons or dataset means',parameters=vars(args)))
    print(name,'COMPLETE',spans,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=GO/'output/mip360_foreground_gallery_v1')
    p.add_argument('--scenes',nargs='+',choices=['bonsai','garden','kitchen'],default=['bonsai','garden','kitchen'])
    p.add_argument('--steps',type=int,default=10000); p.add_argument('--check',action='store_true'); p.add_argument('--child',action='store_true'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); a.out=a.out.resolve(); os.chdir(GO)
    if a.smoke:
        if not a.child: p.error('--smoke requires --child')
        a.steps=10
    if not a.smoke and a.steps<1000: p.error('--steps must be >=1000')
    if a.check: check(a); return
    if a.child: train_scene(a,a.scenes[0]); return
    import fcntl
    check(a); a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=dict(steps=a.steps,scenes=a.scenes,script=digest(Path(__file__)),
        inputs={name:{str(f.relative_to(GO/'data/mip360'/name)):digest(f) for f in list((GO/'data/mip360'/name/'sparse/0').glob('*.bin'))+list((GO/'data/mip360'/name).glob('sparse*.txt'))+list(f for f in (GO/'data/mip360'/name/'masks').glob('*.png') if not f.name.startswith('._'))} for name in a.scenes})
    cfg=a.out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise RuntimeError('Configuration changed; use new --out')
    json_out(cfg,config)
    for name in a.scenes:
        with (a.out/f'{name}.log').open('a') as log:
            print('[start]',name,'log:',log.name,flush=True)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child','--scenes',name,'--steps',str(a.steps),'--out',str(a.out)],stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__': main()
