"""Isolated paired project-3DGS/Drop diagnostic. No synthetic experiment data."""
import argparse,csv,json,os,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
GO=Path(__file__).resolve().parents[2]
ROOT=GO.parents[1]
sys.path[:0]=[str(GO),str(GO/'experiments/llff_scene_gallery')]
# Import preparation by unique module name to avoid this run.py name collision.
import importlib.util
spec=importlib.util.spec_from_file_location('llff_prepare',GO/'experiments/llff_scene_gallery/run.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
from analysis import bins,table,plot
def child(a):
    import torch,train_gs as train
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from utils.full_scene import full_scene_options
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    import lpips
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('GPU required; use --check on CPU')
    leaf=a.out/f'v{a.views}'/a.scene/a.method
    if leaf.exists():raise RuntimeError(f'Existing run preserved: {leaf}; choose new --out')
    leaf.mkdir(parents=True)
    parser=argparse.ArgumentParser();mp,op,pp=ModelParams(parser),OptimizationParams(parser),PipelineParams(parser);b=parser.parse_args([])
    b.source_path=str(a.out/'prepared'/a.scene);b.model_path=str(leaf);b.full_scene=True;b.eval=True;b.resolution=1;b.sh_degree=2
    b.iterations=a.steps;b.position_lr_max_steps=a.steps;b.densify_until_iter=min(a.steps-1,5000);b.max_num_splats=200000
    b.mono_depth_weight=0.;b.mono_loss_type='mid';b.lambda_t_norm=0.;b.use_dust3r=False;b.sparse_view_num=-1;b.init_pcd_name='origin';b.transform_the_world=False
    b.use_drop=a.method=='drop';b.drop_rate=.2;b.drop_schedule='linear';full_scene_options(b)
    counts=np.zeros(10,dtype=np.int64);sums=np.zeros(10);above=np.zeros(10,dtype=np.int64);zeros=np.zeros(10,dtype=np.int64);accn=np.zeros(10,dtype=np.int64);acca=np.zeros(10,dtype=np.int64);events=[];bounds={};positive=np.zeros(10,dtype=np.int64);clipped=[0,0]
    import scene.dataset_readers as readers
    original=readers.sceneLoadTypeCallbacks['Colmap']
    def reader(*args,**kwargs):
        info=original(*args,**kwargs)
        cams=info.train_cameras
        if len(cams)<9:raise ValueError('Need >=9 non-test cameras')
        ids9=np.linspace(0,len(cams)-1,9).round().astype(int)
        ids=ids9[[0,4,8]] if a.views==3 else ids9
        return info._replace(train_cameras=[cams[i] for i in ids],render_cameras=[])
    readers.sceneLoadTypeCallbacks['Colmap']=reader
    Scene=train.Scene
    def scene(*args,**kwargs):
        result=Scene(*args,**kwargs);xyz=result.gaussians.get_xyz.detach()
        for cam in result.getTrainCameras():
            z=(xyz@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3])[:,2];z=z[z>0]
            bounds[cam.image_name]=[float(z.min()),float(z.max())]
        path=a.out/f'v{a.views}'/a.scene/'depth_bounds.json'
        if path.exists():
            prior=json.loads(path.read_text())
            if prior!=bounds:raise AssertionError('Different initial camera/depth bounds')
        else:dump(path,bounds)
        dump(leaf/'config.json',dict(arguments=vars(b),seed=0,train_cameras=[c.image_name for c in result.getTrainCameras()],test_cameras=[c.image_name for c in result.getTestCameras()],views=a.views,protocol_version=2,camera_selection='original uniform9 non-test camera positions [0,4,8] for v3; same full9 for v9',trainer_sha256=hashlib.sha256((GO/'train_gs.py').read_bytes()).hexdigest()))
        return result
    train.Scene=scene
    @torch.no_grad()
    def observe(step,gaussians,cam,pkg):
        if step<1000 or step%50 or step>=b.densify_until_iter:return
        grad=pkg['viewspace_points'].grad
        if grad is None:raise RuntimeError('Missing retained screen-space gradient')
        g=grad[:,:2].norm(dim=-1);xyz=gaussians.get_xyz.detach()
        z=(xyz@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3])[:,2]
        v=pkg['visibility_filter']
        if not torch.isfinite(g[v]).all():raise RuntimeError('Nonfinite visible gradient')
        z=z.cpu().numpy();g=g.cpu().numpy();v=v.cpu().numpy()
        event=bins(z,g,v,*bounds[cam.image_name],b.densify_grad_threshold)
        counts[:]+=event['count'];sums[:]+=event['sum'];above[:]+=event['above_count'];zeros[:]+=event['zero_count']
        positive[:]+=bins(z,np.zeros_like(g),np.ones_like(v,dtype=bool),*bounds[cam.image_name],b.densify_grad_threshold)['count']
        clipped[0]+=event['clipped_low'];clipped[1]+=event['clipped_high']
        events.append(dict(step=step,camera=cam.image_name,**event,densification_accumulator_active=bool(len(gaussians.get_xyz)<b.max_num_splats)))
    @torch.no_grad()
    def accumulated(step,gaussians,cam):
        if step<1000 or step>=b.densify_until_iter or step<=b.densify_from_iter or step%b.densification_interval:return
        denom=gaussians.denom[:,0];value=gaussians.xyz_gradient_accum[:,0]/denom.clamp_min(1)
        xyz=gaussians.get_xyz.detach();z=(xyz@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3])[:,2]
        result=bins(z.cpu().numpy(),value.cpu().numpy(),(denom>0).cpu().numpy(),*bounds[cam.image_name],b.densify_grad_threshold,inclusive=True)
        accn[:]+=result['count'];acca[:]+=result['above_count']
    train.depth_observe=observe;train.depth_accumulated=accumulated
    source=(GO/'train_gs.py').read_text();needle='        loss.backward()'
    if source.count(needle)!=1:raise RuntimeError('Trainer changed; review hook position')
    source=source.replace(needle,needle+'\n        depth_observe(iteration,gaussians,viewpoint_cam,render_pkg)')
    add='                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)'
    if source.count(add)!=1:raise RuntimeError('Densification hook changed')
    source=source.replace(add,add+'\n                depth_accumulated(iteration,gaussians,viewpoint_cam)')
    exec(compile(source,str(GO/'train_gs.py'),'exec'),train.__dict__)
    train.Scene=scene;train.prepare_output_and_logger=lambda dataset:None
    @torch.no_grad()
    def report(tb,step,l1,loss,lf,elapsed,testing,sc,render,render_args):
        if step!=a.steps:return
        metric=lpips.LPIPS(net='vgg').cuda().eval();rows=[]
        for i,cam in enumerate(sc.getTestCameras()):
            rgb=render(cam,sc.gaussians,*render_args)['render'].clamp(0,1);gt=cam.original_image[:3].cuda().clamp(0,1)
            rgb=(rgb*255).round()/255;gt=(gt*255).round()/255
            rows.append(dict(camera=cam.image_name,PSNR=float(psnr(rgb,gt).mean()),SSIM=float(ssim(rgb,gt).mean()),LPIPS=float(metric(rgb[None],gt[None],normalize=True))))
            if i==0:
                for name,img in [('render',rgb),('GT',gt)]:Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(leaf/f'{name}.png')
        table(leaf/'per_view_metrics.csv',rows)
        dump(leaf/'metrics.json',dict(test_mean={key:float(np.mean([r[key] for r in rows])) for key in ['PSNR','SSIM','LPIPS']},displayed_view=rows[0],LPIPS_backbone='VGG',images='8-bit full background'))
    train.training_report=report;safe_state(False)
    train.training(b,mp.extract(b),op.extract(b),pp.extract(b),[],[],[],None,-1)
    if not counts.sum():raise RuntimeError('No observations captured')
    protocol=dict(version=2,views=a.views,steps=a.steps,seed=0,threshold=b.densify_grad_threshold,bin_edges=np.linspace(0,1,11).tolist(),statistics_start=1000,statistics_stride=50,statistics_stop_exclusive=b.densify_until_iter,depth_normalization='per-camera common initial positive Z min/max; clipped; no refit',unit='Gaussian-view-training-step observation; visible zero gradients retained')
    dump(leaf/'statistics.json',dict(protocol=protocol,count=counts.tolist(),sum=sums.tolist(),above_count=above.tolist(),zero_count=zeros.tolist(),accumulator_count=accn.tolist(),accumulator_above_count=acca.tolist(),events=events,diagnostics=dict(positive_depth_count=positive.tolist(),clipped_low=clipped[0],clipped_high=clipped[1])))
    dump(leaf/'complete.json',dict(status='complete'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=GO/'output/gradient_depth_drop_v1');p.add_argument('--scenes',nargs='+',default=['horns','fortress'],choices=['fern','horns','fortress']);p.add_argument('--views',type=int,choices=[3,9],default=3);p.add_argument('--plot-only',action='store_true');p.add_argument('--steps',type=int,default=10000);p.add_argument('--check',action='store_true');p.add_argument('--scene');p.add_argument('--method',choices=['3DGS','drop']);a=p.parse_args();a.out=a.out.resolve()
    if a.steps<2500:p.error('steps >=2500 required for statistics window')
    os.chdir(GO)
    if a.method:child(a);return
    if a.plot_only:
        for n in a.scenes:plot(a.out/f'v{a.views}'/n)
        return
    import torch
    for n in a.scenes:prep.load_source(GO/'data/LLFF'/n)
    prep.guard_disk();weights=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    if not weights.exists():raise FileNotFoundError(weights)
    if a.check:
        import pycolmap
        for name in a.scenes:
            rec=prep.load_source(GO/'data/LLFF'/name)
            allnames=sorted(Path(im.name).stem for im in rec.images.values())
            train=[n for i,n in enumerate(allnames) if i%8!=0]
            ids=np.linspace(0,len(train)-1,9).round().astype(int)
            chosen=[train[i] for i in (ids[[0,4,8]] if a.views==3 else ids)]
            assert len(set(chosen))==a.views
            print(name,'training cameras:',chosen,flush=True)
        source=(GO/'scene/gaussian_model.py').read_text()
        assert 'viewspace_point_tensor.grad[update_filter,:2]' in source
        assert (GO/'train_gs.py').read_text().count('        loss.backward()')==1
        n=bins(np.array([1.,1.5,2.,3.,0.]),np.array([0.,.0002,.0003,.0004,9.]),np.ones(5,dtype=bool),1.,3.,.0002)
        assert sum(n['count'])==4 and n['count'][0]==1 and n['count'][2]==1 and n['count'][5]==1 and n['count'][9]==1
        assert sum(n['above_count'])==2 and sum(n['zero_count'])==1
        print('CPU preflight passed: source cameras/images, offline VGG, bin edges and zero-gradient inclusion; no GPU execution.');return
    if not torch.cuda.is_available():raise RuntimeError('No GPU; use --check')
    ranking=[]
    for n in a.scenes:
        root=a.out/f'v{a.views}'/n;root.mkdir(parents=True,exist_ok=True);prep.prepare(GO/'data/LLFF'/n,a.out/'prepared'/n,4)
        for method in ['3DGS','drop']:
            if not (root/method/'complete.json').exists():subprocess.run([sys.executable,str(Path(__file__).resolve()),'--out',str(a.out),'--scene',n,'--method',method,'--steps',str(a.steps),'--views',str(a.views)],check=True)
        cb=json.loads((root/'3DGS/config.json').read_text());cd=json.loads((root/'drop/config.json').read_text())
        assert cb['train_cameras']==cd['train_cameras'] and cb['test_cameras']==cd['test_cameras']
        assert len(cb['train_cameras'])==a.views and not set(cb['train_cameras'])&set(cb['test_cameras'])
        b=json.loads((root/'3DGS/statistics.json').read_text());d=json.loads((root/'drop/statistics.json').read_text())
        if b['protocol']['steps']!=a.steps or d['protocol']['steps']!=a.steps:raise AssertionError('Existing run steps differ')
        assert [(e['step'],e['camera']) for e in b['events']]==[(e['step'],e['camera']) for e in d['events']], 'Observation schedules differ; do not compare'
        ranking.append(dict(scene=n,mean_absolute_curve_difference_x1000=plot(root)))
    table(a.out/f'v{a.views}'/'scene_differences.csv',sorted(ranking,key=lambda x:x['mean_absolute_curve_difference_x1000'],reverse=True))
if __name__=='__main__':main()
