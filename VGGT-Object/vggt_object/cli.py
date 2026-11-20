import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from .common import PROJECT,REPO,config,json_dump,environment,seed_all,require_gpu


def check(cfg):
    from .model import create
    from .bundle import sources
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch not in ('main','feat/sparse-vcar-seg'):
        raise RuntimeError(f'Expected main or legacy feat/sparse-vcar-seg; found {branch}; no automatic branch switch')
    # No allocation of real 1B parameters and no CUDA calls. Strictly compare local checkpoint key shapes.
    model,stats=create(cfg,device='cpu',meta_only=True)
    state=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    state=state.get('model',state.get('state_dict',state))
    expected=model.state_dict();missing=[]
    for name,value in expected.items():
        if name.endswith(('.lora_A','.lora_B')):
            continue
        base_name=name.replace('.base.', '.')
        if base_name not in state or state[base_name].shape!=value.shape:
            missing.append(name)
    if missing:
        raise ValueError(f'Base checkpoint mismatch: {missing[:10]}')
    compact={k:v for k,v in stats.items() if k!='trainable_names'}
    print(json.dumps(dict(environment=environment(cfg),parameters=compact,
                         free_GiB=shutil.disk_usage(PROJECT).free/1024**3,
                         datasets_present={d:len(list((PROJECT/'data/raw'/d).iterdir())) for d in ('omniobject3d','co3dv2','smithsonian')},
                         configured_data_sources={k:str(v) for k,v in sources().items()},
                         note='CPU/meta check only; raw counts above do not include separately configured existing upload directories'),indent=2),flush=True)
    json_dump(PROJECT/'reports/cpu_preflight.json',dict(environment=environment(cfg),parameters=compact,checkpoint_shapes_verified=True))
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PROJECT/'tests'),'-v'],cwd=PROJECT)
    if result.returncode:
        raise RuntimeError('CPU tests failed')
    json_dump(PROJECT/'reports/cpu_preflight.json',dict(environment=environment(cfg),parameters=compact,
                                                      checkpoint_shapes_verified=True,cpu_tests_passed=True))


def main():
    p=argparse.ArgumentParser(description='VGGT depth adaptation; optional last-six global-attention LoRA')
    p.add_argument('--config')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    agg_check=sub.add_parser('aggregator-check')
    agg_check.add_argument('--rank',type=int,choices=[8,16],default=16)
    sub.add_parser('data-status')
    bridge=sub.add_parser('import-bundle')
    bridge.add_argument('--datasets',nargs='+',choices=['smithsonian','co3dv2'],default=['smithsonian'])
    bridge.add_argument('--review-co3d-depth',action='store_true')
    raw=sub.add_parser('prepare');raw.add_argument('--dataset',choices=['omniobject3d','co3dv2'],required=True)
    raw.add_argument('--raw',type=Path);raw.add_argument('--processed',type=Path,default=PROJECT/'data/processed')
    raw.add_argument('--depth-type',choices=['camera_z_depth','ray_distance'])
    raw.add_argument('--max-objects',type=int,default=20);raw.add_argument('--max-views',type=int,default=48)
    raw.add_argument('--max-side',type=int,default=518);raw.add_argument('--sequence-splits',type=Path)
    raw.add_argument('--co3d-depth-reviewed',action='store_true')
    m=sub.add_parser('manifest');m.add_argument('--processed',type=Path,default=PROJECT/'data/processed')
    m.add_argument('--output',type=Path,default=PROJECT/'data/manifests/objects.json')
    v=sub.add_parser('validate');v.add_argument('--limit',type=int,default=100)
    t=sub.add_parser('train');t.add_argument('--experiment',default='F0',choices=['E1','E2','E3','E4','E5_no_conf','E6','F0','F1','A16','A8'])
    t.add_argument('--phase',default='smoke',choices=['smoke','overfit','pilot','stage_a','stage_b','extended','review','agg_smoke','agg_smoke_2v','agg_smoke_336'])
    t.add_argument('--run',required=True);t.add_argument('--resume',action='store_true');t.add_argument('--init-adapter',type=Path)
    for name in ('eval','export-points'):
        e=sub.add_parser(name);e.add_argument('--adapter',type=Path);e.add_argument('--output',type=Path,required=True)
        e.add_argument('--split',choices=['train','val','test'],default='val')
        e.add_argument('--views',type=int,default=4);e.add_argument('--limit',type=int,default=20)
        e.add_argument('--resolution',type=int,choices=[336,392,518],default=518)
        if name=='export-points':
            e.add_argument('--object-id',required=True);e.add_argument('--confidence-quantile',type=float,default=.5)
    g=sub.add_parser('gs');g.add_argument('--data',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    g.add_argument('--views',type=int,choices=[4,6,9],required=True);g.add_argument('--steps',type=int,default=10000)
    geo=sub.add_parser('geometry');geo.add_argument('--pred',required=True);geo.add_argument('--gt',required=True)
    geo.add_argument('--threshold',type=float,default=.01);geo.add_argument('--output',type=Path,required=True)
    merge=sub.add_parser('merge-depth');merge.add_argument('--adapter',type=Path,required=True);merge.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=config(a.config,getattr(a,'experiment',None),getattr(a,'phase',None))
    if a.command=='check':
        check(cfg)
    elif a.command=='aggregator-check':
        import os
        os.environ['CUDA_VISIBLE_DEVICES']=''
        cfg=config(a.config,'A16' if a.rank==16 else 'A8','agg_smoke')
        check(cfg)
    elif a.command=='data-status':
        from .bundle import sources,upload_status
        paths=sources();upload_status(paths['omni3d'],paths['bundle'],PROJECT/'reports/upload_status.json')
    elif a.command=='import-bundle':
        from .bundle import sources,import_bundle
        paths=sources();import_bundle(paths['bundle'],paths['omni3d'],paths['manifest'],a.datasets,a.review_co3d_depth)
    elif a.command=='prepare':
        from .prepare import omni,co3d
        raw=(a.raw or PROJECT/'data/raw'/a.dataset).resolve()
        if a.max_objects<1 or a.max_views<6 or a.max_side<28:
            p.error('Need max-objects >=1, max-views >=6, max-side >=28')
        if a.dataset=='omniobject3d':
            omni(raw,a.processed,a.depth_type,a.max_objects,a.max_views,a.max_side)
        else:
            co3d(raw,a.processed,a.max_objects,a.max_views,a.max_side,a.co3d_depth_reviewed,a.sequence_splits)
    elif a.command=='manifest':
        from .prepare import build_manifest
        build_manifest(a.processed,a.output,cfg['seed'])
    elif a.command=='validate':
        from .prepare import validate
        validate(cfg['manifest'],PROJECT/'reports/data_quality',limit=a.limit)
    elif a.command=='train':
        from .train import train
        if a.resume and a.init_adapter:
            p.error('--resume and --init-adapter are mutually exclusive')
        train(cfg,a.run,a.resume,a.init_adapter)
    elif a.command in ('eval','export-points','merge-depth'):
        if a.command!='merge-depth':
            require_gpu()
        if a.adapter:
            info=json.loads((a.adapter/'model_info.json').read_text());cfg=info['config']
        else:
            cfg['rank']=0;cfg['adapter_kind']='depth_conv_lora'
        from .model import create
        from .checkpoint import load_adapter
        seed_all(cfg['seed']);model,stats=create(cfg,device='cpu' if a.command=='merge-depth' else 'cuda')
        if a.adapter:
            load_adapter(a.adapter,model,cfg)
        model.eval()
        if a.command=='merge-depth':
            from .model import LoRAConv2d
            for name,module in list(model.depth_head.named_modules()):
                if isinstance(module,LoRAConv2d):
                    delta=(module.lora_B[:,:,0,0]@module.lora_A.flatten(1)).reshape_as(module.base.weight)*module.scale
                    with torch.no_grad():module.base.weight.add_(delta)
                    parent,_,attr=name.rpartition('.');setattr(model.depth_head.get_submodule(parent),attr,module.base)
            from safetensors.torch import save_file
            if a.output.exists():raise RuntimeError('Output already exists')
            a.output.parent.mkdir(parents=True,exist_ok=True)
            save_file({k:v.detach().cpu().contiguous() for k,v in model.depth_head.state_dict().items()},str(a.output))
        else:
            cfg['views']=a.views
            cfg['resolution']=a.resolution
            from .data import objects
            pool=objects(cfg['manifest'],a.split)
            if a.command=='eval':
                from .evaluate import evaluate
                evaluate(model,pool,cfg,a.output,a.limit)
            else:
                from .downstream import export_scene
                matches=[o for o in pool if o['id']==a.object_id]
                if len(matches)!=1:raise ValueError('Object ID must identify one object in selected split')
                if not 0<=a.confidence_quantile<1:raise ValueError('Confidence quantile must be in [0,1)')
                export_scene(model,matches[0],cfg,a.output,a.views,a.confidence_quantile)
    elif a.command=='gs':
        require_gpu()
        from .downstream import train_gaussians
        train_gaussians(a.data.resolve(),a.output.resolve(),a.views,a.steps)
    elif a.command=='geometry':
        from .metrics import geometry_metrics
        json_dump(a.output,geometry_metrics(a.pred,a.gt,a.threshold))


if __name__=='__main__':
    main()
