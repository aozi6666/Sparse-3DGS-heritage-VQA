import json
import math
import os
from pathlib import Path
import time
import torch
from .common import PROJECT,config,seed_all,digest,json_dump,space_guard,environment,require_gpu
from .data import objects,SceneSampler,to_device
from .model import create,predict,train_mode
from .losses import objective
from .checkpoint import save,restore,load_adapter
from .evaluate import evaluate


def _train(cfg,run,resume=False,init_adapter=None):
    from .aggregator_lora import enabled,parameter_groups,grad_norm,UpdateInspector
    agg_enabled=enabled(cfg)
    if agg_enabled:
        initial=str(init_adapter.resolve()) if init_adapter else None
        if resume:
            initial=json.loads((PROJECT/'experiments'/run/'config.json').read_text()).get('initialization_adapter')
        cfg={**cfg,'initialization_adapter':initial}
    if not Path(cfg['manifest']).is_file():
        raise FileNotFoundError('Training manifest not ready. Finish upload first, then run '
                                '`bash VGGT-Object/run.sh data-status`, `import-bundle --datasets smithsonian`, '
                                'and `validate`. Do not start training against an incomplete upload.')
    quality=PROJECT/'reports/data_quality/quality.json'
    if not quality.is_file() or json.loads(quality.read_text()).get('manifest_sha256')!=digest(cfg['manifest']):
        raise RuntimeError('Current training manifest has not passed CPU data validation. Run '
                           '`bash VGGT-Object/run.sh validate --limit 100` and inspect the quality previews first.')
    require_gpu()
    if cfg['batch_scenes']!=1 or cfg['resolution']%14:
        raise ValueError('First implementation supports one scene per batch and patch-14 resolution')
    if not cfg['rank'] and cfg.get('adapter_kind')!='depth_full':
        raise ValueError('E0 is frozen baseline: use eval instead of train')
    if cfg['phase']=='stage_b' and not (resume or init_adapter):
        raise ValueError('Stage B requires --init-adapter from Stage A (or --resume)')
    space_guard();seed_all(cfg['seed'])
    root=PROJECT/'experiments'/run
    if not run or Path(run).name!=run:
        raise ValueError('--run must be a simple directory name')
    if root.exists() and not resume:
        raise RuntimeError(f'Run already exists: {root}; use --resume or a new --run name')
    if resume and not (root/'latest/trainer.pt').is_file():
        raise FileNotFoundError(root/'latest/trainer.pt')
    manifest_hash=digest(cfg['manifest']);sampler=SceneSampler(cfg['manifest'],cfg)
    if cfg['phase']=='stage_b' and 'co3dv2' not in sampler.groups:
        raise ValueError('Stage B requires reviewed CO3D data; do not label Smithsonian-only training as real-domain adaptation')
    try:
        validation=[o for o in objects(cfg['manifest'],'val') if o['dataset'] in sampler.groups
                    and (o['dataset']!='co3dv2' or o.get('depth_reviewed',False))]
        if not validation:
            raise ValueError('No validation objects from active reviewed datasets')
        val_scope='val'
    except ValueError:
        if cfg['phase'] not in ('smoke','overfit'):
            raise ValueError('Formal training requires disjoint validation objects; no test-set fallback')
        validation=[o for o in objects(cfg['manifest'],'train') if o['dataset'] in sampler.groups][:2]
        val_scope='train_diagnostic_only'
    model,stats=create(cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    groups=parameter_groups(model,cfg) if agg_enabled else params
    optimizer=torch.optim.AdamW(groups,lr=cfg['depth_lr'],weight_decay=cfg['lora_weight_decay'],betas=(.9,.999),eps=1e-8)
    warm=max(1,round(cfg['steps']*cfg['warmup_ratio']));floor=cfg['min_lr']/cfg['depth_lr']
    def factor(step):
        if step<warm:
            return max(1e-3,(step+1)/warm)
        return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,cfg['steps']-warm))))
    if agg_enabled:
        def schedule(base_lr):
            def multiplier(step):
                if step<warm:return max(1e-3,(step+1)/warm)
                floor_lr=cfg['min_lr']/base_lr
                return floor_lr+(1-floor_lr)*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,cfg['steps']-warm))))
            return multiplier
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[schedule(group['lr']) for group in optimizer.param_groups])
    else:
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    start=0;best=float('inf')
    if resume:
        start,best=restore(root/'latest',model,optimizer,scheduler,cfg,sampler,manifest_hash)
    elif init_adapter:
        load_adapter(init_adapter,model,cfg,allow_new_aggregator=agg_enabled)
    root.mkdir(parents=True,exist_ok=True)
    json_dump(root/'config.json',cfg);json_dump(root/'parameters.json',stats);json_dump(root/'environment.json',environment(cfg))
    if agg_enabled:json_dump(root/'parameter_report.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if k!='trainable_names'},indent=2),flush=True)
    # Baseline is evaluated before updates. A failed adaptation cannot silently
    # replace a better original model simply because it is the first checkpoint.
    if resume:
        baseline=json.loads((root/'baseline_validation.json').read_text())
        control=json.loads((root/'latest/model_info.json').read_text()).get('training_control',{})
        bad_validations=control.get('bad_validations',0)
        patience_score=control.get('patience_score',best)
    else:
        baseline=evaluate(model,validation,cfg,limit=cfg['validation_objects'])
        json_dump(root/'baseline_validation.json',baseline)
        best=baseline['summary'][cfg['validation_key']]['AbsRel'];bad_validations=0
        patience_score=best
        save(root/'best',model,optimizer,scheduler,0,cfg,sampler,best,manifest_hash,resume=False)
    baseline_raw=baseline['summary']['canonical_unaligned']['AbsRel']
    train_mode(model);torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    actual_steps=start;stopped_early=False
    probe=None;update_evidence={}
    with (root/'train.jsonl').open('a',buffering=1) as log:
        for step in range(start,cfg['steps']):
            step_started=time.monotonic()
            inspector=UpdateInspector(model,optimizer) if agg_enabled and step==start else None
            optimizer.zero_grad(set_to_none=True);terms={};should_stop=False
            for micro in range(cfg['accumulation']):
                batch=to_device(sampler.next(),'cuda')
                if agg_enabled and probe is None:probe=batch['images'].detach().cpu().clone()
                out=predict(model,batch['images']);components={}
                loss,detail=objective(out,batch,cfg,step,components)
                if cfg.get('component_gradient_diagnostics',True) and micro==0 and step%cfg.get('gradient_diagnostics_every',50)==0:
                    # Infrequent diagnostics, not four extra backwards at every step.
                    for key,component in components.items():
                        if component.requires_grad:
                            gradients=torch.autograd.grad(component,params,retain_graph=True,allow_unused=True)
                            norms=[g.detach().float().square().sum() for g in gradients if g is not None]
                            detail['grad_'+key]=float(torch.stack(norms).sum().sqrt()) if norms else 0.
                            del gradients,norms
                (loss/cfg['accumulation']).backward()
                for key,value in detail.items():
                    terms[key]=terms.get(key,0)+value/cfg['accumulation']
            if agg_enabled:
                lora_norm=grad_norm(optimizer.param_groups[1]['params']);dpt_norm=grad_norm(optimizer.param_groups[0]['params'])
                if lora_norm<=0 or dpt_norm<=0:raise AssertionError('Missing LoRA or DPT gradients')
                terms.update(lora_gradient_norm=lora_norm,dpt_gradient_norm=dpt_norm)
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_norm'],error_if_nonfinite=True)
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in params):
                raise RuntimeError('All trainable gradients are zero')
            used_lrs=[group["lr"] for group in optimizer.param_groups]
            optimizer.step();scheduler.step()
            if inspector is not None:
                update_evidence=inspector.verify();del inspector
                json_dump(root/'gradient_verification.json',update_evidence)
            actual_steps=step+1
            terms.update(step=step+1,gradient_norm=float(norm),lr=scheduler.get_last_lr()[0],
                         elapsed_seconds=time.monotonic()-began,peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3)
            if agg_enabled:
                terms.update(depth_learning_rate=used_lrs[0],lora_learning_rate=used_lrs[1],
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,step_seconds=time.monotonic()-step_started)
            log.write(json.dumps(terms,allow_nan=False)+'\n')
            if agg_enabled or (step+1)%10==0 or step==start:
                print(json.dumps(terms),flush=True)
            if (step+1)%cfg['validate_every']==0 or step+1==cfg['steps']:
                result=evaluate(model,validation,cfg,limit=cfg['validation_objects'])
                score=result['summary'][cfg['validation_key']]['AbsRel']
                raw=result['summary']['canonical_unaligned']['AbsRel']
                eligible=raw<=baseline_raw*cfg.get('canonical_regression_limit',1.25)
                json_dump(root/'validation_latest.json',dict(step=step+1,scope=val_scope,**result))
                with (root/'validation_history.jsonl').open('a') as history:
                    history.write(json.dumps(dict(step=step+1,score=score,canonical_AbsRel=raw,eligible=eligible))+'\n')
                if eligible and score<patience_score-cfg.get('early_stopping_min_delta',.001):
                    patience_score=score;bad_validations=0
                else:
                    bad_validations+=1
                # Always save the lowest eligible score; min_delta controls patience only.
                if eligible and score<best:
                    best=score
                    save(root/'best',model,optimizer,scheduler,step+1,cfg,sampler,best,manifest_hash,resume=False)
                should_stop=bad_validations>=cfg.get('early_stopping_patience',4)
                print(f'validation step={step+1} aligned AbsRel={score:.6f} canonical={raw:.6f} eligible={eligible} best={best:.6f} scope={val_scope}',flush=True)
            if (step+1)%cfg['save_every']==0 or step+1==cfg['steps'] or should_stop:
                save(root/'latest',model,optimizer,scheduler,step+1,cfg,sampler,best,manifest_hash,
                     control={'bad_validations':bad_validations,'patience_score':patience_score})
            if should_stop:
                stopped_early=True
                print(f'Early stop at {actual_steps}: validation did not improve; original baseline remains a valid best.',flush=True)
                break
    # Round-trip selected adapter tensor values; end-to-end predictions tested separately by eval.
    from safetensors.torch import load_file
    if cfg.get('adapter_kind')=='depth_full':
        saved=load_file(str(root/'latest/depth_head.safetensors'));named=model.depth_head.state_dict()
    else:
        saved=load_file(str(root/'latest/depth_head_lora.safetensors'));named=dict(model.named_parameters())
    if not all(torch.equal(named[k].detach().cpu(),v) for k,v in saved.items()):
        raise AssertionError('Saved adapter is not identical to in-memory parameters')
    completion=dict(steps=actual_steps,planned_steps=cfg['steps'],stopped_early=stopped_early,
                                      baseline_validation=baseline['summary'][cfg['validation_key']]['AbsRel'],
                                      best_validation=best,validation_scope=val_scope,
                                      duration_seconds=time.monotonic()-began,adapter_roundtrip=True,
                                      peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3)
    if agg_enabled:
        # Release last backward references before the fresh-model inference check.
        optimizer.zero_grad(set_to_none=True)
        if actual_steps>start:
            del out,loss,components,batch
        del named,saved,params,optimizer,scheduler,groups
        if probe is None:probe=sampler.next()['images'].detach().cpu().clone()
        if not update_evidence and (root/'gradient_verification.json').is_file():
            update_evidence=json.loads((root/'gradient_verification.json').read_text())
        from .aggregator_smoke import finish
        finish(model,cfg,root,probe,update_evidence)
        completion['aggregator_smoke_verification']=True
    json_dump(root/'complete.json',completion)


def train(cfg,run,resume=False,init_adapter=None):
    from .aggregator_lora import enabled
    if enabled(cfg):
        if not run or Path(run).name!=run:raise ValueError('Invalid run name')
        if (PROJECT/'experiments'/run).exists() and not resume:
            raise RuntimeError('Existing run preserved; use --resume or a new run name')
    try:
        return _train(cfg,run,resume,init_adapter)
    except Exception as error:
        root=PROJECT/'experiments'/run
        # Never write failure reports into pre-existing unrelated experiments.
        if enabled(cfg) and (root/'config.json').is_file():
            recorded=json.loads((root/'config.json').read_text())
            if recorded.get('aggregator_lora',{}).get('enabled'):
                json_dump(root/'memory_report.json',dict(status='failed',error=str(error),
                    oom=isinstance(error,torch.cuda.OutOfMemoryError),
                    nonfinite_detected='non-finite' in str(error).lower() or 'nan' in str(error).lower(),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else None))
        raise
