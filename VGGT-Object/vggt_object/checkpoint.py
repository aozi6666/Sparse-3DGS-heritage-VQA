import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from .common import json_dump,environment,space_guard


def save(folder,model,optimizer,scheduler,step,cfg,sampler,best,manifest_hash,resume=True,control=None):
    space_guard()
    folder=Path(folder);tmp=folder.with_name(folder.name+'.writing')
    if tmp.exists():
        raise RuntimeError(f'Interrupted checkpoint preserved: {tmp}; inspect it before removing')
    tmp.mkdir(parents=True)
    lora={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n.startswith('depth_head.') and n.endswith(('.lora_A','.lora_B'))}
    if lora:
        save_file(lora,str(tmp/'depth_head_lora.safetensors'))
    agg={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n.startswith('aggregator.') and n.endswith(('.lora_A','.lora_B'))}
    if agg:save_file(agg,str(tmp/'aggregator_lora.safetensors'))
    if cfg.get('adapter_kind')=='depth_full':
        save_file({n:v.detach().cpu().contiguous() for n,v in model.depth_head.state_dict().items()},str(tmp/'depth_head.safetensors'))
    info=dict(config=cfg,step=step,manifest_sha256=manifest_hash,best_validation=best,training_control=control or {},
              environment=environment(cfg),base_checkpoint=dict(path=cfg['base_checkpoint'],size=Path(cfg['base_checkpoint']).stat().st_size),
              targets=cfg['depth_targets'],adapter_kind=cfg['adapter_kind'],aggregator=cfg.get('aggregator_lora',{'enabled':False}),
              gauge='first input camera; mean GT foreground world-point distance canonical scale',
              license='Base model and datasets retain their original licenses; no rights changed by adapters.')
    json_dump(tmp/'model_info.json',info)
    if resume:
        torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,best=best,
                        sampler_rng=sampler.rng.getstate(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        sampler_state=sampler.state_dict() if hasattr(sampler,'state_dict') else None,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp/'trainer.pt')
    # Keep previous checkpoint until fully written; interruption never leaves no recoverable version.
    previous=folder.with_name(folder.name+'.previous')
    if previous.exists():
        raise RuntimeError(f'Previous interrupted swap preserved: {previous}')
    if folder.exists():
        os.replace(folder,previous)
    os.replace(tmp,folder)
    if previous.exists():
        shutil.rmtree(previous)  # Only this run's superseded, fully replaced checkpoint.


def load_adapter(folder,model,cfg,allow_new_aggregator=False):
    folder=Path(folder);info=json.loads((folder/'model_info.json').read_text())
    for key in ('rank','alpha','depth_targets','adapter_kind'):
        if info['config'][key]!=cfg[key]:
            raise ValueError(f'Checkpoint architecture mismatch {key}')
    current=cfg.get('aggregator_lora',{'enabled':False});previous=info['config'].get('aggregator_lora',{'enabled':False})
    current_on=bool(current.get('enabled'));previous_on=bool(previous.get('enabled'))
    if current_on!=previous_on and not (allow_new_aggregator and current_on and not previous_on):
        raise ValueError('Aggregator LoRA enabled state differs from checkpoint')
    if previous_on:
        for key in ('rank','alpha','dropout','target_blocks','target_modules'):
            if current.get(key)!=previous.get(key):raise ValueError(f'Aggregator LoRA mismatch: {key}')
        state=load_file(str(folder/'aggregator_lora.safetensors'))
        parameters=dict(model.named_parameters())
        expected={n for n in parameters if n.startswith('aggregator.') and n.endswith(('.lora_A','.lora_B'))}
        if set(state)!=expected:raise ValueError('Aggregator adapter keys mismatch')
        with torch.no_grad():
            for name,value in state.items():parameters[name].copy_(value)
    if cfg.get('adapter_kind')=='depth_full':
        model.depth_head.load_state_dict(load_file(str(folder/'depth_head.safetensors')),strict=True)
    elif cfg['rank']:
        state=load_file(str(folder/'depth_head_lora.safetensors'))
        expected={n for n,p in model.named_parameters() if n.endswith(('.lora_A','.lora_B'))}
        if set(state)!=expected:
            raise ValueError('Adapter keys do not match injected modules')
        parameters=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in state.items():
                parameters[name].copy_(value)
    return info


import json


def restore(folder,model,optimizer,scheduler,cfg,sampler,manifest_hash):
    info=load_adapter(folder,model,cfg)
    if info['manifest_sha256']!=manifest_hash:
        raise ValueError('Manifest changed; resume would not reproduce the same experiment')
    for key in ('resolution','views','accumulation','steps','mix','experiment','phase','protocol_version',
                'grad_weight','confidence_weight','mv_weight','view_sampling','validation_cases'):
        if info['config'][key]!=cfg[key]:
            raise ValueError(f'Resume config mismatch: {key}; use --init-adapter for a new stage')
    for key in ('aggregator_lr','depth_lr'):
        if info['config'].get(key)!=cfg.get(key):raise ValueError(f'Resume LR mismatch: {key}')
    if info['config'].get('rgb_augmentation',False)!=cfg.get('rgb_augmentation',False):
        raise ValueError('Resume photometric augmentation differs from checkpoint')
    # trainer.pt is generated by this project; contains Python RNG state, not an external model.
    state=torch.load(Path(folder)/'trainer.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    sampler.rng.setstate(state['sampler_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    if state.get('sampler_state') and hasattr(sampler,'load_state_dict'):
        sampler.load_state_dict(state['sampler_state'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'],state['best']
