"""Opt-in LoRA on six real VGGT global-attention blocks; no upstream edits."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def options(cfg):
    return cfg.get('aggregator_lora',{'enabled':False})


def enabled(cfg):
    return bool(options(cfg).get('enabled',False))


class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha,dropout):
        super().__init__()
        if not isinstance(base,nn.Linear):raise TypeError('LoRA target must be an actual Linear')
        self.base=base;self.base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.dropout=nn.Dropout(dropout);self.scale=alpha/rank;self.enabled=True

    def forward(self,x):
        y=self.base(x)
        if not self.enabled:return y
        a=torch.nn.functional.linear(self.dropout(x).to(self.lora_A.dtype),self.lora_A)
        return y+torch.nn.functional.linear(a,self.lora_B).to(y.dtype)*self.scale


def block_checkpoint(block):
    """Instance-local wrapper of original forward, with unchanged parameter names."""
    if getattr(block,'_object_checkpointed',False):raise ValueError('Block checkpoint already installed')
    original=block.forward
    def forward(*args,**kwargs):
        if torch.is_grad_enabled():
            return checkpoint(original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
        return original(*args,**kwargs)
    block.forward=forward;block._object_checkpointed=True


def inject(model,cfg):
    opt=options(cfg)
    if not enabled(cfg):return []
    if cfg['adapter_kind']!='depth_full':raise ValueError('Aggregator LoRA requires full DPT in this experiment')
    rank=int(opt['rank']);alpha=float(opt['alpha']);drop=float(opt.get('dropout',.05))
    indices=opt.get('target_blocks',list(range(18,24)))
    targets=opt.get('target_modules',['attn.qkv','attn.proj'])
    if indices!=list(range(18,24)) or targets!=['attn.qkv','attn.proj']:
        raise ValueError('Only global_blocks[18:24] attn.qkv/proj are allowed')
    if rank not in (8,16) or alpha<=0 or not 0<=drop<1:raise ValueError('Invalid LoRA hyperparameters')
    blocks=model.aggregator.global_blocks
    if len(blocks)!=24 or len(model.aggregator.frame_blocks)!=24:
        raise ValueError('Expected 24 frame blocks and 24 global blocks')
    report=[]
    for i in indices:
        for attr in ('qkv','proj'):
            base=getattr(blocks[i].attn,attr)
            if not isinstance(base,nn.Linear):raise TypeError(f'Unexpected/repeated injection: global {i} {attr}')
            report.append(dict(name=f'aggregator.global_blocks.{i}.attn.{attr}',
                               in_features=base.in_features,out_features=base.out_features))
            if opt.get('verify_vggt_dimensions',True):
                if (base.in_features,base.out_features)!=((1024,3072) if attr=='qkv' else (1024,1024)):
                    raise ValueError(f'Unexpected real VGGT projection dimensions: {report[-1]}')
            setattr(blocks[i].attn,attr,LoRALinear(base,rank,alpha,drop))
    if opt.get('gradient_checkpointing',True):
        for i in indices:block_checkpoint(blocks[i])
        # Frozen frame blocks after the first LoRA must still carry input gradients.
        for i in range(19,24):block_checkpoint(model.aggregator.frame_blocks[i])
    return report


def validate(model,cfg):
    if not enabled(cfg):return
    for name,p in model.named_parameters():
        expected=name.startswith('depth_head.') or (name.startswith('aggregator.global_blocks.') and name.endswith(('.lora_A','.lora_B')))
        if p.requires_grad!=expected:raise AssertionError(f'Incorrect freezing: {name}')
    if getattr(model,'camera_head',None) is not None and any(p.requires_grad for p in model.camera_head.parameters()):
        raise AssertionError('Camera head unexpectedly trainable')
    if getattr(model,'point_head',None) is not None or getattr(model,'track_head',None) is not None:
        raise AssertionError('Point/track heads must be absent')
    actual=sum(p.numel() for n,p in model.named_parameters() if n.startswith('aggregator.') and p.requires_grad)
    if options(cfg).get('verify_vggt_dimensions',True) and actual!=36864*int(options(cfg)['rank']):
        raise AssertionError(f'Unexpected LoRA count: {actual}')


def parameter_groups(model,cfg):
    depth=[p for n,p in model.named_parameters() if n.startswith('depth_head.') and p.requires_grad]
    agg=[p for n,p in model.named_parameters() if n.startswith('aggregator.') and p.requires_grad]
    if not depth or not agg:raise AssertionError('Both DPT and Aggregator LoRA must train')
    return [dict(params=depth,lr=cfg['depth_lr'],name='depth'),
            dict(params=agg,lr=cfg['aggregator_lr'],name='aggregator_lora')]


def grad_norm(parameters):
    norms=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.


@contextmanager
def disabled(model):
    layers=[m for m in model.modules() if isinstance(m,LoRALinear)]
    old=[m.enabled for m in layers]
    try:
        for m in layers:m.enabled=False
        yield
    finally:
        for m,value in zip(layers,old):m.enabled=value


class UpdateInspector:
    """First-step update evidence; no multi-GB frozen-weight disk copies."""
    def __init__(self,model,optimizer):
        self.model=model
        self.frozen={n:(p,p._version) for n,p in model.named_parameters() if not p.requires_grad}
        train_ids={id(p) for group in optimizer.param_groups for p in group['params']}
        if any(id(p) in train_ids for p,_ in self.frozen.values()):raise AssertionError('Frozen weights in optimizer')
        self.before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    def verify(self):
        for n,(p,version) in self.frozen.items():
            if p.grad is not None or p._version!=version:raise AssertionError(f'Frozen parameter changed: {n}')
        changed={'aggregator':False,'depth_head':False}
        for n,p in self.model.named_parameters():
            if n in self.before and not torch.equal(p.detach(),self.before[n]):changed[n.split('.')[0]]=True
        if not all(changed.values()):raise AssertionError(f'Incomplete optimizer update: {changed}')
        self.before.clear()
        return dict(lora_updated=changed['aggregator'],dpt_updated=changed['depth_head'],
                    frozen_versions_unchanged=True,frozen_gradients_absent=True,
                    frozen_check='All frozen parameter version counters + optimizer exclusion; CPU tests additionally compare exact tensors.')
