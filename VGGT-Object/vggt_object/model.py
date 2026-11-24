import math
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from .common import setup_vggt


class LoRAConv2d(nn.Module):
    """Low-rank flattened-convolution delta; zero B preserves base output."""
    def __init__(self,base,rank,alpha,dropout):
        super().__init__()
        if base.groups!=1 or base.padding_mode!='zeros':
            raise ValueError('Only standard dense zero-padded Conv2d is supported')
        self.base=base;self.base.requires_grad_(False)
        self.scale=alpha/rank;self.dropout=nn.Dropout2d(dropout)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_channels,*base.kernel_size,device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_channels,rank,1,1,device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,x):
        a=torch.nn.functional.conv2d(self.dropout(x),self.lora_A,None,self.base.stride,self.base.padding,self.base.dilation)
        return self.base(x)+torch.nn.functional.conv2d(a,self.lora_B)*self.scale


def configure(model,cfg):
    model.requires_grad_(False)
    injected=[]
    if cfg.get('adapter_kind')=='depth_full':
        model.depth_head.requires_grad_(True)
    elif cfg['rank']:
        for name,base in list(model.depth_head.named_modules()):
            if (isinstance(base,nn.Conv2d) and any(name.startswith(p) for p in cfg['depth_targets'])
                and name not in cfg.get('lora_exclude',[])):
                parent_name,_,attr=name.rpartition('.')
                parent=model.depth_head.get_submodule(parent_name)
                setattr(parent,attr,LoRAConv2d(base,cfg['rank'],cfg['alpha'],cfg['lora_dropout']))
                injected.append('depth_head.'+name)
        if not injected:
            raise ValueError('No Depth Head Conv2d matched requested LoRA targets')
    from .aggregator_lora import inject,validate,enabled
    aggregator_targets=inject(model,cfg)
    train=[n for n,p in model.named_parameters() if p.requires_grad]
    if not enabled(cfg) and any(not n.startswith('depth_head.') for n in train):
        raise AssertionError('Unexpected trainable parameter')
    if cfg.get('adapter_kind')!='depth_full' and any(not n.endswith(('.lora_A','.lora_B')) for n in train):
        raise AssertionError('LoRA experiment unexpectedly trains base weights')
    stats={'total':sum(p.numel() for p in model.parameters()),
           'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
           'lora':sum(p.numel() for n,p in model.named_parameters() if n.endswith(('.lora_A','.lora_B'))),
           'depth_head':sum(p.numel() for p in model.depth_head.parameters() if p.requires_grad),
           'aggregator_trainable':sum(p.numel() for p in model.aggregator.parameters() if p.requires_grad),
           'injected_modules':injected,'trainable_names':train}
    stats['frozen']=stats['total']-stats['trainable']
    stats['trainable_ratio']=stats['trainable']/stats['total']
    stats['aggregator_lora_targets']=aggregator_targets
    stats['aggregator_lora_parameters']=sum(p.numel() for n,p in model.named_parameters() if n.startswith('aggregator.') and p.requires_grad)
    validate(model,cfg)
    return stats


def create(cfg,device='cuda',meta_only=False):
    setup_vggt(cfg)
    from vggt.models.vggt import VGGT
    # Local DINO constructs its stochastic-depth Python list via linspace(...).item().
    # Those 24 scalar values must be CPU tensors even during meta initialization.
    # Scoped patch affects construction only; existing VGGT files remain unchanged.
    from unittest.mock import patch
    linspace=torch.linspace
    def cpu_linspace(*args,**kwargs):
        kwargs.setdefault('device','cpu')
        return linspace(*args,**kwargs)
    with torch.device('meta'),patch('torch.linspace',cpu_linspace):
        model=VGGT(enable_camera=True,enable_point=False,enable_depth=True,enable_track=False)
    if not meta_only:
        path=Path(cfg['base_checkpoint'])
        if not path.is_file():
            raise FileNotFoundError(f'Local pretrained weights required: {path}')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        state=state.get('model',state.get('state_dict',state))
        # Disabled heads are the only accepted extra base-checkpoint keys.
        state={k:v for k,v in state.items() if not k.startswith(('point_head.','track_head.'))}
        model.load_state_dict(state,strict=True,assign=True)
        del state
        # Nonpersistent normalization buffers are absent from state_dict and cannot
        # be left on meta. Restore the official constants before moving the model.
        from vggt.models.aggregator import _RESNET_MEAN,_RESNET_STD
        model.aggregator._resnet_mean=torch.tensor(_RESNET_MEAN).view(1,1,3,1,1)
        model.aggregator._resnet_std=torch.tensor(_RESNET_STD).view(1,1,3,1,1)
        if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
            raise RuntimeError('Unmaterialized meta tensors remain after local checkpoint load')
        model.to(device)
    stats=configure(model,cfg)
    if cfg.get('aggregator_lora',{}).get('enabled') and stats['depth_head']!=32654562:
        raise AssertionError('Full original DPT must remain trainable (32,654,562 parameters)')
    return model,stats


def train_mode(model):
    model.train()
    model.aggregator.eval()
    model.camera_head.eval()
    from .aggregator_lora import LoRALinear
    for module in model.aggregator.modules():
        if isinstance(module,LoRALinear):module.dropout.train()


def predict(model,images,pose=False):
    # Keep official alternating aggregator unchanged, avoid unused camera training branch.
    needs_grad=any(p.requires_grad for p in model.aggregator.parameters())
    with (nullcontext() if needs_grad else torch.no_grad()),torch.autocast(device_type=images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
        tokens,start=model.aggregator(images)
    # Official heads run in FP32. Casting also preserves adapter gradients.
    with torch.autocast(device_type=images.device.type,enabled=False):
        selected={4,11,17,23}
        tokens=[t.float() if i in selected else t for i,t in enumerate(tokens)]
        depth,conf=model.depth_head(tokens,images=images.float(),patch_start_idx=start,frames_chunk_size=1)
        out={'depth':depth[...,0],'confidence':conf}
        if pose:
            # CameraHead may use only final tokens, official implementation handles this list.
            with torch.no_grad():
                out['pose_enc']=model.camera_head([t.float() for t in tokens])[-1]
    return out
