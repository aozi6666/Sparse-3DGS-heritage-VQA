import torch
from .geometry import multiview


def gradient_loss(pred,gt,mask):
    parts=[]
    for step in (1,2,4):
        diff=(pred-gt)[...,::step,::step]
        m=mask[...,::step,::step]
        for axis in (-1,-2):
            if axis==-1:
                delta=diff[...,1:]-diff[...,:-1];valid=m[...,1:]&m[...,:-1]
            else:
                delta=diff[...,1:,:]-diff[...,:-1,:];valid=m[...,1:,:]&m[...,:-1,:]
            if valid.any():
                parts.append(delta[valid].abs().mean())
    return torch.stack(parts).mean() if parts else pred.sum()*0


def objective(out,batch,cfg,step,components=None):
    pred=out['depth'].float(); conf=out['confidence'].float()
    gt=batch['depth'].float();mask=batch['valid'].bool()
    if not torch.isfinite(pred).all() or not torch.isfinite(conf).all():
        raise FloatingPointError('Non-finite depth/confidence output')
    if int(mask.sum())<100:
        raise ValueError('Fewer than 100 valid depth pixels; cannot supervise this batch')
    err=(pred-gt).abs()
    # Scalar depth version of official vector-norm regression and confidence term.
    reg=err[mask].mean()
    confidence=(err[mask]*conf[mask]-cfg['confidence_alpha']*conf[mask].clamp_min(1e-6).log()).mean()
    grad=gradient_loss(pred.clamp_min(1e-6).log(),gt.clamp_min(1e-6).log(),mask) if cfg.get('gradient_log',False) else gradient_loss(pred,gt,mask)
    weight=cfg['mv_weight']*min(1.,max(0.,(step-cfg['mv_warmup'])/max(cfg['mv_ramp'],1)))
    if weight:
        mv,n=multiview(pred[0],gt[0],batch['K'][0],batch['E'][0],mask[0],cfg['mv_stride'],cfg['mv_occlusion_relative'])
    else:
        mv=pred.sum()*0;n=0
    total=(reg+cfg['confidence_weight']*confidence+cfg['grad_weight']*grad+weight*mv)*batch['supervision_weight']
    if components is not None:
        components.update(reg=reg,confidence=cfg['confidence_weight']*confidence,
                          gradient=cfg['grad_weight']*grad,multiview=weight*mv)
    terms={'total':total,'reg':reg,'confidence':confidence,'gradient':grad,'multiview':mv}
    if not all(torch.isfinite(x).all() for x in terms.values()):
        raise FloatingPointError('Non-finite objective; abort rather than silently replace')
    return total,{**{k:float(v.detach()) for k,v in terms.items()},'mv_weight':weight,'mv_valid_pairs_pixels':n}
