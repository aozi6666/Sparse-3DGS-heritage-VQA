import torch
import torch.nn.functional as F


def pixel_grid(h,w,device,dtype=torch.float32):
    y,x=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(w,device=device,dtype=dtype),indexing='ij')
    return torch.stack((x,y,torch.ones_like(x)),-1)


def backproject(depth,K,E):
    """Z-depth -> world points; E is OpenCV camera-from-world, column vectors."""
    h,w=depth.shape[-2:]
    rays=pixel_grid(h,w,depth.device)@torch.linalg.inv(K).T
    cam=rays*depth[...,None]
    return (cam-E[:3,3])@E[:3,:3]


def normalize_scene(depth,K,E,mask):
    """Match VGGT first-camera gauge and mean valid world-point distance scaling."""
    first=E[0].clone()
    rel=E@torch.linalg.inv(first)
    points=torch.stack([backproject(d,k,e) for d,k,e in zip(depth,K,E)])
    first_points=points@first[:3,:3].T+first[:3,3]
    scale=first_points.norm(dim=-1)[mask].mean().clamp(1e-6,1e6)
    if not torch.isfinite(scale):
        raise ValueError('No valid geometric scale')
    rel[:,:3,3]/=scale
    return depth/scale,rel,scale


def warp(depth,K0,E0,K1,E1):
    h,w=depth.shape
    points=backproject(depth,K0,E0)
    target=points@E1[:3,:3].T+E1[:3,3]
    z=target[...,2]
    projected=target@K1.T
    xy=projected[...,:2]/z[...,None].clamp_min(1e-6)
    grid=xy*xy.new_tensor([2/max(w-1,1),2/max(h-1,1)])-1
    valid=(z>1e-6)&torch.isfinite(grid).all(-1)&(grid.abs()<=1).all(-1)
    return grid,z,valid


def sample(image,grid,nearest=False):
    return F.grid_sample(image[None,None].float(),grid[None].float(),mode='nearest' if nearest else 'bilinear',
                         padding_mode='zeros',align_corners=True)[0,0]


def multiview(pred,gt,K,E,mask,stride=4,tolerance=.03):
    """GT-visible fixed correspondences; predicted errors cannot turn supervision off.

    Only GT geometry/masks define visibility and correspondence. Compare predicted
    source Z transformed into the target camera with target predicted Z at that
    detached correspondence. Relative residual is normalized by target GT Z.
    """
    values=[];count=0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[0]):
            if i==j:continue
            with torch.no_grad():
                grid,z_gt,inside=warp(gt[i],K[i],E[i],K[j],E[j])
                target_gt=sample(gt[j],grid)
                visible=(mask[i]&inside&(sample(mask[j],grid)>.999)&
                         ((z_gt-target_gt).abs()<=tolerance*target_gt.clamp_min(1e-6)))
                visible=visible[::stride,::stride]
                reference=target_gt[::stride,::stride].clamp_min(1e-6)
            n=int(visible.sum())
            if not n:continue
            world=backproject(pred[i],K[i],E[i])
            transformed=world@E[j,:3,:3].T+E[j,:3,3]
            z=transformed[...,2][::stride,::stride]
            target=sample(pred[j],grid)[::stride,::stride]
            delta=(z[visible]-target[visible])/reference[visible]
            values.append(F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.05,reduction='sum'))
            count+=n
    return (torch.stack(values).sum()/count if count else pred.sum()*0),count


def co3d_camera(view,size_hw):
    import numpy as np
    h,w=size_hw
    fmt=view.get('intrinsics_format','ndc_norm_image_bounds')
    if fmt=='ndc_norm_image_bounds':
        scale=np.array([w,h])/2
    elif fmt=='ndc_isotropic':
        scale=np.array([min(w,h),min(w,h)])/2
    else:
        raise ValueError(f'Unsupported CO3D intrinsics: {fmt}')
    f=np.asarray(view['focal_length'])*scale
    c=np.array([w,h])/2-np.asarray(view['principal_point'])*scale
    K=np.array([[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1.]])
    flip=np.diag([-1.,-1.,1.])
    E=np.eye(4);E[:3,:3]=flip@np.asarray(view['R']).T;E[:3,3]=flip@np.asarray(view['T'])
    return K,E
