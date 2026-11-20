"""Offline-only adapters. Unknown depth semantics are rejected, never guessed."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
from .common import PROJECT,json_dump,space_guard
from .geometry import co3d_camera


def read_float(path,packed=False):
    path=Path(path)
    if path.suffix=='.npy':
        a=np.load(path,allow_pickle=False)
    elif packed:
        # CO3D official uint16 PNG stores float16 bit patterns, not integer millimetres.
        a=np.asarray(Image.open(path)).astype(np.uint16).view(np.float16).astype(np.float32)
    else:
        os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
        a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f'Cannot load floating-point depth {path}')
        if a.dtype not in (np.float16,np.float32,np.float64):
            raise ValueError(f'Refusing non-floating depth without explicit dataset decoder: {path}')
    if a.ndim==3:
        if not np.allclose(a[...,0],a[...,1],equal_nan=True):
            raise ValueError(f'Ambiguous multichannel depth: {path}')
        a=a[...,0]
    return np.asarray(a,dtype=np.float32)


def find_file(root,name,folder=None,extensions=('.png','.jpg','.jpeg','.exr','.npy')):
    root=Path(root);p=root/name
    if p.is_file():
        return p
    bases=[p.parent,(root/folder) if folder else root,root/'images']
    matches=[]
    for base in bases:
        for ext in extensions:
            matches.extend(base.glob(Path(name).stem+ext))
    matches=sorted(set(matches))
    if len(matches)!=1:
        raise FileNotFoundError(f'Expected unique {name} under {base}; found {matches}')
    return matches[0]


def emit_object(dest,identifier,dataset,frames,depth_reviewed,max_side,provenance):
    if dest.exists():
        if (dest/'cameras.json').is_file():
            print(f'[skip prepared] {dest}',flush=True)
            return
        raise RuntimeError(f'Partial conversion preserved: {dest}; choose another processed root')
    dest.mkdir(parents=True)
    for folder in ('images','depths','masks'):
        (dest/folder).mkdir()
    output=[]
    for i,record in enumerate(frames):
        space_guard(3)
        image,dep,mask,K,E=record()
        image=image.convert('RGB');w,h=image.size
        if dep.shape!=(h,w):
            dep=cv2.resize(dep,(w,h),interpolation=cv2.INTER_NEAREST)
        if mask.shape!=(h,w):
            mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        valid=mask&np.isfinite(dep)&(dep>1e-6)&(dep<1e8)
        if valid.sum()<100:
            raise ValueError(f'Insufficient valid depth: {identifier}/{i}')
        ratio=min(1.,max_side/max(w,h));nw,nh=round(w*ratio),round(h*ratio)
        image=image.resize((nw,nh),Image.Resampling.LANCZOS)
        dep=cv2.resize(np.where(valid,dep,0),(nw,nh),interpolation=cv2.INTER_NEAREST)
        valid=cv2.resize(valid.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)>0
        K=K.copy();K[0,:]*=nw/w;K[1,:]*=nh/h
        stem=f'{i:05d}'
        image.save(dest/'images'/f'{stem}.jpg',quality=95)
        np.save(dest/'depths'/f'{stem}.npy',np.where(valid,dep,0).astype(np.float32))
        Image.fromarray(valid.astype(np.uint8)*255).save(dest/'masks'/f'{stem}.png')
        output.append(dict(image=f'images/{stem}.jpg',depth=f'depths/{stem}.npy',mask=f'masks/{stem}.png',
                           K=K.tolist(),E=E.tolist()))
    if len(output)<6:
        raise ValueError('Need at least six valid views per object')
    json_dump(dest/'cameras.json',dict(schema_version=1,id=identifier,group_id=identifier,dataset=dataset,
              depth_type='camera_z_depth',coordinate_convention='opencv_camera_from_world',
              depth_reviewed=depth_reviewed,provenance=provenance,frames=output))


def omni(raw,out,depth_type,max_objects,max_views,max_side):
    transforms=sorted(raw.rglob('transforms.json'))[:max_objects]
    if not transforms:
        raise FileNotFoundError('No transforms.json. Copy blender_renders (RGB-D) or render raw_scans first.')
    for path in transforms:
        root=path.parent;meta=json.loads(path.read_text())
        semantics=meta.get('depth_type',depth_type)
        if semantics not in ('camera_z_depth','ray_distance'):
            raise ValueError('Omni depth meaning is unknown. Confirm metadata and pass --depth-type camera_z_depth or ray_distance; no guessing.')
        # Include relative category/object directory to prevent category-local ID collisions.
        rel=str(root.relative_to(raw)).replace('/','__')
        ident=rel if rel!='.' else raw.name
        rows=meta['frames'];chosen=np.linspace(0,len(rows)-1,min(max_views,len(rows))).round().astype(int)
        loaders=[]
        for idx in chosen:
            frame=rows[idx]
            def load(frame=frame):
                rgbpath=find_file(root,frame['file_path'])
                im=Image.open(rgbpath).convert('RGBA');w,h=im.size
                fx=frame.get('fl_x',meta.get('fl_x'))
                fy=frame.get('fl_y',meta.get('fl_y',fx))
                if fx is None:
                    fx=.5*w/np.tan(float(meta['camera_angle_x'])/2);fy=fx
                K=np.array([[fx,0,frame.get('cx',meta.get('cx',w/2))],
                            [0,fy,frame.get('cy',meta.get('cy',h/2))],[0,0,1.]],dtype=float)
                # Standard NeRF/Blender c2w: x right, y up, z backward -> OpenCV.
                c2w=np.asarray(frame['transform_matrix'],dtype=float)
                E=np.diag([1.,-1.,-1.,1.])@np.linalg.inv(c2w)
                if 'depth_file_path' in frame:
                    dp=root/frame['depth_file_path']
                else:
                    candidates=[p for p in (root/'depths').glob(rgbpath.stem+'*')
                                if p.suffix.lower() in ('.npy','.exr') and not p.name.startswith('._')]
                    if len(candidates)!=1:
                        raise ValueError(f'Ambiguous/missing depth for {rgbpath}: {candidates}')
                    dp=candidates[0]
                depth=read_float(dp)
                if depth.shape!=(h,w):
                    raise ValueError('Omni RGB/depth must be registered at same resolution')
                if semantics=='ray_distance':
                    yy,xx=np.mgrid[:h,:w]
                    rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
                    depth=depth/np.linalg.norm(rays,axis=-1)
                alpha=np.asarray(im)[...,3]>127
                if frame.get('mask_file_path'):
                    alpha=alpha&(np.asarray(Image.open(root/frame['mask_file_path']).convert('L'))>127)
                # RGB alpha may be opaque; depth validity still excludes Blender infinity background.
                return im,depth,alpha,K,E
            loaders.append(load)
        emit_object(out/'omniobject3d'/ident,ident,'omniobject3d',loaders,True,max_side,
                    dict(source=str(path),native_depth_type=semantics,units='source_scene_units'))


def co3d(raw,out,max_objects,max_views,max_side,reviewed=False,sequence_splits=None):
    annotations=sorted(raw.rglob('frame_annotations.jgz'))
    if not annotations:
        raise FileNotFoundError('No CO3Dv2 frame_annotations.jgz; copy local annotations too')
    # Official sequence-disjoint assignment must be provided. No random mixing of official test sequences.
    if not sequence_splits:
        raise ValueError('CO3D needs --sequence-splits JSON: {category/sequence: train|val|test}, derived from official split. See DATA_FORMAT.md.')
    splits=json.loads(Path(sequence_splits).read_text())
    emitted=0
    for ann in annotations:
        category=ann.parent.name
        with gzip.open(ann,'rt') as f:
            rows=json.load(f)
        groups={}
        for row in rows:
            if row.get('depth') and row.get('mask') and row.get('viewpoint'):
                groups.setdefault(row['sequence_name'],[]).append(row)
        for name,rows in sorted(groups.items()):
            ident=category+'/'+name
            if ident not in splits:
                continue
            if splits[ident] not in ('train','val','test'):
                raise ValueError(f'Bad sequence split: {ident}')
            if emitted>=max_objects:
                return
            emitted+=1
            rows=sorted(rows,key=lambda x:x['frame_number'])
            chosen=np.linspace(0,len(rows)-1,min(max_views,len(rows))).round().astype(int)
            loaders=[]
            for idx in chosen:
                row=rows[idx]
                def load(row=row):
                    # Official paths relative to data root; support a copied category root as well.
                    def locate(rel):
                        for candidate in (raw/rel,ann.parent.parent/rel,ann.parent/Path(rel).relative_to(category) if rel.startswith(category+'/') else ann.parent/rel):
                            if candidate.is_file():
                                return candidate
                        raise FileNotFoundError(raw/rel)
                    im=Image.open(locate(row['image']['path'])).convert('RGB')
                    if im.size!=tuple(reversed(row['image']['size'])):
                        raise ValueError('CO3D image/annotation mismatch')
                    dp=row['depth'];depth=read_float(locate(dp['path']),True)*float(dp['scale_adjustment'])
                    mask=np.asarray(Image.open(locate(row['mask']['path'])).convert('L'))>127
                    if dp.get('mask_path'):
                        dm=np.asarray(Image.open(locate(dp['mask_path'])).convert('L'))>0
                        dm=cv2.resize(dm.astype(np.uint8),im.size,interpolation=cv2.INTER_NEAREST)>0
                        mask=mask&dm
                    K,E=co3d_camera(row['viewpoint'],row['image']['size'])
                    return im,depth,mask,K,E
                loaders.append(load)
            dest=out/'co3dv2'/ident.replace('/','__')
            emit_object(dest,ident,'co3dv2',loaders,reviewed,max_side,
                        dict(source=str(ann),units='co3d_reconstruction_units',official_split=splits[ident]))


def build_manifest(processed,out,seed=42):
    entries=[];seen=set()
    for path in sorted(processed.rglob('cameras.json')):
        m=json.loads(path.read_text())
        key=(m['dataset'],m.get('group_id',m['id']))
        if key in seen:
            raise ValueError(f'Duplicate object group {key}; group multiple scans explicitly before indexing')
        seen.add(key)
        # Stable object-level split independent of directory order. CO3D uses explicit official assignment.
        score=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:12],16)/16**12
        a,b=(.7,.85) if m['dataset']=='smithsonian' else (.8,.9)
        split=(m.get('split') or m.get('provenance',{}).get('source_split') or
               m.get('provenance',{}).get('official_split') or ('train' if score<a else 'val' if score<b else 'test'))
        entries.append(dict(id=m['id'],group_id=key[1],dataset=m['dataset'],split=split,path=str(path.parent.resolve()),
                            frames=len(m['frames']),depth_reviewed=m.get('depth_reviewed',False)))
    if not entries:
        raise ValueError('No processed cameras.json; prepare local raw data first')
    json_dump(out,dict(schema_version=1,seed=seed,objects=entries))
    print({s:sum(e['split']==s for e in entries) for s in ('train','val','test')},flush=True)


def validate(manifest,report,size=336,limit=100):
    import random
    from .data import read_object
    from .geometry import multiview
    import torch
    all_objects=json.loads(Path(manifest).read_text())['objects'];groups={};results=[]
    for o in all_objects:
        key=(o['dataset'],o['group_id'])
        if key in groups and groups[key]!=o['split']:
            raise ValueError('Object/sequence split leakage')
        groups[key]=o['split']
    for idx,o in enumerate(all_objects[:limit]):
        b=read_object(o,min(4,o['frames']),size,random.Random(42))
        m=json.loads((Path(o['path'])/'cameras.json').read_text())
        if m['depth_type']!='camera_z_depth' or m['coordinate_convention']!='opencv_camera_from_world':
            raise ValueError(f'Unknown geometry convention {o["id"]}')
        for e,k in zip(b['E'][0],b['K'][0]):
            R=e[:3,:3]
            if not torch.allclose(R@R.T,torch.eye(3),atol=1e-3) or abs(float(torch.det(R))-1)>1e-3:
                raise ValueError('Invalid rotation')
            if (k.diag()<=0).any():
                raise ValueError('Invalid intrinsics')
        _,pixels=multiview(b['depth'][0],b['depth'][0],b['K'][0],b['E'][0],b['valid'][0],4,.05)
        results.append(dict(id=o['id'],split=o['split'],valid_pixels=int(b['valid'].sum()),
                            gt_consistent_projection_pixels=pixels,scale=b['scale']))
        # RGB, normalized GT depth and valid mask contact sheet for first 20 objects.
        if idx<20:
            report.mkdir(parents=True,exist_ok=True)
            sheet=Image.new('RGB',(size*3,size*b['images'].shape[1]))
            for v in range(b['images'].shape[1]):
                rgb=(b['images'][0,v].permute(1,2,0).numpy()*255).astype(np.uint8)
                d=b['depth'][0,v].numpy();mask=b['valid'][0,v].numpy()
                span=np.percentile(d[mask],[2,98]);color=cv2.applyColorMap((np.clip((d-span[0])/max(span[1]-span[0],1e-6),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1].copy();color[~mask]=0
                for col,im in enumerate((rgb,color,np.repeat((mask*255).astype(np.uint8)[...,None],3,-1))):
                    sheet.paste(Image.fromarray(im),(col*size,v*size))
            sheet.save(report/f'{idx:03d}.jpg',quality=90)
    manifest_hash=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    json_dump(report/'quality.json',dict(objects_checked=len(results),results=results,manifest_sha256=manifest_hash,passed=True,
                note='GT reprojection support is a quality diagnostic; inspect previews before training.'))
    print(f'Validated {len(results)} objects; previews and quality.json: {report}',flush=True)
