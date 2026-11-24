"""Zero-copy bridge for the user's already prepared VGGT_finetune_v1 package."""
import json
from pathlib import Path
import hashlib
import time
import uuid
import yaml
from .common import PROJECT,json_dump


def sources():
    cfg=yaml.safe_load((PROJECT/'configs/data_sources.yaml').read_text())
    return {k:(PROJECT/Path(v)).resolve() for k,v in cfg.items()}


def remap(row,bundle,omni):
    if row['dataset']=='omni3d':
        # Source JSON contains a Mac absolute path; resolve by object ID, not by that path.
        root=Path(omni)/row['id']
    else:
        recorded=Path(row['root'])
        root=(Path(bundle)/recorded).resolve()
        if Path(bundle).resolve() not in root.parents:
            raise ValueError(f'Unexpected path outside uploaded bundle: {recorded}')
    return root


def upload_status(omni,bundle,report=None):
    result={'captured_unix':time.time(),'sources':{}}
    for name,root in [('omni3d',Path(omni)),('VGGT_finetune_v1',Path(bundle))]:
        files=[]
        if root.is_dir():
            for p in root.rglob('*'):
                if p.is_file() and not p.name.startswith('.'):
                    try:files.append((p,p.stat()))
                    except FileNotFoundError:pass
        changed=sorted(files,key=lambda x:x[1].st_ctime,reverse=True)
        result['sources'][name]=dict(path=str(root),exists=root.is_dir(),files=len(files),
                   zero_bytes=sum(s.st_size==0 for p,s in files),bytes=sum(s.st_size for p,s in files),
                   latest_change_age_seconds=round(time.time()-changed[0][1].st_ctime) if changed else None,
                   recent_files=[dict(file=str(p.relative_to(root)),bytes=s.st_size) for p,s in changed[:3]],
                   note='File count/size snapshot only; not a source checksum or upload completion percentage.')
    if report:json_dump(report,result)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    return result


def read_catalog(bundle,omni):
    path=Path(bundle)/'manifests/all_objects.json'
    rows=json.loads(path.read_text())['objects'];seen=set();splits={}
    for row in rows:
        if row['id'] in seen:raise ValueError('Duplicate source ID in all_objects.json')
        seen.add(row['id'])
        if row['split'] not in ('train','val','test'):raise ValueError('Invalid source object split')
        splits[row['id']]=row['split']
    # Preserve the package's existing object-disjoint split, do not rehash it.
    declared={}
    for split in ('train','val','test'):
        file=Path(bundle)/f'manifests/{split}.json'
        if not file.is_file() or not file.stat().st_size:
            raise ValueError(f'Upload incomplete: missing split list {file}')
        data=json.loads(file.read_text());ids=data['ids']
        for identifier in ids:
            if identifier in declared:raise ValueError('Source package train/val/test overlap')
            declared[identifier]=split
    if splits!=declared:raise ValueError('all_objects.json and train/val/test lists disagree')
    return [{**row,'resolved_root':str(remap(row,bundle,omni))} for row in rows]


def checked_file(root,relative):
    if not relative:raise ValueError('Missing referenced file path')
    path=(Path(root)/relative).resolve()
    if Path(root).resolve() not in path.parents:raise ValueError('Referenced file outside source object')
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError(f'Upload incomplete: missing or zero-byte file: {path}')
    return str(path)


def import_bundle(bundle,omni,manifest,datasets=('smithsonian',),review_co3d=False):
    import numpy as np
    rows=read_catalog(bundle,omni);prepared=[];catalog=[]
    for row in rows:
        dataset=row['dataset'];root=Path(row['resolved_root'])
        role='pseudo_depth_not_gt' if dataset=='omni3d' else 'mesh_gt' if dataset=='smithsonian' else 'estimated_depth_or_rgb_only'
        entry=dict(id=row['id'],dataset=dataset,split=row['split'],path=str(root),role=role)
        catalog.append(entry)
        if dataset not in datasets:
            entry['training_index']='not_selected';continue
        if dataset not in ('smithsonian','co3dv2'):
            raise ValueError('omni3d ZoeDepth is not GT and is not enabled as supervised data in this importer')
        camera=checked_file(root,'cameras.json');meta=json.loads(Path(camera).read_text())
        if 'OpenCV c2w' not in meta.get('convention',''):
            raise ValueError(f'Unrecognized source camera convention: {camera}')
        views=meta['views']
        if len(views)!=row['n_views']:
            raise ValueError('Incomplete/mismatched source camera view count')
        has_depth=all(v.get('depth') for v in views)
        if not has_depth:
            entry['training_index']='rgb_only_no_depth';continue
        if dataset=='co3dv2' and not review_co3d:
            entry['training_index']='estimated_depth_not_reviewed';continue
        frames=[]
        for view in views:
            fx=view.get('fx',meta.get('fx'));fy=view.get('fy',meta.get('fy'))
            cx=view.get('cx',meta.get('cx'));cy=view.get('cy',meta.get('cy'))
            K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],dtype=float)
            c2w=np.asarray(view['c2w'],dtype=float)
            if c2w.shape!=(4,4) or not np.isfinite(c2w).all() or not np.isfinite(K).all():
                raise ValueError('Invalid camera array')
            depth_type='camera_z_depth'
            if dataset=='co3dv2':
                note=meta.get('depth_note','').lower()
                if 'ray distance' in note:depth_type='ray_distance'
                elif 'z-depth' not in note and 'camera_z_depth' not in note:
                    raise ValueError('CO3D depth semantics not declared; refusing to guess')
            frames.append(dict(image=checked_file(root,view['image']),depth=checked_file(root,view['depth']),
                    mask=checked_file(root,view['mask']),K=K.tolist(),E=np.linalg.inv(c2w).tolist(),
                    depth_encoding='co3d_float16_png' if dataset=='co3dv2' else 'float32_npy',
                    depth_type=depth_type,depth_scale=float(view.get('depth_scale_adjustment',1.)) if dataset=='co3dv2' else 1.,
                    depth_resize_to_rgb=dataset=='co3dv2'))
        normalized=dict(schema_version=1,id=row['id'],group_id=row['id'],dataset=dataset,split=row['split'],
                        depth_type='camera_z_depth',coordinate_convention='opencv_camera_from_world',
                        depth_reviewed=dataset=='smithsonian' or review_co3d,
                        supervision_source=role,frames=frames,
                        provenance=dict(source=str(root),source_split=row['split'],original_eval_views=row.get('eval_views'),
                                        scene_scale_m=meta.get('scene_scale_m'),no_media_copy=True))
        prepared.append((row,normalized));entry['training_index']='included'
    if not prepared:
        raise ValueError('No eligible depth-supervised objects; finish uploads or review depth sources first')
    # Check all references before publishing any new training manifest.
    index=PROJECT/'data/processed'/('bundle_index_'+uuid.uuid4().hex[:10]);objects=[]
    for row,normalized in prepared:
        token=hashlib.sha256((row['dataset']+':'+row['id']).encode()).hexdigest()[:16]
        path=index/row['dataset']/token
        json_dump(path/'cameras.json',normalized)
        objects.append(dict(id=row['id'],group_id=row['id'],dataset=row['dataset'],split=row['split'],path=str(path),
                            frames=len(normalized['frames']),depth_reviewed=normalized['depth_reviewed'],
                            supervision_source=normalized['supervision_source']))
    json_dump(manifest,dict(schema_version=1,source='VGGT_finetune_v1 zero-copy bridge',objects=objects))
    json_dump(PROJECT/'data/manifests/source_catalog.json',dict(objects=catalog,source_bundle=str(bundle),source_omni3d=str(omni)))
    result=dict(index=str(index),manifest=str(manifest),eligible_objects=len(objects),
                split_counts={s:sum(o['split']==s for o in objects) for s in ('train','val','test')},
                copied_media_bytes=0,note='Original files and source manifests unchanged. CO3D geometric depth is estimated, not sensor GT.')
    json_dump(PROJECT/'reports/bundle_import.json',result);print(json.dumps(result,indent=2),flush=True)
    return result
