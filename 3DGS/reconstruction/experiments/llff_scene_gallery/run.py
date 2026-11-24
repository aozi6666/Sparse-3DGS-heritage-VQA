#!/usr/bin/env python3
"""Isolated full-background stage galleries; no source data writes or GPU in check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GO = Path(__file__).resolve().parents[2]
ROOT = GO.parents[1]
GIB = 1024**3


def guard_disk(minimum=4):
    if shutil.disk_usage(GO).free < minimum * GIB:
        raise RuntimeError(f'Need at least {minimum} GiB free; existing files preserved')


def fingerprint(source):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source/'sparse/0').glob('*.bin'))}


def load_source(source):
    import pycolmap
    from PIL import Image
    rec = pycolmap.Reconstruction(str(source/'sparse/0'))
    if len(rec.images) < 9 or not len(rec.points3D):
        raise ValueError(f'Missing usable calibrated data: {source}')
    stems = [Path(im.name).stem.casefold() for im in rec.images.values()]
    if len(stems) != len(set(stems)):
        raise ValueError('Image stems must be unique')
    for cam in rec.cameras.values():
        if cam.model_name not in ('SIMPLE_RADIAL','RADIAL','OPENCV','PINHOLE','SIMPLE_PINHOLE'):
            raise ValueError(f'Unsupported camera: {cam.model_name}')
    for im in rec.images.values():
        p = source/'images'/im.name
        if Path(im.name).is_absolute() or '..' in Path(im.name).parts:
            raise ValueError('Unsafe image name')
        with Image.open(p) as rgb:
            cam = rec.cameras[im.camera_id]
            if rgb.size != (cam.width,cam.height):
                raise ValueError(f'Image/calibration size mismatch: {p}')
    return rec


def calibration(cam, factor):
    import cv2
    import numpy as np
    p = cam.params
    if cam.model_name == 'SIMPLE_RADIAL':
        f,cx,cy,k=p; fx=fy=f; dist=np.array([k,0,0,0.])
    elif cam.model_name == 'RADIAL':
        f,cx,cy,k,k2=p; fx=fy=f; dist=np.array([k,k2,0,0.])
    elif cam.model_name == 'OPENCV':
        fx,fy,cx,cy=p[:4]; dist=np.asarray(p[4:])
    elif cam.model_name == 'PINHOLE':
        fx,fy,cx,cy=p; dist=np.zeros(4)
    else:
        f,cx,cy=p; fx=fy=f; dist=np.zeros(4)
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    size=(cam.width,cam.height)
    newK,_=cv2.getOptimalNewCameraMatrix(K,dist,size,0,size,centerPrincipalPoint=True)
    small=(round(cam.width/factor),round(cam.height/factor))
    newK[0,:] *= small[0]/cam.width
    newK[1,:] *= small[1]/cam.height
    maps=cv2.initUndistortRectifyMap(K,dist,None,newK,small,cv2.CV_32FC1)
    return K,dist,newK,small,maps


def rectify_one(path, maps):
    import cv2
    rgb=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f'Cannot decode {path}')
    return cv2.remap(rgb,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)


def prepare(source, dest, factor):
    import cv2
    import numpy as np
    import pycolmap
    expected={'source':str(source),'factor':factor,'colmap_sha256':fingerprint(source),
              'images':{im.name:(source/'images'/im.name).stat().st_size
                        for im in load_source(source).images.values()},'version':1}
    ready=dest/'prepared.json'
    if ready.is_file():
        if json.loads(ready.read_text()) != expected:
            raise RuntimeError(f'Prepared data changed: {dest}; use a new RUN_TAG')
        load_source(dest)
        return
    if dest.exists():
        raise RuntimeError(f'Partial preparation preserved: {dest}; use a new RUN_TAG')
    guard_disk()
    dest.mkdir(parents=True)
    (dest/'images').mkdir()
    rec=load_source(source)
    for cid,cam in rec.cameras.items():
        K,dist,newK,size,maps=calibration(cam,factor)
        for im in rec.images.values():
            if im.camera_id != cid:
                continue
            guard_disk(2)
            rgb=rectify_one(source/'images'/im.name,maps)
            path=dest/'images'/im.name
            path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),rgb,[cv2.IMWRITE_JPEG_QUALITY,95]):
                raise IOError(path)
            if len(im.points2D):
                xy=np.array([p.xy for p in im.points2D],dtype=np.float64)
                fixed=cv2.undistortPoints(xy.reshape(-1,1,2),K,dist,P=newK).reshape(-1,2)
                for p,newxy in zip(im.points2D,fixed):
                    p.xy=newxy
        cam.model=pycolmap.CameraModelId.PINHOLE
        cam.params=[newK[0,0],newK[1,1],newK[0,2],newK[1,2]]
        cam.width,cam.height=size
    (dest/'sparse/0').mkdir(parents=True)
    rec.write(str(dest/'sparse/0'))
    load_source(dest)
    ready.write_text(json.dumps(expected,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenes',nargs='+',choices=['horns','fortress'],default=['horns','fortress'])
    p.add_argument('--out',type=Path,default=GO/'output/llff_horns_fortress_gallery_v1')
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--factor',type=int,choices=[4,8],default=4)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.steps<2000:
        p.error('--steps must be >=2000')
    a.out=a.out.resolve()
    for scene in a.scenes:
        source=(GO/'data/LLFF'/scene).resolve()
        if a.out == source or source in a.out.parents:
            raise ValueError('Output must be outside source data')
    import cv2
    import torch
    cv2.setNumThreads(2)
    print(f'Free: {shutil.disk_usage(GO).free/GIB:.2f} GiB; CUDA: {torch.cuda.is_available()}',flush=True)
    guard_disk()
    weights=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    if not weights.is_file():
        raise FileNotFoundError(f'Offline VGG LPIPS weights missing: {weights}')
    # Verify imports/extensions on CPU; do not create any CUDA tensors here.
    sys.path.insert(0,str(GO))
    import train_gs
    import lpips
    for scene in a.scenes:
        source=(GO/'data/LLFF'/scene).resolve()
        rec=load_source(source)
        chosen=sorted(rec.images.values(),key=lambda im:im.name)[len(rec.images)//2]
        cam=rec.cameras[chosen.camera_id]
        _,_,_,size,maps=calibration(cam,a.factor)
        preview=rectify_one(source/'images'/chosen.name,maps)
        if preview.shape[:2] != size[::-1]:
            raise ValueError('Rectification dimension mismatch')
        print(f'{scene}: {len(rec.images)} images, {len(rec.points3D)} points; '
              f'{cam.model_name} -> PINHOLE {size}; fixed camera={chosen.name}',flush=True)
    if a.check:
        print('CPU check passed, including one in-memory rectification per scene. No dataset writes or training.',flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for run; CPU --check is available')
    a.out.mkdir(parents=True,exist_ok=True)
    # New subprocess per scene releases all CUDA/host memory before the next.
    for scene in a.scenes:
        leaf=a.out/scene
        if (leaf/'complete.json').is_file():
            report=json.loads((leaf/'complete.json').read_text())
            if report['parameters']['iterations'] != a.steps:
                raise RuntimeError('Completed output has different steps; use new RUN_TAG')
            print(f'[skip completed] {scene}',flush=True)
            continue
        if leaf.exists():
            raise RuntimeError(f'Partial training preserved: {leaf}; use a new RUN_TAG')
        guard_disk()
        source=(GO/'data/LLFF'/scene).resolve()
        data=a.out/'prepared'/scene
        prepare(source,data,a.factor)
        rec=load_source(data)
        camera=Path(sorted(rec.images.values(),key=lambda im:im.name)[len(rec.images)//2].name).stem
        command=[sys.executable,'-u',str(Path(__file__).with_name('worker.py')),
                 '--root',str(ROOT),'--out',str(a.out),'--scene',scene,
                 '--data',str(data),'--camera',camera,'--steps',str(a.steps)]
        with (a.out/f'{scene}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=GO)
        if result.returncode:
            raise RuntimeError(f'{scene} failed; see {a.out/scene}.log; partial data preserved')
        report=json.loads((leaf/'complete.json').read_text())
        print(f'{scene} COMPLETE: {report["count"]} renders + GT; '
              f'PSNR span={report["psnr_span"]:.3f} dB; target_met={report["target_met"]}',flush=True)
    print(f'All requested scenes complete: {a.out}',flush=True)


if __name__=='__main__':
    main()
