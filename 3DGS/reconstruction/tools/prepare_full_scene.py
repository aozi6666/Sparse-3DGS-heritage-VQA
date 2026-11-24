#!/usr/bin/env python3
"""CPU-only COLMAP undistortion and immutable full-scene data preparation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def serial_undistort(reconstruction, source, stage):
    """Bounded-memory rectification, updating pycolmap cameras and observations.

    COLMAP's automatic thread count can allocate dozens of full-resolution maps
    on a CPU-only 2 GiB instance. Process one calibration and image at a time.
    """
    import cv2
    import pycolmap
    cv2.setNumThreads(2)
    (stage/"images").mkdir()
    for camera_id,camera in reconstruction.cameras.items():
        name=camera.model_name
        p=np.asarray(camera.params).copy()
        if name=="SIMPLE_RADIAL":
            f,cx,cy,k1=p;fx=fy=f;dist=np.array([k1,0,0,0.])
        elif name=="RADIAL":
            f,cx,cy,k1,k2=p;fx=fy=f;dist=np.array([k1,k2,0,0.])
        elif name=="OPENCV":
            fx,fy,cx,cy=p[:4];dist=p[4:]
        elif name=="PINHOLE":
            fx,fy,cx,cy=p;dist=np.zeros(4)
        elif name=="SIMPLE_PINHOLE":
            f,cx,cy=p;fx=fy=f;dist=np.zeros(4)
        else:
            raise ValueError(f"Serial rectification does not support {name}; provide an undistorted PINHOLE dataset")
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        size=(camera.width,camera.height)
        newK,_=cv2.getOptimalNewCameraMatrix(K,dist,size,0,size,centerPrincipalPoint=True)
        mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,newK,size,cv2.CV_32FC1)
        for image in reconstruction.images.values():
            if image.camera_id!=camera_id:continue
            src=source/"images"/image.name;dst=stage/"images"/image.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            rgb=cv2.imread(str(src),cv2.IMREAD_COLOR)
            if rgb is None or (rgb.shape[1],rgb.shape[0])!=size:
                raise ValueError(f"Image/calibration mismatch: {src}")
            warped=cv2.remap(rgb,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            if not cv2.imwrite(str(dst),warped,[cv2.IMWRITE_JPEG_QUALITY,95]):
                raise IOError(dst)
            xy=np.array([point.xy for point in image.points2D],dtype=np.float64)
            if len(xy):
                fixed=cv2.undistortPoints(xy.reshape(-1,1,2),K,dist,P=newK).reshape(-1,2)
                for point,newxy in zip(image.points2D,fixed):point.xy=newxy
            print(f"[undistort] {image.name}",flush=True)
        camera.model=pycolmap.CameraModelId.PINHOLE
        camera.params=[newK[0,0],newK[1,1],newK[0,2],newK[1,2]]
        del mapx,mapy
    (stage/"sparse/0").mkdir(parents=True)
    reconstruction.write(str(stage/"sparse/0"))


def prepare(source, output):
    import pycolmap
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or source in output.parents:
        raise ValueError("Prepared data must be outside the original source directory")
    reconstruction=pycolmap.Reconstruction(str(source/"sparse/0"))
    names=sorted(im.name for im in reconstruction.images.values())
    if len(names)<2 or len({Path(n).stem.casefold() for n in names})!=len(names):
        raise ValueError("Need at least two uniquely named registered images")
    inventory={n:digest(source/"images"/n) for n in names}
    camera_files={p.name:digest(p) for p in (source/"sparse/0").glob("*.bin")}
    fingerprint={"images":inventory,"colmap":camera_files,"mode":"full_scene","version":2,
                 "rectification":"serial_opencv_with_pycolmap_camera_and_observation_update"}
    if output.exists():
        manifest=output/"full_scene_manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text())["source_fingerprint"]!=fingerprint:
            raise RuntimeError(f"Refusing to reuse unrecognized or changed output: {output}")
        validate(output)
        print(f"[reuse] verified prepared data: {output}")
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+".preparing-",dir=output.parent))
    try:
        serial_undistort(reconstruction,source,stage)
        sparse=stage/"sparse"
        if (sparse/"cameras.bin").exists():
            (sparse/"0").mkdir(exist_ok=True)
            for p in list(sparse.glob("*.bin")):
                p.rename(sparse/"0"/p.name)
        rec=pycolmap.Reconstruction(str(sparse/"0"))
        if len(rec.images)!=len(names):
            raise RuntimeError("Undistortion lost registered images")
        for sub in ("masks","images_2","images_4","images_8"):
            (stage/sub).mkdir(exist_ok=True)
        for name in names:
            p=stage/"images"/name
            with Image.open(p) as im:
                rgb=im.convert("RGB")
                for factor in (2,4,8):
                    resized=rgb.resize((round(rgb.width/factor),round(rgb.height/factor)),Image.Resampling.LANCZOS)
                    resized.save(stage/f"images_{factor}"/name,quality=95)
                Image.new("L",rgb.size,255).save(stage/"masks"/(Path(name).stem+".png"))
        (stage/f"sparse_{len(names)}.txt").write_text("".join(f"{i}\n" for i in range(len(names))))
        (stage/"sparse_test.txt").write_text("")
        centres=np.stack([np.asarray(im.cam_from_world().inverse().translation) for im in rec.images.values()])
        extent=float(np.linalg.norm(centres-centres.mean(axis=0),axis=1).max()*1.1)
        for licence in source.glob("license*"):
            if licence.is_file(): shutil.copyfile(licence,stage/licence.name)
        manifest={
            "version":1,"source":str(source),"source_fingerprint":fingerprint,
            "mode":"full_scene","mask_semantics":"all_pixels_valid_not_object_segmentation",
            "train_count":len(names),"test_count":0,"names":names,"scene_extent":extent,
            "display_camera":Path(names[len(names)//2]).stem,
            "evaluation_scope":"all_images_used_for_training_demonstration_only",
        }
        (stage/"full_scene_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        validate(stage)
        os.replace(stage,output)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    print(f"[ready] {len(names)} undistorted full-RGB images: {output}")


def validate(root):
    import pycolmap
    root=Path(root)
    rec=pycolmap.Reconstruction(str(root/"sparse/0"))
    for camera in rec.cameras.values():
        if camera.model_name not in ("PINHOLE","SIMPLE_PINHOLE"):
            raise ValueError(f"Unsupported prepared camera {camera.model_name}")
        if camera.model_name=="PINHOLE":
            cx,cy=camera.params[2:4]
        else:
            cx,cy=camera.params[1:3]
        if abs(cx-camera.width/2)>1.0 or abs(cy-camera.height/2)>1.0:
            raise ValueError("Prepared camera principal point is not centered; requires off-axis renderer support")
    for im in rec.images.values():
        with Image.open(root/"images"/im.name) as image:
            camera=rec.cameras[im.camera_id]
            if image.size!=(camera.width,camera.height): raise ValueError("Image/calibration size mismatch")
            for factor in (2,4,8):
                with Image.open(root/f"images_{factor}"/im.name) as small:
                    assert small.size==(round(image.width/factor),round(image.height/factor))
            with Image.open(root/"masks"/(Path(im.name).stem+".png")) as mask:
                assert mask.size==image.size and mask.getextrema()==(255,255)
    print(f"[validate] {len(rec.images)} matching images/cameras and full-validity masks")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    prepare(a.source,a.output)
