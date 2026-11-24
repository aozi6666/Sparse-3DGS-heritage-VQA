"""Opt-in full RGB scene support. Object-mode defaults are unchanged."""
from pathlib import Path
import numpy as np
from PIL import Image


def resolve_image_file(folder, recorded_name):
    """Resolve an actual image by exact name, then unique case-insensitive stem."""
    folder = Path(folder)
    exact = folder / Path(recorded_name).name
    if exact.is_file():
        return str(exact)
    stem = Path(recorded_name).stem.casefold()
    matches = [p for p in folder.iterdir() if p.is_file()
               and p.stem.casefold() == stem
               and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {recorded_name} in {folder}; found {matches}")
    return str(matches[0])


def read_depth(path):
    """Float NPY avoids uint16 depth overflow; historical millimetre PNG still works."""
    path = Path(path)
    npy = path.with_suffix(".npy")
    if npy.is_file():
        depth = np.load(npy, allow_pickle=False).astype(np.float32)
    elif path.is_file():
        depth = np.asarray(Image.open(path), dtype=np.float32) / 1000.0
    else:
        raise FileNotFoundError(f"Missing depth: {npy} or {path}")
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError(f"Invalid depth array: {path}")
    return depth


def full_scene_options(args):
    """Apply only after CLI/cfg merge, before extracting model/optimizer parameters."""
    if getattr(args, "full_scene", False):
        args.use_mask = False
        args.white_background = False
        args.random_background = False
        args.lambda_silhouette = 0.0
    return args


def effective_mask(image_size, object_mask_path, full_scene=False):
    """Full-scene mask means all pixels are valid, NOT object segmentation."""
    if full_scene:
        return np.full((image_size[1], image_size[0]), 255, dtype=np.uint8)
    with Image.open(object_mask_path) as im:
        return np.asarray(im.convert("L").resize(image_size, Image.Resampling.NEAREST))


def fit_full_image(image, size, signed=False):
    """Letterbox the complete image; source RGB uses 0 padding, target [-1,1] -1."""
    height,width,channels=image.shape
    scale=min(size/width,size/height)
    nh,nw=max(1,round(height*scale)),max(1,round(width*scale))
    y,x=(size-nh)//2,(size-nw)//2
    fill=-1.0 if signed else 0.0
    if isinstance(image,np.ndarray):
        import cv2
        resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
        result=np.full((size,size,channels),fill,dtype=image.dtype)
    else:
        import torch
        import torch.nn.functional as F
        resized=F.interpolate(image.permute(2,0,1)[None],size=(nh,nw),
                              mode="bilinear",align_corners=False)[0].permute(1,2,0)
        result=torch.full((size,size,channels),fill,dtype=image.dtype,device=image.device)
    result[y:y+nh,x:x+nw]=resized
    return result
