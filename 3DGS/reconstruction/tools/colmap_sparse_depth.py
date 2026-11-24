#!/usr/bin/env python3
"""COLMAP points3D → per-image sparse depth maps (shared by stats + finetune)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_GO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GO_DIR not in sys.path:
    sys.path.insert(0, _GO_DIR)

# Load colmap_loader by path to avoid importing heavy scene/__init__.py
import importlib.util

_cl_path = os.path.join(_GO_DIR, "scene", "colmap_loader.py")
_cl_spec = importlib.util.spec_from_file_location("go_colmap_loader", _cl_path)
_cl = importlib.util.module_from_spec(_cl_spec)
assert _cl_spec.loader is not None
_cl_spec.loader.exec_module(_cl)
Camera = _cl.Camera
Image = _cl.Image
qvec2rotmat = _cl.qvec2rotmat
read_extrinsics_binary = _cl.read_extrinsics_binary
read_extrinsics_text = _cl.read_extrinsics_text
read_intrinsics_binary = _cl.read_intrinsics_binary
read_intrinsics_text = _cl.read_intrinsics_text
read_next_bytes = _cl.read_next_bytes


def read_points3D_binary_dict(path: str) -> Dict[int, np.ndarray]:
    """Return {point3D_id: xyz} from COLMAP points3D.bin."""
    points: Dict[int, np.ndarray] = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point_id = int(props[0])
            xyz = np.array(props[1:4], dtype=np.float64)
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            if track_length:
                read_next_bytes(
                    fid,
                    num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length,
                )
            points[point_id] = xyz
    return points


def read_points3D_text_dict(path: str) -> Dict[int, np.ndarray]:
    points: Dict[int, np.ndarray] = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = np.array(tuple(map(float, elems[1:4])), dtype=np.float64)
            points[point_id] = xyz
    return points


def load_points3D_dict(sparse_dir: str) -> Dict[int, np.ndarray]:
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    if os.path.isfile(bin_path):
        return read_points3D_binary_dict(bin_path)
    if os.path.isfile(txt_path):
        return read_points3D_text_dict(txt_path)
    raise FileNotFoundError(f"No points3D.bin/txt under {sparse_dir}")


def load_colmap_cameras_images(
    sparse_dir: str,
) -> Tuple[Dict[int, Camera], Dict[int, Image]]:
    try:
        cameras = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except Exception:
        cameras = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
        images = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    return cameras, images


def camera_K(camera: Camera) -> np.ndarray:
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = camera.params[0], camera.params[1], camera.params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    if camera.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = camera.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")


def sorted_image_entries(images: Dict[int, Image]) -> List[Tuple[int, Image]]:
    """Sort like GaussianObject dataset_readers: by basename stem (image_name)."""

    def key(item: Tuple[int, Image]) -> str:
        name = os.path.basename(item[1].name)
        return os.path.splitext(name)[0]

    return sorted(images.items(), key=key)


def load_exclude_indices(path: Optional[str]) -> Set[int]:
    if not path:
        return set()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ids = np.loadtxt(path, dtype=np.int32)
    if ids.ndim == 0:
        return {int(ids)}
    return {int(x) for x in ids.tolist()}


@dataclass
class FrameSparseDepth:
    index: int
    image_id: int
    name: str
    image_path: str
    width: int
    height: int
    depth: np.ndarray  # float32 HxW, 0 = invalid
    n_valid: int
    excluded: bool


def build_sparse_depth_from_tracks(
    image: Image,
    camera: Camera,
    points3D: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Sparse depth at COLMAP-observed 2D locations; closer z wins on collision."""
    h, w = int(camera.height), int(camera.width)
    depth = np.zeros((h, w), dtype=np.float32)
    R = qvec2rotmat(image.qvec)
    t = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    n_valid = 0
    for xy, pid in zip(image.xys, image.point3D_ids):
        if pid < 0:
            continue
        xyz = points3D.get(int(pid))
        if xyz is None:
            continue
        z = float((R @ xyz + t)[2])
        if z <= 0:
            continue
        u = int(round(float(xy[0])))
        v = int(round(float(xy[1])))
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        prev = depth[v, u]
        if prev == 0.0 or z < prev:
            if prev == 0.0:
                n_valid += 1
            depth[v, u] = z
    return depth, n_valid


def apply_mask_to_depth(depth: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Zero out depth outside foreground mask (mask > 0.5)."""
    out = depth.copy()
    if mask.shape != out.shape:
        raise ValueError(f"mask shape {mask.shape} != depth {out.shape}")
    out[mask <= 0.5] = 0.0
    n_valid = int((out > 0).sum())
    return out, n_valid


def load_mask_for_image(data_dir: str, image_name: str, width: int, height: int) -> Optional[np.ndarray]:
    import cv2

    stem = os.path.splitext(os.path.basename(image_name))[0]
    candidates = [
        os.path.join(data_dir, "masks", f"{stem}.png"),
        os.path.join(data_dir, "masks", os.path.basename(image_name).replace(
            os.path.splitext(image_name)[1], ".png"
        )),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask.astype(np.float32) / 255.0)
    return None


def resolve_image_path(data_dir: str, image_name: str) -> str:
    base = os.path.basename(image_name)
    for sub in ("images", "images_2", "images_4", "images_8"):
        path = os.path.join(data_dir, sub, base)
        if os.path.isfile(path):
            return path
    # try case variants
    images_dir = os.path.join(data_dir, "images")
    if os.path.isdir(images_dir):
        lower = {f.lower(): f for f in os.listdir(images_dir)}
        key = base.lower()
        if key in lower:
            return os.path.join(images_dir, lower[key])
    raise FileNotFoundError(f"Image not found for {image_name} under {data_dir}/images")


def iter_scene_sparse_depths(
    data_dir: str,
    exclude_indices: Optional[Sequence[int]] = None,
    use_mask: bool = False,
    images_subdir: str = "images",
) -> List[FrameSparseDepth]:
    """Build sparse depth for every COLMAP image (sorted). Indices match sparse_*.txt."""
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    cameras, images = load_colmap_cameras_images(sparse_dir)
    points3D = load_points3D_dict(sparse_dir)
    exclude: Set[int] = set(int(x) for x in (exclude_indices or []))
    frames: List[FrameSparseDepth] = []

    for index, (image_id, image) in enumerate(sorted_image_entries(images)):
        camera = cameras[image.camera_id]
        depth, n_valid = build_sparse_depth_from_tracks(image, camera, points3D)
        if use_mask:
            mask = load_mask_for_image(
                data_dir, image.name, int(camera.width), int(camera.height)
            )
            if mask is not None:
                depth, n_valid = apply_mask_to_depth(depth, mask)
        try:
            image_path = resolve_image_path(data_dir, image.name)
            # prefer explicitly requested subdir if present
            preferred = os.path.join(data_dir, images_subdir, os.path.basename(image.name))
            if os.path.isfile(preferred):
                image_path = preferred
        except FileNotFoundError:
            image_path = ""
        frames.append(
            FrameSparseDepth(
                index=index,
                image_id=int(image_id),
                name=image.name,
                image_path=image_path,
                width=int(camera.width),
                height=int(camera.height),
                depth=depth,
                n_valid=n_valid,
                excluded=index in exclude,
            )
        )
    return frames


def resize_sparse_depth(depth: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Down/up-scale a sparse depth map by scattering valid pixels.

    ``cv2.INTER_NEAREST`` on a mostly-zero map collapses many COLMAP hits into
    few pixels (often <50), which then fails median-scale losses. Scatter keeps
    one sample per source point (last write wins on collisions).
    """
    h, w = depth.shape
    if h == out_h and w == out_w:
        return depth
    ys, xs = np.nonzero(depth > 0)
    out = np.zeros((out_h, out_w), dtype=depth.dtype)
    if ys.size == 0:
        return out
    xs2 = np.clip(np.round(xs.astype(np.float64) * (out_w / float(w))).astype(np.int64), 0, out_w - 1)
    ys2 = np.clip(np.round(ys.astype(np.float64) * (out_h / float(h))).astype(np.int64), 0, out_h - 1)
    out[ys2, xs2] = depth[ys, xs]
    return out
