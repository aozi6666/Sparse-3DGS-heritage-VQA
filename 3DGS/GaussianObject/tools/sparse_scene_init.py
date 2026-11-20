#!/usr/bin/env python3
"""Training-view-only geometry and frozen VGGT for the trex sparse pilot.

Uses provided rectified COLMAP poses and feature-track identities. Point
coordinates, colors, depth scale and depth consistency use training views only.
This is not a reconstruction of camera poses/tracks from sparse images alone.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

GO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GO))
sys.path.insert(0, str(GO.parent / "vggt"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def calibration(camera, width=None, height=None):
    if camera.model_name == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model_name == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        fy = fx
    else:
        raise ValueError("Use rectified PINHOLE cameras")
    sx = width / camera.width if width else 1
    sy = height / camera.height if height else 1
    return np.array([fx * sx, fy * sy, cx * sx, cy * sy])


def write_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    fields = [(x, "f4") for x in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(x, "u1") for x in ("red", "green", "blue")]
    values = np.zeros(len(xyz), dtype=fields)
    for i, name in enumerate(("x", "y", "z")):
        values[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        values[name] = rgb[:, i]
    PlyData([PlyElement.describe(values, "vertex")]).write(str(path))


def frames_for(source, ids):
    import pycolmap
    rec = pycolmap.Reconstruction(str(source / "sparse/0"))
    all_frames = sorted(rec.images.values(), key=lambda x: Path(x.name).stem)
    return rec, [all_frames[i] for i in ids]


def triangulate(observations, max_error=4., min_angle=1.):
    """DLT with selected observations; never reads the supplied dense xyz."""
    rows = []
    for xy, k, R, t in observations:
        P = np.column_stack((R, t))
        u, v = (xy - k[2:]) / k[:2]
        rows.extend((u * P[2] - P[0], v * P[2] - P[1]))
    _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    if abs(vt[-1, 3]) < 1e-10:
        return None
    xyz = vt[-1, :3] / vt[-1, 3]
    if not np.isfinite(xyz).all():
        return None
    rays = []
    for xy, k, R, t in observations:
        cam = R @ xyz + t
        if cam[2] <= 1e-6:
            return None
        uv = cam[:2] / cam[2] * k[:2] + k[2:]
        if np.linalg.norm(uv - xy) > max_error:
            return None
        ray = xyz + R.T @ t
        rays.append(ray / np.linalg.norm(ray))
    cosines = np.clip(np.asarray(rays) @ np.asarray(rays).T, -1, 1)
    if np.degrees(np.arccos(cosines.min())) < min_angle:
        return None
    return xyz


def training_geometry(source, data, ids, resolution):
    rec, frames = frames_for(source, ids)
    tracks = defaultdict(list)
    cache = []
    for fi, im in enumerate(frames):
        camera = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        t = np.asarray(im.cam_from_world().translation)
        rgb = np.asarray(Image.open(data / f"images_{resolution}" / im.name).convert("RGB"))
        h, w = rgb.shape[:2]
        k = calibration(camera)
        cache.append((rgb, w / camera.width, h / camera.height, R, t, k))
        for p in im.points2D:
            if p.has_point3D():
                tracks[int(p.point3D_id)].append((fi, np.asarray(p.xy)))
    pids, xyzs, colors = [], [], []
    anchors = [0] * len(frames)
    for pid, observations in sorted(tracks.items()):
        if len(observations) < 2:
            continue
        obs = [(xy, cache[i][5], cache[i][3], cache[i][4]) for i, xy in observations]
        xyz = triangulate(obs)
        if xyz is None:
            continue
        samples = []
        for i, xy in observations:
            rgb, sx, sy, *_ = cache[i]
            u, v = np.rint(xy * [sx, sy]).astype(int)
            if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                samples.append(rgb[v, u])
                anchors[i] += 1
        if not samples:
            continue
        pids.append(pid)
        xyzs.append(xyz)
        colors.append(np.rint(np.median(samples, axis=0)).astype(np.uint8))
    if len(xyzs) < 100 or min(anchors) < 30:
        raise RuntimeError(f"Insufficient selected-view geometry: {len(xyzs)} points, anchors {anchors}")
    xyz = np.asarray(xyzs, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    np.savez_compressed(data / "training_geometry.npz", ids=np.asarray(pids), xyz=xyz, rgb=rgb)
    write_ply(data / "sparse/0/points3D.ply", xyz, rgb)
    centers = np.asarray([-c[3].T @ c[4] for c in cache])
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    report = {"train_ids": ids, "train_names": [im.name for im in frames],
              "triangulated_points": len(xyz), "anchors_per_frame": anchors,
              "scene_extent": extent, "dense_point_coordinates_used": False,
              "geometry": "DLT from selected-view observations; >=2 views; reprojection <=4px; angle >=1deg",
              "provided_calibration": "all-view COLMAP poses and track identities; fixed, not estimated from sparse images"}
    write_json(data / "training_geometry.json", report)
    print(f"[geometry] {len(frames)} views: {len(xyz)} points; anchors {anchors}", flush=True)


def frozen_depth(source, data, ids, resolution, weights):
    import torch
    import torch.nn.functional as F
    from vggt.models.vggt import VGGT
    from vggt_depthft_ckpt import load_vggt_with_optional_depth_head
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen VGGT inference")
    rec, frames = frames_for(source, ids)
    geom = np.load(data / "training_geometry.npz", allow_pickle=False)
    point_map = dict(zip(geom["ids"].tolist(), geom["xyz"]))
    geometry_sha = sha(data / "training_geometry.npz")
    weight_sha = sha(weights)
    depths, confs = data / "zoe_depth", data / "vggt_conf"
    depths.mkdir(exist_ok=True)
    confs.mkdir(exist_ok=True)
    model = None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for im in frames:
        stem = Path(im.name).stem
        image = data / f"images_{resolution}" / im.name
        signature = {"backend": "frozen_pretrained_VGGT_single_frame_train_only_scale_v1",
                     "weights_sha256": weight_sha, "image_sha256": sha(image),
                     "geometry_sha256": geometry_sha, "train_ids": ids}
        info = depths / f"{stem}.json"
        dp, cp = depths / f"{stem}.npy", confs / f"{stem}.npy"
        if info.exists():
            if json.loads(info.read_text())["signature"] != signature:
                raise RuntimeError(f"Depth config changed: {info}; use a new run tag")
            if dp.exists() and cp.exists():
                d, c = np.load(dp), np.load(cp)
                if d.shape == c.shape and np.isfinite(d).all() and (d > 0).all() and np.isfinite(c).all():
                    print(f"[reuse] frozen depth {stem}", flush=True)
                    continue
        if model is None:
            model = VGGT()
            mode, _ = load_vggt_with_optional_depth_head(model, str(weights))
            if mode != "full":
                raise RuntimeError("This frozen-pretrained run rejects adapted depth-head overlay checkpoints")
            model.requires_grad_(False).eval().cuda()
        rgb = np.asarray(Image.open(image).convert("RGB"))
        h, w = rgb.shape[:2]
        ratio = min(1., 518 / max(h, w))
        nh, nw = [max(14, round(side * ratio / 14) * 14) for side in (h, w)]
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255
        tensor = F.interpolate(tensor[None], size=(nh, nw), mode="bilinear", align_corners=False)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            result = model(tensor[None])
        dep, conf = [F.interpolate(result[key].float().squeeze()[None, None], size=(h, w),
                                  mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                     for key in ("depth", "depth_conf")]
        camera = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        t = np.asarray(im.cam_from_world().translation)
        ratios = []
        for point in im.points2D:
            xyz = point_map.get(int(point.point3D_id)) if point.has_point3D() else None
            if xyz is None:
                continue
            z = (R @ xyz + t)[2]
            u, v = np.rint(np.asarray(point.xy) * [w / camera.width, h / camera.height]).astype(int)
            if z > 0 and 0 <= u < w and 0 <= v < h and np.isfinite(dep[v, u]) and dep[v, u] > 0:
                ratios.append(z / dep[v, u])
        if len(ratios) < 30:
            raise RuntimeError(f"Insufficient training-only scale anchors: {im.name}: {len(ratios)}")
        scale = float(np.median(ratios))
        dep = (dep * scale).astype(np.float32)
        if not np.isfinite(dep).all() or (dep <= 0).any() or not np.isfinite(conf).all():
            raise ValueError(f"Invalid frozen VGGT depth: {im.name}")
        np.save(dp, dep, allow_pickle=False)
        np.save(cp, conf.astype(np.float32), allow_pickle=False)
        write_json(info, {"signature": signature, "scale": scale, "scale_anchors": len(ratios),
                          "shape": [h, w], "fine_tuned": False, "not_ground_truth": True})
        print(f"[VGGT] {im.name}: scale={scale:.5g}, anchors={len(ratios)}", flush=True)
        del result, tensor
    if model is not None:
        del model
        torch.cuda.empty_cache()


def enhance(source, data, ids, resolution, max_new_points):
    rec, frames = frames_for(source, ids)
    geom = np.load(data / "training_geometry.npz", allow_pickle=False)
    base, colors = geom["xyz"], geom["rgb"]
    extent = json.loads((data / "training_geometry.json").read_text())["scene_extent"]
    lo, hi = np.quantile(base, [.005, .995], axis=0)
    pad = (hi - lo) * .1
    cache = []
    for im in frames:
        stem = Path(im.name).stem
        depth = np.load(data / "zoe_depth" / f"{stem}.npy")
        conf = np.load(data / "vggt_conf" / f"{stem}.npy")
        rgb = np.asarray(Image.open(data / f"images_{resolution}" / im.name).convert("RGB"))
        if depth.shape != rgb.shape[:2] or conf.shape != depth.shape:
            raise ValueError(f"Depth/RGB mismatch: {im.name}")
        R = np.asarray(im.cam_from_world().rotation.matrix())
        t = np.asarray(im.cam_from_world().translation)
        k = calibration(rec.cameras[im.camera_id], depth.shape[1], depth.shape[0])
        cache.append((depth, conf, rgb, R, t, k, -R.T @ t))
    points, rgbs, records = [base], [colors], []
    for index, (depth, conf, rgb, R, t, k, center) in enumerate(cache):
        y, x = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
        x, y = x.ravel(), y.ravel()
        z = depth[y, x]
        good = np.isfinite(z) & (z > 0) & (conf[y, x] >= np.median(conf))
        x, y, z = x[good], y[good], z[good]
        cam = np.stack([(x-k[2])/k[0]*z, (y-k[3])/k[1]*z, z], axis=1)
        world = (cam - t) @ R
        valid = ((world >= lo-pad) & (world <= hi+pad)).all(1)
        support = np.zeros(len(world), dtype=int)
        neighbours = np.argsort([np.linalg.norm(c[6]-center) for c in cache])
        for ni in [int(i) for i in neighbours if i != index][:3]:
            d2, _, _, r2, t2, k2, _ = cache[ni]
            proj = world @ r2.T + t2
            zz = proj[:, 2]
            u = np.rint(proj[:, 0]/np.maximum(zz, 1e-6)*k2[0]+k2[2]).astype(int)
            v = np.rint(proj[:, 1]/np.maximum(zz, 1e-6)*k2[1]+k2[3]).astype(int)
            inside = (zz > 0) & (u >= 0) & (v >= 0) & (u < d2.shape[1]) & (v < d2.shape[0])
            js = np.flatnonzero(inside)
            support[js] += np.abs(d2[v[js], u[js]]-zz[js]) <= np.maximum(.05*zz[js], .01*extent)
        chosen = np.flatnonzero(valid & (support >= 2))
        cap = max_new_points // len(frames)
        if len(chosen) > cap:
            chosen = chosen[np.linspace(0, len(chosen)-1, cap, dtype=int)]
        points.append(world[chosen].astype(np.float32))
        rgbs.append(rgb[y[chosen], x[chosen]])
        records.append({"name": frames[index].name, "candidates": len(world), "accepted": len(chosen)})
    xyz, rgb = np.concatenate(points), np.concatenate(rgbs)
    if len(xyz) > len(base):
        voxels = np.floor(xyz[len(base):] / max(extent / 500, 1e-5)).astype(np.int64)
        _, unique = np.unique(voxels, axis=0, return_index=True)
        keep = np.r_[np.arange(len(base)), len(base) + np.sort(unique)]
        xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) == len(base):
        raise RuntimeError("VGGT added no reliable points; do not silently label this as VGGT enhancement")
    write_ply(data / "scene_sparse_vggt.ply", xyz, rgb)
    write_json(data / "scene_init_report.json", {"train_ids": ids, "base_points": len(base),
               "new_points": len(xyz)-len(base), "total_points": len(xyz), "frames": records,
               "visual_hull_used": False, "fine_tuned": False, "test_depth_used": False})
    print(f"[init] {len(base)} triangulated + {len(xyz)-len(base)} VGGT points", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--max-new-points", type=int, default=60000)
    a = p.parse_args()
    ids = json.loads((a.data / "split.json").read_text())["train_ids"]
    frozen_depth(a.source, a.data, ids, a.resolution, a.weights)
    enhance(a.source, a.data, ids, a.resolution, a.max_new_points)
