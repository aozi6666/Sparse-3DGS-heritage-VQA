"""Render one Smithsonian GLB to a 24-view RGB-D dataset (bpy headless, Cycles CPU).

Reproducible renderer used to build smith_render/ (10 objects x 24 views RGB + GT depth).
See smith_render/README.md for the dataset layout and conventions.

Requirements:
  - Python 3.11 venv with: bpy==4.5.9 (Blender as a Python module), numpy<2 (1.26.x), Pillow.
    e.g.  python -m venv .venv_bpy && \
          .venv_bpy/bin/pip install bpy==4.5.9 "numpy==1.26.4" Pillow
    (bpy 4.5.9 requires numpy<2.0,>=1.26; numpy 2.x breaks the bundled glTF/importer extensions.)
  - No display/UI needed; run headless.

Usage:
  .venv_bpy/bin/python render_smith.py -- <slug> <glb_path>

  e.g. .venv_bpy/bin/python render_smith.py -- ewer-349e8fa2 \
         vggt_audit/smith/glb/ewer-349e8fa2.glb

  Output: smith_render/<slug>/ (images/, depth/, masks/, cameras.json, meta.json, preview.jpg).
  One Blender process per object is recommended (isolates crashes): loop over the GLBs.
  Scratch EXRs: vggt_audit/smith/render_scratch/<slug>/exr/.

Pipeline: one Cycles CPU render per view produces both RGB (compositor .Image -> Composite,
saved as 512x512 RGBA PNG via write_still) and the GT Z pass (Render Layers.Depth ->
File Output OPEN_EXR 32-bit); afterwards each EXR is loaded back and converted to float32
camera-space Z (.npy, background 0), alpha channel -> 0/255 mask PNG, depth preview PNG,
cameras.json (OpenCV c2w), meta.json and an 8x3 contact sheet.
Metric scale: the Smithsonian engine GLBs use per-object units; the unit (m/mm/cm/inch) is
inferred by matching the rendered bbox to the Smithsonian catalog dimensions (CATALOG_DIMS).

Outputs under smith_render/<slug>/:
  images/%05d.png  512x512 RGBA (transparent bg)
  depth/%05d.npy   float32 camera-space Z (scene units; bg=0)
  depth/%05d.png   16-bit normalized preview
  masks/%05d.png   0/255 foreground (alpha>0.5)
  cameras.json, meta.json, preview.jpg

Scratch EXRs under vggt_audit/smith/render_scratch/<slug>/exr/
"""
import bpy
import sys
import os
import json
import math
import time
import shutil
import glob

import numpy as np
from mathutils import Vector, Matrix
from PIL import Image

# Workspace root: this script normally lives in <workspace>/vggt_finetune_v1/scripts/.
# Override with SMITH_WORKSPACE if you moved it.
_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("SMITH_WORKSPACE") or os.path.dirname(os.path.dirname(_HERE))
if not os.path.isdir(os.path.join(WS, "vggt_audit", "smith")):
    WS = "/Users/zhangao/Documents/deepseek-harness/default-workspace"
OUT_ROOT = os.path.join(WS, "smith_render")
SCRATCH_ROOT = os.path.join(WS, "vggt_audit", "smith", "render_scratch")
SEL10 = os.path.join(WS, "vggt_audit", "smith", "selected10.json")

# Smithsonian catalog dimensions (record_json "Dimensions" from
# vggt_audit/smith/meta_parquet/), used to resolve the GLB's source units.
# The "engine" GLBs in this pack are NOT in consistent units.
CATALOG_DIMS = {
    "baluster-vase-one-of-three-in-a-five-piece-garniture-f1980-1-d8c62634":
        ("fsg_F1980.192a-c", "H x Diam (assembled): 45.5 x 17.1 cm (17 15/16 x 6 3/4 in)"),
    "beaker-shaped-vase-from-a-five-piece-garniture-d8c64b96":
        ("fsg_F1980.194a-b", "H x Diam: 43 x 20.6 cm (16 15/16 x 8 1/8 in)"),
    "buddha-draped-in-robes-portraying-the-realms-of-desire-d8c62be8":
        ("fsg_F1923.15", "H x W x D: 151.3 x 62.9 x 31.3 cm (59 9/16 x 24 3/4 x 12 5/16 in)"),
    "colonoware-pot-from-cooper-river-charleston-county-sc-79da3e3f":
        ("nmaahc_2017.108.1.1", "H x W: 9 1/16 x 10 1/2 in. (23 x 26.7 cm)"),
    "ewer-349e8fa2":
        ("chndm_1986-61-53", "H x W x D: 23 x 25.4 x 13.8cm (9 1/16 x 10 x 5 7/16in.)"),
    "kneeling-winged-monster-0dc68216":
        ("fsg_F1916.345", "H x W x D: 88.4 x 47.3 x 28.5 cm (34 13/16 x 18 5/8 x 11 1/4 in)"),
    "lidded-incense-burner-xianglu-with-geometric-decoration-and-ce850625":
        ("fsg_F1947.15a-b", "H x W: 17.9 x 10 cm (7 1/16 x 3 15/16 in)"),
    "model-of-the-greek-slave-8edffe56":
        ("saam_1968.155.8", "66 3/8 x 21 1/2 x 18 3/8 in. (168.6 x 54.6 x 46.6 cm)"),
    "ritual-wine-container-fangyi-with-masks-taotie-serpents-and-d8c62f94":
        ("fsg_F1930.54a-b", "H x W x D: 35.3 x 24.8 x 23.3 cm (13 7/8 x 9 3/4 x 9 3/16 in)"),
    "ritual-wine-ewer-gong-with-masks-taotie-dragons-and-real-ani-d8c646aa":
        ("fsg_F1961.33a-b", "H x W x D: 32.2 x 32.2 x 15.7 cm (12 11/16 x 12 11/16 x 6 3/16 in)"),
}
CANDIDATE_UNIT_FACTORS = [1.0, 0.001, 0.01, 0.0254]  # metres, millimetres, centimetres, inches
UNIT_FACTOR_NAME = {1.0: "m", 0.001: "mm", 0.01: "cm", 0.0254: "inch"}


def _parse_catalog_dims(raw):
    import re as _re
    nums = None
    m = _re.search(r"\(([\d\.]+(?:\s*[x\u00d7]\s*[\d\.]+){1,2})\s*cm\)", raw)
    if m:
        nums = [float(x) / 100.0 for x in _re.split(r"\s*[x\u00d7]\s*", m.group(1))]
    if nums is None:
        m = _re.search(r"([\d\.]+(?:\s*[x\u00d7]\s*[\d\.]+){1,2})\s*cm\b", raw)
        if m:
            nums = [float(x) / 100.0 for x in _re.split(r"\s*[x\u00d7]\s*", m.group(1))]
    if nums is None:
        # imperial-only fallback, e.g. '66 3/8 x 21 1/2 x 18 3/8 in.'
        parts = _re.findall(r"(\d+\s+\d+/\d+|\d+(?:\.\d+)?)", raw)
        if parts:
            nums = []
            for v in parts:
                try:
                    nums.append(float(eval(v.replace(" ", "+"))) * 0.0254)
                except Exception:
                    pass
    return nums


def resolve_units(slug, measured_dims):
    """Infer the GLB's source unit by matching the rendered bbox to the Smithsonian catalog dims."""
    import math as _math
    info = CATALOG_DIMS.get(slug)
    if not info:
        return dict(scene_scale_m=max(measured_dims), gltf_units="unknown",
                    gltf_unit_factor_to_m=1.0, confident=False,
                    source=None, bbox_original_meters=[float(x) for x in measured_dims])
    record_id, dim_str = info
    catalog = _parse_catalog_dims(dim_str)
    if not catalog:
        return dict(scene_scale_m=max(measured_dims), gltf_units="unknown",
                    gltf_unit_factor_to_m=1.0, confident=False,
                    source=dict(record_id=record_id, catalog_dimensions=dim_str),
                    bbox_original_meters=[float(x) for x in measured_dims])
    best = None
    for f in CANDIDATE_UNIT_FACTORS:
        dists = [min(abs(_math.log(float(d) * f / c)) for c in catalog) for d in measured_dims]
        score = sum(dists) / len(dists)
        if best is None or score < best[1]:
            best = (f, score, dists)
    f, score, dists = best
    return dict(
        scene_scale_m=float(max(measured_dims) * f),
        gltf_units=UNIT_FACTOR_NAME[f],
        gltf_unit_factor_to_m=f,
        confident=score < _math.log(1.35),
        bbox_original_meters=[float(x) * f for x in measured_dims],
        source=dict(record_id=record_id, catalog_dimensions=dim_str,
                    catalog_dimensions_m=catalog, method=(
                        "compare rendered bbox against Smithsonian catalog dimensions over candidate "
                        "unit factors [m, mm, cm, inch]; choose best log-space fit"),
                    unit_factor=f, unit_name=UNIT_FACTOR_NAME[f],
                    mean_log_fit_error=score, per_dim_log_error=dists),
    )

W = H = 512
LENS = 50.0
SENSOR = 36.0
FX = LENS / SENSOR * W  # 711.111...
FOV = 2.0 * math.atan(SENSOR / (2.0 * LENS))
N_AZ = 8
ELEVS = [15.0, 35.0, 60.0]
N_VIEWS = 24
SAMPLES = 16
BG_SENTINEL = 1e9  # Cycles Z pass background value (observed ~1e10)
BG_STORE = 0.0     # convention stored in .npy for background


def log(msg):
    print(f"[render] {msg}", flush=True)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Keep factory prefs but ensure gltf addon is enabled
    addon = "io_scene_gltf2"
    try:
        bpy.ops.preferences.addon_enable(module=addon)
    except Exception as e:  # noqa: BLE001
        log(f"addon enable warning: {e}")
    return bpy.context.scene


def build_scene(glb_path, margin):
    """Import GLB, normalize to unit max-bbox-dim, setup camera/lights/render. Returns info dict."""
    scene = reset_scene()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no MESH objects imported")

    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    nverts = 0
    for o in meshes:
        mw = o.matrix_world
        for v in o.data.vertices:
            w = mw @ v.co
            for i in range(3):
                if w[i] < mins[i]:
                    mins[i] = w[i]
                if w[i] > maxs[i]:
                    maxs[i] = w[i]
        nverts += len(o.data.vertices)
        if o.data.vertices == 0:
            log(f"warning: empty mesh {o.name}")

    dims = maxs - mins
    center = (mins + maxs) * 0.5
    smax = max(dims.x, dims.y, dims.z)
    if not (smax > 0):
        raise RuntimeError(f"degenerate bbox dims {tuple(dims)}")
    scale = 1.0 / smax
    M = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)
    for o in meshes:
        o.matrix_world = M @ o.matrix_world
    dims_n = dims * scale

    r = 0.5 * math.sqrt(dims_n.x ** 2 + dims_n.y ** 2 + dims_n.z ** 2)
    d = r / math.sin(FOV / 2.0) * margin

    # ---- camera ----
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = LENS
    cam_data.sensor_width = SENSOR
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # ---- lights (fixed world-space even lighting) ----
    def add_area(name, loc, energy, size):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        direction = (Vector((0.0, 0.0, 0.0)) - Vector(loc)).normalized()
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return lo

    add_area("key", (2.5, -2.5, 3.0), 150.0, 3.0)
    add_area("fill", (-3.0, 2.0, 1.5), 80.0, 3.0)

    # ---- world ambient ----
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.55, 0.55, 1.0)
    bg.inputs["Strength"].default_value = 0.7

    # ---- render settings ----
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.threads_mode = "AUTO"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.frame_start = 0
    scene.frame_end = N_VIEWS - 1

    # ---- passes + compositor: RGBA -> Composite (write_still), Z -> File Output EXR ----
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    fo = nt.nodes.new("CompositorNodeOutputFile")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    nt.links.new(rl.outputs["Depth"], fo.inputs[0])

    return dict(
        scene=scene,
        cam=cam,
        meshes=meshes,
        mins=mins,
        maxs=maxs,
        dims=dims,
        dims_n=dims_n,
        center=center,
        smax=smax,
        scale=scale,
        r=r,
        d=d,
        margin=margin,
        nverts=nverts,
        fo=fo,
        nt=nt,
    )


def set_view(info, frame):
    az_deg = (frame % N_AZ) * 45.0
    el_deg = ELEVS[frame // N_AZ]
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    d = info["d"]
    pos = Vector((d * math.cos(el) * math.cos(az),
                  d * math.cos(el) * math.sin(az),
                  d * math.sin(el)))
    zcam = pos.normalized()
    up = Vector((0.0, 0.0, 1.0))
    xcam = up.cross(zcam)
    if xcam.length < 1e-8:  # degenerate, should not happen for el<90
        xcam = Vector((1.0, 0.0, 0.0))
    xcam.normalize()
    ycam = zcam.cross(xcam)
    R = Matrix((xcam, ycam, zcam)).transposed().to_4x4()
    c2w_b = Matrix.Translation(pos) @ R
    info["cam"].matrix_world = c2w_b
    R_cv = R.to_3x3() @ Matrix.Diagonal((1.0, -1.0, -1.0))
    c2w_cv = np.eye(4, dtype=np.float64)
    c2w_cv[:3, :3] = np.array([[R_cv[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    c2w_cv[:3, 3] = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
    return az_deg, el_deg, c2w_cv, pos


def render_views(info, outdir, exrdir):
    scene = info["scene"]
    info["fo"].base_path = exrdir + "/"
    info["fo"].format.file_format = "OPEN_EXR"
    info["fo"].format.color_depth = "32"
    info["fo"].format.exr_codec = "ZIP"
    info["fo"].file_slots[0].path = "####"
    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    views = []
    t0 = time.time()
    for frame in range(N_VIEWS):
        az, el, c2w_cv, pos = set_view(info, frame)
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(imgdir, "%05d.png" % frame)
        t = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"  view {frame:02d}/{N_VIEWS} az={az:.0f} el={el:.0f} {time.time()-t:.2f}s")
        views.append(dict(frame=frame, azimuth_deg=az, elevation_deg=el,
                          c2w=c2w_cv.tolist(), camera_position=[float(x) for x in pos]))
    log(f"render done in {time.time()-t0:.1f}s")
    return views


def fill_edge_depth(z, mask, iters=4):
    """Fill mask pixels whose Z was the background sentinel from positive neighbors."""
    z = z.copy()
    for _ in range(iters):
        holes = (z <= 0) & mask
        if not holes.any():
            break
        pos = z > 0
        neigh_sum = np.zeros_like(z)
        neigh_cnt = np.zeros_like(z)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(np.roll(z, dy, axis=0), dx, axis=1)
            sp = np.roll(np.roll(pos, dy, axis=0), dx, axis=1)
            neigh_sum += np.where(sp, shifted, 0.0)
            neigh_cnt += sp.astype(np.float32)
        fill = holes & (neigh_cnt > 0)
        z[fill] = (neigh_sum[fill] / neigh_cnt[fill]).astype(np.float32)
    return z


def compose(slug, info, views, outdir, exrdir):
    """Convert EXR depth to .npy, alpha->mask, depth 16-bit preview. Returns stats."""
    depth_dir = os.path.join(outdir, "depth")
    mask_dir = os.path.join(outdir, "masks")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    fg_fracs = []
    depth_mins = []
    depth_maxs = []
    mask_depth_coverage = []
    clipped = []
    for v in views:
        fr = v["frame"]
        png_path = os.path.join(outdir, "images", "%05d.png" % fr)
        im = Image.open(png_path).convert("RGBA")
        rgba = np.asarray(im, dtype=np.uint8)
        alpha = rgba[:, :, 3]
        mask = (alpha > 127).astype(np.uint8)
        fg = float(mask.mean())
        fg_fracs.append(fg)
        Image.fromarray(mask * 255, mode="L").save(os.path.join(mask_dir, "%05d.png" % fr))

        # border touching -> possible clipping
        border = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
        if border > 0:
            clipped.append(fr)

        exr_path = os.path.join(exrdir, "%04d.exr" % fr)
        exr = bpy.data.images.load(exr_path)
        try:
            exr.colorspace_settings.name = "Non-Color"
            w, h = exr.size
            nch = exr.channels
            buf = np.empty(w * h * nch, dtype=np.float32)
            exr.pixels.foreach_get(buf)
            arr = buf.reshape(h, w, nch)
            z = arr[:, :, 0].copy()
        finally:
            bpy.data.images.remove(exr)
        # bpy image rows go bottom-up; store top-down like the PNG
        z = z[::-1, :]
        if z.shape != (H, W):
            raise RuntimeError(f"EXR shape {z.shape} != {(H, W)}")
        z = np.where(np.isfinite(z) & (z < BG_SENTINEL) & (z > 0), z, 0.0).astype(np.float32)
        mb = mask.astype(bool)
        z = np.where(mb, z, 0.0).astype(np.float32)
        z = fill_edge_depth(z, mb)
        np.save(os.path.join(depth_dir, "%05d.npy" % fr), z)

        mvals = z[mb]
        pos = mvals[mvals > 0]
        if pos.size:
            depth_mins.append(float(pos.min()))
            depth_maxs.append(float(pos.max()))
            mask_depth_coverage.append(float((mvals > 0).mean()))
        else:
            depth_mins.append(0.0)
            depth_maxs.append(0.0)
            mask_depth_coverage.append(0.0)

        # 16-bit depth preview (near = bright), background black
        zmin = float(pos.min()) if pos.size else 0.0
        zmax = float(pos.max()) if pos.size else 1.0
        rng = max(zmax - zmin, 1e-6)
        vis = np.clip(1.0 - (z - zmin) / rng, 0.0, 1.0)
        vis = np.where(mask.astype(bool), vis, 0.0)
        Image.fromarray((vis * 65535.0).round().astype(np.uint16)).save(
            os.path.join(depth_dir, "%05d.png" % fr))

    stats = dict(
        fg_fractions=fg_fracs,
        fg_median=float(np.median(fg_fracs)),
        fg_min=float(np.min(fg_fracs)),
        fg_max=float(np.max(fg_fracs)),
        n_nonempty=int(sum(1 for f in fg_fracs if f >= 0.002)),
        depth_min=float(np.min([x for x in depth_mins if x > 0])) if any(x > 0 for x in depth_mins) else 0.0,
        depth_max=float(np.max(depth_maxs)),
        mask_depth_coverage=float(np.mean(mask_depth_coverage)),
        clipped_views=clipped,
    )
    return stats


def make_preview(outdir):
    cols, rows, tile = 8, 3, 168
    sheet = Image.new("RGB", (cols * tile, rows * tile), (28, 28, 30))
    for fr in range(N_VIEWS):
        p = os.path.join(outdir, "images", "%05d.png" % fr)
        im = Image.open(p).convert("RGBA")
        im = im.resize((tile, tile), Image.LANCZOS)
        bg = Image.new("RGBA", (tile, tile), (28, 28, 30, 255))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), ((fr % cols) * tile, (fr // cols) * tile))
    path = os.path.join(outdir, "preview.jpg")
    sheet.save(path, quality=92)
    return path


def write_cameras(slug, info, views, outdir, units):
    cams = {
        "object_id": slug,
        "convention": "OpenCV c2w (x right, y down, z forward)",
        "width": W,
        "height": H,
        "fx": FX,
        "fy": FX,
        "cx": W / 2.0,
        "cy": H / 2.0,
        "camera_model": "pinhole, 50mm lens on 36mm sensor, square 512 image",
        "fov_deg": math.degrees(FOV),
        "background_depth": BG_STORE,
        "background_depth_note": "background/empty pixels stored as 0.0 in depth/*.npy; raw Cycles Z background sentinel was ~1e10",
        "depth_unit": "scene units; multiply by scene_scale_m for meters",
        "scene_scale_m": units["scene_scale_m"],
        "scene_scale_raw_gltf_units": info["smax"],
        "gltf_units": units["gltf_units"],
        "gltf_unit_factor_to_m": units["gltf_unit_factor_to_m"],
        "gltf_units_confidence": "high" if units["confident"] else "low",
        "gltf_units_source": units["source"],
        "depth_scale": units["scene_scale_m"],
        "depth_scale_note": "meters = depth_value * depth_scale",
        "normalization": "object centered at origin, max bbox dim scaled to 1.0",
        "bbox_dim_original_gltf_units": [float(x) for x in info["dims"]],
        "bbox_dim_original_meters": units["bbox_original_meters"],
        "bbox_dim_normalized": [float(x) for x in info["dims_n"]],
        "bounding_sphere_radius_normalized": info["r"],
        "camera_distance": info["d"],
        "margin_factor": info["margin"],
        "n_views": len(views),
        "elevations_deg": ELEVS,
        "azimuths_deg": [i * 45.0 for i in range(N_AZ)],
        "views": [],
    }
    for v in views:
        fr = v["frame"]
        cams["views"].append({
            "frame": fr,
            "image": "images/%05d.png" % fr,
            "depth": "depth/%05d.npy" % fr,
            "mask": "masks/%05d.png" % fr,
            "depth_preview": "depth/%05d.png" % fr,
            "azimuth_deg": v["azimuth_deg"],
            "elevation_deg": v["elevation_deg"],
            "camera_position": v["camera_position"],
            "c2w": [[float(x) for x in row] for row in v["c2w"]],
            "fx": FX,
            "fy": FX,
            "cx": W / 2.0,
            "cy": H / 2.0,
            "depth_scale": units["scene_scale_m"],
        })
    p = os.path.join(outdir, "cameras.json")
    with open(p, "w") as f:
        json.dump(cams, f, indent=1)
    return p


def write_meta(slug, info, glb_path, outdir, stats, units):
    sel = []
    if os.path.exists(SEL10):
        with open(SEL10) as f:
            sel = json.load(f)
    entry = next((e for e in sel if e.get("id") == slug), {})
    meta = {
        "id": slug,
        "title": entry.get("title"),
        "authors": entry.get("authors"),
        "kind": entry.get("kind"),
        "license": entry.get("license"),
        "triangles": entry.get("triangles"),
        "source_file_name": entry.get("file_name"),
        "source_glb": os.path.relpath(glb_path, WS),
        "source_glb_bytes": os.path.getsize(glb_path),
        "scene_scale_m": units["scene_scale_m"],
        "scene_scale_raw_gltf_units": info["smax"],
        "gltf_units": units["gltf_units"],
        "gltf_unit_factor_to_m": units["gltf_unit_factor_to_m"],
        "catalog_dimensions": (units["source"] or {}).get("catalog_dimensions"),
        "catalog_record_id": (units["source"] or {}).get("record_id"),
        "scene_units_note": ("glTF units are not consistent across this dataset; per-object units were "
                             "inferred by matching the rendered bounding box to the Smithsonian catalog "
                             "dimensions. normalized_scene: max bbox dim = 1.0; meters = depth * scene_scale_m."),
        "normalization": "object centered at origin, max bbox dim scaled to 1.0",
        "render": {
            "engine": "CYCLES (CPU)",
            "samples": SAMPLES,
            "denoiser": "OPENIMAGEDENOISE",
            "resolution": [W, H],
            "film_transparent": True,
            "n_views": len(stats["fg_fractions"]),
            "elevations_deg": ELEVS,
            "azimuth_step_deg": 45,
            "fov_deg": math.degrees(FOV),
            "fx": FX,
            "camera_distance": info["d"],
            "margin_factor": info["margin"],
            "view_transform": "Standard",
            "blender_version": bpy.app.version_string,
            "depth": "GT from Cycles Z pass (compositor File Output, 32-bit EXR), camera-space Z, scene units",
            "background_depth": BG_STORE,
            "mask": "render alpha channel thresholded at 0.5",
        },
        "qc": {
            "foreground_fraction_median": stats["fg_median"],
            "foreground_fraction_min": stats["fg_min"],
            "foreground_fraction_max": stats["fg_max"],
            "nonempty_views": stats["n_nonempty"],
            "depth_min": stats["depth_min"],
            "depth_max": stats["depth_max"],
            "mask_depth_coverage": stats["mask_depth_coverage"],
            "clipped_views": stats["clipped_views"],
        },
    }
    p = os.path.join(outdir, "meta.json")
    with open(p, "w") as f:
        json.dump(meta, f, indent=1)
    return p


def validate(info, stats):
    problems = []
    if stats["n_nonempty"] < 22:
        problems.append(f"only {stats['n_nonempty']}/24 non-empty views")
    if not (0.02 <= stats["fg_median"] <= 0.85):
        problems.append(f"median foreground fraction {stats['fg_median']:.4f} outside [0.02,0.85]")
    if stats["mask_depth_coverage"] < 0.98:
        problems.append(f"mask/depth coverage {stats['mask_depth_coverage']:.3f} < 0.98")
    if stats["depth_max"] <= 0 or stats["depth_max"] > info["d"] + info["r"] * 2.0:
        problems.append(f"depth max {stats['depth_max']:.3f} out of sensible range")
    if stats["depth_min"] <= 0:
        problems.append("no positive depth values")
    if stats["clipped_views"]:
        problems.append(f"mask touches image border on views {stats['clipped_views']}")
    return problems


def run(slug, glb_path):
    outdir = os.path.join(OUT_ROOT, slug)
    scratch = os.path.join(SCRATCH_ROOT, slug)
    exrdir = os.path.join(scratch, "exr")
    margins = [1.15, 1.35, 0.95]
    result = None
    for attempt, margin in enumerate(margins):
        log(f"attempt {attempt+1} margin={margin}")
        # clean output dirs from previous attempts
        for sub in ("images", "depth", "masks"):
            p = os.path.join(outdir, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
        if os.path.isdir(scratch):
            shutil.rmtree(scratch)
        os.makedirs(exrdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        info = build_scene(glb_path, margin)
        log(f"imported {len(info['meshes'])} meshes, orig bbox dims "
            f"{tuple(round(x,4) for x in info['dims'])}, scale={info['scale']:.4f}, "
            f"r={info['r']:.4f}, d={info['d']:.4f}")
        units = resolve_units(slug, [float(x) for x in info["dims"]])
        log(f"units: {units['gltf_units']} (factor {units['gltf_unit_factor_to_m']}), "
            f"scene_scale_m={units['scene_scale_m']:.4f}, confident={units['confident']}")
        views = render_views(info, outdir, exrdir)
        stats = compose(slug, info, views, outdir, exrdir)
        log(f"qc: median_fg={stats['fg_median']:.4f} range=({stats['fg_min']:.4f},"
            f"{stats['fg_max']:.4f}) nonempty={stats['n_nonempty']}/24 "
            f"depth=({stats['depth_min']:.4f},{stats['depth_max']:.4f}) "
            f"coverage={stats['mask_depth_coverage']:.4f} clipped={stats['clipped_views']}")
        problems = validate(info, stats)
        if not problems:
            cams = write_cameras(slug, info, views, outdir, units)
            meta = write_meta(slug, info, glb_path, outdir, stats, units)
            prev = make_preview(outdir)
            result = dict(slug=slug, ok=True, attempts=attempt + 1, margin=margin,
                          stats={k: v for k, v in stats.items() if k != "fg_fractions"},
                          scene_scale_m=units["scene_scale_m"],
                          gltf_units=units["gltf_units"],
                          original_dims=[float(x) for x in info["dims"]],
                          cameras=cams, meta=meta, preview=prev,
                          glb=glb_path, exr=exrdir,
                          blender=bpy.app.version_string)
            break
        log("validation problems: " + "; ".join(problems) + " -> retry")
        result = dict(slug=slug, ok=False, attempts=attempt + 1, margin=margin,
                      problems=problems, glb=glb_path,
                      stats={k: v for k, v in stats.items() if k != "fg_fractions"})
    with open(os.path.join(SCRATCH_ROOT, "summary_%s.json" % slug), "w") as f:
        json.dump(result, f, indent=1)
    log("RESULT " + json.dumps(result.get("stats", {})))
    log("OK" if result and result.get("ok") else "FAILED")
    return result


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    slug, glb = argv[0], argv[1]
    t0 = time.time()
    try:
        res = run(slug, os.path.abspath(glb))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res = dict(slug=slug, ok=False, error=f"{type(e).__name__}: {e}", glb=glb)
        with open(os.path.join(SCRATCH_ROOT, "summary_%s.json" % slug), "w") as f:
            json.dump(res, f, indent=1)
    log(f"total wall {time.time()-t0:.1f}s")
