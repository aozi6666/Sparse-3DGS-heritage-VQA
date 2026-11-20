#!/usr/bin/env python3
"""trex 4/6/9-view frozen-VGGT + DropGaussian coarse pilot; no repair stages."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from sparse_scene_init import sha, training_geometry, write_json

GO = Path(__file__).resolve().parents[1]
PY = sys.executable


def split_views(count):
    test = list(range(0, count, 8))
    pool = [i for i in range(count) if i not in test]
    if len(pool) < 9:
        raise ValueError("Need at least nine non-test images")
    nine = [pool[i] for i in np.rint(np.linspace(0, len(pool)-1, 9)).astype(int)]
    train = {4: [nine[i] for i in [0, 3, 5, 8]],
             6: [nine[i] for i in [0, 2, 3, 5, 6, 8]], 9: nine}
    assert set(train[4]) < set(train[6]) < set(train[9])
    assert all(not set(ids) & set(test) for ids in train.values())
    return train, test


def execute(command, log):
    command = [str(x) for x in command]
    print("\n[run] " + shlex.join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with log.open("a") as f:
        f.write("\nCOMMAND " + shlex.join(command) + "\n")
        f.flush()
        process = subprocess.Popen(command, cwd=GO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Stage failed ({code}); see {log}. Rerun the same command after fixing it.")


def stage(root, name, command, artifacts):
    marker = root / f".{name}.done.json"
    if marker.exists() and all(p.is_file() and p.stat().st_size > 0 for p in artifacts):
        print(f"[reuse] {root.name}/{name}", flush=True)
        return
    start = time.time()
    execute(command, root / "logs" / f"{name}.log")
    if not all(p.is_file() and p.stat().st_size > 0 for p in artifacts):
        raise RuntimeError(f"Missing output for {name}: {artifacts}")
    write_json(marker, {"command": [str(x) for x in command], "seconds": time.time()-start,
                        "artifacts": [str(p) for p in artifacts]})


def prepare_data(source, data, names, ids, test, resolution):
    split = {"names": names, "train_ids": ids, "test_ids": test,
             "train_names": [names[i] for i in ids], "test_names": [names[i] for i in test],
             "resolution": resolution, "full_scene": True,
             "protocol": "fixed all-view calibration/tracks; selected-view-only triangulation/depth/RGB training"}
    data.mkdir(parents=True, exist_ok=True)
    if (data / "split.json").exists():
        if json.loads((data / "split.json").read_text()) != split:
            raise RuntimeError(f"Existing split differs: {data}")
    for name in ("images", f"images_{resolution}"):
        target, link = source / name, data / name
        if not target.is_dir():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Wrong image link: {link}")
        elif link.exists():
            raise RuntimeError(f"Refuse to overwrite image folder {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    sparse = data / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin"):
        src, dst = source / "sparse/0" / name, sparse / name
        if dst.exists():
            if sha(dst) != sha(src):
                raise RuntimeError(f"Stale calibration: {dst}")
        else:
            shutil.copy2(src, dst)
    # Never copy/link the 55-view points3D or its VGGT depth/init cloud.
    (data / f"sparse_{len(ids)}.txt").write_text("\n".join(map(str, ids))+"\n")
    (data / "sparse_test.txt").write_text("\n".join(map(str, test))+"\n")
    write_json(data / "split.json", split)
    if not all((data / name).exists() for name in
               ("training_geometry.npz", "training_geometry.json", "sparse/0/points3D.ply")):
        training_geometry(source, data, ids, resolution)
    if not (data / "full_scene_manifest.json").exists():
        geom = json.loads((data / "training_geometry.json").read_text())
        write_json(data / "full_scene_manifest.json", {"mode": "sparse_full_scene", "names": names,
                   "train_count": len(ids), "test_count": len(test), "scene_extent": geom["scene_extent"]})


def summarize(run, data_root, iterations):
    rows = []
    for n in (4, 6, 9):
        metric = run / f"v{n}/test/ours_{iterations}/results.json"
        if not metric.exists():
            continue
        values = json.loads(metric.read_text())
        if not all(math.isfinite(values[k]) for k in ("PSNR", "SSIM", "LPIPS")):
            raise RuntimeError(f"Invalid metrics: {metric}")
        split = json.loads((data_root / f"v{n}/split.json").read_text())
        rows.append({"views": n, "test_views": len(split["test_ids"]), "iterations": iterations,
                     **{k: values[k] for k in ("PSNR", "SSIM", "LPIPS")}})
    if not rows:
        return
    with (run / "metrics_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# trex sparse coarse comparison", "",
             "Fixed shared held-out views. Full RGB background. Frozen pretrained VGGT + DropGaussian.",
             "Provided all-view COLMAP poses/track identities; point coordinates re-triangulated using training views only.",
             "This pilot is not the published SynGS/Visual Hull protocol and is not a module ablation.", "",
             "| Training views | Test views | Iterations | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['views']} | {r['test_views']} | {iterations} | {r['PSNR']:.4f} | {r['SSIM']:.6f} | {r['LPIPS']:.6f} |")
    (run / "metrics_comparison.md").write_text("\n".join(lines)+"\n")
    # A true rendered-image comparison, not synthesized or quality-ranked images.
    from PIL import Image, ImageDraw, ImageOps
    first = json.loads((data_root / f"v{rows[0]['views']}/split.json").read_text())
    cols, w, h, label = len(rows)+1, 420, 315, 30
    canvas = Image.new("RGB", (cols*w, (h+label)*len(first["test_ids"])), "white")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(first["test_names"]):
        entries = [("GT "+Path(name).stem,
                    run/f"v{rows[0]['views']}/test/ours_{iterations}/gt/{i:05d}.png")]
        entries += [(f"{r['views']} views", run/f"v{r['views']}/test/ours_{iterations}/renders/{i:05d}.png") for r in rows]
        for col, (title, path) in enumerate(entries):
            with Image.open(path) as im:
                tile = ImageOps.contain(im.convert("RGB"), (w, h))
            x, y = col*w, i*(h+label)
            canvas.paste(tile, (x+(w-tile.width)//2, y+label+(h-tile.height)//2))
            draw.text((x+7, y+8), title, fill="black")
    canvas.save(run / "render_comparison.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, metric, direction in zip(axes, ("PSNR", "SSIM", "LPIPS"), ("higher", "higher", "lower")):
        ax.plot([r["views"] for r in rows], [r[metric] for r in rows], marker="o")
        ax.set_xticks([4, 6, 9])
        ax.set_xlabel("Training views")
        ax.set_title(f"{metric} ({direction} is better)")
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(run / "metrics_comparison.png", dpi=180)
    plt.close(fig)
    print("\n"+"\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=GO / "data/LLFF/trex_fullscene")
    p.add_argument("--run-tag", default="trex_sparse469_v1")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--resolution", type=int, choices=[2, 4, 8], default=4)
    p.add_argument("--max-gaussians", type=int, default=500000)
    p.add_argument("--max-new-points", type=int, default=60000)
    p.add_argument("--drop-rate", type=float, default=.2)
    p.add_argument("--plan", action="store_true", help="read-only plan, no training")
    p.add_argument("--prepare-only", action="store_true", help="CPU splits and selected-view triangulation; no VGGT/training")
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    if Path(a.run_tag).name != a.run_tag or a.run_tag in ("", ".", ".."):
        raise ValueError("run-tag must be one directory name")
    if a.iterations < 1000 or a.max_gaussians < 1000 or a.max_new_points < 9 or not 0 <= a.drop_rate < 1:
        raise ValueError("Invalid iteration, point budget or drop rate")
    source = a.source.resolve()
    meta = json.loads((source / "full_scene_manifest.json").read_text())
    names = sorted(meta["names"], key=lambda name: Path(name).stem)
    if len({Path(name).stem for name in names}) != len(names):
        raise ValueError("Ambiguous image names")
    train, test = split_views(len(names))
    run, data_root = GO / "output" / a.run_tag, GO / "data/LLFF" / a.run_tag
    weights = GO.parent / "vggt/models/model.pt"
    plan = {"source": str(source), "output": str(run), "data": str(data_root),
            "train_ids": {str(n): ids for n, ids in train.items()}, "test_ids": test, "test_names": [names[i] for i in test],
            "iterations": a.iterations, "resolution": a.resolution, "drop_rate": a.drop_rate,
            "pretrained_vggt": str(weights), "fine_tune": False, "LOO_LoRA_repair": False,
            "background": "real full RGB", "save": "one final PLY/group; shared-test PNGs; no video/optimizer snapshots",
            "calibration": "provided all-view COLMAP poses and track identities",
            "geometry": "training-only DLT, training-only frozen VGGT depth scaling and consistency",
            "mono_depth_loss_weight": 0., "max_gaussians_soft_cap": a.max_gaussians,
            "max_new_points": a.max_new_points}
    if a.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if a.summarize_only:
        summarize(run, data_root, a.iterations)
        return
    if not a.prepare_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Enable the GPU and rerun the command")
        if shutil.disk_usage(GO).free < 3*1024**3:
            raise RuntimeError("Need at least 3 GiB free disk")
        if not weights.is_file():
            raise FileNotFoundError(weights)
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another runner is active: {run}")
        source_files = [source / "full_scene_manifest.json", source / "sparse/0/cameras.bin",
                        source / "sparse/0/images.bin"]
        used = sorted(set(test) | set(train[9]))
        source_files += [source / f"images_{a.resolution}" / names[i] for i in used]
        config = {**plan, "source_hashes": {str(path): sha(path) for path in source_files},
                  "code_hashes": {name: sha(GO/name) for name in
                      ("tools/run_sparse_469.py", "tools/sparse_scene_init.py", "train_gs.py", "render.py",
                       "scene/dataset_readers.py", "scene/cameras.py", "utils/camera_utils.py",
                       "gaussian_renderer/__init__.py", "scene/gaussian_model.py")}}
        # Separate CPU preparation fingerprint allows GPU execution after --prepare-only.
        config_file = run / "run_config.json"
        if config_file.exists():
            if json.loads(config_file.read_text()) != config:
                raise RuntimeError("Config/code/source changed; choose a NEW --run-tag to preserve the old experiment")
        else:
            write_json(config_file, config)
        if not a.prepare_only:
            weight_config = {"pretrained_weights": str(weights), "sha256": sha(weights)}
            previous = run / "vggt_weight.json"
            if previous.exists() and json.loads(previous.read_text()) != weight_config:
                raise RuntimeError("VGGT weights changed; use a new run tag")
            write_json(previous, weight_config)
        for n in (4, 6, 9):
            data, out = data_root / f"v{n}", run / f"v{n}"
            prepare_data(source, data, names, train[n], test, a.resolution)
            if a.prepare_only:
                continue
            out.mkdir(parents=True, exist_ok=True)
            stage(out, "depth_init", [PY, GO/"tools/sparse_scene_init.py", "--source", source,
                  "--data", data, "--weights", weights, "--resolution", a.resolution,
                  "--max-new-points", a.max_new_points], [data/"scene_sparse_vggt.ply", data/"scene_init_report.json"])
            final = out / f"point_cloud/iteration_{a.iterations}/point_cloud.ply"
            stage(out, "train", [PY, "train_gs.py", "--full_scene", "-s", data, "-m", out,
                  "-r", a.resolution, "--sparse_view_num", n, "--init_pcd_name", "scene_sparse_vggt",
                  "--iterations", a.iterations, "--position_lr_max_steps", a.iterations,
                  "--densify_until_iter", int(.6*a.iterations), "--max_num_splats", a.max_gaussians,
                  "--save_iterations", a.iterations, "--test_iterations", -1,
                  "--mono_depth_weight", 0, "--use_drop", "--drop_rate", a.drop_rate,
                  "--drop_schedule", "linear"], [final, out/"cfg_args"])
            # input.ply duplicates a large initialized Gaussian model. Only remove that
            # regenerable copy in THIS run after successful final-model verification.
            (out / "input.ply").unlink(missing_ok=True)
            images = out / f"test/ours_{a.iterations}"
            artifacts = [images/"results.json"]
            artifacts += [images/f"{sub}/{i:05d}.png" for sub in ("renders", "gt") for i in range(len(test))]
            stage(out, "evaluate", [PY, "render.py", "--full_scene", "-s", data, "-m", out,
                  "-r", a.resolution, "--sparse_view_num", n, "--init_pcd_name", "scene_sparse_vggt",
                  "--iteration", a.iterations, "--skip_train", "--skip_all", "--not_generate_video"], artifacts)
            write_json(images/"camera_names.json", {f"{i:05d}.png": names[idx] for i, idx in enumerate(test)})
            summarize(run, data_root, a.iterations)
        if a.prepare_only:
            print("CPU preparation complete; no VGGT inference or training started.", flush=True)
        else:
            write_json(run/"COMPLETE.json", {"views": [4, 6, 9], "iterations": a.iterations,
                                           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"\nCOMPLETE: {run}", flush=True)


if __name__ == "__main__":
    main()
