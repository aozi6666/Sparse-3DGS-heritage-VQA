#!/usr/bin/env python3
"""Matched project-3DGS baseline: disable VGGT and Drop, retain other controls."""
import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from run_sparse_469 import GO, stage
from sparse_scene_init import sha, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def set_option(command, option, value):
    result = list(command)
    index = result.index(option)
    result[index+1] = str(value)
    return result


def baseline_command(original, data, out):
    """Reuse the successful command; change only dataset/output/init and Drop."""
    command = [sys.executable] + original[1:]
    for key, value in (("-s", data), ("-m", out), ("--init_pcd_name", "origin")):
        command = set_option(command, key, value)
    if "--use_drop" in command:
        command.remove("--use_drop")
    for key in ("--drop_rate", "--drop_schedule"):
        if key in command:
            index = command.index(key)
            del command[index:index+2]
    assert "--use_drop" not in command
    assert command[command.index("--init_pcd_name")+1] == "origin"
    return command


def link_input(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Wrong input link: {target}")
    elif target.exists():
        raise RuntimeError(f"Refuse to replace existing input: {target}")
    else:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def check_reference(reference):
    from plyfile import PlyData
    config = read_json(reference / "run_config.json")
    if not (reference / "COMPLETE.json").is_file():
        raise RuntimeError("Reference VGGT+Drop run is not complete")
    for name, expected in config["code_hashes"].items():
        if sha(GO/name) != expected:
            raise RuntimeError(f"Reference training code changed: {name}. Matched comparison requires review.")
    for name, expected in config["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError(f"Reference input changed: {name}")
    hashes = {str(reference/"run_config.json"): sha(reference/"run_config.json")}
    data_root = Path(config["data"])
    for n in (4, 6, 9):
        data, model = data_root/f"v{n}", reference/f"v{n}"
        split = read_json(data/"split.json")
        assert split["train_ids"] == config["train_ids"][str(n)]
        assert split["test_ids"] == config["test_ids"]
        assert not set(split["train_ids"]) & set(split["test_ids"])
        assert len(split["train_ids"]) == n
        geometry_sha = sha(data/"training_geometry.npz")
        # The frozen-depth sidecars saved at reference training time fingerprint
        # the original base geometry; require that exact geometry now.
        for name in split["train_names"]:
            sidecar = data/"zoe_depth"/f"{Path(name).stem}.json"
            assert read_json(sidecar)["signature"]["geometry_sha256"] == geometry_sha, sidecar
        with np.load(data/"training_geometry.npz", allow_pickle=False) as geometry:
            vertices = PlyData.read(str(data/"sparse/0/points3D.ply"))["vertex"]
            xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
            rgb = np.column_stack([vertices[k] for k in ("red", "green", "blue")])
            assert np.array_equal(xyz, geometry["xyz"]), "Base PLY does not match original geometry"
            assert np.array_equal(rgb, geometry["rgb"]), "Base PLY colors changed"
        for kind in ("train", "evaluate"):
            command = read_json(model/f".{kind}.done.json")["command"]
            assert Path(command[command.index("-s")+1]) == data
            assert int(command[command.index("--sparse_view_num")+1]) == n
        training = read_json(model/".train.done.json")["command"]
        assert "--use_drop" in training
        assert float(training[training.index("--mono_depth_weight")+1]) == 0
        assert int(training[training.index("--iterations")+1]) == config["iterations"]
        for relative in ("split.json", "training_geometry.npz", "training_geometry.json",
                         "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.ply"):
            hashes[str(data/relative)] = sha(data/relative)
        for relative in (".train.done.json", ".evaluate.done.json",
                         f"test/ours_{config['iterations']}/results.json"):
            hashes[str(model/relative)] = sha(model/relative)
    return config, hashes


def prepare_baseline(original, target, n, resolution):
    target.mkdir(parents=True, exist_ok=True)
    if (target/"zoe_depth").exists() or (target/"scene_sparse_vggt.ply").exists():
        raise RuntimeError(f"Baseline must not contain VGGT depth/cloud: {target}")
    for name in ("images", f"images_{resolution}", "split.json", "full_scene_manifest.json",
                 f"sparse_{n}.txt", "sparse_test.txt", "training_geometry.npz", "training_geometry.json",
                 "sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.ply"):
        link_input(original/name, target/name)
    write_json(target/"baseline_manifest.json", {"variant": "project_3DGS_no_VGGT_no_Drop",
               "base_geometry_source": str(original/"training_geometry.npz"),
               "VGGT_used": False, "Drop_used": False, "depth_loss_used": False,
               "provided_all_view_calibration_and_tracks": True,
               "official_upstream_3DGS_reproduction": False})


def metrics(path):
    value = read_json(path)
    if not all(math.isfinite(value[k]) for k in ("PSNR", "SSIM", "LPIPS")):
        raise RuntimeError(f"Invalid metrics: {path}")
    return {k: value[k] for k in ("PSNR", "SSIM", "LPIPS")}


def compare(run, reference, iterations):
    rows, paired = [], []
    for n in (4, 6, 9):
        out = run/f"v{n}/test/ours_{iterations}"
        if not (out/"results.json").exists():
            continue
        base = metrics(out/"results.json")
        combo = metrics(reference/f"v{n}/test/ours_{iterations}/results.json")
        paired.append((n, base, combo))
        for method, values in (("project_3DGS_baseline", base), ("VGGT_plus_Drop", combo)):
            rows.append({"views": n, "method": method, "iterations": iterations, **values})
    if not rows:
        return
    with (run/"baseline_vs_vggt_drop.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Project 3DGS baseline versus VGGT + Drop", "",
            "Same split, provided COLMAP calibration/tracks, base geometry, resolution, trainer and training steps.",
            "Baseline disables BOTH VGGT enhancement and Drop. All other project training controls retained.",
            "This is a matched project baseline, not an independent official-upstream 3DGS reproduction.", "",
            "| Views | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
            "|---:|---|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['views']} | {row['method']} | {row['PSNR']:.4f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} |")
    text += ["", "## VGGT + Drop minus baseline (signed differences)", "",
             "| Views | ΔPSNR (+ better) | ΔSSIM (+ better) | ΔLPIPS (- better) |",
             "|---:|---:|---:|---:|"]
    for n, base, combo in paired:
        text.append(f"| {n} | {combo['PSNR']-base['PSNR']:+.4f} | {combo['SSIM']-base['SSIM']:+.6f} | {combo['LPIPS']-base['LPIPS']:+.6f} |")
    text += ["", "This single-seed scene comparison measures the combined effect; it cannot separate VGGT and Drop contributions."]
    (run/"baseline_vs_vggt_drop.md").write_text("\n".join(text)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, key in zip(axes, ("PSNR", "SSIM", "LPIPS")):
        ax.plot([p[0] for p in paired], [p[1][key] for p in paired], "o-", label="3DGS project baseline")
        ax.plot([p[0] for p in paired], [p[2][key] for p in paired], "s-", label="VGGT + Drop")
        ax.set_title(key+(" (lower better)" if key == "LPIPS" else " (higher better)"))
        ax.set_xlabel("Training views")
        ax.set_xticks([4, 6, 9])
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run/"metrics_comparison.png", dpi=180)
    plt.close(fig)
    from PIL import Image, ImageDraw, ImageOps
    w, h, label = 420, 315, 30
    overview = Image.new("RGB", (3*w, len(paired)*(h+label)), "white")
    for row, (n, _, _) in enumerate(paired):
        base_dir = run/f"v{n}/test/ours_{iterations}"
        ref_dir = reference/f"v{n}/test/ours_{iterations}"
        mapping = read_json(ref_dir/"camera_names.json")
        files = sorted(mapping)
        sheet = Image.new("RGB", (3*w, len(files)*(h+label)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, filename in enumerate(files):
            paths = [("GT", ref_dir/"gt"/filename), ("3DGS baseline", base_dir/"renders"/filename),
                     ("VGGT + Drop", ref_dir/"renders"/filename)]
            for col, (name, path) in enumerate(paths):
                with Image.open(path) as im:
                    tile = ImageOps.contain(im.convert("RGB"), (w, h))
                x, y = col*w, i*(h+label)
                sheet.paste(tile, (x+(w-tile.width)//2, y+label+(h-tile.height)//2))
                draw.text((x+6, y+8), f"{n} views | {name} | {Path(mapping[filename]).stem}", fill="black")
        sheet.save(run/f"v{n}/render_comparison.png")
        # Fixed first held-out camera, selected before looking at quality.
        overview.paste(sheet.crop((0, 0, 3*w, h+label)), (0, row*(h+label)))
    overview.save(run/"render_comparison.png")
    print("\n"+"\n".join(text), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tag", default="trex_sparse469_v1")
    parser.add_argument("--run-tag", default="trex_sparse469_baseline_v1")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    for tag in (args.reference_tag, args.run_tag):
        if Path(tag).name != tag or tag in ("", ".", ".."):
            raise ValueError("Tags must be one directory name")
    if args.run_tag == args.reference_tag:
        raise ValueError("Baseline output must be separate from reference")
    reference, run = GO/"output"/args.reference_tag, GO/"output"/args.run_tag
    config, hashes = check_reference(reference)
    iterations, resolution = config["iterations"], config["resolution"]
    data_root = GO/"data/LLFF"/args.run_tag
    commands = {}
    for n in (4, 6, 9):
        commands[str(n)] = {kind: baseline_command(read_json(reference/f"v{n}/.{kind}.done.json")["command"],
                              data_root/f"v{n}", run/f"v{n}") for kind in ("train", "evaluate")}
    fingerprint = {"variant": "project_3DGS_baseline_no_VGGT_no_Drop", "reference": str(reference),
                   "iterations": iterations, "resolution": resolution, "commands": commands,
                   "reference_hashes": hashes, "code_hashes": {
                       "tools/run_sparse_469_baseline.py": sha(Path(__file__)),
                       "tools/run_sparse_469.py": sha(GO/"tools/run_sparse_469.py")},
                   "test_ids": config["test_ids"], "train_ids": config["train_ids"],
                   "official_upstream_reproduction": False}
    if args.plan:
        print(json.dumps(fingerprint, indent=2))
        return
    if args.summarize_only:
        compare(run, reference, iterations)
        return
    if not args.prepare_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Enable GPU before starting training")
        if shutil.disk_usage(GO).free < 2*1024**3:
            raise RuntimeError("Need at least 2 GiB free disk")
    run.mkdir(parents=True, exist_ok=True)
    with (run/".runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another baseline runner is active: {run}")
        saved = run/"run_config.json"
        if saved.exists() and read_json(saved) != fingerprint:
            raise RuntimeError("Inputs/code/reference changed; choose a new --run-tag")
        write_json(saved, fingerprint)
        for n in (4, 6, 9):
            data, out = data_root/f"v{n}", run/f"v{n}"
            prepare_baseline(Path(config["data"])/f"v{n}", data, n, resolution)
            if args.prepare_only:
                continue
            out.mkdir(parents=True, exist_ok=True)
            final = out/f"point_cloud/iteration_{iterations}/point_cloud.ply"
            stage(out, "train", commands[str(n)]["train"], [final, out/"cfg_args"])
            (out/"input.ply").unlink(missing_ok=True)
            images = out/f"test/ours_{iterations}"
            artifacts = [images/"results.json"]
            artifacts += [images/f"{kind}/{i:05d}.png" for kind in ("renders", "gt")
                          for i in range(len(config["test_ids"]))]
            stage(out, "evaluate", commands[str(n)]["evaluate"], artifacts)
            write_json(images/"camera_names.json", read_json(reference/f"v{n}/test/ours_{iterations}/camera_names.json"))
            compare(run, reference, iterations)
        if args.prepare_only:
            print("Baseline CPU preparation complete. NO training or VGGT inference started.", flush=True)
        else:
            write_json(run/"COMPLETE.json", {"baseline_views": [4, 6, 9], "reference": str(reference),
                       "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"\nCOMPLETE: {run}", flush=True)


if __name__ == "__main__":
    main()
