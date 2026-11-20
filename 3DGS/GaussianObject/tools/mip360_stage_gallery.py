"""Full-background training-stage galleries; execute on the AutoDL GPU host."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', default='/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
    cli.add_argument('--out', required=True)
    cli.add_argument('--scenes', nargs='+', default=['bonsai', 'garden', 'kitchen'])
    cli.add_argument('--steps', type=int, default=10000)
    cli.add_argument('--check', action='store_true')
    c = cli.parse_args()
    go = Path(c.root) / '3DGS/GaussianObject'
    os.chdir(go)
    sys.path.insert(0, str(go))
    import torch
    import numpy as np
    from PIL import Image
    import lpips
    import train_gs as train
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.full_scene import full_scene_options
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    out = Path(c.out).resolve()
    free = shutil.disk_usage(go).free / 1024**3
    print(f'Free disk: {free:.2f} GiB; CUDA: {torch.cuda.is_available()}', flush=True)
    for name in c.scenes:
        data = go / 'data/mip360' / name
        for item in ['images', 'sparse/0/cameras.bin', 'sparse/0/images.bin', 'sparse/0/points3D.bin']:
            if not (data / item).exists():
                raise FileNotFoundError(data / item)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    # VGG LPIPS, matching the project render.py backbone. No automatic download.
    weights = Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth'
    if not weights.exists():
        raise FileNotFoundError(f'Existing VGG weight required; will not download: {weights}')
    if c.check:
        print('Preflight passed; no training started.', flush=True)
        return
    if free < 4:
        raise RuntimeError('Need at least 4 GiB free; no existing files will be deleted')
    out.mkdir(parents=True, exist_ok=True)
    metric = lpips.LPIPS(net='vgg').cuda().eval()
    for name in c.scenes:
        leaf = out / name
        if (leaf / 'complete.json').exists():
            print(f'[reuse] completed {name}', flush=True)
            continue
        if leaf.exists():
            raise RuntimeError(f'Partial run preserved: {leaf}. Use a NEW --out path to restart.')
        if shutil.disk_usage(go).free < 4 * 1024**3:
            raise RuntimeError('Free disk below 4 GiB; stopping before next scene')
        leaf.mkdir()
        parser = argparse.ArgumentParser()
        mp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
        a = parser.parse_args([])
        a.source_path = str(go / 'data/mip360' / name)
        a.model_path = str(leaf / 'model')
        a.full_scene = True
        a.eval = True
        a.resolution = 4
        a.sh_degree = 2
        a.iterations = c.steps
        a.position_lr_max_steps = c.steps
        a.densify_until_iter = min(5000, c.steps // 2)
        a.max_num_splats = 200000
        a.mono_depth_weight = 0.0
        a.mono_loss_type = 'mid'
        a.lambda_t_norm = 0.0
        a.use_dust3r = False
        a.sparse_view_num = -1
        a.init_pcd_name = 'origin'
        a.transform_the_world = False
        a.use_drop = False
        a.drop_rate = 0.0
        a.drop_schedule = 'linear'
        full_scene_options(a)
        records, buffers = [], {}
        camera_name = None

        def setup(dataset):
            Path(dataset.model_path).mkdir()
            (Path(dataset.model_path) / 'cfg_args').write_text(str(argparse.Namespace(**vars(dataset))))
            return None  # no TensorBoard image/event duplication

        @torch.no_grad()
        def capture(tb, step, l1, loss, lf, elapsed, testing, scene, render, render_args):
            nonlocal camera_name
            if step < 1000 or step % 100:
                return
            if shutil.disk_usage(go).free < 2 * 1024**3:
                raise RuntimeError('Free disk below 2 GiB; aborting safely')
            view = scene.getTestCameras()[0]
            camera_name = view.image_name
            rgb = render(view, scene.gaussians, *render_args)['render'].clamp(0, 1)
            arr = (rgb.permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
            # Metrics use the actual 8-bit exported image, full frame including background.
            saved = torch.from_numpy(arr.copy()).permute(2, 0, 1).cuda().float() / 255
            gt = view.original_image[:3].cuda().clamp(0, 1)
            rec = dict(step=step, camera=camera_name,
                       PSNR=psnr(saved, gt).mean().item(), SSIM=ssim(saved, gt).mean().item())
            records.append(rec)
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format='PNG')
            buffers[step] = b.getvalue()
            if not (leaf / 'GT.png').exists():
                gtarr = (gt.permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
                Image.fromarray(gtarr).save(leaf / 'GT.png')
            print(f'[gallery] {name} step={step} PSNR={rec["PSNR"]:.3f}', flush=True)

        train.prepare_output_and_logger = setup
        train.training_report = capture
        safe_state(False)  # fixed project seed (0)
        train.training(a, mp.extract(a), op.extract(a), pp.extract(a), [], [c.steps], [], None, -1)
        ordered = sorted(records, key=lambda r: r['PSNR'])
        choices = []
        for i, low in enumerate(ordered):
            window = [r for r in ordered[i:] if r['PSNR'] - low['PSNR'] <= 2.0]
            if len(window) >= 9 and window[-1]['PSNR'] - window[0]['PSNR'] >= .5:
                indices = np.linspace(0, len(window) - 1, 9).round().astype(int)
                pick = [window[j] for j in indices]
                choices.append(pick)
        # Prefer later stages among valid windows; never modify pixels or metrics to fit.
        selected = max(choices, key=lambda p: sum(r['step'] for r in p)) if choices else ordered[-9:]
        selected = sorted(selected, key=lambda r: r['PSNR'], reverse=True)
        gt = torch.from_numpy(np.array(Image.open(leaf / 'GT.png')).copy()).permute(2, 0, 1).cuda().float() / 255
        for rank, rec in enumerate(selected, 1):
            dest = leaf / f'{rank:02d}_iter{rec["step"]:05d}.png'
            dest.write_bytes(buffers[rec['step']])
            arr = np.array(Image.open(dest)).copy()
            image = torch.from_numpy(arr).permute(2, 0, 1).cuda().float() / 255
            with torch.no_grad():
                rec['LPIPS'] = metric(image[None], gt[None], normalize=True).item()
            rec['file'] = dest.name
        span = selected[0]['PSNR'] - selected[-1]['PSNR']
        with (leaf / 'metrics.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=['file', 'step', 'camera', 'PSNR', 'SSIM', 'LPIPS'])
            w.writeheader()
            w.writerows(selected)
        (leaf / 'candidates.json').write_text(json.dumps(records, indent=2))
        report = dict(scene=name, camera=camera_name, count=len(selected), psnr_span=span,
                      target_met=bool(choices), protocol='training-stage illustration; full RGB background; first every-8th held-out camera; SfM uses existing calibration; not 9 methods; not SynGS evaluation',
                      parameters=vars(a), lpips='vgg, normalize=True', seed=0)
        (leaf / 'complete.json').write_text(json.dumps(report, indent=2))
        print(f'{name}: {len(selected)} PNGs; PSNR span={span:.3f} dB; target_met={bool(choices)}', flush=True)
        del buffers, metric
        torch.cuda.empty_cache()
        metric = lpips.LPIPS(net='vgg').cuda().eval()


if __name__ == '__main__':
    main()
