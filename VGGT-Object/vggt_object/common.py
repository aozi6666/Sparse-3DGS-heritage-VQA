import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config(path=None, experiment=None, phase=None):
    cfg = yaml.safe_load((PROJECT/'configs/base.yaml').read_text())
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()))
    if experiment:
        cfg.update(yaml.safe_load((PROJECT/'configs/experiments.yaml').read_text())[experiment])
        cfg['experiment'] = experiment
    if phase:
        cfg.update(yaml.safe_load((PROJECT/'configs/phases.yaml').read_text())[phase])
        cfg['phase'] = phase
    for key in ('base_checkpoint','vggt_source','manifest'):
        p = Path(cfg[key])
        cfg[key] = str((REPO/p if not p.is_absolute() else p).resolve())
    return cfg


def setup_vggt(cfg):
    source = Path(cfg['vggt_source'])
    if not (source/'vggt/models/vggt.py').is_file():
        raise FileNotFoundError(f'Existing local VGGT required: {source}')
    sys.path.insert(0,str(source))
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'


def space_guard(min_gib=5):
    if shutil.disk_usage(PROJECT).free < min_gib*1024**3:
        raise RuntimeError(f'Need {min_gib} GiB free for atomic checkpoints; no files removed')


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this command; use check for CPU-only verification')
    major,minor=torch.cuda.get_device_capability()
    arch=f'{major}{minor}';compiled=torch.cuda.get_arch_list()
    if major>=10 and not any(a.startswith('sm_'+arch) or a.startswith('compute_'+arch) for a in compiled):
        raise RuntimeError(f'GPU {torch.cuda.get_device_name()} needs sm_{arch}; current PyTorch {torch.__version__} '
                           f'CUDA {torch.version.cuda} has {compiled}. Use a compatible PyTorch/CUDA build before training; environment was not changed automatically.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU support required; no silent precision downgrade')


def environment(cfg):
    def git(*args):
        p=subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True)
        return p.stdout.strip() if not p.returncode else None
    return dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,
                branch=git('branch','--show-current'),repository_commit=git('rev-parse','HEAD'),
                vggt_source=cfg['vggt_source'],vggt_code_sha256=digest(Path(cfg['vggt_source'])/'vggt/models/vggt.py'))
