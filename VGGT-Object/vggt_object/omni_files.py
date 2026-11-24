"""Exact Omni RGB/depth lookup; no Torch or GPU side effects."""
from pathlib import Path

def checked(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:raise ValueError(f'Missing/empty upload: {path}')
    return path

def exact_depth(root,stem):
    folder=Path(root)/'depths'
    matches=[folder/(stem+suffix) for suffix in ('_depth.exr','_depth.npy','.exr','.npy') if (folder/(stem+suffix)).is_file()]
    if len(matches)!=1:raise ValueError(f'Need one exact depth for {stem}; found {matches}')
    return checked(matches[0])

def valid_depth(array):
    import numpy as np
    return np.isfinite(array)&(array>1e-6)&(array<65504.)
