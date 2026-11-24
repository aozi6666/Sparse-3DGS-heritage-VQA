"""Populate a prepared dataset with regular files, reusing read-only media inodes."""
import filecmp
import os
import shutil
import tempfile
from pathlib import Path


def _copy_file(source, destination):
    if Path(source).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _matches(source, destination):
    if destination.is_symlink():
        return False
    if source.is_dir():
        return (destination.is_dir() and
                {p.name for p in source.iterdir()} == {p.name for p in destination.iterdir()} and
                all(_matches(p, destination / p.name) for p in source.iterdir()))
    return destination.is_file() and (source.samefile(destination) or
                                      filecmp.cmp(source, destination, shallow=False))


def materialize(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if destination.exists():
        if _matches(source, destination):
            return
        raise ValueError(f'Existing prepared input differs: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / 'input'
        if source.is_dir():
            shutil.copytree(source, staged, copy_function=_copy_file)
        else:
            _copy_file(source, staged)
        staged.rename(destination)
