"""User-started, same-filesystem hardlink reuse; no network or torch import."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import tomllib
from local_assets import PROJECT, branch_guard, require, save_report

SOURCE = Path('/root/autodl-tmp/Mage_ViT/mage/lib/python3.11/site-packages')
VERSION = '2.11.0+cu128'


def ensure_no_sync():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if args and Path(args[0]).name=='uv' and 'sync' in args and (entry/'cwd').resolve()==PROJECT:
                raise RuntimeError('当前uv sync仍在运行；请先在安装终端按Ctrl+C，再运行复用命令。')
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass


def main():
    branch_guard();ensure_no_sync()
    require(sys.version_info[:2]==(3,11),'需要新环境的Python3.11')
    require(Path(sys.prefix).resolve()==(PROJECT/'.venv').resolve(),'请使用Project/.venv/bin/python')
    lock=tomllib.loads((PROJECT/'uv.lock').read_text())
    expected=next(x for x in lock['package'] if x['name']=='torch')['version']
    require(expected==VERSION,'锁文件torch版本改变，停止')
    source_info=SOURCE/('torch-'+VERSION+'.dist-info')
    dist=md.Distribution.at(source_info)
    require(dist.version==VERSION,'旧torch版本不匹配')
    dest=PROJECT/'.venv/lib/python3.11/site-packages'
    dest.mkdir(parents=True,exist_ok=True)
    require(SOURCE.stat().st_dev==dest.stat().st_dev,'不在同一文件系统，不能硬链接，停止')
    linked=0
    for name in ('torch','torchgen','functorch',source_info.name):
        top=SOURCE/name
        require(top.is_dir(),'旧安装缺组件：'+name)
        for src in [top,*sorted(top.rglob('*'))]:
            if '__pycache__' in src.parts or src.suffix=='.pyc': continue
            dst=dest/src.relative_to(SOURCE)
            if src.is_symlink():
                value=os.readlink(src)
                require(not os.path.isabs(value),'旧包包含绝对符号链接，停止核查')
                if dst.is_symlink():require(os.readlink(dst)==value,'已有符号链接冲突')
                else:
                    require(not dst.exists(),'目标冲突：'+str(dst));dst.symlink_to(value)
            elif src.is_dir():dst.mkdir(parents=True,exist_ok=True)
            elif dst.exists():require(os.path.samefile(src,dst),'目标已存在不同文件，保留不覆盖：'+str(dst))
            else:os.link(src,dst);linked+=1
    for ep in dist.entry_points:
        if ep.group!='console_scripts':continue
        path=PROJECT/'.venv/bin'/ep.name
        script=f'#!{PROJECT}/.venv/bin/python\nfrom importlib.metadata import distribution\nep=next(e for e in distribution("torch").entry_points if e.group=="console_scripts" and e.name=={ep.name!r})\nraise SystemExit(ep.load()())\n'
        if path.exists():require(path.read_text()==script,'已有命令冲突，保留：'+str(path))
        else:
            with path.open('x') as f:f.write(script)
            path.chmod(0o755)
    require(md.Distribution.at(dest/source_info.name).version==VERSION,'新环境metadata校验失败')
    report=dict(version=VERSION,source=str(SOURCE),destination=str(dest),new_hardlinked_files=linked,
        mode='hardlink_not_sys_path_or_symlink',downloads=0,source_files_modified=False)
    path=save_report(report,'local_torch_reuse')
    print('本地torch已硬链接接入，跳过782MiB下载；旧环境内容不改。记录：',path)


if __name__=='__main__':main()
