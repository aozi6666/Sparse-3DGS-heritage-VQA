"""Header-only model validation and explicit asset linking. No torch/network/GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
SOURCE = REPO / 'VLM/mage_vl/models'
BRANCH = 'feat/project-video-deme'
WIDTHS = dict(F64=8, F32=4, F16=2, BF16=2, I64=8, I32=4, I16=2, I8=1, U8=1, BOOL=1)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def branch_guard():
    name = subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    require(name == BRANCH, f'请先创建并进入{BRANCH}；当前{name}，不会自动切分支')


def sha_small(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safetensors_header(path):
    """Read metadata, not 10GB tensor payloads. Validate lengths, shapes and spans."""
    with path.open('rb') as f:
        raw = f.read(8)
        require(len(raw) == 8, '权重文件头缺失：'+str(path))
        length = struct.unpack('<Q',raw)[0]
        require(0 < length < 32*1024**2, '非法header长度：'+str(path))
        header = json.loads(f.read(length))
    tensors = {k:v for k,v in header.items() if k != '__metadata__'}
    require(bool(tensors), '无张量：'+str(path))
    spans = []
    for name, entry in tensors.items():
        start, end = entry['data_offsets']
        require(0 <= start <= end, '非法偏移：'+name)
        require(entry['dtype'] in WIDTHS, '未覆盖的数据类型：'+entry['dtype'])
        require(end-start == math.prod(entry['shape'])*WIDTHS[entry['dtype']], '张量形状/字节数不符：'+name)
        spans.append((start,end))
    spans.sort()
    require(spans[0][0] == 0 and all(a[1] == b[0] for a,b in zip(spans,spans[1:])), '张量区间有缺口/重叠')
    payload = spans[-1][1]
    require(8+length+payload == path.stat().st_size, '文件截断或尾部多余：'+str(path))
    return tensors, dict(file=path.name, bytes=path.stat().st_size, payload_bytes=payload, tensors=len(tensors))


def validate(model):
    model = Path(model).resolve(strict=True)
    index = json.loads((model/'model.safetensors.index.json').read_text())
    actual, shards = {}, []
    for name in sorted(set(index['weight_map'].values())):
        require(Path(name).name == name, '分片路径非法')
        tensors, info = safetensors_header(model/name)
        shards.append(info)
        for key in tensors:
            require(key not in actual, '分片重复张量：'+key)
            actual[key] = name
    require(actual == index['weight_map'], '权重索引与实际分片张量不一致')
    require(sum(s['payload_bytes'] for s in shards) == index['metadata']['total_size'], '索引总大小不符')
    _, gate = safetensors_header(model/'streammind_gate.safetensors')
    required = ['config.json','generation_config.json','configuration_mage_vl.py','modeling_mage_vl.py',
        'processing_mage_vl.py','codec_video_processing_mage_vl.py','video_processing_mage_vl.py',
        'tokenizer.json','tokenizer_config.json','preprocessor_config.json','video_preprocessor_config.json',
        'chat_template.jinja','streammind_gate.py','special_tokens_map.json','vocab.json','merges.txt']
    for name in required:
        require((model/name).is_file() and (model/name).stat().st_size > 0, '模型所需文件缺失：'+name)
    return dict(model_path=str(model), structural_check_passed=True, shards=shards, gate=gate,
        total_main_tensors=len(actual), payload_bytes=index['metadata']['total_size'],
        configuration_sha256={name:sha_small(model/name) for name in required},
        note='Header/index/offset/length validation only; no full-payload checksum, model weight load, GPU or download.')


def link_model():
    target = PROJECT/'models/mage-vl'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        require(target.resolve() == SOURCE.resolve(), '已有模型软链接指向不同位置，保留不覆盖')
    else:
        require(not target.exists(), '目标已存在，保留不覆盖：'+str(target))
        target.symlink_to(os.path.relpath(SOURCE,target.parent),target_is_directory=True)
    return dict(path=str(target), resolves_to=str(target.resolve()), kind='temporary_symlink')


def link_tools():
    """Hardlinks keep a standalone Project filename even if external source is removed."""
    result = {}
    for name in ('ffmpeg','ffprobe'):
        source = Path('/root/autodl-tmp/VLMEvalKit/ffmpeg_linux')/name
        require(source.is_file(), '缺少现有工具，请报告，不自动安装：'+str(source))
        target = PROJECT/'tools/bin'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            require(not target.is_symlink() and os.path.samefile(source,target), '工具目标冲突，保留不覆盖：'+str(target))
        else:
            require(source.stat().st_dev == target.parent.stat().st_dev, '无法省盘硬链接，需单独安排工具安装')
            os.link(source,target)
        result[name] = dict(path=str(target), kind='hardlink', bytes=target.stat().st_size,
            version=subprocess.check_output([str(target),'-version'],text=True).splitlines()[0])
    return result


def save_report(report, stem):
    directory=PROJECT/'storage/audits';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    with path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--link',action='store_true',help='After validating, create the explicit model symlink and ffmpeg hardlinks')
    args=parser.parse_args();branch_guard()
    report=validate(SOURCE)
    if args.link:
        report['model_link']=link_model();report['tools']=link_tools()
    path=save_report(report,'model_structure')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('检查记录：',path)


if __name__ == '__main__': main()
