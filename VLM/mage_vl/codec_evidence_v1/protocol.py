"""Frozen, resumable protocol. Importing this module never imports torch."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

PROJECT = Path(os.environ.get('CODEC_EVIDENCE_PROJECT', '/root/autodl-tmp/Sparse-3DGS-heritage-VQA'))
CODE = Path(__file__).resolve().parent
MODEL = PROJECT / 'VLM/mage_vl/models'
OLD = PROJECT / 'VLM/evaluation/results/practical_mage_v1/24_codec_full714_32768_v1'
OUT = PROJECT / 'VLM/evaluation/results/practical_mage_v1/codec_evidence_v1'
BLIP = Path(os.environ.get('BLIP_MODEL_DIR', '/root/autodl-tmp/models/blip-itm-large-coco'))
BRANCH = 'feat/air-visual-evidence'
PRIMARY = 'codec_complete32768'
CODEC = dict(engine='hevc', target_canvas=32, patch=16, group_size=32,
             images_per_group=4, spatial_mask_mode='off', max_pixels=150000)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    # Deliberately identical to the historical practical_common.signature.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def seal(value):
    return dict(value, record_sha256=digest(value))


def unseal(value):
    x = dict(value)
    h = x.pop('record_sha256', None)
    require(h == digest(x), '断点校验失败；原文件保留，不自动修复')
    return x


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    """Write once with atomic publication; existing conflicting data is preserved."""
    path = Path(path)
    if path.exists():
        require(read(path) == value, '文件冲突，保留不覆盖：' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.own_', delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(read(path) == value, '并发写冲突：' + str(path))
    finally:
        tmp.unlink()  # Only our temporary file; never historical files.


def branch_guard():
    b = subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip()
    require(b == BRANCH, '分支不符，停止；不会自动切换分支：' + b)


def disk_guard():
    free = shutil.disk_usage(PROJECT).free
    size = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) if OUT.exists() else 0
    require(free >= 10 * 1024**3, '空闲不足10GiB，停止，不清理')
    require(size <= 256 * 1024**2, '本实验结果超过256MiB，停止，不清理')
    return dict(free_gib=round(free / 1024**3, 3), result_mib=round(size / 1024**2, 3))


@contextlib.contextmanager
def operation_lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'operation.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def code_identity():
    return {str(f): sha(f) for f in sorted(CODE.iterdir()) if f.suffix in ('.py', '.sh')}


def scope_rows(m, scope):
    require(scope in ('pilot60', 'remaining120'), '只支持pilot60或remaining120，不自动扩714')
    rows = [r for r in m['rows'] if r['scope'] == scope]
    require(len(rows) == (60 if scope == 'pilot60' else 120), '划分题数不符')
    return rows


def uniform(ids, k=4):
    require(len(ids) >= k and ids == sorted(set(ids)), '候选不足或非递增')
    # Centers of four equal-count strata; deterministic, no score access.
    return [ids[min(len(ids)-1, int((j + .5) * len(ids) / k))] for j in range(k)]


def semantic(ids, scores, total_frames, k=4):
    require(len(scores) == len(ids) and len(ids) >= k, '分数/帧数不符')
    require(all(math.isfinite(s) for s in scores), '分数包含NaN/Inf')
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    selected = []
    gap = .02 * max(1, total_frames - 1)
    for i in order:
        if all(abs(ids[i] - x) >= gap for x in selected):
            selected.append(ids[i])
        if len(selected) == k:
            break
    for i in order:
        if len(selected) == k:
            break
        if ids[i] not in selected:
            selected.append(ids[i])
    return sorted(selected)


def query(row):
    return row['question'] + '\n' + '\n'.join(c + '.' + b for c, b in zip('ABCD', row['bodies']))


def init():
    """Metadata only: no torch, no video decode, no GPU, no new Codec caches."""
    import numpy as np
    from tqdm import tqdm
    old_p = read(OLD / 'protocol.json')
    old_b = unseal(read(OLD / 'bundle.json'))
    require(digest(old_p) == old_b['signature'], '旧协议/数据包签名不符')
    rows = [dict(r) for r in old_b['rows'] if r['stage'] == 'dev180']
    require(len(rows) == 180 and len({r['video_id'] for r in rows}) == 60, '开发池不符')
    require(all(not {'answer', 'gt', 'pred'}.intersection(r) for r in rows), '推理清单含答案')
    by_duration = {}
    for r in rows:
        by_duration.setdefault(r['duration'], set()).add(r['video_id'])
    rng = random.Random(20261109)
    pilot = set()
    for duration in ('medium', 'long'):
        pool = sorted(by_duration[duration])
        require(len(pool) >= 10, '时长分层不足10视频')
        pilot.update(rng.sample(pool, 10))
    for r in rows:
        r['scope'] = 'pilot60' if r['video_id'] in pilot else 'remaining120'
    rows.sort(key=lambda r: (r['video_id'], int(r['index'])))
    sources = {str(OLD / n): sha(OLD / n) for n in ('protocol.json', 'bundle.json')}
    videos = {}
    for vid in tqdm(sorted({r['video_id'] for r in rows}), desc='冻结开发元数据', unit='视频', dynamic_ncols=True):
        disk_guard()
        v = old_b['videos'][vid]
        require(identity(v['video_path']) == v['source_video_identity'], '视频身份改变：' + vid)
        cache = Path(v['cache_dir'])
        meta = read(cache / 'meta.json')
        require(meta['video'] == v['video_path'] and meta['fps'] == v['fps'], 'Codec元数据身份改变')
        for f in (cache / 'meta.json', cache / 'src_patch_position.npy'):
            require(sha(f) == v['cache_files_sha256'][str(f)], 'Codec来源改变：' + str(f))
        ids = sorted({int(i) for g in meta['groups'] for i in g['frame_ids']})
        require(len(ids) >= 4 and ids[0] >= 0 and ids[-1] < v['total_frames'], '候选帧非法')
        a = np.load(cache / 'src_patch_position.npy', allow_pickle=False)
        require(a.ndim == 2 and a.shape[1] == 3, 'patch位置形状非法')
        require(set(a[a[:, 0] >= 0, 0].tolist()) <= set(ids), '保留patch不在候选池')
        require(v['source_indices'] == np.unique(a[a[:, 0] >= 0, 0]).astype(int).tolist(), '字幕时间来源改变')
        grid = meta['hb_wb']
        require(all(i > 0 and i % 2 == 0 for i in grid), 'Codec网格未按2×2对齐')
        coverage = {str(i): len(np.unique(a[a[:, 0] == i, 1:], axis=0)) / (grid[0] * grid[1]) for i in ids}
        evidence = v['evidence'][PRIMARY]['continuous']
        videos[vid] = {k: v[k] for k in ('video_path', 'cache_dir', 'fps', 'total_frames',
            'source_indices', 'cache_files_sha256', 'canvas_files', 'source_video_identity')}
        videos[vid].update(candidate_ids=ids, coverage=coverage, subtitle_text=evidence['text'],
            subtitle_tokens=evidence['text_tokens'], subtitle_sha256=digest(evidence['text']))
    references = {}
    for r in rows:
        path = OLD / 'units' / (r['index'] + '_' + PRIMARY + '.json')
        u = unseal(read(path))
        require(u['signature'] == old_b['signature'] and u['index'] == r['index'] and u['native_equal'], '旧单元身份不符')
        references[r['index']] = dict(path=str(path), file_sha256=sha(path),
            **{k: u[k] for k in ('tensor_sha256', 'input_sha256', 'input_tokens', 'prompt_sha256')})
    p = dict(version='codec_evidence_v1', old_signature=old_b['signature'], codec=CODEC,
        model_identity=old_p['model'], runtime=old_p['runtime'], seed=20261109,
        source_sha256=sources, code_sha256=code_identity(), n_extra_images=4, gap_ratio=.02,
        query='unmodified_question_plus_all_four_options_no_truncation',
        candidate_pool='sorted_unique_meta_groups_frame_ids', scopes=['pilot60', 'remaining120'],
        note='Seen-development effect study; no new independent test. Images are local-coordinate stills with source-time text. No ROI/D/automatic full714.')
    m = dict(signature=digest(p), protocol=p, rows=rows, videos=videos, references=references)
    scope_rows(m, 'pilot60'); scope_rows(m, 'remaining120')
    atomic(OUT / 'manifest.json', seal(m))
    print('已冻结180题/60视频，pilot60=60题/20视频；只写元数据，无视频解码/GPU。')


def context():
    branch_guard(); disk_guard()
    m = unseal(read(OUT / 'manifest.json'))
    require(m['signature'] == digest(m['protocol']), '协议签名错误')
    require(m['protocol']['code_sha256'] == code_identity(), '代码已改变；停止，保留旧协议和结果')
    for f, h in m['protocol']['source_sha256'].items():
        require(sha(f) == h, '旧来源改变：' + f)
    for name, expected in m['protocol']['model_identity']['files'].items():
        require(sha(MODEL / name) == expected, 'Mage配置/源码改变：' + name)
    for name, expected in m['protocol']['model_identity']['weights'].items():
        require(identity(MODEL / name) == expected, 'Mage权重身份改变')
    for v in m['videos'].values():
        require(identity(v['video_path']) == v['source_video_identity'], '来源视频改变')
    return m


def baseline(m, row):
    ref = m['references'][row['index']]
    require(sha(ref['path']) == ref['file_sha256'], '旧基线单元改变')
    u = unseal(read(ref['path']))
    require(u['signature'] == m['protocol']['old_signature'] and u['arm'] == PRIMARY, '旧基线路线不符')
    return u


def gpu_guard():
    require(os.environ.get('CODEC_EVIDENCE_GPU_CONFIRMED') == '1', 'GPU必须由用户显式设置CODEC_EVIDENCE_GPU_CONFIRMED=1')
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    busy = [x for x in r.stdout.splitlines() if x.strip()]
    require(not busy, '检测到已有GPU计算进程，停止且不操作它们：' + '; '.join(busy))

