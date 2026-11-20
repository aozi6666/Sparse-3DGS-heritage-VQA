#!/usr/bin/env python3
"""路线 B：固定原顺序，只增加 Frames 帧数；不改写已归档的换序协议。"""
import argparse
from copy import copy
import datetime
import fcntl
import json
import math
from pathlib import Path
import uuid

from practical_common import (ROOT, STAGES, LETTERS, sha, signature, write_once,
    ballot_key, load_ballots)
from practical_vote import context, prerequisites, run_gpu, session_log
from practical_analyze import comparison

BASE_SIG = '86f337f035388253a22e9dcb7d7a6977005aae4ab1273fa75313735ba5ab2c16'
ARCHIVE = ROOT/'07_archive/option_vote_dev60_v1.json'


def archive():
    """小文件保存快照，大日志/逐题原始记录保留原位并登记校验和，不复制缓存。"""
    files = []
    for stage in ['smoke', 'dev30', 'dev60']:
        folder = ROOT/STAGES[stage]/'px150000_f32_frames4_codec1'
        files.extend(p for p in folder.iterdir() if p.is_file())
    store = ROOT/'cache/ballots'/BASE_SIG
    files.extend(p for p in store.iterdir() if p.is_file())
    files.extend(p for p in (ROOT/'01_manifests').iterdir() if p.is_file())
    records = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(files)}
    snapshots = {str(p.relative_to(ROOT)): json.loads(p.read_text()) for p in files
                 if p.name in ['summary.json', 'predictions.json', 'protocol.json']}
    summary = snapshots['03_dev60/px150000_f32_frames4_codec1/summary.json']
    if summary['signature'] != BASE_SIG or summary['audit']['N'] != 180:
        raise ValueError('待归档来源不符')
    write_once(ARCHIVE, dict(source_signature=BASE_SIG, files=records, snapshots=snapshots,
        conclusion='dev60: Codec 111, Frames 110, vote 108, mean 109 / 180. '
                   'Vote versus Frames: rescue 4, harm 6; no gain. Pause voting; no val/test/full.',
        retention='Original records/logs/cache remain in place. Hash inventory is not an off-machine backup.'))
    print('归档索引和结果快照:', ARCHIVE)


def verify_archive():
    data = json.loads(ARCHIVE.read_text())
    for name, item in data['files'].items():
        p = ROOT/name
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError(f'归档源发生变化，停止且不覆盖: {p}')


def make_context(args):
    verify_archive()
    # 原脚本不改动，因此历史配置签名和可复核性不受路线 B 影响。
    base_args = copy(args)
    base_args.frames = 32
    base = context(base_args)
    if base['signature'] != BASE_SIG:
        raise ValueError('模型/数据/旧代码漂移，不能与既有 Frames-32 配对比较')
    required = {ballot_key(r['index'], b, 0) for r in base['rows'] for b in ['codec', 'frames']}
    if not required.issubset(base['units']):
        raise ValueError('同批题目的 32 帧基线不完整，不自动补跑')
    ctx = context(args)
    protocol = dict(ctx['protocol'], version='practical_coverage_v1',
        permutations='identity only for both backends; no voting or score fusion',
        baseline_signature=BASE_SIG, coverage_script_sha256=sha(__file__),
        intent='Change Frames count only. Codec identity rerun is a paired reproducibility control.')
    sig = signature(protocol)
    store = ROOT/'cache/coverage'/sig
    ctx.update(protocol=protocol, signature=sig, store=store,
        tag=f'px150000_f{args.frames}_identity_v1', chosen='frames_identity',
        expected=required, units=load_ballots(store/'units.jsonl', sig, ctx['all_rows']))
    ctx['stage_dir'] = ROOT/'08_coverage'/args.stage/ctx['tag']
    return ctx, base


def audit_units(rows, units, baseline, frames, smoke=False):
    """审查真实网格/token/像素和时间端点；任何审计失败都禁止进入开发池。"""
    errors = []
    for r in rows:
        idx = r['index']
        c, f = [units[ballot_key(idx, b, 0)] for b in ['codec', 'frames']]
        old_c, old_f = [baseline[ballot_key(idx, b, 0)] for b in ['codec', 'frames']]
        if any(u['display_pred'] not in LETTERS or not u['display_pred'] for u in [c, f]):
            errors.append([idx, 'invalid_answer'])
        # Codec 不改帧数配置；需复现输入签名、答案及视觉布局，排除环境漂移。
        for name in ['semantic_pred', 'prefix_sha256', 'grid_thw', 'pv_shape', 'patch_time_range', 'n_tok']:
            if c[name] != old_c[name]:
                errors.append([idx, 'codec_control_mismatch', name])
        grid = f['grid_thw']
        valid_grid = (len(grid) == frames and all(len(g) == 3 and g[0] == 1
            and g[1] > 0 and g[2] > 0 and g[1] % 2 == 0 and g[2] % 2 == 0
            and 3136 <= g[1]*g[2]*256 <= 150000 for g in grid))
        if not valid_grid:
            errors.append([idx, 'frame_grid_or_pixel_budget'])
        raw = sum(t*h*w for t,h,w in grid)
        if raw != f['n_tok']*4 or f['pv_shape'] != [raw, 768]:
            errors.append([idx, 'token_or_tensor_shape'])
        if f['frames_max_pixels'] != 150000 or f['frames_min_pixels'] != 3136:
            errors.append([idx, 'applied_pixel_budget'])
        # 同样逐帧空间预算，token 应按帧数比例增加；不是等总 token 的算法对照。
        if f['n_tok']*32 != old_f['n_tok']*frames:
            errors.append([idx, 'unexpected_token_scaling'])
        times, old_times = f['patch_time_range'], old_f['patch_time_range']
        if (len(times) != 2 or not all(math.isfinite(t) for t in times)
                or not times[1] > times[0] or len(old_times) != 2
                or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-6) for a,b in zip(times,old_times))):
            errors.append([idx, 'time_endpoints_changed'])
        if smoke and any(u['audit']['direct_input_equal'] is not True for u in [c,f]):
            errors.append([idx, 'native_input_tensor_comparison'])
    return dict(passed=not errors, errors=errors, N=len(rows),
        note='Time endpoint audit does not prove every intermediate timestamp; native sampling is unchanged.')


def summarize(args, ctx, base):
    if not ctx['expected'].issubset(ctx['units']):
        raise ValueError(f'尚未完成 {len(ctx["expected"] & set(ctx["units"]))}/{len(ctx["expected"])} 单元')
    m = ROOT/'01_manifests'
    lock = json.loads((m/'dataset_lock.json').read_text())
    if sha(m/'evaluation_labels.json') != lock['labels_sha256']:
        raise ValueError('标签清单漂移')
    labels = json.loads((m/'evaluation_labels.json').read_text())
    rows, units = ctx['rows'], ctx['units']
    preds = {name: {r['index']: source[ballot_key(r['index'], backend, 0)]['semantic_pred'] for r in rows}
             for name, source, backend in [
                ('codec32', base['units'], 'codec'), ('frames32', base['units'], 'frames'),
                (f'frames{args.frames}', units, 'frames')]}
    report = dict(signature=ctx['signature'], baseline_signature=BASE_SIG,
        stage=args.stage, interpretation='Development screening only; not independent evidence of gains.',
        audit=audit_units(rows, units, base['units'], args.frames, args.stage=='smoke'), methods={})
    for name, pred in preds.items():
        old = name in ['codec32', 'frames32']
        source = base['units'] if old else units
        backend = 'codec' if name=='codec32' else 'frames'
        costs = [source[ballot_key(r['index'], backend, 0)] for r in rows]
        metric = dict(N=len(rows), correct=sum(pred[r['index']]==labels[r['index']] for r in rows),
            invalid=sum(pred[r['index']] not in LETTERS or not pred[r['index']] for r in rows),
            vs={b: comparison(rows,pred,preds[b],labels) for b in ['codec32','frames32']},
            cost=dict(model_calls=len(costs),visual_tokens=sum(u['n_tok'] for u in costs),
                inference_s=sum(u['inference_s'] for u in costs),
                preprocess_s=sum(u['preprocess_s'] for u in costs),
                peak_allocated_gib=max(u['peak_allocated_gib'] for u in costs),
                note='Historical preprocessing was shared across votes; wall times are not a controlled speed benchmark.'),
            by_duration={})
        metric['accuracy'] = metric['correct']/len(rows)
        for duration in sorted({r['duration'] for r in rows}):
            subset = [r for r in rows if r['duration']==duration]
            metric['by_duration'][duration] = dict(N=len(subset),correct=sum(pred[r['index']]==labels[r['index']] for r in subset))
        report['methods'][name] = metric
    details = [dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],
                    predictions={n:p[r['index']] for n,p in preds.items()}) for r in rows]
    write_once(ctx['stage_dir']/'predictions.json', details)
    write_once(ctx['stage_dir']/'summary.json', report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('结果路径:', ctx['stage_dir'])
    if not report['audit']['passed']:
        raise ValueError('审计未通过，保存现场并停止；不能据此宣称涨点')


def preflight(args, ctx):
    # 使用原有依赖只读检查，但由本路线独立的 smoke 锁控制后续阶段。
    check = copy(args)
    check.stage = 'smoke'
    prerequisites(check, ctx)
    if args.stage != 'smoke':
        path = ROOT/'08_coverage/smoke'/ctx['tag']/'summary.json'
        result = json.loads(path.read_text())
        if result['signature'] != ctx['signature'] or not result['audit']['passed']:
            raise ValueError('必须先完成当前帧数 smoke 和汇总审计')
    if args.stage == 'dev60':
        path = ROOT/'08_coverage/dev30'/ctx['tag']/'summary.json'
        result = json.loads(path.read_text())
        if result['signature'] != ctx['signature'] or not result['audit']['passed']:
            raise ValueError('必须先完成 dev30；dev60 由用户查看结果后决定')


def run(args, ctx):
    import os
    preflight(args, ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED') != '1':
        raise RuntimeError('需用户设置 PRACTICAL_STAGE_CONFIRMED=1；助手不启动推理')
    with (ROOT/'cache/gpu.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx, _ = make_context(args)
        ctx['store'].mkdir(parents=True,exist_ok=True)
        ctx['stage_dir'].mkdir(parents=True,exist_ok=True)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],
            baseline_signature=BASE_SIG,stage=args.stage,indices=[r['index'] for r in ctx['rows']],
            store=str(ctx['store']),protocol=ctx['protocol'],chosen='frames_identity'))
        if ctx['expected'].issubset(ctx['units']):
            print('必要单元已落盘，不加载模型；请 summarize。')
            return
        log=ctx['stage_dir']/('run_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.log')
        print('每题仅 Codec 原顺序审计 + Frames 原顺序；复用旧执行器的进度条标题仍显示 Mage 换序。')
        with session_log(log):
            # chosen=frames_identity 使原执行器 jobs_for 只生成两个恒等排列单元。
            # 复用完全相同的生成和解析实现，避免为新路线悄悄改变基线。
            run_gpu(args,ctx)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['archive','preflight','run','summarize'])
    p.add_argument('--stage',choices=['smoke','dev30','dev60'],default='smoke')
    p.add_argument('--frames',type=int,choices=[48,64],default=48)
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    args.root=ROOT
    args.codec_vote=False
    if args.min_free < 1 or args.budget <= 0:
        p.error('磁盘保留至少 1 GiB，预算必须为正')
    if args.action=='archive':
        # 连 archive 也检查分支和磁盘，但绝不导入模型。
        a=copy(args); a.frames=32; context(a)
        archive(); return
    ctx,base=make_context(args)
    print(f'路线 B: Frames-{args.frames}, {args.stage}: {len(ctx["rows"])}题，'
          f'{len(ctx["expected"] & set(ctx["units"]))}/{len(ctx["expected"])}单元已完成')
    if args.action=='preflight':
        preflight(args,ctx)
        print('只读预检通过；未加载模型或推理')
    elif args.action=='run':
        run(args,ctx)
    else:
        summarize(args,ctx,base)


if __name__=='__main__':
    main()
