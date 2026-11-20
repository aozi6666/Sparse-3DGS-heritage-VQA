#!/usr/bin/env python3
"""仅用标准库统计完整配对；不加载模型，不读取历史 Mage 结果。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from r2v_pair_infer import dump_new


def metrics(rows):
    counts = Counter()
    for r in rows:
        c, f = r['codec_ok'], r['frames_ok']
        counts['both_ok' if c and f else 'codec_only' if c else 'frames_only' if f else 'both_bad'] += 1
    n = len(rows)
    q = {k: counts[k] for k in ['both_ok', 'codec_only', 'frames_only', 'both_bad']}
    wrong = q['frames_only'] + q['both_bad']
    return dict(N=n, **q,
        codec_accuracy=(q['both_ok']+q['codec_only'])/n if n else None,
        frames_accuracy=(q['both_ok']+q['frames_only'])/n if n else None,
        oracle_accuracy=(n-q['both_bad'])/n if n else None,
        oracle_gain_pp=100*q['frames_only']/n if n else None,
        rescue_rate=q['frames_only']/wrong if wrong else None,
        agreement_rate=sum(r['codec_pred']==r['frames_pred'] for r in rows)/n if n else None,
        invalid_codec=sum(r['codec_pred']=='' for r in rows),
        invalid_frames=sum(r['frames_pred']=='' for r in rows),
        **{f'{field}_{s}_total': sum(r[f'{field}_{s}'] for r in rows)
           for field in ['t', 'n_tok'] for s in ['c', 'f']},
        **{f'peak_mem_{s}_max_gib': max((r[f'peak_mem_{s}'] for r in rows), default=None) for s in ['c', 'f']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    lock = json.loads((args.output/'config_lock.json').read_text())
    # 统计分母由配置锁固定；不允许把未跑完的子集冒充完整 short 结果。
    rows = []
    seen = set()
    with (args.output/'pair_short.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError('残缺尾行；不自动截断')
            r = json.loads(line)
            if r['index'] in seen:
                raise ValueError('重复 index')
            seen.add(r['index'])
            if r['gt'] not in list('ABCD'):
                raise ValueError('非法 gt')
            for b in ['codec', 'frames']:
                if r[f'{b}_pred'] not in ['', 'A', 'B', 'C', 'D'] or r[f'{b}_ok'] != (r[f'{b}_pred']==r['gt']):
                    raise ValueError('预测或正确性字段不一致')
            rows.append(r)
    if seen != set(lock['indices']):
        raise ValueError(f'配对尚未完整：{len(seen)}/{len(lock["indices"])}；不生成正式汇总')
    result = dict(phase=lock['phase'], metrics=metrics(rows), slices={})
    for field in ['task_type', 'sub_category']:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result['slices'][field] = {k: metrics(v) for k, v in sorted(groups.items())}
    target = args.output/'summary_short.json'
    if target.exists():
        if json.loads(target.read_text()) != result:
            raise ValueError('已有汇总不同；拒绝覆盖')
    else:
        dump_new(target, result)
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
    print(f'汇总路径: {target}')


if __name__ == '__main__':
    main()
