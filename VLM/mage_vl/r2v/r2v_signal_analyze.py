#!/usr/bin/env python3
"""离线诊断；不拟合温度/阈值，不把 short219 当独立测试集。"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from r2v_pair_infer import digest, dump_new
from r2v_signal_score import LETTERS, features, logadd, read_jsonl


def choose_frames(codec_feature, frames_feature, feature):
    # 纯信号决策：无真值入口；相等时保留 Codec。不是正式校准后的 Gate2。
    return frames_feature[feature] > codec_feature[feature]


def outcomes(rows, changed):
    rescue = harm = 0
    correct = sum(r['phase1']['codec_ok'] for r in rows)
    for r in rows:
        if r['index'] in changed:
            c,f = r['phase1']['codec_ok'],r['phase1']['frames_ok']
            rescue += int(not c and f)
            harm += int(c and not f)
    n = len(rows)
    return dict(N=n, changes=len(changed), rescue=rescue,harm=harm,net=rescue-harm,
                accuracy=(correct+rescue-harm)/n,delta_pp=100*(rescue-harm)/n)


def auc(positives,negatives):
    # 正类=救回题，负类=误伤题；只衡量可分性，不寻找最佳阈值。
    if not positives or not negatives:
        return None
    return sum(1 if p>n else 0.5 if p==n else 0 for p in positives for n in negatives)/(len(positives)*len(negatives))


def diagnose(rows,scheme):
    n = len(rows)
    conflict = [r for r in rows if r['phase1']['codec_pred'] != r['phase1']['frames_pred']]
    rescue_pool = [r for r in rows if not r['phase1']['codec_ok'] and r['phase1']['frames_ok']]
    harm_pool = [r for r in rows if r['phase1']['codec_ok'] and not r['phase1']['frames_ok']]
    wrong = sum(not r['phase1']['codec_ok'] for r in rows)
    result = dict(conflict_N=len(conflict),rescue_pool=len(rescue_pool),harm_pool=len(harm_pool),
                  selectors={}, gate1_budget_diagnostic=[],separation_auc={})
    for feature in ['margin','answer_margin','answer_p']:
        def signal(r):
            return r['frames'][scheme][feature] - r['codec'][scheme][feature]
        changes = {r['index'] for r in conflict if choose_frames(r['codec'][scheme],r['frames'][scheme],feature)}
        stats = outcomes(rows,changes)
        # 与“在全部冲突题中随机选同样 K 题替换”比较，给出解析期望而非抽一次随机数。
        k = len(changes)
        stats['random_same_changes_expected_rescue'] = k*len(rescue_pool)/len(conflict) if conflict else 0
        stats['random_same_changes_expected_harm'] = k*len(harm_pool)/len(conflict) if conflict else 0
        stats['random_same_changes_expected_delta_pp'] = 100*k*(len(rescue_pool)-len(harm_pool))/(len(conflict)*n) if conflict else 0
        result['selectors'][feature] = stats
        result['separation_auc'][feature] = auc([signal(r) for r in rescue_pool],[signal(r) for r in harm_pool])
    # Gate1 只依赖 Codec 的 margin。固定预算网格，无标签调参，不从中挑最好点。
    ranked = sorted(rows,key=lambda r:(r['codec'][scheme]['margin'],str(r['index'])))
    for fraction in [0,0.1,0.25,0.5,0.75,1]:
        k = math.ceil(n*fraction)
        triggered = {r['index'] for r in ranked[:k]}
        rescued_available = sum(r['index'] in triggered for r in rescue_pool)
        untriggered_wrong = sum(not r['phase1']['codec_ok'] and r['index'] not in triggered for r in rows)
        changes = {r['index'] for r in conflict if r['index'] in triggered and
                   choose_frames(r['codec'][scheme],r['frames'][scheme],'margin')}
        result['gate1_budget_diagnostic'].append(dict(requested_fraction=fraction,triggered=k,
            trigger_rate=k/n,frames_only_recalled=rescued_available,
            frames_only_recall=rescued_available/len(rescue_pool) if rescue_pool else None,
            oracle_trigger_accuracy=(sum(r['phase1']['codec_ok'] for r in rows)+rescued_available)/n,
            random_same_trigger_budget_expected_rescue=k*len(rescue_pool)/n,
            codec_wrong_not_triggered=untriggered_wrong,
            overconfident_error_fraction=untriggered_wrong/wrong if wrong else None,
            simulated_margin_selection=outcomes(rows,changes)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args = p.parse_args()
    lock = json.loads((args.output/'signal_config_lock.json').read_text())
    for name,sha in lock['scripts_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('分析/打分脚本变化；拒绝用新逻辑覆盖旧汇总')
    source = Path(lock['source'])
    if digest(source/'config_lock.json') != lock['source_lock_sha256'] or digest(source/'pair_short.jsonl') != lock['source_pairs_sha256']:
        raise ValueError('阶段 1 来源发生变化')
    phase1 = {r['index']:r for r in read_jsonl(source/'pair_short.jsonl')}
    rows = read_jsonl(args.output/'signal_short.jsonl')
    if {r['index'] for r in rows} != set(lock['indices']) or len(rows) != len(lock['indices']):
        raise ValueError(f'分数未完整或 index 重复: {len(rows)}/{len(lock["indices"])}')
    probes = json.loads((args.output/'token_probe.json').read_text())
    for r in rows:
        if phase1[r['index']] != r['phase1']:
            raise ValueError('分数记录与阶段 1 原题不一致')
        for b in ['codec','frames']:
            value = r[b]
            answer = r['phase1'][f'{b}_pred']
            if value['plain'] != features(value['plain_logits'],answer) or value['union'] != features(
                    [logadd(a,z) for a,z in zip(value['plain_logits'],value['space_logits'])],answer):
                raise ValueError('分数派生字段不一致')
            letter = next((c for c in LETTERS if value['generated_token_id'] in [probes[c]['plain'],probes[c]['space']]),None)
            if value['first_letter'] != letter or value['matches_phase1'] != (letter == answer):
                raise ValueError('原答案一致性字段错误')
    mismatches = {b:[r['index'] for r in rows if not r[b]['matches_phase1']] for b in ['codec','frames']}
    plain_top1_mismatch = {b:[r['index'] for r in rows if r[b]['plain']['top1'] != r['phase1'][f'{b}_pred']] for b in ['codec','frames']}
    audit = dict(N=len(rows),first_token_mismatch=mismatches,plain_top1_mismatch=plain_top1_mismatch,
        smoke_interface_pass=lock['scope']=='smoke' and not any(mismatches.values()) and
            all(r[b]['forward_check'] is not None and r[b]['forward_check']['passed'] for r in rows for b in ['codec','frames']),
        letter_mass_range={b:{scheme:[min(r[b][f'letter_mass_{scheme}'] for r in rows),
                                           max(r[b][f'letter_mass_{scheme}'] for r in rows)]
                              for scheme in ['plain','union']} for b in ['codec','frames']},
        interpretation='Conditional letter softmax is NOT calibrated correctness probability. Check mismatch indices before interpreting selectors.')
    result = dict(status='exploratory_posthoc_short_only',audit=audit,
        baselines=dict(codec_accuracy=sum(r['phase1']['codec_ok'] for r in rows)/len(rows),
                       frames_accuracy=sum(r['phase1']['frames_ok'] for r in rows)/len(rows),
                       oracle_accuracy=sum(r['phase1']['codec_ok'] or r['phase1']['frames_ok'] for r in rows)/len(rows)),
        primary_plain=diagnose(rows,'plain'),space_union_sensitivity=diagnose(rows,'union'))
    path = args.output/'signal_summary.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('已有汇总不同；拒绝覆盖')
    else:
        dump_new(path,result)
    print(json.dumps(dict(status=result['status'],audit=audit,baselines=result['baselines'],
        primary_selectors=result['primary_plain']['selectors'],
        primary_auc=result['primary_plain']['separation_auc'],
        gate1_budget_diagnostic=result['primary_plain']['gate1_budget_diagnostic']),ensure_ascii=False,indent=2))
    print(f'完整汇总（含空格变体敏感性分析）: {path}')


if __name__ == '__main__':
    main()
