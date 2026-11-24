#!/usr/bin/env python3
"""离线汇总和冻结入口；只有这里读取独立的评测标签文件。"""
import argparse
from collections import Counter,defaultdict
import json
import math

from practical_common import ROOT,STAGES,LETTERS,sha,write_once,stage_rows,ballot_key,aggregate
from practical_vote import context

METHODS={'codec_identity':('codec','identity'),'frames_identity':('frames','identity'),
         'frames_vote':('frames','vote'),'frames_mean':('frames','mean'),
         'codec_vote':('codec','vote'),'codec_mean':('codec','mean')}


def predictions(rows,units,method):
    backend,rule=METHODS[method]
    results={}
    for row in rows:
        ks=[0] if row['no_permute'] or rule=='identity' else range(4)
        keys=[ballot_key(row['index'],backend,k) for k in ks]
        if any(key not in units for key in keys):
            return None
        results[row['index']]=aggregate([units[key] for key in keys],rule)
    return results


def comparison(rows,pred,base,labels):
    b=c=0
    effects=defaultdict(lambda:[0,0])
    for r in rows:
        idx=r['index']; correct=pred[idx]==labels[idx]; original=base[idx]==labels[idx]
        b+=int(correct and not original); c+=int(original and not correct)
        effects[r['video_id']][0]+=int(correct)-int(original)
        effects[r['video_id']][1]+=1
    # 以视频为聚类单元，避免把同视频三题当成完全独立样本。
    import numpy as np
    array=np.array(list(effects.values()))
    draws=np.random.default_rng(0).integers(0,len(array),size=(5000,len(array)))
    sampled=array[draws].sum(axis=1)
    ci=np.quantile(100*sampled[:,0]/sampled[:,1],[.025,.975]).tolist()
    discord=b+c
    p=min(1,2*sum(math.comb(discord,k) for k in range(min(b,c)+1))/2**discord) if discord else 1.0
    return dict(rescue=b,harm=c,net=b-c,delta_pp=100*(b-c)/len(rows),
                video_cluster_bootstrap_95ci_pp=ci,questionwise_mcnemar_exact_p=p,
                note='McNemar is questionwise; clustered CI is the primary dependence-aware interval')


def summarize(args,ctx):
    if not ctx['expected'].issubset(ctx['units']):
        raise ValueError(f'尚未完成：{len(ctx["expected"] & set(ctx["units"]))}/{len(ctx["expected"])} 单元')
    m=args.root/'01_manifests'
    dataset_lock=json.loads((m/'dataset_lock.json').read_text())
    if sha(m/'evaluation_labels.json')!=dataset_lock['labels_sha256']:
        raise ValueError('评测标签发生变化')
    labels=json.loads((m/'evaluation_labels.json').read_text())
    rows=ctx['rows']; units=ctx['units']
    all_predictions={name:predictions(rows,units,name) for name in METHODS}
    all_predictions={name:value for name,value in all_predictions.items() if value is not None}
    allowed={'codec_identity','frames_identity',ctx['chosen']} if ctx['chosen'] else (
        set(METHODS) if args.codec_vote else {'codec_identity','frames_identity','frames_vote','frames_mean'})
    all_predictions={k:v for k,v in all_predictions.items() if k in allowed}
    selected_units=[units[k] for k in sorted(ctx['expected'])]
    identity=[u for u in selected_units if u['permutation_id']==0]
    smoke_pass=(args.stage=='smoke' and all(u['audit']['direct_input_equal'] is True for u in identity)
        and all(u['audit']['legacy_equal'] is True for u in identity if u['backend']=='codec' or args.frames==32)
        and all(u['display_pred'] in list(LETTERS) for u in selected_units)
        and all(len(u['grid_thw'])==args.frames for u in identity if u['backend']=='frames')
        and all(u['frames_max_pixels']==150000 and u['frames_min_pixels']==3136 for u in selected_units))
    report=dict(signature=ctx['signature'],stage=args.stage,
        interpretation='714 overall includes development data' if args.stage=='full' else
            ('locked independent test; no retuning' if args.stage=='test' else 'smoke/development/validation; not final proof'),
        audit=dict(smoke_pass=smoke_pass if args.stage=='smoke' else None,
            N=len(rows),videos=len({r['video_id'] for r in rows}),complete_units=len(selected_units),
            invalid_ballots=sum(u['display_pred']=='' for u in selected_units),
            unscorable_first_positions=sum(not u['score_eligible'] for u in selected_units),
            position_sensitive_fallbacks=[r['index'] for r in rows if r['no_permute']]),
        methods={})
    for method,pred in all_predictions.items():
        backend,rule=METHODS[method]
        metric=dict(N=len(rows),correct=sum(pred[r['index']]==labels[r['index']] for r in rows),
                    invalid=sum(pred[r['index']]=='' for r in rows),by_duration={},vs={})
        metric['accuracy']=metric['correct']/len(rows)
        for duration in sorted({r['duration'] for r in rows}):
            subset=[r for r in rows if r['duration']==duration]
            metric['by_duration'][duration]=dict(N=len(subset),correct=sum(pred[r['index']]==labels[r['index']] for r in subset))
        keys={ballot_key(r['index'],backend,k) for r in rows for k in
              ([0] if rule=='identity' or r['no_permute'] else range(4))}
        costs=[units[k] for k in keys]
        metric['cost']=dict(model_calls=len(costs),visual_tokens=sum(u['n_tok'] for u in costs),
            inference_s=sum(u['inference_s'] for u in costs),charged_preprocess_s=sum(u['preprocess_s'] for u in costs),
            peak_allocated_gib=max(u['peak_allocated_gib'] for u in costs),
            note='actual unit costs, shared preprocessing charged to first unit; inspect run logs for total wall time')
        for base in ['codec_identity','frames_identity']:
            metric['vs'][base]=comparison(rows,pred,all_predictions[base],labels)
        report['methods'][method]=metric
    final_rows=[]
    for r in rows:
        idx=r['index']
        final_rows.append(dict(index=idx,video_id=r['video_id'],duration=r['duration'],gt=labels[idx],
            no_permute=r['no_permute'],predictions={k:p[idx] for k,p in all_predictions.items()}))
    ctx['stage_dir'].mkdir(parents=True,exist_ok=True)
    write_once(ctx['stage_dir']/'predictions.json',final_rows)
    write_once(ctx['stage_dir']/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('结果路径:',ctx['stage_dir'])


def freeze(args,ctx):
    if args.stage!='val':
        raise ValueError('仅允许从 val 冻结')
    summary_path=ctx['stage_dir']/'summary.json'
    summary=json.loads(summary_path.read_text())
    if summary['signature']!=ctx['signature'] or args.method not in summary['methods']:
        raise ValueError('候选方法未完成相同协议验证')
    test_indices={r['index'] for r in stage_rows(args.root,'test')}
    if any(u['index'] in test_indices for u in ctx['units'].values()):
        raise ValueError('测试集已有推理单元，拒绝重新选型/冻结')
    frozen=dict(signature=ctx['signature'],method=args.method,validation_summary_sha256=sha(summary_path),
                validation_summary=str(summary_path),protocol=ctx['protocol'],
                note='method explicitly chosen before test; no per-question ground-truth fallback')
    target=args.root/'01_manifests'/f'frozen_{ctx["tag"]}.json'
    write_once(target,frozen)
    print('冻结完成:',target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['summarize','freeze'])
    p.add_argument('--stage',choices=list(STAGES),required=True)
    p.add_argument('--root',type=__import__('pathlib').Path,default=ROOT)
    p.add_argument('--frames',type=int,choices=[32,48,64],default=32)
    p.add_argument('--codec-vote',action='store_true')
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    p.add_argument('--method',choices=list(METHODS),default='frames_vote')
    args=p.parse_args()
    if args.root.resolve()!=ROOT:
        p.error('固定结果根目录')
    ctx=context(args)
    if args.action=='summarize':
        summarize(args,ctx)
    else:
        freeze(args,ctx)


if __name__=='__main__':
    main()
