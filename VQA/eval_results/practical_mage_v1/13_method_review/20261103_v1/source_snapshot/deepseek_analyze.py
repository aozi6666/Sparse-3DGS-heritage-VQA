#!/usr/bin/env python3
"""API独立回答和组合系统的开发统计；唯一读取GT的API模块。"""
import argparse
import json
from practical_common import ROOT, LETTERS, sha, write_once, read_lines
from practical_analyze import comparison
from deepseek_review import context, STAGES


def group(rows,ctx,labels):
    if not rows: return dict(N=0,methods={})
    preds={name:{} for name in ['mage_no_sub','deepseek_independent','mage_deepseek_review']}
    for row in rows:
        idx=row['index']; u=ctx['units'][idx]
        preds['mage_no_sub'][idx]=ctx['continuous']['old']['base']['units'][f'{idx}:frames:p0']['semantic_pred']
        preds['deepseek_independent'][idx]=u['api_pred']; preds['mage_deepseek_review'][idx]=u['system_pred']
    methods={name:dict(correct=sum(p[r['index']]==labels[r['index']] for r in rows),
        accuracy=sum(p[r['index']]==labels[r['index']] for r in rows)/len(rows),
        invalid_or_abstain=sum(p[r['index']] not in list(LETTERS) for r in rows)) for name,p in preds.items()}
    for name in ['deepseek_independent','mage_deepseek_review']:
        methods[name]['vs_mage']=comparison(rows,preds[name],preds['mage_no_sub'],labels)
    return dict(N=len(rows),videos=len({r['video_id'] for r in rows}),methods=methods)


def summarize(args,ctx):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完成，不读取GT或发布部分统计')
    manifest=ROOT/'01_manifests'; lock=json.loads((manifest/'dataset_lock.json').read_text())
    if sha(manifest/'evaluation_labels.json')!=lock['labels_sha256']: raise ValueError('标签漂移')
    labels=json.loads((manifest/'evaluation_labels.json').read_text()); rows=ctx['rows']
    bad=sum(bool(ctx['units'][r['index']]['errors']['independent']) for r in rows)
    report=dict(signature=ctx['signature'],stage=args.stage,N=len(rows),
        interpretation='Development only. Pilot enriched for prediction disagreements, not a representative overall score. Independent API first pass uses sparse real frames; verification may add dense windows and a second call.',
        audit=dict(passed=not bad,complete=len(rows),invalid_independent_schema_or_citation=bad),overall=group(rows,ctx,labels),
        by_duration={d:group([r for r in rows if r['duration']==d],ctx,labels) for d in sorted({r['duration'] for r in rows})},
        overridden=sum(ctx['units'][r['index']]['overridden'] for r in rows))
    selected={r['index'] for r in rows}; calls=[]
    for path in (ctx['store']/'responses').glob('*.json'):
        if path.stem.split('_',1)[0] in selected: calls.append(json.loads(path.read_text()))
    report['cost']=dict(successful_calls=len(calls),total_tokens=sum((c['response'].get('usage') or {}).get('total_tokens',0) for c in calls),
        request_time_s=sum(c['elapsed_s'] for c in calls),
        stage_attempts_including_failures=sum(e['event']=='start' and e['key'].split('_',1)[0] in selected
            for e in read_lines(ctx['store']/'attempts.jsonl')),
        note='Shared cache reuses identical completed requests across stages. Token stop line is checked before a request and may overshoot by one response; this is not a currency spending cap. Image tokens included in provider usage.')
    details=[dict(index=r['index'],video_id=r['video_id'],gt=labels[r['index']],
        api=ctx['units'][r['index']]['api_pred'],system=ctx['units'][r['index']]['system_pred'],record_sha256=ctx['units'][r['index']]['record_sha256']) for r in rows]
    write_once(ctx['stage_dir']/'predictions.json',details); write_once(ctx['stage_dir']/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('汇总路径:',ctx['stage_dir']/'summary.json')
    if bad: raise ValueError('API结构/引用审计失败，保存结果，不自动重试')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=STAGES,default='smoke'); p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args(); summarize(args,context(args))
