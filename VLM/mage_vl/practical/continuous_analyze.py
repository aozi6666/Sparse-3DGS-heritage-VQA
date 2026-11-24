#!/usr/bin/env python3
"""连续字幕开发统计；仅离线汇总读取GT。"""
import argparse
import json
from practical_common import ROOT, LETTERS, sha, write_once
from practical_analyze import comparison
from continuous_common import context,STAGES


def group(rows,ctx,labels):
    if not rows: return dict(N=0,methods={})
    preds={name:{} for name in ['frames32_no_sub','aligned_sub_v1','continuous_time_v1']}
    for row in rows:
        idx=row['index']; preds['frames32_no_sub'][idx]=ctx['old']['base']['units'][f'{idx}:frames:p0']['semantic_pred']
        preds['aligned_sub_v1'][idx]=ctx['old']['units'][idx]['pred']; preds['continuous_time_v1'][idx]=ctx['units'][idx]['pred']
    methods={name:dict(correct=sum(pred[r['index']]==labels[r['index']] for r in rows),
        accuracy=sum(pred[r['index']]==labels[r['index']] for r in rows)/len(rows),
        invalid=sum(pred[r['index']] not in list(LETTERS) for r in rows)) for name,pred in preds.items()}
    methods['continuous_time_v1']['vs']={name:comparison(rows,preds['continuous_time_v1'],preds[name],labels)
        for name in ['frames32_no_sub','aligned_sub_v1']}
    return dict(N=len(rows),videos=len({r['video_id'] for r in rows}),methods=methods)


def summarize(args,ctx):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完整，不读取标签')
    manifest=ROOT/'01_manifests'; lock=json.loads((manifest/'dataset_lock.json').read_text())
    if sha(manifest/'evaluation_labels.json')!=lock['labels_sha256']: raise ValueError('标签漂移')
    labels=json.loads((manifest/'evaluation_labels.json').read_text()); rows=ctx['rows']
    invalid=sum(ctx['units'][r['index']]['pred'] not in list(LETTERS) for r in rows)
    report=dict(signature=ctx['signature'],stage=args.stage,N=len(rows),interpretation='Development experiment; extra neighboring subtitle context and timestamp formatting jointly changed; no claim of independent effects.',
        audit=dict(passed=not invalid,invalid=invalid,complete_units=len(rows)),overall=group(rows,ctx,labels),
        by_duration={d:group([r for r in rows if r['duration']==d],ctx,labels) for d in sorted({r['duration'] for r in rows})},
        cost=dict(new_calls=sum(ctx['units'][r['index']]['new_answer_calls'] for r in rows),
            logical_visual_tokens=sum(ctx['units'][r['index']]['visual']['n_tok'] for r in rows),
            subtitle_tokens=sum(ctx['units'][r['index']]['subtitle_tokens'] for r in rows),
            inference_s=sum(ctx['units'][r['index']]['inference_s'] for r in rows)),
        budget_omitted_neighbor_videos=sum(bool(ctx['evidence'][v]['omitted_neighbor_ids']) for v in {r['video_id'] for r in rows}))
    details=[dict(index=r['index'],video_id=r['video_id'],gt=labels[r['index']],pred=ctx['units'][r['index']]['pred'],
        record_sha256=ctx['units'][r['index']]['record_sha256']) for r in rows]
    write_once(ctx['stage_dir']/'predictions.json',details); write_once(ctx['stage_dir']/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('汇总路径:',ctx['stage_dir']/'summary.json')
    if invalid: raise ValueError('非法回答保留分母，审计不通过')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=STAGES,default='smoke'); p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args(); summarize(args,context(args))
