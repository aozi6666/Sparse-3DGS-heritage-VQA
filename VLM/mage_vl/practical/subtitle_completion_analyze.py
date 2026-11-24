#!/usr/bin/env python3
"""预算对照/短视频汇总和显式配置冻结；本模块才读取GT。"""
import argparse
from copy import copy
import json
from collections import Counter

from practical_common import ROOT, LETTERS, sha, write_once, stage_rows
from practical_analyze import comparison
from subtitle_completion_common import RESULTS,context,key,get_evidence


def group(rows,ctx,labels):
    if not rows: return dict(N=0,methods={})
    if ctx['scope']=='budget':
        preds={name:{} for name in ['no_sub','aligned','continuous_2048','continuous_3072']}
        for r in rows:
            idx=r['index']; parent=ctx['parent']
            preds['no_sub'][idx]=parent['old']['base']['units'][f'{idx}:frames:p0']['semantic_pred']
            preds['aligned'][idx]=parent['old']['units'][idx]['pred']
            preds['continuous_2048'][idx]=parent['units'][idx]['pred']
            preds['continuous_3072'][idx]=ctx['units'][key(idx,'continuous')]['pred']
        target='continuous_3072'; bases=['no_sub','aligned','continuous_2048']
    else:
        preds={arm:{r['index']:ctx['units'][key(r['index'],arm)]['pred'] for r in rows} for arm in ctx['arms']}
        target='continuous'; bases=['no_sub','aligned']
    metrics={name:dict(correct=sum(pred[r['index']]==labels[r['index']] for r in rows),
        accuracy=sum(pred[r['index']]==labels[r['index']] for r in rows)/len(rows),
        invalid=sum(pred[r['index']] not in list(LETTERS) for r in rows)) for name,pred in preds.items()}
    metrics[target]['vs']={name:comparison(rows,preds[target],preds[name],labels) for name in bases}
    return dict(N=len(rows),videos=len({r['video_id'] for r in rows}),methods=metrics)


def build_report(args,ctx,labels):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完成，不发布部分报告')
    rows=ctx['rows']; items=[ctx['units'][key(r['index'],arm)] for r in rows for arm in ctx['arms']]
    origins=Counter(u['origin'] for u in items); invalid=sum(u['pred'] not in list(LETTERS) for u in items)
    errors=[]
    if invalid: errors.append(f'{invalid} invalid answers retained in denominator')
    if ctx['scope']=='budget' and (origins['generated']!=24 or origins['reused_old']!=156): errors.append('budget scope must be 24 generated + 156 verified reused')
    report=dict(signature=ctx['signature'],scope=ctx['scope'],tokens=ctx['tokens'],N=len(rows),
        interpretation='Development/coverage only, not independent test. Budget changes never selected per question.',
        audit=dict(passed=not errors,errors=errors,complete_units=len(items),invalid=invalid,origins=dict(origins)),
        overall=group(rows,ctx,labels),by_duration={},groups={},cost={})
    report['by_duration']={d:group([r for r in rows if r['duration']==d],ctx,labels) for d in sorted({r['duration'] for r in rows})}
    if ctx['scope']=='budget':
        report['groups']['changed24']=group([r for r in rows if r['index'] in ctx['changed']],ctx,labels)
        report['groups']['verified_reuse156']=group([r for r in rows if r['index'] not in ctx['changed']],ctx,labels)
        report['budget_control']=dict(old_tokens=2048,new_tokens=3072,changed_videos=8,changed_questions=24,
            reused_questions=156,all_reused_tensor_equal=True,
            note='Reused questions reconstruct inputs without new answer generation; budget does not retrieve evidence away from sampled frames.')
    else:
        evidence={r['video_id']:get_evidence(ctx,r['video_id']) for r in rows}
        for status in ['used','missing','empty','no_hit']:
            report['groups'][status]=group([r for r in rows if evidence[r['video_id']]['aligned']['status']==status],ctx,labels)
        smoke_ids={r['index'] for r in stage_rows(ROOT,'smoke')}
        report['groups']['beyond_smoke213']=group([r for r in rows if r['index'] not in smoke_ids],ctx,labels)
        report['subtitle_video_status']=dict(Counter(e['aligned']['status'] for e in evidence.values()))
        report['budget_omitted_neighbor_videos']=sum(bool(e['continuous']['omitted_neighbor_ids']) for e in evidence.values())
    for arm in ctx['arms']:
        selected=[ctx['units'][key(r['index'],arm)] for r in rows]
        report['cost'][arm]=dict(logical_answers=len(rows),new_generation_calls=sum(u['new_answer_calls'] for u in selected),
            reused_answers=sum(u['origin']!='generated' for u in selected),logical_visual_tokens=sum(u['visual']['n_tok'] for u in selected),
            subtitle_tokens=sum(u['subtitle_tokens'] for u in selected),input_tokens=sum(u['input_tokens'] for u in selected),
            new_inference_s=sum(u['inference_s'] for u in selected),
            peak_new_generation_gib=max(u['peak_allocated_gib'] for u in selected))
    report['cost']['shared_preprocess_s']=sum(u['preprocess_s'] for u in items)
    report['cost']['note']='Only newly completed generation time; preprocessing includes extra native-input audits. Reused answers retain logical visual/text cost; not a controlled deployment speed benchmark.'
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],
        predictions={arm:ctx['units'][key(r['index'],arm)]['pred'] for arm in ctx['arms']},
        record_sha256={arm:ctx['units'][key(r['index'],arm)]['record_sha256'] for arm in ctx['arms']}) for r in rows]
    return report,details


def summarize(args,ctx):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完成，不读取GT')
    manifests=ROOT/'01_manifests'; lock=json.loads((manifests/'dataset_lock.json').read_text())
    path=manifests/'evaluation_labels.json'
    if sha(path)!=lock['labels_sha256']: raise ValueError('标签文件漂移')
    labels=json.loads(path.read_text()); report,details=build_report(args,ctx,labels)
    write_once(ctx['stage_dir']/'predictions.json',details); write_once(ctx['stage_dir']/'summary.json',report)
    archive=dict(version='subtitle_completion_stage_v1',signature=ctx['signature'],scope=ctx['scope'],tokens=ctx['tokens'],
        summary=report,predictions=details,source_sha256={n:sha(ctx['stage_dir']/n) for n in ['summary.json','predictions.json']},
        journal_sha256=sha(ctx['store']/'units.jsonl'))
    write_once(ROOT/'07_archive'/f'subtitle_completion_{ctx["scope"]}_t{ctx["tokens"]}_v1.json',archive)
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('汇总路径:',ctx['stage_dir']/'summary.json')
    if not report['audit']['passed']: raise ValueError('审计失败；保留完整结果，不自动重跑')


def freeze_short(args):
    a=copy(args); a.scope='budget'; ctx=context(a)
    path=ctx['stage_dir']/'summary.json'; summary=json.loads(path.read_text())
    if (not ctx['expected'].issubset(ctx['units']) or summary['signature']!=ctx['signature']
            or not summary['audit']['passed'] or summary['audit']['origins'].get('generated')!=24
            or summary['audit']['origins'].get('reused_old')!=156):
        raise ValueError('先完成3072预算对照并通过完整汇总')
    if args.tokens not in [2048,3072]: raise ValueError('显式指定 --tokens 2048 或 3072')
    source=RESULTS/'short219_sources.json'
    if not source.exists(): raise ValueError('先prepare-short审计字幕源')
    chosen=dict(version='fixed_short_subtitle_configuration_v1',tokens=args.tokens,
        budget_candidate_sha256=sha(RESULTS/'budget3072_candidate.json'),budget_summary_sha256=sha(path),
        source_script_sha256=ctx['protocol']['scripts'],short_sources_sha256=sha(source),
        selection='User explicitly selects one development configuration before any short219 score is inspected; no per-question selection or automatic budget scan.')
    write_once(RESULTS/'selected_short_config.json',chosen)
    print('短视频配置已固定：',args.tokens,'；再init short、run short、summarize short。')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['summarize','freeze-short'])
    p.add_argument('--scope',choices=['budget','short'],default='budget'); p.add_argument('--tokens',type=int,choices=[2048,3072])
    p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0: p.error('磁盘预算无效')
    if args.action=='freeze-short': freeze_short(args)
    else: summarize(args,context(args))


if __name__=='__main__': main()
