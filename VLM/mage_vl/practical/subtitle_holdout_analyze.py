#!/usr/bin/env python3
"""val90 只有全部配对完成才读取GT；不选阈值，不根据结果修改候选。"""
import argparse
from collections import Counter
import json

from practical_common import ROOT, LETTERS, sha, write_once
from practical_analyze import comparison
from subtitle_holdout import ARMS, make_context, unit_key, complete


def group_metrics(rows,units,labels):
    if not rows:
        return dict(N=0,videos=0,methods={})
    preds={arm:{r['index']:units[unit_key(r['index'],arm)]['pred'] for r in rows} for arm in ARMS}
    methods={arm:dict(correct=sum(preds[arm][r['index']]==labels[r['index']] for r in rows),
        accuracy=sum(preds[arm][r['index']]==labels[r['index']] for r in rows)/len(rows),
        invalid=sum(preds[arm][r['index']] not in list(LETTERS) for r in rows)) for arm in ARMS}
    methods[ARMS[1]]['vs']={ARMS[0]:comparison(rows,preds[ARMS[1]],preds[ARMS[0]],labels)}
    a=methods[ARMS[1]]['vs'][ARMS[0]]
    methods[ARMS[1]]['quadrants']=dict(both_ok=methods[ARMS[0]]['correct']-a['harm'],
        rescued=a['rescue'],harmed=a['harm'],both_wrong=len(rows)-methods[ARMS[0]]['correct']-a['rescue'])
    return dict(N=len(rows),videos=len({r['video_id'] for r in rows}),methods=methods)


def build_report(ctx,labels):
    if not complete(ctx):
        raise ValueError('配对单元未全部完成，拒绝部分汇总')
    rows=ctx['rows']; units=ctx['units']
    selected=[units[unit_key(r['index'],arm)] for r in rows for arm in ARMS]
    evidence={r['video_id']:ctx['evidence_loader'](r['video_id'])['evidence'] for r in rows}
    invalid=sum(u['pred'] not in list(LETTERS) for u in selected)
    report=dict(signature=ctx['signature'],stage='val',N=len(rows),
        interpretation='Preregistered video-disjoint validation of one subtitle configuration; not a final test or algorithm novelty claim. Historical benchmark use must be disclosed.',
        audit=dict(passed=invalid==0,invalid=invalid,complete_units=len(selected),
            errors=[] if not invalid else ['Invalid answers retained as errors in denominator'],
            native_both_prompt_checks=True,
            subtitle_question_status=dict(Counter(units[unit_key(r['index'],ARMS[1])]['subtitle_status'] for r in rows)),
            subtitle_video_status=dict(Counter(e['status'] for e in evidence.values())),
            text_tokens_per_video_max=max(e['text_tokens'] for e in evidence.values()),
            native_sampled_frames=sum(len(e['frame_indices']) for e in evidence.values()),
            sampled_frames_with_subtitle=sum(sum(bool(ids) for ids in e['frame_cue_ids']) for e in evidence.values()),
            note='Native-input equality and timestamp overlap do not certify decoded pixel correctness or subtitle transcription.'),
        overall=group_metrics(rows,units,labels),by_duration={},groups={},cost={})
    report['by_duration']={d:group_metrics([r for r in rows if r['duration']==d],units,labels)
        for d in sorted({r['duration'] for r in rows})}
    report['by_task_type']={t:group_metrics([r for r in rows if r['task_type']==t],units,labels)
        for t in sorted({r['task_type'] for r in rows})}
    for status in ['used','missing','empty','no_hit']:
        report['groups'][status]=group_metrics([r for r in rows if evidence[r['video_id']]['status']==status],units,labels)
    for arm in ARMS:
        items=[units[unit_key(r['index'],arm)] for r in rows]
        report['cost'][arm]=dict(logical_answer_calls=len(rows),new_answer_calls=sum(u['cost']['new_answer_calls'] for u in items),
            logical_visual_tokens=sum(u['visual']['n_tok'] for u in items),
            input_tokens_sum=sum(u['input_tokens'] for u in items),subtitle_tokens_sum=sum(u['subtitle_tokens'] for u in items),
            incremental_input_tokens_vs_no_sub=sum(u['input_tokens']-units[unit_key(u['index'],ARMS[0])]['input_tokens'] for u in items),
            new_inference_s=sum(u['cost']['inference_s'] for u in items),
            peak_new_generation_gib=max(u['cost']['peak_allocated_gib'] for u in items),
            logical_inference_s=sum((units[unit_key(u['index'],ARMS[0])] if not u['cost']['new_answer_calls'] else u)['cost']['inference_s'] for u in items))
    report['cost']['shared_preprocess_s']=sum(u['cost']['preprocess_s'] for u in selected)
    report['cost']['note']='Paired experiment shares visual preprocessing and performs extra CPU native-input audits; new calls differ from logical deployment costs. Not a controlled speed benchmark.'
    details=[]
    for row in rows:
        idx=row['index']; e=evidence[row['video_id']]
        details.append(dict(index=idx,video_id=row['video_id'],duration=row['duration'],task_type=row['task_type'],gt=labels[idx],
            subtitle_status=e['status'],subtitle_tokens=e['text_tokens'],
            predictions={arm:units[unit_key(idx,arm)]['pred'] for arm in ARMS},
            record_sha256={arm:units[unit_key(idx,arm)]['record_sha256'] for arm in ARMS},
            alignment=str(ctx['store']/'alignments'/f'{row["video_id"]}.json')))
    return report,details


def summarize(ctx):
    # 先完整性检查，再开标签文件，防止中途窥探验证分数影响继续运行的决定。
    if not complete(ctx):
        raise ValueError(f'只完成 {len(ctx["units"])}/180 单元，不读取标签')
    manifest=ROOT/'01_manifests'; path=manifest/'evaluation_labels.json'
    lock=json.loads((manifest/'dataset_lock.json').read_text())
    if sha(path)!=lock['labels_sha256']:
        raise ValueError('标签清单漂移')
    labels=json.loads(path.read_text())
    report,details=build_report(ctx,labels)
    write_once(ctx['stage_dir']/'predictions.json',details)
    write_once(ctx['stage_dir']/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('汇总路径:',ctx['stage_dir']/'summary.json')
    if not report['audit']['passed']:
        raise ValueError('审计失败，保留现场；不自动重跑')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['val'],default='val')
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:
        p.error('磁盘预算无效')
    summarize(make_context(args))


if __name__=='__main__':
    main()
