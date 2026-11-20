#!/usr/bin/env python3
"""字幕路线的离线配对统计；只有本模块读取GT。"""
import argparse
from collections import Counter
import json

from practical_common import ROOT, LETTERS, sha, write_once, stage_rows
from practical_analyze import comparison
from practical_subtitles import make_context
from subtitle_common import STAGES, TAG, RESULTS


def group_metrics(rows,units,baseline,labels):
    if not rows:
        return dict(N=0,methods={})
    preds={name:{} for name in ['codec32_no_sub','frames32_no_sub','frames32_aligned_sub']}
    for r in rows:
        idx=r['index']
        preds['codec32_no_sub'][idx]=baseline[f'{idx}:codec:p0']['semantic_pred']
        preds['frames32_no_sub'][idx]=baseline[f'{idx}:frames:p0']['semantic_pred']
        preds['frames32_aligned_sub'][idx]=units[idx]['pred']
    metrics={}
    for name,pred in preds.items():
        correct=sum(pred[r['index']]==labels[r['index']] for r in rows)
        metrics[name]=dict(correct=correct,accuracy=correct/len(rows),
            invalid=sum(pred[r['index']] not in list(LETTERS) for r in rows))
    metrics['frames32_aligned_sub']['vs']={name:comparison(rows,preds['frames32_aligned_sub'],preds[name],labels)
                for name in ['frames32_no_sub','codec32_no_sub']}
    return dict(N=len(rows),videos=len({r['video_id'] for r in rows}),methods=metrics)


def build_report(args,ctx,labels):
    rows=ctx['rows']; units=ctx['units']; base=ctx['base']['units']
    if not ctx['expected'].issubset(units):
        raise ValueError(f'尚未完成 {len(ctx["expected"] & set(units))}/{len(ctx["expected"])} 单元，不做部分汇总')
    selected=[units[r['index']] for r in rows]
    evidence={r['video_id']:ctx['evidence_loader'](r['video_id']) for r in rows}
    invalid=sum(u['pred'] not in list(LETTERS) for u in selected)
    statuses=Counter(u['subtitle_status'] for u in selected)
    errors=[]
    if invalid:
        errors.append(f'{invalid} invalid answers; retained in denominator')
    if args.stage=='smoke':
        if not statuses['used']:
            errors.append('smoke did not exercise any nonempty aligned subtitle')
        if not all(u['checks'].get('native_input_equal') is True and
                   u['checks'].get('no_sub_answer_equal') is True for u in selected):
            errors.append('native input / no-sub control audit not passed')
    report=dict(signature=ctx['signature'],stage=args.stage,N=len(rows),
        interpretation='Additional subtitle modality, development only; not a pure-visual algorithm improvement.',
        audit=dict(passed=not errors,errors=errors,complete_units=len(selected),invalid=invalid,
            subtitle_question_status=dict(sorted(statuses.items())),
            subtitle_video_status=dict(sorted(Counter(e['status'] for e in evidence.values()).items())),
            native_sampled_frames=sum(len(e['frame_indices']) for e in evidence.values()),
            sampled_frames_with_subtitle=sum(sum(bool(ids) for ids in e['frame_cue_ids']) for e in evidence.values()),
            subtitle_cues_available=sum(e['source']['nonempty_cues'] for e in evidence.values()),
            subtitle_cues_selected=sum(len(e['selected_cues']) for e in evidence.values()),
            text_tokens_per_video_max=max(e['text_tokens'] for e in evidence.values()),
            note='Timestamp overlap is audited; it does not certify subtitle transcription or decoded pixel correctness.'),
        overall=group_metrics(rows,units,base,labels),groups={},cost={})
    for status in ['used','missing','empty','no_hit']:
        subset=[r for r in rows if units[r['index']]['subtitle_status']==status]
        report['groups'][status]=group_metrics(subset,units,base,labels)
    report['by_duration']={duration:group_metrics([r for r in rows if r['duration']==duration],units,base,labels)
                           for duration in sorted({r['duration'] for r in rows})}
    if args.stage=='dev60':
        old={r['index'] for r in stage_rows(ROOT,'dev30')}
        report['groups']['new90']=group_metrics([r for r in rows if r['index'] not in old],units,base,labels)
    report['cost']=dict(logical_answer_calls=len(rows),new_answer_calls=sum(u['cost']['new_answer_calls'] for u in selected),
        reused_no_sub_answers=sum(u['cost']['new_answer_calls']==0 for u in selected),
        extra_smoke_control_calls=sum(u['cost']['extra_smoke_control_calls'] for u in selected),
        logical_visual_tokens=sum(u['visual']['n_tok'] for u in selected),
        newly_generated_answer_visual_tokens=sum(u['visual']['n_tok']*u['cost']['new_answer_calls'] for u in selected),
        subtitle_tokens_sum_over_questions=sum(u['subtitle_tokens'] for u in selected),
        new_answer_inference_s=sum(u['cost']['answer_s'] for u in selected),
        extra_smoke_control_s=sum(u['cost']['control_s'] for u in selected),
        preprocess_s=sum(u['cost']['preprocess_s'] for u in selected),
        peak_allocated_gib=max(u['cost']['peak_allocated_gib'] for u in selected),
        note='Baseline answers reused only when input has no aligned subtitle. Logical costs still count those visual answers. '
             'Cached preprocessing is charged to first question; these times are not a controlled speedup benchmark.')
    details=[]
    for r in rows:
        idx=r['index']; e=evidence[r['video_id']]; u=units[idx]
        details.append(dict(index=idx,video_id=r['video_id'],duration=r['duration'],task_type=r['task_type'],gt=labels[idx],
            subtitle_status=e['status'],subtitle_tokens=e['text_tokens'],selected_cue_ids=[c['cue_id'] for c in e['selected_cues']],
            predictions=dict(codec32_no_sub=base[f'{idx}:codec:p0']['semantic_pred'],
                frames32_no_sub=base[f'{idx}:frames:p0']['semantic_pred'],frames32_aligned_sub=u['pred']),
            unit_sha256=u['record_sha256'],alignment=str(ctx['store']/'alignments'/f'{r["video_id"]}.json')))
    return report,details


def summarize(args,ctx):
    if not ctx['expected'].issubset(ctx['units']):
        raise ValueError('该阶段尚未完成，不读取标签或发布部分汇总')
    manifest=ROOT/'01_manifests'
    lock=json.loads((manifest/'dataset_lock.json').read_text())
    if sha(manifest/'evaluation_labels.json')!=lock['labels_sha256']:
        raise ValueError('GT清单发生变化')
    labels=json.loads((manifest/'evaluation_labels.json').read_text())
    report,details=build_report(args,ctx,labels)
    write_once(ctx['stage_dir']/'predictions.json',details)
    write_once(ctx['stage_dir']/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('汇总路径:',ctx['stage_dir']/'summary.json')
    if not report['audit']['passed']:
        raise ValueError('审计未通过，保留记录并停止后续阶段')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:
        p.error('无效磁盘预算')
    summarize(args,make_context(args))


if __name__=='__main__':
    main()
