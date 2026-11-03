"""只做CPU重算与归档；不修正校验、不读密钥、不调用模型或网络。"""
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from collections import Counter

PROJECT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
sys.path.insert(0,str(PROJECT/'VQA/mage_vl/practical'))
from practical_common import ROOT,sha,signature,write_once,stage_rows
from practical_analyze import comparison
from continuous_analyze import group as mage_group
from deepseek_analyze import group as api_group
from subtitle_common import subtitle_source
import deepseek_review as api

OUT=ROOT/'13_method_review/20261103_v1'
args=SimpleNamespace(stage='pilot',min_free=10,budget=1)
ctx=api.context(args); mc=ctx['continuous']; old=mc['old']
manifest=ROOT/'01_manifests'; lock=json.loads((manifest/'dataset_lock.json').read_text())
assert sha(manifest/'evaluation_labels.json')==lock['labels_sha256']
labels=json.loads((manifest/'evaluation_labels.json').read_text())


def text_once(path,text):
    path=Path(path)
    if path.exists():
        # CSV使用CRLF；read_text会归一化换行，必须比较原始UTF-8字节以保持幂等。
        if path.read_bytes()!=text.encode('utf-8'): raise ValueError(f'归档冲突，不覆盖: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream: stream.write(text)


def predictions(rows,include_api=False):
    methods={name:{} for name in ['mage_codec_no_sub','mage_frames32_no_sub','aligned_sub_v1','continuous_time_v1']}
    if include_api: methods.update(deepseek_independent={},mage_deepseek_review={})
    for row in rows:
        idx=row['index']
        methods['mage_codec_no_sub'][idx]=old['base']['units'][f'{idx}:codec:p0']['semantic_pred']
        methods['mage_frames32_no_sub'][idx]=old['base']['units'][f'{idx}:frames:p0']['semantic_pred']
        methods['aligned_sub_v1'][idx]=old['units'][idx]['pred']
        methods['continuous_time_v1'][idx]=mc['units'][idx]['pred']
        if include_api:
            methods['deepseek_independent'][idx]=ctx['units'][idx]['api_pred']
            methods['mage_deepseek_review'][idx]=ctx['units'][idx]['system_pred']
    return methods


csv_rows=[]; cohort_details={}


def cohort(name,rows,scope,include_api=False):
    preds=predictions(rows,include_api); results={}
    for duration in ['overall',*sorted({r['duration'] for r in rows})]:
        subset=rows if duration=='overall' else [r for r in rows if r['duration']==duration]
        results[duration]={}
        for method,pred in preds.items():
            correct=sum(pred[r['index']]==labels[r['index']] for r in subset)
            entry=dict(correct=correct,N=len(subset),videos=len({r['video_id'] for r in subset}),accuracy=correct/len(subset),
                invalid_or_abstain=sum(pred[r['index']] not in list('ABCD') for r in subset))
            if method!='mage_frames32_no_sub':
                entry['vs_no_sub_frames32']=comparison(subset,pred,preds['mage_frames32_no_sub'],labels)
            if method=='mage_deepseek_review':
                entry['vs_continuous_time']=comparison(subset,pred,preds['continuous_time_v1'],labels)
            results[duration][method]=entry
            csv_rows.append(dict(cohort=name,scope=scope,duration=duration,method=method,**{k:entry[k] for k in ['correct','N','videos','accuracy','invalid_or_abstain']}))
    cohort_details[name]=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],task_type=r['task_type'],
        gt=labels[r['index']],predictions={method:p[r['index']] for method,p in preds.items()}) for r in rows]
    return dict(scope=scope,indices_sha256=signature([r['index'] for r in rows]),metrics=results)


dev=cohort('dev180',mc['rows'],'development, already used for method selection')
pilot=cohort('api_pilot40',ctx['rows'],'development; selected by old predictions, enriched for disagreements',True)
smoke_rows=stage_rows(ROOT,'smoke')
smoke=cohort('short_smoke6',smoke_rows,'interface smoke only, not representative short performance')

# 从源日志重算，绝不改已有summary或校验规则。
source_paths=[]
for stage in ['smoke','dev60']:
    for base in [ROOT/'09_subtitles'/stage/'f32_px150000_aligned_sub_v1',ROOT/'11_continuous_subtitles'/stage/'f32_px150000_neighbor1_time_t2048_v1']:
        source_paths += [base/'summary.json',base/'predictions.json']
assert mage_group(mc['rows'],mc,labels)==json.loads((mc['stage_dir']/'summary.json').read_text())['overall']
api_summary=json.loads((ctx['stage_dir']/'summary.json').read_text())
assert api_group(ctx['rows'],ctx,labels)==api_summary['overall']
for stage in ['smoke','pilot']:
    source_paths += [ROOT/'12_deepseek_review'/stage/n for n in ['summary.json','predictions.json']]
val=ROOT/'10_subtitle_holdout/val90/f32_px150000_aligned_sub_v1'
source_paths += [val/n for n in ['summary.json','predictions.json','diagnostic_rescue7_harm8.json']]
short_path=PROJECT/'VQA/eval_results/r2v_mage/px150000_f32/short/summary_short.json'
short=json.loads(short_path.read_text())['metrics']; source_paths.append(short_path)
short_metrics=dict(N=short['N'],codec_correct=short['both_ok']+short['codec_only'],frames_correct=short['both_ok']+short['frames_only'],
    codec_accuracy=short['codec_accuracy'],frames_accuracy=short['frames_accuracy'],
    scope='historical R2V short219; do not combine with dev180 as an independent test; new subtitle/API full short runs absent')

long_flow=Counter(); long_cases=[]
for row in ctx['rows']:
    idx=row['index']; u=ctx['units'][idx]
    if row['duration']!='long': continue
    base=old['base']['units'][f'{idx}:frames:p0']['semantic_pred']
    if u['errors']['independent']: reason='invalid_first_pass'
    elif not u['api_pred']: reason='valid_abstention'
    elif u['api_pred']==base: reason='same_as_base'
    elif u['verify'] is not None: reason='verification_rejected'
    else: reason='no_eligible_dense_window'
    long_flow[reason]+=1
    times=[f['seconds'] for f in u['frames']['independent']]
    long_cases.append(dict(index=idx,video_id=row['video_id'],question=row['question'],options=row['bodies'],task_type=row['task_type'],gt=labels[idx],
        baseline=base,aligned=old['units'][idx]['pred'],continuous=mc['units'][idx]['pred'],api=u['api_pred'],system=u['system_pred'],
        reason=reason,independent=u['independent'],verify=u['verify'],errors=u['errors'],had_dense=u['had_dense'],
        overridden=u['overridden'],duration_s=u['duration'],global_max_gap_s=max(b-a for a,b in zip(times,times[1:])),
        dense_frames=sum(f['dense'] for f in u['frames']['verify']),response_sha256=u['response_sha256']))
assert sum(long_flow.values())==20 and not any(c['overridden'] for c in long_cases)

# 保存关键诊断的来源位置；全文字幕只读作错误分析，不注入新推理。
footrow=next(r for r in ctx['rows'] if r['index']=='595')
source,cues=subtitle_source(footrow['video_id']); evidence=mc['evidence'][footrow['video_id']]
input_ids={i for b in evidence['blocks'] for i in b['cue_ids']}
source_diagnostics=dict(video_id=footrow['video_id'],source=source,continuous_text_tokens=evidence['text_tokens'],
    omitted_budget_neighbor_ids=evidence['omitted_neighbor_ids'],
    campaign_cues=[dict(cue_id=c['cue_id'],start_ms=c['start_ms'],end_ms=c['end_ms'],was_supplied=c['cue_id'] in input_ids)
        for c in cues if c['cue_id'] in range(51,62)],
    midfielder_cues=[dict(cue_id=c['cue_id'],start_ms=c['start_ms'],end_ms=c['end_ms'],was_supplied=c['cue_id'] in input_ids)
        for c in cues if c['cue_id'] in range(748,752)],
    interpretation='Direct evidence exists in full local SRT but is outside the frame-hit+neighbor context. No budget trimming for this video. This is diagnostic evidence, not a tested Agent improvement.')
assert not evidence['omitted_neighbor_ids']
u=ctx['units']['579']; supplied={f['id'] for f in u['frames']['independent']}
format_issue=dict(index='579',frame_citations=len(u['independent']['frame_ids']),implemented_cap=16,
    unsupplied_frame_ids=[x for x in u['independent']['frame_ids'] if x not in supplied],
    errors=u['errors']['independent'],status='not fixed or rescored; retain audit.passed=false')

budget=[]
for video,e in mc['evidence'].items():
    if not e['omitted_neighbor_ids']: continue
    full=__import__('continuous_common').continuous_evidence(e['original'],old['cues'][video],old['count_tokens'],budget=1000000)
    budget.append(dict(video_id=video,selected_tokens=e['text_tokens'],full_neighbor_tokens=full['text_tokens'],
        omitted_neighbor_ids=e['omitted_neighbor_ids'],status='CPU size audit only; no 3072-token inference experiment'))

summary=dict(version='method_review_20261103_v1',date='2026-11-03 Asia/Shanghai',
    current_branch='vlm',mage_signature=mc['signature'],api_signature=ctx['signature'],
    dev180=dev,api_pilot40=pilot,short_smoke6=smoke,historical_short219=short_metrics,
    val90_original_aligned=json.loads((val/'summary.json').read_text()),
    api_pilot_original_summary=api_summary,long_flow=dict(long_flow),long_cases=long_cases,
    direct_source_diagnostics=source_diagnostics,format_issue=format_issue,budget_audit=budget,
    conclusions=['Retain codec and no-sub Frames32 baselines.',
        'Retain aligned subtitles as standard-protocol comparator; previous val90 did not sustain gain.',
        'Retain continuous subtitles as leading measured development candidate, not a confirmed test result.',
        'API review currently underperforms continuous subtitles on the same pilot: 16 vs 23; do not claim additive gain.',
        'Keep pilot audit failure and all abstentions; no citation correction or new experiment in this archive.',
        'No formal short219 subtitle/API evaluation; six-question smoke must not be extrapolated.'],
    task_boundary='archive and discussion only; no source-code changes, inference, API calls, key reads, downloads, deletion or test225 evaluation')
write_once(OUT/'comparison.json',summary)
write_once(OUT/'matched_predictions.json',cohort_details)
write_once(OUT/'long20_diagnostics.json',dict(long_flow=dict(long_flow),cases=long_cases,
    direct_source_diagnostics=source_diagnostics,format_issue=format_issue))
buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
text_once(OUT/'comparison.csv',buffer.getvalue())

# 补齐历史short与原val90的已报告分数，仍保留独立cohort标签，不拼接Overall。
all_rows=list(csv_rows)
short_videos=len({r['video_id'] for r in stage_rows(ROOT,'full') if r['duration']=='short'})
for method,correct in [('mage_codec_no_sub',short_metrics['codec_correct']),('mage_frames32_no_sub',short_metrics['frames_correct'])]:
    all_rows.append(dict(cohort='historical_short219',scope='historical R2V development pool, enhanced full short runs absent',
        duration='short',method=method,correct=correct,N=short['N'],videos=short_videos,accuracy=correct/short['N'],invalid_or_abstain=0))
vs=summary['val90_original_aligned']
for duration,metrics in [('overall',vs['overall']),*vs['by_duration'].items()]:
    for source,method in [('frames32_no_sub','mage_frames32_no_sub'),('frames32_aligned_sub','aligned_sub_v1')]:
        metric=metrics['methods'][source]
        all_rows.append(dict(cohort='val90_original_aligned',scope='validation subsequently used for diagnosis; no continuous/API evaluation',
            duration=duration,method=method,correct=metric['correct'],N=metrics['N'],videos=metrics['videos'],accuracy=metric['accuracy'],invalid_or_abstain=metric['invalid']))
buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(all_rows)
text_once(OUT/'all_reported_metrics.csv',buffer.getvalue())

# 小型summary/预测快照及当时源码单独保存；大日志/缓存原位保留，避免复制约147MB成果。
for path in source_paths:
    relative=path.relative_to(PROJECT)
    text_once(OUT/'stage_snapshots'/relative,path.read_text())
code_dir=PROJECT/'VQA/mage_vl/practical'
code_files=[p for p in sorted(code_dir.iterdir()) if p.suffix in ['.py','.sh','.md'] and p.is_file()]
for path in code_files: text_once(OUT/'source_snapshot'/path.name,path.read_text())
sources={str(p.relative_to(PROJECT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [*source_paths,*code_files]}
write_once(OUT/'source_manifest.json',sources)

# 白名单范围只含实验结果根；不会访问项目根deepseek.env或其他密钥。
inventory=[]
for root in [ROOT,PROJECT/'VQA/eval_results/r2v_mage']:
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.is_relative_to(ROOT/'13_method_review'): continue
        if path.name.endswith('.lock'): continue
        inventory.append(dict(path=str(path.relative_to(PROJECT)),bytes=path.stat().st_size,sha256=sha(path)))
write_once(OUT/'original_results_inventory.json',dict(scope=['practical_mage_v1','r2v_mage'],
    excluded=['13_method_review self','symlinks','lock files'],files=inventory,total_bytes=sum(p['bytes'] for p in inventory)))
assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'decord' not in sys.modules
print(json.dumps(dict(archive=str(OUT),cohort_rows=len(csv_rows),result_files_preserved=len(inventory),
    original_bytes=sum(p['bytes'] for p in inventory),long_flow=dict(long_flow),same_pilot_overall={k:v['correct'] for k,v in pilot['metrics']['overall'].items()},
    source_diagnostics=source_diagnostics,format_issue=format_issue,code_sha256={p.name:sha(p) for p in code_files}),ensure_ascii=False,indent=2))
