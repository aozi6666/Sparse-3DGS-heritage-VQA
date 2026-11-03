"""纯CPU整理冻结字幕成果；源文件只读，不调用模型/API/视频解码。"""
import csv,json,sys,os
from pathlib import Path
from types import SimpleNamespace
PROJECT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA');sys.path.insert(0,str(PROJECT/'VQA/mage_vl/practical'))
from practical_common import ROOT,sha,write_once,signature,stage_rows
from practical_vote import disk_guard
from continuous_common import context as dev_context
from subtitle_completion_common import context as short_context,load_alignment,key
from continuous_val90 import context as val_context
from continuous_test225 import context as test_context
OUT=ROOT/'17_subtitle_final_report/20261103_v1'
disk_guard(ROOT,10,1)
contexts={'short219':short_context(SimpleNamespace(scope='short',min_free=10,budget=1)),
 'dev180':dev_context(SimpleNamespace(stage='dev60',min_free=10,budget=1)),
 'val90':val_context(SimpleNamespace(min_free=10,budget=1)),
 'test225':test_context(SimpleNamespace(min_free=10,budget=1))}
ARMS=('no_sub','aligned','continuous'); ROLES={'short219':'historical development coverage','dev180':'development','val90':'previously diagnosed supplementary validation','test225':'frozen final evaluation with historical-use disclosure'}
metrics=[];paired=[];predictions=[];sources={};snapshots={};test_rows={}
for stage,c in contexts.items():
    summary=c['stage_dir']/'summary.json'; details=c['stage_dir']/'predictions.json'
    s=json.loads(summary.read_text()); ds=json.loads(details.read_text())
    assert s['audit']['passed'] and len(ds)==len(c['rows']) and c['expected'].issubset(c['units'])
    for path in [summary,details,c['store']/'units.jsonl']:
        sources[str(path.relative_to(PROJECT))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    write_once(OUT/'source_snapshots'/stage/'summary.json',s)
    names=dict(zip(ARMS,['frames32_no_sub','aligned_sub_v1','continuous_time_v1'])) if stage=='dev180' else {a:a for a in ARMS}
    for group,rs in [('overall',c['rows'])]+[(d,[r for r in c['rows'] if r['duration']==d]) for d in sorted({r['duration'] for r in c['rows']})]:
        old=s['overall'] if group=='overall' else s['by_duration'][group]
        for arm in ARMS:
            m=old['methods'][names[arm]]
            metrics.append(dict(stage=stage,role=ROLES[stage],duration=group,N=len(rs),videos=len({r['video_id'] for r in rs}),method=arm,correct=m['correct'],accuracy_pct=100*m['correct']/len(rs)))
        for baseline in ARMS[:2]:
            m=old['methods'][names['continuous']]['vs'][names[baseline]]
            paired.append(dict(stage=stage,role=ROLES[stage],duration=group,N=len(rs),baseline=baseline,**m))
    for d in ds:
        idx=d['index'];row=next(r for r in c['rows'] if r['index']==idx)
        if stage=='dev180':
            actual=dict(no_sub=c['old']['base']['units'][f'{idx}:frames:p0']['semantic_pred'],aligned=c['old']['units'][idx]['pred'],continuous=c['units'][idx]['pred'])
            assert actual['continuous']==d['pred']
        else:
            actual={a:c['units'][key(idx,a)]['pred'] for a in ARMS};assert actual==d['predictions']
        predictions.append(dict(stage=stage,role=ROLES[stage],index=idx,video_id=row['video_id'],duration=row['duration'],task_type=row['task_type'],gt=d['gt'],**actual))
        if stage=='test225':test_rows[idx]=row
    for m in [x for x in metrics if x['stage']==stage]:
        pp=[r for r in predictions if r['stage']==stage and (m['duration']=='overall' or r['duration']==m['duration'])]
        assert sum(r[m['method']]==r['gt'] for r in pp)==m['correct']
    for m in [x for x in paired if x['stage']==stage]:
        pp=[r for r in predictions if r['stage']==stage and (m['duration']=='overall' or r['duration']==m['duration'])]
        assert sum(r['continuous']==r['gt'] and r[m['baseline']]!=r['gt'] for r in pp)==m['rescue']
        assert sum(r['continuous']!=r['gt'] and r[m['baseline']]==r['gt'] for r in pp)==m['harm']
    snapshots[stage]=dict(summary=str(summary.relative_to(PROJECT)),signature=s['signature'],N=s['N'],role=ROLES[stage])
indices=[r['index'] for r in predictions];allrows=stage_rows(ROOT,'full')
assert len(indices)==len(set(indices))==len(allrows)==714 and set(indices)=={r['index'] for r in allrows}
assert len({r['video_id'] for r in predictions})==238
stages=list(contexts)
for i,a in enumerate(stages):
    for b in stages[i+1:]:assert not ({r['video_id'] for r in predictions if r['stage']==a}&{r['video_id'] for r in predictions if r['stage']==b})
test=json.loads((contexts['test225']['stage_dir']/'summary.json').read_text())
cost=[dict(method=a,questions=225,visual_tokens=test['cost'][a]['logical_visual_tokens'],subtitle_tokens=test['cost'][a]['subtitle_tokens'],input_tokens=test['cost'][a]['input_tokens'],input_tokens_per_question=test['cost'][a]['input_tokens']/225,**{k:test['cost'][a][k] for k in ['new_answer_calls','reused_answers','new_inference_s','peak_new_generation_gib']}) for a in ARMS]
manifest=dict(version='subtitle_final_report_v1',date='2026-11-03 Asia/Shanghai',stages=snapshots,coverage=dict(questions=714,videos=238,no_duplicate_questions=True,video_disjoint=True,official_full_benchmark=False),sources=sources)
write_once(OUT/'source_manifest.json',manifest);write_once(OUT/'stage_metrics.json',metrics);write_once(OUT/'paired_comparisons.json',paired);write_once(OUT/'all_predictions.json',predictions);write_once(OUT/'test225_token_cost.json',cost)
def csv_once(name,rows):
    import io
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    path=OUT/name;data=buf.getvalue()
    if path.exists():assert path.read_text()==data.replace('\r\n','\n')
    else:path.write_text(data,encoding='utf-8')
for name,rows in [('stage_duration_metrics.csv',metrics),('paired_comparisons.csv',paired),('all_predictions.csv',predictions),('test225_token_cost.csv',cost)]:csv_once(name,rows)
# 按固定索引排序选案例，明确为看过GT后的定性示例，不用案例选择调规则。
c=contexts['test225'];pp=sorted([r for r in predictions if r['stage']=='test225'],key=lambda r:int(r['index']));cases=[];seen=set()
rules=[('rescue_beyond_aligned',lambda r:r['no_sub']!=r['gt'] and r['aligned']!=r['gt'] and r['continuous']==r['gt'],3),('subtitle_rescue',lambda r:r['no_sub']!=r['gt'] and r['aligned']==r['gt'] and r['continuous']==r['gt'],2),('harm_vs_no_sub',lambda r:r['no_sub']==r['gt'] and r['continuous']!=r['gt'],3),('harm_vs_aligned',lambda r:r['aligned']==r['gt'] and r['continuous']!=r['gt'],2)]
for kind,rule,n in rules:
    selected=[r for r in pp if rule(r) and r['index'] not in seen][:n]
    for p in selected:
        idx=p['index'];row=test_rows[idx];e=load_alignment(c,row['video_id']);seen.add(idx)
        cases.append(dict(kind=kind,**p,question=row['question'],options=row['bodies'],video_path=row['video_path'],raw_outputs={a:c['units'][key(idx,a)]['raw_text'] for a in ARMS},record_sha256={a:c['units'][key(idx,a)]['record_sha256'] for a in ARMS},evidence=e,interpretation='Prediction transition only; quoted subtitles and actual sampled timestamps retained for human evidence review. No causal or visual correctness certification.'))
write_once(OUT/'qualitative_cases.json',cases)
lines=['# 成功与失败案例','看过真值后按规则及索引排序选出的定性示例，不代表随机样本，也不用于调参。仅整理字幕和采样位置，没有抽取或重新检查画面；不能仅凭这些记录断言因果机制。完整证据与单元哈希见 qualitative_cases.json。']
for p in cases:
    e=p['evidence'];lines.extend([f"\n## {p['index']} · {p['kind']} · {p['duration']} · {p['task_type']}",f"视频：`{p['video_id']}`；真值 {p['gt']}；无字幕 {p['no_sub']} → 原对齐 {p['aligned']} → 连续 {p['continuous']}。",p['question'],json.dumps(p['options'],ensure_ascii=False),f"实际帧秒数：{e['aligned']['frame_seconds']}",f"原对齐字幕（以下至多1200字符，展示截取不影响模型输入）：\n\n{e['aligned']['text'][:1200]}",f"连续字幕（展示至多1800字符）：\n\n{e['continuous']['text'][:1800]}",f"模型原始输出：{json.dumps(p['raw_outputs'],ensure_ascii=False)}"])
(OUT/'qualitative_cases.md').write_text('\n\n'.join(lines)+'\n')
assert not any(n in sys.modules for n in ['torch','transformers','decord'])
print(json.dumps(dict(out=str(OUT),questions=714,videos=238,cases=len(cases),source_files=len(sources)),ensure_ascii=False))
