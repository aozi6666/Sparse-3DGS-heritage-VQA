"""完整30题后离线评分；主要测相对noobs的增量，保留历史uniform差异。"""
import json
from pathlib import Path
import sys

import air_protocol as p
import condition_clips as cohorts
import leaf_qa30_run as r
import leaf_qa30_protocol as k
from leaf_probe import write_once


def legacy_comparison():
    sys.path.insert(0,str(p.PROJECT/'VQA/mage_vl/practical'))
    from practical_analyze import comparison
    return comparison


def paired(rows,pred,base,labels,compare):
    out=compare(rows,pred,base,labels)
    out.update(both_correct=sum(pred[x['index']]==labels[x['index']]==base[x['index']] for x in rows),
               both_wrong=sum(pred[x['index']]!=labels[x['index']] and base[x['index']]!=labels[x['index']] for x in rows),
               agreement=sum(pred[x['index']]==base[x['index']] for x in rows))
    return out


def analyze(c,labels,compare):
    k.require_complete(c)
    if any(i not in labels or labels[i] not in list('ABCD') for i in c['protocol']['indices']):raise ValueError('标签不全或非法，完整30题不丢样')
    src=c['noobs'];rows=[c['rows'][i] for i in c['protocol']['indices']]
    preds=dict(uniform32={i:u['pred'] for i,u in src['bases'].items()},noobs={i:u['pred'] for i,u in src['units'].items()},
               frozen_leaf7_fallback23={i:u['pred'] for i,u in c['units'].items()})
    for i in c['protocol']['indices']:
        if not c['plans'][i]['generate'] and c['units'][i]['raw_text']!=src['units'][i]['raw_text']:raise ValueError('23题复用答案发生改变')
    def group(rr):
        if not rr:return dict(N=0,videos=0,methods={})
        return dict(N=len(rr),videos=len({x['video_id'] for x in rr}),
                    methods={name:dict(correct=sum(pp[x['index']]==labels[x['index']] for x in rr),accuracy=sum(pp[x['index']]==labels[x['index']] for x in rr)/len(rr),invalid=sum(pp[x['index']] not in list('ABCD') for x in rr)) for name,pp in preds.items()},
                    versus_noobs=paired(rr,preds['frozen_leaf7_fallback23'],preds['noobs'],labels,compare),
                    versus_uniform=paired(rr,preds['frozen_leaf7_fallback23'],preds['uniform32'],labels,compare),
                    historical_noobs_versus_uniform=paired(rr,preds['noobs'],preds['uniform32'],labels,compare))
    units=list(c['units'].values());invalid={name:sum(v not in list('ABCD') for v in pp.values()) for name,pp in preds.items()}
    report=dict(signature=c['signature'],audit=dict(passed=not any(invalid.values()),source_input_audits=len(c['audits']),complete_predictions=30,
                                                   invalid_answers=invalid,reused_answers_unchanged=True,full_tensor_source_parity=True),
                overall=group(rows),cohorts={name:group([row for row in rows if (row['index'] in cohorts.IDS)==prior]) for name,prior in [('prior12',True),('extra18',False)]},
                by_duration={duration:group([row for row in rows if row['duration']==duration]) for duration in ['medium','long']},
                by_scope={name:group([row for row in rows if row['index'] in ids]) for name,ids in c['protocol']['groups'].items()},
                execution=dict(planned_new_generations=sum(x['generate'] for x in c['plans'].values()),saved_new_generations=len(c['generations']),
                               reused_answers=sum(u['origin']=='reused_noobs' for u in units),
                               changed_inputs_by_cohort={name:sum(c['plans'][row['index']]['generate'] for row in rows if (row['index'] in cohorts.IDS)==prior) for name,prior in [('prior12',True),('extra18',False)]},
                               frame_counts={i:dict(old=len(plan['old_noobs_indices']),new=len(plan['indices'])) for i,plan in c['plans'].items() if plan['group']=='candidate'}),
                new_stage_cost=dict(source_input_audit_s=sum(a['preprocess_s'] for a in c['audits'].values()),
                                    candidate_input_preprocess_s=sum(u['preprocess_s'] for u in units),
                                    saved_generation_inference_s=sum(u['inference_s'] for u in units),
                                    generated_tokens=sum(u['generated_tokens'] for u in units),
                                    generated_visual_tokens=sum(u['input']['visual_tokens'] for u in units if u['new_calls']),
                                    logical_all30_visual_tokens=sum(u['input']['visual_tokens'] for u in units),
                                    logical_all30_QA_answers=30,
                                    reused_answers_source_recorded_inference_s=sum(src['units'][u['index']]['inference_s'] for u in units if not u['new_calls']),
                                    peak_new_generation_gib=max(u['peak_gib'] for u in units),at_token_cap=sum(u['generated_tokens']==32 for u in units)),
                interpretation='Single frozen seen-development30 diagnosis with manually chosen7 scope, not independent test or general routing. '
                               'Primary increment is versus_noobs; versus_uniform includes historical noobs effects. '
                               'Whole30 and extra18 retained, including484. No retuning or automatic expansion.',
                cost_note='Saved successful generation costs across resumptions; interrupted unsaved attempts may add unmeasured work. '
                          '30 source audits and extra native-parity preprocessing are validation overhead, not a deployment speed benchmark. '
                          'Historical reports retained separately, do not add their overlapping costs blindly.')
    details=[dict(index=row['index'],video_id=row['video_id'],duration=row['duration'],cohort='prior12' if row['index'] in cohorts.IDS else 'extra18',
                  group=c['plans'][row['index']]['group'],gt=labels[row['index']],predictions={name:pp[row['index']] for name,pp in preds.items()},
                  origin=c['units'][row['index']]['origin'],new_calls=c['units'][row['index']]['new_calls'],
                  old_noobs_indices=c['plans'][row['index']]['old_noobs_indices'],indices=c['plans'][row['index']]['indices'],
                  source_input_audit_sha=c['audits'][row['index']]['record_sha256'],unit_sha=c['units'][row['index']]['record_sha256']) for row in rows]
    return report,details


def summarize(c):
    k.require_complete(c)
    labels,sha=r.read_labels(c)
    report,details=analyze(c,labels,legacy_comparison());report['evaluation_labels_sha256']=sha
    # 只在预测完整后读取这些历史报告，保留实测来源成本，不作为生成/路由输入。
    paths=[r.selector.OUT/'results/summary.json',r.selector.text_source.OUT/'results/summary.json',r.noobs.OUT/'results/summary.json']
    report['historical_pipeline_reports']={str(path):dict(sha256=p.sha(path),costs={name:value for name,value in json.loads(path.read_text()).items() if 'cost' in name}) for path in paths}
    report['retained_failure_events']={str(path):dict(sha256=p.sha(path),phase=json.loads(path.read_text())['phase'],index=json.loads(path.read_text())['index']) for path in sorted((r.OUT/'failures').glob('*.json'))}
    if any(p.sha(path)!=digest for path,digest in c['protocol']['source_files'].items()):raise ValueError('运行中来源文件发生变化，保留现场，不发布汇总')
    write_once(r.OUT/'results/summary.json',report);write_once(r.OUT/'results/predictions.json',details)
    write_once(r.OUT/'results/records_snapshot.json',[c['units'][i] for i in c['protocol']['indices']])
    print(json.dumps(report,ensure_ascii=False,indent=2));print('冻结7题QA汇总：',r.OUT/'results/summary.json')
    if not report['audit']['passed']:raise ValueError('非法答案已保留计错，不重跑；先诊断，不扩样')
