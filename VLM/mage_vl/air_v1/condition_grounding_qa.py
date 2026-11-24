"""定位审阅后的最终QA；精确复用不变输入，不把解析文字输入答案模型。"""
import json
from pathlib import Path
from contextlib import contextmanager
import air_protocol as p
import condition_grounding_run as r
import condition_clips_qa30 as qa
import condition_clips_noobs_qa30 as n
import condition_clips as old

OUT=r.OUT/'qa'
ORIGINAL_VALIDATE=qa.validate

def context(c):
 gs=r.load_ground(c);src=n.context()
 if len(src['units'])!=30:raise ValueError('旧noobs QA须完整')
 plans={i:dict(indices=u['result']['indices'],fps=c['meta'][i]['fps'],count=c['meta'][i]['frame_count'],reuse_noobs=u['same_as_noobs']) for i,u in gs.items()}
 protocol=dict(version='condition_grounding_v1_QA',parent_signature=c['signature'],indices=c['protocol']['indices'],plans=plans,grounding_files={i:p.sha(r.OUT/'grounding'/f'{i}.json') for i in plans},source_noobs_signature=src['signature'],source_noobs_records={i:u['record_sha256'] for i,u in src['units'].items()},policy='Pixels plus original QA only; no parsed text,subtitles,API or answer override')
 p.write_once(OUT/'candidate.json',protocol)
 out=dict(signature=p.digest(protocol),protocol=protocol,rows=c['rows'],plans=plans,bases=src['bases'],source=src,units={})
 for u in p.read_journal(OUT/'units.jsonl'):validate(out,u,out['units']);out['units'][u['index']]=u
 return out

def validate(c,u,seen):
 if u.get('grounding_reuse_record') is None:
  if c['plans'].get(u['index'],{}).get('reuse_noobs'):raise ValueError('相同输入必须复用')
  ORIGINAL_VALIDATE(c,u,seen);return
 p.unseal(u);i=u['index']
 if i not in c['rows'] or i in seen or u['signature']!=c['signature'] or not c['plans'][i]['reuse_noobs']:raise ValueError('复用身份异常')
 src=c['source']['units'][i]
 if u['grounding_reuse_record']!=src['record_sha256'] or u['indices']!=c['plans'][i]['indices']:raise ValueError('复用来源不符')
 for key in ['raw_text','pred','prompt_sha','tensor_sha','baseline_tensor_sha','baseline_pred','baseline_record_sha','input_tokens','visual_tokens','grid_thw','native_equal']:
  if u[key]!=src[key]:raise ValueError('复用输入/答案变化:'+key)
 if any(u[x]!=0 for x in ['new_calls','generated_tokens','inference_s','preprocess_s','peak_gib']):raise ValueError('复用不得计新生成')

@contextmanager
def scope():
 original_out,original_validate=qa.OUT,qa.validate;qa.OUT,qa.validate=OUT,validate
 try:yield
 finally:qa.OUT,qa.validate=original_out,original_validate

def run(c):
 cc=context(c)
 for i in cc['protocol']['indices']:
  if i in cc['units'] or not cc['plans'][i]['reuse_noobs']:continue
  d=p.unseal(cc['source']['units'][i]);d.update(signature=cc['signature'],grounding_reuse_record=cc['source']['units'][i]['record_sha256'],new_calls=0,generated_tokens=0,inference_s=0.,preprocess_s=0.,peak_gib=0.)
  # 旧source_reuse_record若存在只属历史来源，新阶段单独使用grounding_reuse_record。
  u=p.seal(d);validate(cc,u,cc['units']);p.append(OUT/'units.jsonl',u);cc['units'][i]=u
 with scope():qa.run(cc)

def summarize(c):
 cc=context(c)
 if len(cc['units'])!=30:raise ValueError('QA未完整，不读标签')
 path=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((path.parent/'dataset_lock.json').read_text())
 if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签身份漂移')
 labels=json.loads(path.read_text());import sys
 sys.path.insert(0,str(p.PROJECT/'VLM/mage_vl/practical'))
 from practical_analyze import comparison
 src=cc['source'];rows=list(cc['rows'].values());preds={name:{i:u['pred'] for i,u in units.items()} for name,units in [('uniform32',cc['bases']),('v2',src['source']['units']),('noobs',src['units']),('condition_grounding',cc['units'])]}
 def group(rr):
  if not rr:return dict(N=0)
  return dict(N=len(rr),videos=len({x['video_id'] for x in rr}),methods={name:dict(correct=sum(pp[x['index']]==labels[x['index']] for x in rr),accuracy=sum(pp[x['index']]==labels[x['index']] for x in rr)/len(rr)) for name,pp in preds.items()},versus_uniform=comparison(rr,preds['condition_grounding'],preds['uniform32'],labels),versus_noobs=comparison(rr,preds['condition_grounding'],preds['noobs'],labels))
 bad={name:sum(v not in list('ABCD') for v in pp.values()) for name,pp in preds.items()};gs=r.load_ground(c);us=list(cc['units'].values())
 report=dict(signature=cc['signature'],audit=dict(passed=not any(bad.values()),invalid_answers=bad,native_equal=all(u['native_equal'] for u in us)),overall=group(rows),by_duration={d:group([x for x in rows if x['duration']==d]) for d in ['medium','long']},cohorts={name:group([x for x in rows if (x['index'] in old.IDS)==prior]) for name,prior in [('prior12',True),('extra18',False)]},fallbacks={i:u['fallback_reason'] for i,u in gs.items() if u['fallback_reason']},new_QA_cost=dict(calls=sum(u['new_calls'] for u in us),reused_noobs=sum(u.get('grounding_reuse_record') is not None for u in us),inference_s=sum(u['inference_s'] for u in us),preprocess_s=sum(u['preprocess_s'] for u in us),peak_gib=max(u['peak_gib'] for u in us)),note='Development30,not independent test. Condition parsing and leaf text encoding add costs recorded separately; shared image features still have nonzero original cost. No semantic or accuracy guarantee.')
 details=[dict(index=x['index'],video_id=x['video_id'],duration=x['duration'],gt=labels[x['index']],predictions={name:pp[x['index']] for name,pp in preds.items()},fallback=gs[x['index']]['fallback_reason'],indices=cc['plans'][x['index']]['indices']) for x in rows]
 p.write_once(OUT/'results/summary.json',report);p.write_once(OUT/'results/predictions.json',details);p.write_once(OUT/'results/units_snapshot.json',us)
 print(json.dumps(report,ensure_ascii=False,indent=2));print('条件定位QA汇总:',OUT/'results/summary.json')
 if any(bad.values()):raise ValueError('非法答案保留，请诊断，不自动重跑')
