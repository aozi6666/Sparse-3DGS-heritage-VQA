"""单因素消融：精确复用共享向量，仅移除全局短观察查询；不调用证据模型。"""
import argparse,json,os,fcntl,time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import air_protocol as p
import air_hybrid as h
import condition_clips as old
import condition_clips_v2_run as v2
import condition_clips_qa30 as qa

OUT=p.ROOT/'condition_clips_noobs_qa30'
ORIGINAL_VALIDATE=qa.validate

def plan_one(row,meta,coarse_ids,features,tv):
    queries=old.query_texts(p.query(row),row['question'],None)
    if len(set(tv['texts']))!=len(tv['texts']) or not all(q in tv['texts'] for q in queries):raise ValueError('共享原查询/子事件向量缺失，不自动编码')
    vectors=np.asarray([tv['vectors'][tv['texts'].index(q)] for q in queries],dtype='float32');norm=np.linalg.norm(vectors,axis=1,keepdims=True)
    if vectors.shape!=(len(queries),768) or not np.isfinite(vectors).all() or (norm<=1e-8).any():raise ValueError('文本缓存非法')
    windows=old.propose(coarse_ids,features,vectors/norm,meta['fps'],meta['frame_count'],old.requirements(row['question']))
    return dict(indices=old.combined_frames(meta['frame_count'],windows),fps=meta['fps'],count=meta['frame_count'],queries=queries,windows=windows)

def context(init=False):
    start=time.perf_counter();source=qa.context();s=v2.context()
    if len(source['units'])!=30:raise ValueError('原v2 QA30须先完整')
    plans={};files={}
    for idx,row in source['rows'].items():
        tp=v2.OUT/'text_features'/f'{idx}.json';tv=json.loads(tp.read_text());p.unseal(tv)
        if tv['signature']!=s['signature']:raise ValueError('共享向量来源异常')
        ids,f=v2.coarse(s,idx);plan=plan_one(row,s['meta'][idx],ids,f,tv)
        plan['reuse_v2']=plan['indices']==source['plans'][idx]['indices'];plans[idx]=plan;files[str(tp)]=p.sha(tp);files[s['features'][idx]]=p.sha(s['features'][idx])
    folder=Path(__file__).parent
    protocol=dict(version='condition_clips_noobs_qa30',indices=source['protocol']['indices'],source_qa_signature=source['signature'],source_v2_signature=s['signature'],plans=plans,files=files,
        source_QA_records={i:u['record_sha256'] for i,u in source['units'].items()},source_baselines=source['protocol']['baseline_records'],generation=source['protocol']['generation'],
        code={n:p.sha(folder/n) for n in ['condition_clips_noobs_qa30.py','run_condition_clips_noobs_qa30.sh']},
        policy='Only remove automatic global observation query; reuse exact shared vectors and image features. No evidence calls,API,subtitles,QA prompt changes. Posthoc development30.')
    if init:
        p.write_once(OUT/'candidate.json',protocol)
        if not (OUT/'planning_cost.json').exists():p.write_once(OUT/'planning_cost.json',dict(first_CPU_planning_wall_s=time.perf_counter()-start,note='Includes source validation/file hashing,not deployment selector benchmark'))
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；来源/代码漂移不覆盖')
    c=dict(signature=p.digest(protocol),protocol=protocol,rows=source['rows'],plans=plans,bases=source['bases'],source=source,units={})
    for u in p.read_journal(OUT/'units.jsonl'):validate(c,u,c['units']);c['units'][u['index']]=u
    return c

def validate(c,u,seen):
    if u.get('source_reuse_record') is None:
        if c['plans'].get(u['index'],{}).get('reuse_v2'):raise ValueError('相同输入必须复用原QA')
        ORIGINAL_VALIDATE(c,u,seen);return
    p.unseal(u);idx=u['index']
    if idx not in c['rows'] or idx in seen or u['signature']!=c['signature'] or not c['plans'][idx]['reuse_v2']:raise ValueError('复用身份异常')
    src=c['source']['units'][idx]
    if u['source_reuse_record']!=src['record_sha256'] or u['indices']!=c['plans'][idx]['indices']:raise ValueError('复用来源漂移')
    for name in ['raw_text','pred','prompt_sha','tensor_sha','baseline_tensor_sha','baseline_pred','baseline_record_sha','input_tokens','visual_tokens','grid_thw','native_equal']:
        if u[name]!=src[name]:raise ValueError('复用完整输入/答案字段变化:'+name)
    if any(u[name]!=0 for name in ['new_calls','generated_tokens','inference_s','preprocess_s','peak_gib']):raise ValueError('复用不得计新生成/预处理成本')

@contextmanager
def runtime_scope():
    # 只在本进程重定向独立输出与校验；旧源文件及磁盘结果不修改。
    original_out,original_validate=qa.OUT,qa.validate
    qa.OUT,qa.validate=OUT,validate
    try:yield
    finally:qa.OUT,qa.validate=original_out,original_validate

def run(c):
    # 未变输入直接复用已审计完整张量身份与答案，不加载模型或再次解码。
    for idx in c['protocol']['indices']:
        if idx in c['units'] or not c['plans'][idx]['reuse_v2']:continue
        d=p.unseal(c['source']['units'][idx]);d.update(signature=c['signature'],source_reuse_record=c['source']['units'][idx]['record_sha256'],new_calls=0,generated_tokens=0,inference_s=0.,preprocess_s=0.,peak_gib=0.)
        u=p.seal(d);validate(c,u,c['units']);p.append(OUT/'units.jsonl',u);c['units'][idx]=u
        print('复用相同v2输入/答案',idx,flush=True)
    with runtime_scope():qa.run(c)

def summarize(c):
    if len(c['units'])!=30:raise ValueError('未完整，不读取标签')
    path=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((path.parent/'dataset_lock.json').read_text())
    if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签身份漂移')
    labels=json.loads(path.read_text());import sys
    sys.path.insert(0,str(p.PROJECT/'VQA/mage_vl/practical'))
    from practical_analyze import comparison
    rows=list(c['rows'].values());preds={name:{i:u['pred'] for i,u in units.items()} for name,units in [('uniform32',c['bases']),('v2_with_observation',c['source']['units']),('no_observation',c['units'])]}
    def group(rr):
        if not rr:return dict(N=0)
        return dict(N=len(rr),videos=len({r['video_id'] for r in rr}),methods={name:dict(correct=sum(pp[r['index']]==labels[r['index']] for r in rr),accuracy=sum(pp[r['index']]==labels[r['index']] for r in rr)/len(rr)) for name,pp in preds.items()},versus_uniform=comparison(rr,preds['no_observation'],preds['uniform32'],labels),versus_v2=comparison(rr,preds['no_observation'],preds['v2_with_observation'],labels))
    bad={name:sum(v not in list('ABCD') for v in pp.values()) for name,pp in preds.items()};units=list(c['units'].values())
    report=dict(signature=c['signature'],audit=dict(passed=not any(bad.values()),invalid_answers=bad,native_parity_all=all(u['native_equal'] for u in units)),overall=group(rows),by_duration={d:group([r for r in rows if r['duration']==d]) for d in ['medium','long']},cohorts={name:group([r for r in rows if (r['index'] in old.IDS)==prior]) for name,prior in [('prior12',True),('extra18',False)]},
        inputs=dict(same_as_v2=sum(x['reuse_v2'] for x in c['plans'].values()),changed_from_v2=sum(not x['reuse_v2'] for x in c['plans'].values())),
        cost=dict(new_QA_calls=sum(u['new_calls'] for u in units),reused_v2_answers=sum(u.get('source_reuse_record') is not None for u in units),reused_uniform_answers=sum(not u['new_calls'] and u.get('source_reuse_record') is None for u in units),new_QA_inference_s=sum(u['inference_s'] for u in units),new_QA_preprocess_s=sum(u['preprocess_s'] for u in units),new_QA_visual_tokens=sum(u['visual_tokens'] for u in units if u['new_calls']),peak_new_generation_gib=max(u['peak_gib'] for u in units),evidence_calls=0,text_encoding_calls=0,CPU_planning=json.loads((OUT/'planning_cost.json').read_text())),
        archived_v2_selection_cost=json.loads((v2.OUT/'results/diag30/summary.json').read_text())['cost'],note='Fixed development30 single ablation,NOT independent test. Archived v2 evidence calls are not required by no-observation deployment; shared image/text features still have nonzero original encoding cost. New QA/preprocessing costs not deployment benchmark.')
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],predictions={a:pp[r['index']] for a,pp in preds.items()},input_unchanged_from_v2=c['plans'][r['index']]['reuse_v2'],queries=c['plans'][r['index']]['queries'],indices=c['plans'][r['index']]['indices']) for r in rows]
    p.write_once(OUT/'results/summary.json',report);p.write_once(OUT/'results/predictions.json',details);p.write_once(OUT/'results/units_snapshot.json',units)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('消融汇总:',OUT/'results/summary.json')
    if any(bad.values()):raise ValueError('非法答案保留，请诊断，不自动重跑')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','run','summarize']);a=ap.parse_args();print('SPACE',p.disk_guard());c=context(a.action=='init')
    same=sum(x['reuse_v2'] for x in c['plans'].values());new=sum(not x['reuse_v2'] and x['indices']!=c['bases'][i]['indices'] for i,x in c['plans'].items());print('同30题；复用v2',same,'最多新增QA',new,'完成',len(c['units']))
    if a.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认才run')
        with (p.OLD/'cache/gpu.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run(context())
    elif a.action=='summarize':summarize(c)

if __name__=='__main__':main()
