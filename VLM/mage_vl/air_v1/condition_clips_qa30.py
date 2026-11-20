"""冻结v2画面直接QA；短观察/条件状态均不进入答案提示。"""
import argparse,json,os,fcntl,time,math
from pathlib import Path
import air_protocol as p
import air_hybrid as h
import condition_clips as old
import condition_clips_v2_run as v2
from air_mage_adapter import selected_video_inputs,assert_inputs_equal

OUT=p.ROOT/'condition_clips_v2_qa30'

def context(init=False):
    src=v2.context()
    if len(src['units'])!=30:raise ValueError('v2冻结30题须先完整')
    bases={};stores={Path(fp).parents[2] for fp in src['features'].values()}
    for store in stores:
        seen={};bc=dict(signature=src['protocol']['source_signature'],lookup=src['rows'],store=store)
        for u in p.read_journal(store/'units.jsonl'):
            if u['index'] not in src['rows'] or u['arm']!='uniform32':continue
            key=p.validate_unit(bc,u,seen);seen[key]=u
            if u['index'] in bases:raise ValueError('重复基线来源')
            bases[u['index']]=u
    if set(bases)!=set(src['rows']):raise ValueError('缺少旧同题均匀基线，不自动补跑')
    plans={}
    for idx,u in src['units'].items():
        m=src['meta'][idx];ids=u['joint_indices']
        if not 1<=len(ids)<=32 or ids!=old.combined_frames(m['frame_count'],u['windows']):raise ValueError('冻结选帧不符')
        plans[idx]=dict(indices=ids,fps=m['fps'],count=m['frame_count'],v2_record_sha=u['record_sha256'])
    folder=Path(__file__).parent;summary=v2.OUT/'results/diag30/summary.json'
    protocol=dict(version='condition_clips_v2_qa30',indices=src['protocol']['indices'],v2_signature=src['signature'],v2_candidate_sha=p.sha(v2.OUT/'candidate.json'),
        plans=plans,baseline_records={i:u['record_sha256'] for i,u in bases.items()},historical_selector_summary_sha=p.sha(summary),
        code={n:p.sha(folder/n) for n in ['condition_clips_qa30.py','run_condition_clips_qa30.sh','air_hybrid.py','air_mage_adapter.py']},
        prompt_header=p.ANSWER_HEADER,generation=dict(seed=0,max_new_tokens=32,greedy=True,max_pixels=150000,min_pixels=3136),
        policy='Same frozen30,only final QA. No short observations/condition states/manual references/GT in answer inputs; posthoc development diagnostic.')
    if init:p.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；来源/代码漂移保留现场')
    c=dict(signature=p.digest(protocol),protocol=protocol,rows=src['rows'],plans=plans,bases=bases,units={})
    for u in p.read_journal(OUT/'units.jsonl'):
        validate(c,u,c['units']);c['units'][u['index']]=u
    return c

def validate(c,u,seen):
    p.unseal(u);idx=u['index']
    if idx not in c['rows'] or idx in seen or u['signature']!=c['signature']:raise ValueError('断点身份/重复异常')
    base=c['bases'][idx];plan=c['plans'][idx]
    if u['indices']!=plan['indices'] or u['baseline_record_sha']!=base['record_sha256'] or u['baseline_pred']!=base['pred'] or u['pred']!=p.parse_answer(u['raw_text']):raise ValueError('断点来源/答案不符')
    if u['prompt_sha']!=p.digest(p.prompt(c['rows'][idx])) or u['native_equal'] is not True or len(u['tensor_sha'])!=64 or len(u['baseline_tensor_sha'])!=64:raise ValueError('输入审计异常')
    same=plan['indices']==base['indices']
    if u['new_calls']!=int(not same) or (same and (u['raw_text']!=base['raw_text'] or u['generated_tokens']!=0 or u['inference_s']!=0)):raise ValueError('复用非法')
    if not same and not 1<=u['generated_tokens']<=32:raise ValueError('生成长度非法')
    for n in ['inference_s','preprocess_s','peak_gib']:
        if not math.isfinite(u[n]) or u[n]<0:raise ValueError('成本非法')

def input_pair(proc,vr,row,plan,base):
    # 唯一QA文本是旧统一提示+原题/选项，绝不加入观察、条件状态或人工定位。
    chat=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':p.prompt(row)}]}],tokenize=False,add_generation_prompt=True)
    def native():return proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)
    def selected(ids):
        rgb=vr.get_batch(ids).asnumpy()
        with selected_video_inputs(proc,rgb,ids,plan['fps'],plan['count'],video_path=row['video_path']):return native()
    original=native();adapt=selected(base['indices']);assert_inputs_equal(original,adapt);del adapt
    if (p.digest(original['input_ids'].tolist())!=base['input_sha256'] or original['image_grid_thw'].tolist()!=base['visual']['grid_thw']
        or original['patch_positions'][:,0].unique_consecutive().tolist()!=base['indices']):raise ValueError('旧基线输入/token/grid/时间身份不符')
    use=original if plan['indices']==base['indices'] else selected(plan['indices'])
    if use['patch_positions'][:,0].unique_consecutive().tolist()!=plan['indices']:raise ValueError('冻结画面时间索引漂移')
    return original,use

def run(c):
    if len(c['units'])==30:print('30题完整，跳过模型加载');return
    import random,numpy as np,torch,transformers
    from transformers import AutoProcessor,AutoModelForCausalLM
    from decord import VideoReader,cpu
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    p.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype)))
    with tqdm(total=30,initial=len(c['units']),desc='冻结选帧 QA30',unit='题',dynamic_ncols=True) as bar:
      for idx in c['protocol']['indices']:
        if idx in c['units']:continue
        p.disk_guard();start=time.perf_counter();row=c['rows'][idx];plan=c['plans'][idx];base=c['bases'][idx]
        vr=VideoReader(row['video_path'],ctx=cpu(0))
        if len(vr)!=plan['count'] or abs(float(vr.get_avg_fps())-plan['fps'])>1e-8:raise ValueError('视频元数据漂移')
        original,use=input_pair(proc,vr,row,plan,base)
        baseline_sha=h.tensor_sha(original);selected_sha=h.tensor_sha(use);prep=time.perf_counter()-start
        grid=use['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//proc.spatial_merge_size**2
        same=plan['indices']==base['indices'];raw=base['raw_text'];generated=0;elapsed=peak=0.
        if not same:
            gpu={n:t.to(device=model.device,dtype=model.dtype) if n=='pixel_values' else t.to(model.device) for n,t in use.items()}
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode():answer=model.generate(**gpu,max_new_tokens=32,do_sample=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
            generated=int(answer.shape[1]-use['input_ids'].shape[1]);raw=proc.tokenizer.decode(answer[0,use['input_ids'].shape[1]:],skip_special_tokens=True).strip();del gpu,answer
        u=p.seal(dict(signature=c['signature'],index=idx,indices=plan['indices'],baseline_record_sha=base['record_sha256'],baseline_pred=base['pred'],baseline_raw_text=base['raw_text'],
            raw_text=raw,pred=p.parse_answer(raw),prompt_sha=p.digest(p.prompt(row)),tensor_sha=selected_sha,baseline_tensor_sha=baseline_sha,native_equal=True,
            new_calls=int(not same),generated_tokens=generated,visual_tokens=visual,input_tokens=int(use['input_ids'].shape[1]),grid_thw=grid,inference_s=elapsed,preprocess_s=prep,peak_gib=peak))
        validate(c,u,c['units']);p.append(OUT/'units.jsonl',u);c['units'][idx]=u;bar.update(1);bar.set_postfix(index=idx,answer=u['pred'] or 'INVALID');print('落盘',idx,u['pred'],flush=True);del original,use,vr
    print('完成，请summarize；未自动扩样。')

def summarize(c):
    if len(c['units'])!=30:raise ValueError('未完整，不读取标签')
    path=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((path.parent/'dataset_lock.json').read_text())
    if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签身份漂移')
    labels=json.loads(path.read_text())
    import sys
    sys.path.insert(0,str(p.PROJECT/'VQA/mage_vl/practical'))
    from practical_analyze import comparison
    rows=list(c['rows'].values());preds={i:u['pred'] for i,u in c['units'].items()};base={i:u['pred'] for i,u in c['bases'].items()}
    def group(rr):
        if not rr:return dict(N=0)
        return dict(N=len(rr),videos=len({r['video_id'] for r in rr}),methods={name:dict(correct=sum(pp[r['index']]==labels[r['index']] for r in rr),accuracy=sum(pp[r['index']]==labels[r['index']] for r in rr)/len(rr)) for name,pp in [('uniform32',base),('frozen_v2_frames',preds)]},versus_uniform=comparison(rr,preds,base,labels))
    units=list(c['units'].values());bad=sum(u['pred'] not in 'ABCD' or len(u['pred'])!=1 for u in units);badbase=sum(c['bases'][i]['pred'] not in list('ABCD') for i in c['rows'])
    report=dict(signature=c['signature'],audit=dict(passed=not(bad or badbase),invalid_selected=bad,invalid_baseline=badbase,native_parity_all=True),overall=group(rows),
        by_duration={d:group([r for r in rows if r['duration']==d]) for d in ['medium','long']},
        cohorts={name:group([r for r in rows if (r['index'] in old.IDS)==prior]) for name,prior in [('prior12',True),('extra18',False)]},
        new_QA_cost=dict(calls=sum(u['new_calls'] for u in units),uniform_answers_reused=30,selected_answers_reused=sum(not u['new_calls'] for u in units),inference_s=sum(u['inference_s'] for u in units),preprocess_s=sum(u['preprocess_s'] for u in units),visual_tokens=sum(u['visual_tokens'] for u in units),peak_gib=max(u['peak_gib'] for u in units)),
        historical_selection_cost=json.loads((v2.OUT/'results/diag30/summary.json').read_text())['cost'],
        note='Posthoc development30,not independent test. Frozen pixels only; no observation/code input. New QA cost excludes cached selector work, historical selection cost retained. Native parity preprocessing is not a deployment speed benchmark.')
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],uniform32=base[r['index']],frozen_v2_frames=preds[r['index']],kind=('both_ok' if preds[r['index']]==labels[r['index']] else 'harm') if base[r['index']]==labels[r['index']] else ('rescue' if preds[r['index']]==labels[r['index']] else 'both_wrong'),indices=c['plans'][r['index']]['indices']) for r in rows]
    p.write_once(OUT/'results/summary.json',report);p.write_once(OUT/'results/predictions.json',details);p.write_once(OUT/'results/units_snapshot.json',units)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('QA汇总:',OUT/'results/summary.json')
    if bad or badbase:raise ValueError('非法答案保留，请先诊断，不自动重跑')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','run','summarize']);a=ap.parse_args();print('SPACE',p.disk_guard());c=context(a.action=='init');print('冻结30题，完成',len(c['units']),'最多新增QA',sum(c['plans'][i]['indices']!=c['bases'][i]['indices'] for i in c['rows']))
    if a.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认才run')
        with (p.OLD/'cache/gpu.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run(context())
    elif a.action=='summarize':summarize(c)

if __name__=='__main__':main()
