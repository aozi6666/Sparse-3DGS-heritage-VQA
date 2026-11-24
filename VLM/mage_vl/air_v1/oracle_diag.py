"""人工式证据定位诊断；只用原画面和原问题，不传定位解释/真值。"""
import argparse,fcntl,json,math,os,time
from pathlib import Path
from collections import Counter
import air_protocol as p
import air_hybrid as h
import air_extension as e
from air_mage_adapter import selected_video_inputs,assert_inputs_equal

OUT=p.ROOT/'oracle_evidence_diag_v1'
GROUPS={'selector_harm':['379','418','484','626'],'prior_rescue':['265','450','486','704'],'all_three_wrong':['325','454','625','655']}
IDS=[i for group in GROUPS.values() for i in group]
ARMS=('uniform32','air_cap32','hybrid32','evidence32')

def validate_inputs(obj):
    # 输入清单只许题号与帧号；排除答案、说明、字幕等附加文本字段。
    if set(obj)!= {'version','rows'} or obj['version']!='oracle_evidence_diag_v1':raise ValueError('诊断输入schema非法')
    if [r.get('index') for r in obj['rows']]!=IDS:raise ValueError('12题固定，不可换题/顺序')
    for r in obj['rows']:
        if set(r)!= {'index','frame_indices'}:raise ValueError('模型输入清单禁止附加答案或解释')
        ix=r['frame_indices']
        if len(ix)!=32 or ix!=sorted(set(ix)) or any(type(i) is not int or i<0 for i in ix):raise ValueError('必须32个不同原始帧号')
    return obj

def bundles():
    first=h.context();source=e.source_context()
    with e.hybrid_scope(source):second=h.context()
    return {r['index']:c for c in [first,second] for r in c['source']['rows']}

def validate_unit(c,u,seen):
    p.unseal(u);idx=u['index']
    if idx not in IDS or idx in seen or u['signature']!=c['signature'] or u['indices']!=c['indices'][idx]:raise ValueError('断点身份/帧号非法')
    if u['prompt_sha256']!=p.digest(p.prompt(c['rows'][idx])) or u['pred']!=p.parse_answer(u['raw_text']):raise ValueError('原始提示词/答案解析变化')
    if not u['native_equal'] or len(u['tensor_sha256'])!=64 or u['visual_tokens']!=c['baselines'][idx]['uniform32']['visual']['n_tok']:raise ValueError('输入审计不符')
    if u['input_tokens']<=u['visual_tokens']:raise ValueError('文本token异常')
    for k in ['inference_s','preprocess_s','peak_gib']:
        if not math.isfinite(u[k]) or u[k]<0:raise ValueError('成本非法')
    arm=u['reuse_arm']
    if arm is not None:
        old=c['baselines'][idx][arm]
        if arm not in ARMS[:3] or c['indices'][idx]!=old['indices'] or u['source_record']!=old['record_sha256'] or u['raw_text']!=old['raw_text'] or u['new_calls']!=0 or u['generated_tokens']!=0 or u['inference_s']!=0:raise ValueError('复用来源非法')
    elif u['source_record'] is not None or u['new_calls']!=1 or type(u['generated_tokens']) is not int or not 1<=u['generated_tokens']<=32:raise ValueError('生成记录非法')

def context(init=False):
    inputs=validate_inputs(json.loads((OUT/'inputs_reviewed_v2.json').read_text()));reviews=json.loads((OUT/'review_v2.json').read_text())
    if set(reviews)!=set(IDS):raise ValueError('评审记录缺题')
    bs=bundles();rows={};meta={};baselines={};sourceids={};indices={r['index']:r['frame_indices'] for r in inputs['rows']}
    for idx in IDS:
        b=bs[idx];src=b['source'];rows[idx]=src['lookup'][idx]
        m=json.loads((src['store']/'video_metadata'/f'{rows[idx]["video_id"]}.json').read_text());meta[idx]=m
        if max(indices[idx])>=m['frame_count']:raise ValueError('定位帧越界')
        baselines[idx]={a:(b['units'][idx] if a=='hybrid32' else src['units'][f'{idx}:{a}']) for a in ARMS[:3]}
        if reviews[idx]['group']!=next(g for g,ids in GROUPS.items() if idx in ids) or reviews[idx]['status'] not in ['explicit','uncertain','visual_insufficient']:raise ValueError('证据评审分组/状态非法')
        sourceids[idx]=dict(hybrid_signature=b['signature'],source_signature=src['signature'],records={a:u['record_sha256'] for a,u in baselines[idx].items()})
    folder=Path(__file__).parent
    protocol=dict(version='oracle_evidence_diag_v1',input_sha256=p.sha(OUT/'inputs_reviewed_v2.json'),review_sha256=p.sha(OUT/'review_v2.json'),sources=sourceids,meta=meta,
        code={n:p.sha(folder/n) for n in ['oracle_diag.py','run_oracle_diag.sh']},
        generation=dict(max_pixels=150000,min_pixels=3136,frames=32,greedy=True,max_new_tokens=32,seed=0),
        model_input_policy='Original source RGB,original frame indices/time,original p.prompt(row) ONLY. No review text,subtitle,answer or explanation. Assistant visual localization,not independently human-certified upper bound.')
    if init:p.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；身份漂移不覆盖旧诊断')
    c=dict(signature=p.digest(protocol),protocol=protocol,rows=rows,meta=meta,baselines=baselines,indices=indices,reviews=reviews,units={})
    for u in p.read_journal(OUT/'units.jsonl'):validate_unit(c,u,c['units']);c['units'][u['index']]=u
    return c

def run(c):
    if len(c['units'])==12:print('12题完成，跳过模型加载');return
    import random,numpy as np,torch,transformers
    from transformers import AutoProcessor,AutoModelForCausalLM
    from decord import VideoReader,cpu
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    p.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,dtype=str(model.dtype),gpu=torch.cuda.get_device_name(0)))
    with tqdm(total=12,initial=len(c['units']),desc='Mage 正确证据到位诊断',unit='题',dynamic_ncols=True) as bar:
        for idx in IDS:
            if idx in c['units']:continue
            p.disk_guard();before=time.perf_counter();row=c['rows'][idx];m=c['meta'][idx];inds=c['indices'][idx]
            vr=VideoReader(row['video_path'],ctx=cpu(0))
            if len(vr)!=m['frame_count'] or abs(float(vr.get_avg_fps())-m['fps'])>1e-8:raise ValueError('视频元数据变化')
            # 唯一文本来自旧清单原问题；不拼入reviews、标签或人工定位说明。
            text=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':p.prompt(row)}]}],tokenize=False,add_generation_prompt=True)
            def native():return proc(text=[text],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)
            def selected(ix):
                raw=vr.get_batch(ix).asnumpy()
                with selected_video_inputs(proc,raw,ix,m['fps'],m['frame_count'],video_path=row['video_path']):return native()
            base=c['baselines'][idx]['uniform32'];actual=native();adapt=selected(base['indices']);assert_inputs_equal(actual,adapt);del adapt
            if p.digest(actual['input_ids'].tolist())!=base['input_sha256'] or actual['image_grid_thw'].tolist()!=base['visual']['grid_thw']:raise ValueError('历史基线输入身份漂移')
            use=actual if inds==base['indices'] else selected(inds)
            if use['patch_positions'][:,0].unique_consecutive().tolist()!=inds:raise ValueError('真实帧号未保留')
            grid=use['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//proc.spatial_merge_size**2
            if visual!=base['visual']['n_tok']:raise ValueError('视觉预算不等于本题基线')
            ids_sha=p.digest(use['input_ids'].tolist());tensor_sha=h.tensor_sha(use);prep=time.perf_counter()-before
            reuse=None;source_record=None;elapsed=peak=0.;generated=0
            for arm,old in c['baselines'][idx].items():
                if inds==old['indices']:
                    expected=old.get('input_ids_sha256',old.get('input_sha256'))
                    if ids_sha!=expected:raise ValueError('复用原输入身份不符')
                    if arm=='hybrid32' and tensor_sha!=old['tensor_sha256']:raise ValueError('复用hybrid像素/位置哈希不符')
                    reuse=arm;source_record=old['record_sha256'];raw=old['raw_text'];break
            if reuse is None:
                gpu={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in use.items()}
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with torch.inference_mode():answer=model.generate(**gpu,max_new_tokens=32,do_sample=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
                generated=int(answer.shape[1]-use['input_ids'].shape[1]);raw=proc.tokenizer.decode(answer[0,use['input_ids'].shape[1]:],skip_special_tokens=True).strip();del gpu,answer
            u=p.seal(dict(signature=c['signature'],index=idx,indices=inds,prompt_sha256=p.digest(p.prompt(row)),input_ids_sha256=ids_sha,tensor_sha256=tensor_sha,visual_tokens=visual,input_tokens=int(use['input_ids'].shape[1]),native_equal=True,raw_text=raw,pred=p.parse_answer(raw),reuse_arm=reuse,source_record=source_record,new_calls=int(reuse is None),generated_tokens=generated,inference_s=elapsed,preprocess_s=prep,peak_gib=peak))
            validate_unit(c,u,c['units']);p.append(OUT/'units.jsonl',u);c['units'][idx]=u;bar.update(1);bar.set_postfix(index=idx,status=c['reviews'][idx]['status'],answer=u['pred'] or 'INVALID');print('落盘',idx,u['pred'],flush=True)
            del use,actual,vr

def summarize(c):
    if len(c['units'])!=12:raise ValueError('12题未完整，不读GT')
    labelpath=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((labelpath.parent/'dataset_lock.json').read_text())
    if p.sha(labelpath)!=lock['labels_sha256']:raise ValueError('GT身份漂移')
    labels=json.loads(labelpath.read_text())
    pred={arm:{idx:(c['units'][idx] if arm=='evidence32' else c['baselines'][idx][arm])['pred'] for idx in IDS} for arm in ARMS}
    def group(ids):
        if not ids:return dict(N=0)
        return dict(N=len(ids),methods={arm:dict(correct=sum(pred[arm][i]==labels[i] for i in ids),accuracy=sum(pred[arm][i]==labels[i] for i in ids)/len(ids),invalid=sum(pred[arm][i] not in list('ABCD') for i in ids)) for arm in ARMS},versus={arm:dict(rescue=sum(pred['evidence32'][i]==labels[i] and pred[arm][i]!=labels[i] for i in ids),harm=sum(pred['evidence32'][i]!=labels[i] and pred[arm][i]==labels[i] for i in ids)) for arm in ARMS[:3]})
    bad=sum(pred['evidence32'][i] not in list('ABCD') for i in IDS);uu=list(c['units'].values())
    report=dict(signature=c['signature'],audit=dict(passed=not bad,invalid_answers=bad,complete=12),overall=group(IDS),explicit_evidence=group([i for i in IDS if c['reviews'][i]['status']=='explicit']),uncertain_evidence=group([i for i in IDS if c['reviews'][i]['status']=='uncertain']),visual_insufficient=group([i for i in IDS if c['reviews'][i]['status']=='visual_insufficient']),by_group={g:group(ids) for g,ids in GROUPS.items()},cost=dict(new_calls=sum(u['new_calls'] for u in uu),reused=sum(not u['new_calls'] for u in uu),inference_s=sum(u['inference_s'] for u in uu),preprocess_s=sum(u['preprocess_s'] for u in uu),visual_tokens=sum(u['visual_tokens'] for u in uu)),interpretation='Posthoc evidence-localization diagnostic on fixed12,not algorithm accuracy/not mathematical upper bound. Explicit evidence reviewed by assistant,not independently human certified. Interpret insufficient cases separately; no significance claim.')
    details=[dict(index=i,group=c['reviews'][i]['group'],evidence_status=c['reviews'][i]['status'],gt=labels[i],predictions={arm:pred[arm][i] for arm in ARMS}) for i in IDS]
    p.write_once(OUT/'results/predictions.json',details);p.write_once(OUT/'results/summary.json',report);p.write_once(OUT/'results/units_snapshot.json',uu)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('诊断汇总:',OUT/'results/summary.json')
    if bad:raise ValueError('非法答案保留，请诊断')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','preflight','run','summarize']);args=parser.parse_args()
    print('SPACE',p.disk_guard());c=context(args.action=='init');print('12题定位状态',dict(Counter(x['status'] for x in c['reviews'].values())),'完成',len(c['units']))
    if args.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认才run')
        with (p.OLD/'cache/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(context())
    elif args.action=='summarize':summarize(c)

if __name__=='__main__':main()
