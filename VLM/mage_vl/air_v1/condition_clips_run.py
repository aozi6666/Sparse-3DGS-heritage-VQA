"""自动运行路径只依赖原视频/问题/CLIP，不读取人工定位或GT。"""
import argparse,json,os,fcntl,time,math
from pathlib import Path
import numpy as np
import air_protocol as p
import air_hybrid as h
import air_extension as e
from air_mage_adapter import selected_video_inputs,assert_inputs_equal
import condition_clips as k

OUT=p.ROOT/k.VERSION

def context(init=False):
    first=p.context('dev30');second=e.source_context()
    sources={r['index']:s for s in [first,second] for r in s['rows']}
    rows={};meta={};features={};files={};conditions={}
    for idx in k.IDS:
        s=sources[idx];row=s['lookup'][idx];rows[idx]=row
        mp=s['store']/'video_metadata'/f'{row["video_id"]}.json';m=json.loads(mp.read_text());meta[idx]=m
        fp=s['store']/'clip_features'/row['video_id']/'coarse.npz';features[idx]=str(fp)
        if m['signature']!=s['signature']:raise ValueError('视频身份不符')
        files[str(mp)]=p.sha(mp);files[str(fp)]=p.sha(fp);conditions[idx]=k.requirements(row['question'])
    folder=Path(__file__).parent
    protocol=dict(version=k.VERSION,indices=list(k.IDS),question_sha256={i:p.digest(p.query(r)) for i,r in rows.items()},source_signature=first['signature'],source_files=files,
        code={n:p.sha(folder/n) for n in ['condition_clips.py','condition_clips_run.py','condition_clips_eval.py','run_condition_clips.sh']},
        conditions=conditions,model=first['protocol']['mage'],clip=first['protocol']['clip'],generation=dict(seed=0,greedy=True,max_new=k.MAX_NEW,pixels=150000),
        budget=dict(questions=12,logical_calls_per_question=8,max_attempts=2,global_frames=8,windows=6,seconds=12,local_frames=4,joint_max_frames=32),
        policy='Automated posthoc feasibility only,no human time ranges/GT/review text read by init/preflight/run. No final QA or answer replacement.')
    if init:p.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；来源/规则/代码漂移不覆盖')
    c=dict(signature=p.digest(protocol),protocol=protocol,rows=rows,meta=meta,features=features,units={})
    for u in p.read_journal(OUT/'units.jsonl'):
        p.unseal(u);idx=u['index']
        if idx not in k.IDS or idx in c['units'] or u['signature']!=c['signature']:raise ValueError('断点身份/重复非法')
        tasks=set()
        for path,expected in u['files'].items():
            if not Path(path).resolve().is_relative_to(OUT.resolve()):raise ValueError('断点引用路径越出本阶段')
            if p.sha(path)!=expected:raise ValueError('逐调用/候选缓存漂移')
            if Path(path).parent.parent.name=='calls':tasks.add(Path(path).stem.rsplit('_',1)[0])
        if tasks!= {'global','joint',*[f'local{i}' for i in range(6)]} or u['logical_calls']!=8:raise ValueError('逐问题8任务不完整')
        if u['global_indices']!=k.uniform_indices(meta[idx]['frame_count'],8) or u['joint_indices']!=k.combined_frames(meta[idx]['frame_count'],u['windows']):raise ValueError('断点候选输入漂移')
        c['units'][idx]=u
    return c

def read_coarse(c,idx):
    path=Path(c['features'][idx])
    with np.load(path,allow_pickle=False) as z:
        ids=z['indices'].tolist();f=z['features'].astype(np.float32);m=json.loads(str(z['meta'].item()))
    if m!=dict(signature=c['protocol']['source_signature'],video_id=c['rows'][idx]['video_id'],kind='coarse_1fps',indices_sha256=p.digest(ids)):raise ValueError('CLIP特征来源身份不符')
    norm=np.linalg.norm(f,axis=1,keepdims=True)
    if f.shape!=(len(ids),768) or not np.isfinite(f).all() or (norm<=1e-8).any():raise ValueError('CLIP特征非法')
    return ids,f/norm

def load_call(path,c,idx,task,prompt,ids,conditions):
    if not path.exists():return None
    u=json.loads(path.read_text());p.unseal(u)
    report,error=k.parse_report(u['raw_text'],ids,conditions)
    if (u['signature']!=c['signature'] or u['index']!=idx or u['task']!=task or u['prompt_sha256']!=p.digest(prompt)
        or u['indices']!=ids or u['report']!=report or u['parse_error']!=error):raise ValueError('逐调用身份/解析漂移')
    if not 1<=u['generated_tokens']<=k.MAX_NEW or len(u['tensor_sha256'])!=64:raise ValueError('生成/输入审计非法')
    for name in ['inference_s','preprocess_s','peak_gib']:
        if not math.isfinite(u[name]) or u[name]<0:raise ValueError('成本非法')
    return u

def run(c):
    if len(c['units'])==12:print('12题自动报告完整，跳过模型加载');return
    import random,torch,transformers
    from transformers import AutoProcessor,AutoModelForCausalLM,CLIPModel,CLIPProcessor
    from decord import VideoReader,cpu
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    clip=CLIPModel.from_pretrained(str(p.CLIP),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    tokenizer=CLIPProcessor.from_pretrained(str(p.CLIP),local_files_only=True).tokenizer
    p.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),mage_dtype=str(model.dtype),clip_dtype=str(clip.dtype)))
    with tqdm(total=12,initial=len(c['units']),desc='完整条件片段验证',unit='题',dynamic_ncols=True) as bar:
      for idx in k.IDS:
        if idx in c['units']:continue
        p.disk_guard();row=c['rows'][idx];meta=c['meta'][idx];fps=meta['fps'];count=meta['frame_count'];conditions=c['protocol']['conditions'][idx]
        vr=VideoReader(row['video_path'],ctx=cpu(0))
        if len(vr)!=count or abs(float(vr.get_avg_fps())-fps)>1e-8:raise ValueError('源视频元数据变化')
        global_ids=k.uniform_indices(count,8);saved={}
        def inputs(ids,prompt):
            chat=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
            rgb=vr.get_batch(ids).asnumpy()
            with selected_video_inputs(proc,rgb,ids,fps,count,video_path=row['video_path']):
                actual=proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=len(ids),max_pixels=150000,return_tensors='pt',padding=True)
            if actual['patch_positions'][:,0].unique_consecutive().tolist()!=ids:raise ValueError('原时间/帧号失真')
            return actual,chat
        def call(task,ids,windows,prior=None):
            original=k.make_prompt(row,conditions,task,ids,global_ids,windows,prior);prompt=original
            selected=None
            for attempt in range(2):
                path=OUT/'calls'/idx/f'{task}_{attempt}.json';u=load_call(path,c,idx,task,prompt,ids,conditions)
                if u is None:
                    p.disk_guard();start=time.perf_counter();actual,chat=inputs(ids,prompt)
                    if task=='global' and attempt==0:
                        native=proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=8,max_pixels=150000,return_tensors='pt',padding=True)
                        assert_inputs_equal(native,actual);del native
                    tensor_sha=h.tensor_sha(actual);prep=time.perf_counter()-start
                    gpu={n:t.to(device=model.device,dtype=model.dtype) if n=='pixel_values' else t.to(model.device) for n,t in actual.items()}
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    with torch.inference_mode():answer=model.generate(**gpu,max_new_tokens=k.MAX_NEW,do_sample=False)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
                    generated=int(answer.shape[1]-actual['input_ids'].shape[1]);raw=proc.tokenizer.decode(answer[0,actual['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                    parsed,error=k.parse_report(raw,ids,conditions);grid=actual['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//proc.spatial_merge_size**2
                    u=p.seal(dict(signature=c['signature'],index=idx,task=task,attempt=attempt,prompt_sha256=p.digest(prompt),indices=ids,tensor_sha256=tensor_sha,input_tokens=int(actual['input_ids'].shape[1]),visual_tokens=visual,generated_tokens=generated,raw_text=raw,report=parsed,parse_error=error,inference_s=elapsed,preprocess_s=prep,peak_gib=peak))
                    p.write_once(path,u);del actual,gpu,answer
                saved[str(path)]=p.sha(path);selected=u
                print('落盘/复用',idx,task,attempt,'valid',u['report'] is not None,flush=True)
                if u['report'] is not None:break
                # 只修复一次格式，旧原文保留；不添加新视觉/人工证据。
                prompt=original+'\nFORMAT REPAIR: Previous response failed schema: '+str(u['parse_error'])+'. Return only the requested compact JSON, no markdown. Use fewer short facts.'
            return selected
        global_record=call('global',global_ids,[])
        queries=k.query_texts(p.query(row),row['question'],global_record['report'])
        textpath=OUT/'text_features'/f'{idx}.json'
        if textpath.exists():
            tv=json.loads(textpath.read_text());p.unseal(tv)
            if tv['signature']!=c['signature'] or tv['texts']!=queries:raise ValueError('自动查询缓存漂移')
        else:
            start=time.perf_counter();vectors=[];token_chunks=[]
            for text in queries:
                ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'];chunks=p.text_chunks(ids,tokenizer.bos_token_id,tokenizer.eos_token_id,77);maximum=max(map(len,chunks))
                pad=[x+[tokenizer.pad_token_id]*(maximum-len(x)) for x in chunks];mask=[[1]*len(x)+[0]*(maximum-len(x)) for x in chunks]
                with torch.inference_mode():v=clip.get_text_features(input_ids=torch.tensor(pad,device='cuda'),attention_mask=torch.tensor(mask,device='cuda'),return_dict=True)
                if not isinstance(v,torch.Tensor):v=v.pooler_output
                if v.shape!=(len(chunks),768) or not torch.isfinite(v).all():raise ValueError('文本投影非法')
                v=v.float();norm=v.norm(dim=-1,keepdim=True)
                if torch.any(norm<=1e-8):raise ValueError('文本向量为零')
                v=(v/norm).mean(0);norm=v.norm()
                if norm<=1e-8:raise ValueError('文本均值向量为零')
                vectors.append((v/norm).cpu().numpy().astype(np.float16).astype(float).tolist());token_chunks.append(chunks)
            tv=p.seal(dict(signature=c['signature'],texts=queries,vectors=vectors,token_chunks=token_chunks,wall_s=time.perf_counter()-start));p.write_once(textpath,tv)
        vectors=np.asarray(tv['vectors'],dtype=np.float32);norm=np.linalg.norm(vectors,axis=1,keepdims=True)
        if vectors.shape!=(len(queries),768) or not np.isfinite(vectors).all() or (norm<=1e-8).any():raise ValueError('文本特征缓存非法')
        coarse_ids,features=read_coarse(c,idx);windows=k.propose(coarse_ids,features,vectors/norm,fps,count,conditions)
        wp=OUT/'windows'/f'{idx}.json';p.write_once(wp,p.seal(dict(signature=c['signature'],index=idx,queries=queries,windows=windows)));saved[str(wp)]=p.sha(wp);saved[str(textpath)]=p.sha(textpath)
        local=[]
        for w in tqdm(windows,desc='片段 '+idx,unit='段',leave=False,dynamic_ncols=True):
            ids=sorted(set(global_ids+w['indices']));r=call('local'+str(w['id']),ids,[w]);local.append(dict(window=w['id'],report=r['report'],parse_error=r['parse_error']))
        joint_ids=k.combined_frames(count,windows);joint=call('joint',joint_ids,windows,local)
        guarded=k.guard(joint['report'],conditions,windows,fps,count,len(k.explicit_items(row['question'])))
        u=p.seal(dict(signature=c['signature'],index=idx,windows=windows,global_indices=global_ids,joint_indices=joint_ids,logical_calls=8,invalid_reports=int(global_record['report'] is None)+sum(r['report'] is None for r in local)+int(joint['report'] is None),joint=joint['report'],guard=guarded,files=saved))
        p.append(OUT/'units.jsonl',u);c['units'][idx]=u;bar.update(1);bar.set_postfix(index=idx,status=guarded['status']);del vr
    print('自动报告完成；请summarize离线评估，未生成最终问答答案。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','run','summarize']);args=ap.parse_args()
    print('SPACE',p.disk_guard());c=context(args.action=='init');print('自动12题，完成',len(c['units']))
    if args.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认后才run')
        with (p.OLD/'cache/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(context())
    elif args.action=='summarize':
        if len(c['units'])!=12:raise ValueError('自动任务未完整，禁止读取人工定位参考')
        from condition_clips_eval import summarize
        summarize(c)

if __name__=='__main__':main()
