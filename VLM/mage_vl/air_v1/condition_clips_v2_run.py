"""独立v2运行：不读取GT/人工定位，不改动v1；每个调用均可恢复。"""
import argparse,fcntl,json,os,time,math
from pathlib import Path
import numpy as np
import air_protocol as p
import air_hybrid as h
import condition_clips as old
import condition_clips_v2 as k
from air_mage_adapter import selected_video_inputs,assert_inputs_equal

OUT=p.ROOT/k.VERSION

def context(init=False):
    import subprocess
    if subprocess.check_output(['git','-C',str(p.PROJECT),'branch','--show-current'],text=True).strip()!=p.BRANCH:raise ValueError('分支不符')
    source_path=p.ROOT/'candidate_native_uniform_fix1.json'
    source=json.loads(source_path.read_text());source_sig=p.digest(source)
    folder=Path(__file__).parent
    for n,sha in source['scripts'].items():
        if p.sha(folder/n)!=sha:raise ValueError('原生基线代码漂移')
    for n,sha in source['sources'].items():
        if p.sha(p.OLD/'01_manifests'/n)!=sha:raise ValueError('原始数据清单漂移')
    if source['mage']!=p.model_identity(p.MODEL) or source['clip']!=p.model_identity(p.CLIP,['model.safetensors']):raise ValueError('本地模型身份变化')
    pool,stages=p.load_rows();rows=k.select_rows(pool)
    first_ids={r['index'] for r in stages['dev30']}
    # 只定位旧特征文件；不调用会读取旧答案账本的source_context。
    stores={r['index']:(p.ROOT/'cache'/source_sig if r['index'] in first_ids else p.ROOT/'extension90_v1/source_cache'/source_sig) for r in rows}
    meta={};features={};files={};conditions={};lookup={r['index']:r for r in rows}
    for r in rows:
        idx=r['index'];store=stores[idx];mp=store/'video_metadata'/f'{r["video_id"]}.json'
        m=json.loads(mp.read_text());fp=store/'clip_features'/r['video_id']/'coarse.npz'
        if m['signature']!=source_sig or m['video_id']!=r['video_id']:raise ValueError('元数据身份异常')
        st=Path(r['video_path']).stat()
        if source['videos'][r['video_id']]!=dict(path=r['video_path'],bytes=st.st_size,mtime_ns=st.st_mtime_ns):raise ValueError('源视频身份变化')
        meta[idx]=m;features[idx]=str(fp);files[str(mp)]=p.sha(mp);files[str(fp)]=p.sha(fp);conditions[idx]=old.requirements(r['question'])
    folder=Path(__file__).parent
    code={n:p.sha(folder/n) for n in [*k.CORE,'condition_clips.py']}
    protocol=dict(version=k.VERSION,indices=[r['index'] for r in rows],question_sha={r['index']:p.digest(p.query(r)) for r in rows},conditions=conditions,
        selection='old12 plus9medium+9long, fixed salted SHA rank, distinct new videos, no GT/predictions',
        source_signature=source_sig,files=files,code=code,source_candidate_sha=p.sha(source_path),mage=source['mage'],clip=source['clip'],
        generation=dict(seed=0,greedy=True,observation_max_new=k.CAP,codes='constrained 2 tokens per condition then EOS; U/P/E + visible unit; no repair'),
        budget=dict(global_frames=8,clips=6,seconds=12,frames_per_clip=4,joint_max_frames=32,calls_per_question=16,max_pixels=150000,min_pixels=3136),
        policy='Posthoc development feasibility; no final QA; auto path no manual time ranges/labels/old answers; constrained syntax NOT semantic correctness')
    if init:p.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；代码/来源漂移不覆盖旧版本')
    c=dict(signature=p.digest(protocol),protocol=protocol,rows=lookup,meta=meta,features=features,units={})
    for u in p.read_journal(OUT/'units.jsonl'):
        idx=u['index']
        if idx not in lookup or idx in c['units'] or u['signature']!=c['signature']:raise ValueError('逐题断点重复/来源异常')
        tasks=set()
        for path,sha in u['files'].items():
            path=Path(path)
            if not path.resolve().is_relative_to(OUT.resolve()) or p.sha(path)!=sha:raise ValueError('断点缓存路径/内容变化')
            if path.parent.parent.name=='calls':tasks.add(path.stem)
        expected={stage+'_'+kind for stage in ['global',*[f'local{i}' for i in range(6)],'joint'] for kind in ['observation','codes']}
        if tasks!=expected or len(u['codes'])!=len(conditions[idx]) or u['joint_indices']!=old.combined_frames(meta[idx]['frame_count'],u['windows']):raise ValueError('断点调用/帧预算不完整')
        c['units'][idx]=u
    return c

def coarse(c,idx):
    with np.load(c['features'][idx],allow_pickle=False) as z:
        ids=z['indices'].tolist();f=z['features'].astype('float32');meta=json.loads(str(z['meta'].item()))
    if meta!=dict(signature=c['protocol']['source_signature'],video_id=c['rows'][idx]['video_id'],kind='coarse_1fps',indices_sha256=p.digest(ids)):raise ValueError('特征来源不符')
    norm=np.linalg.norm(f,axis=1,keepdims=True)
    if f.shape!=(len(ids),768) or not np.isfinite(f).all() or (norm<=1e-8).any():raise ValueError('图像特征非法')
    return ids,f/norm

def read_call(path,c,idx,task,prompt,ids,units,conditions,atoms,eos):
    if not path.exists():return None
    d=json.loads(path.read_text());p.unseal(d)
    if (d['signature']!=c['signature'] or d['index']!=idx or d['task']!=task or d['prompt_sha']!=p.digest(prompt) or d['indices']!=ids or d['evidence_units']!=units):raise ValueError('逐调用身份变化')
    if d['kind']=='codes':
        parsed=k.parse_codes(d['generated_ids'],conditions,list(units),atoms,eos)
        if d['codes']!=parsed:raise ValueError('代码解析漂移')
    elif d['kind']=='observation':
        if d['flags']!=k.observation_flags(d['raw_text'],d['generated_tokens']==k.CAP):raise ValueError('观察审计变化')
    else:raise ValueError('未知调用类型')
    if d['generated_tokens']!=len(d['generated_ids']) or len(d['tensor_sha'])!=64:raise ValueError('token/输入哈希异常')
    for n in ['inference_s','preprocess_s','peak_gib']:
        if not math.isfinite(d[n]) or d[n]<0:raise ValueError('成本非法')
    return d

def run(c,stage):
    requested=list(k.SMOKE) if stage=='smoke3' else c['protocol']['indices']
    if stage=='diag30':k.smoke_gate(c['units'])
    if all(i in c['units'] for i in requested):print('阶段完整，跳过模型加载');return
    import random,torch,transformers
    from transformers import AutoProcessor,AutoModelForCausalLM,CLIPModel,CLIPProcessor
    from decord import VideoReader,cpu
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    atoms,eos=k.atom_tokens(proc.tokenizer)
    model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    clip=CLIPModel.from_pretrained(str(p.CLIP),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    tok=CLIPProcessor.from_pretrained(str(p.CLIP),local_files_only=True).tokenizer
    p.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),atoms=atoms,eos=eos,mage_dtype=str(model.dtype)))
    with tqdm(total=len(requested),initial=sum(i in c['units'] for i in requested),desc='条件证据v2 '+stage,unit='题',dynamic_ncols=True) as bar:
      for idx in requested:
        if idx in c['units']:continue
        p.disk_guard();row=c['rows'][idx];m=c['meta'][idx];fps=m['fps'];count=m['frame_count'];conditions=c['protocol']['conditions'][idx]
        vr=VideoReader(row['video_path'],ctx=cpu(0))
        if len(vr)!=count or abs(float(vr.get_avg_fps())-fps)>1e-8:raise ValueError('视频时间元数据变化')
        globals_=old.uniform_indices(count,8);saved={};allflags=[]
        def call(stage_name,kind,ids,units,prompt):
            task=stage_name+'_'+kind;path=OUT/'calls'/idx/(task+'.json')
            d=read_call(path,c,idx,task,prompt,ids,units,conditions,atoms,eos)
            if d is None:
                p.disk_guard();start=time.perf_counter()
                chat=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)
                rgb=vr.get_batch(ids).asnumpy()
                with selected_video_inputs(proc,rgb,ids,fps,count,video_path=row['video_path']):
                    actual=proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=len(ids),max_pixels=150000,return_tensors='pt',padding=True)
                if actual['patch_positions'][:,0].unique_consecutive().tolist()!=ids:raise ValueError('真实原时间位置失真')
                if stage_name=='global' and kind=='observation':
                    native=proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=8,max_pixels=150000,return_tensors='pt',padding=True)
                    assert_inputs_equal(native,actual);del native
                tensor_sha=h.tensor_sha(actual);prep=time.perf_counter()-start;promptlen=int(actual['input_ids'].shape[1])
                gpu={n:t.to(device=model.device,dtype=model.dtype) if n=='pixel_values' else t.to(model.device) for n,t in actual.items()}
                kwargs=dict(do_sample=False,max_new_tokens=k.CAP)
                if kind=='codes':
                    # 只约束状态和证据编号；不能把强制合法格式解释成模型推理正确。
                    kwargs.update(max_new_tokens=2*len(conditions)+1,eos_token_id=eos,pad_token_id=proc.tokenizer.pad_token_id or eos,
                        prefix_allowed_tokens_fn=lambda batch,seq:k.allowed_next(seq[promptlen:].tolist(),len(conditions),list(units),atoms,eos))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with torch.inference_mode():answer=model.generate(**gpu,**kwargs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
                generated=answer[0,promptlen:].tolist();raw=proc.tokenizer.decode(generated,skip_special_tokens=True).strip()
                codes=k.parse_codes(generated,conditions,list(units),atoms,eos) if kind=='codes' else None
                flags=k.observation_flags(raw,len(generated)==k.CAP) if kind=='observation' else None
                grid=actual['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//proc.spatial_merge_size**2
                d=p.seal(dict(signature=c['signature'],index=idx,task=task,kind=kind,prompt_sha=p.digest(prompt),indices=ids,evidence_units=units,tensor_sha=tensor_sha,
                    raw_text=raw,generated_ids=generated,generated_tokens=len(generated),input_tokens=promptlen,visual_tokens=visual,codes=codes,flags=flags,inference_s=elapsed,preprocess_s=prep,peak_gib=peak))
                p.write_once(path,d);del actual,gpu,answer
            saved[str(path)]=p.sha(path)
            print('落盘/复用',idx,task,d['raw_text'],flush=True)
            return d
        def inspect(stage_name,ids,windows,prior=None):
            units=k.evidence_units(globals_,windows,fps)
            obs=call(stage_name,'observation',ids,units,k.prompt(row,conditions,units,'observation',prior=prior))
            codes=call(stage_name,'codes',ids,units,k.prompt(row,conditions,units,'codes',observation=obs['raw_text'],prior=prior))
            allflags.append(obs['flags'])
            return dict(stage=stage_name,observation=obs['raw_text'],flags=obs['flags'],conditions=codes['codes'])
        overview=inspect('global',globals_,[])
        queries=old.query_texts(p.query(row),row['question'],{'targets':[overview['observation']] if overview['observation'].strip() else []})
        tp=OUT/'text_features'/f'{idx}.json'
        if tp.exists():
            tv=json.loads(tp.read_text());p.unseal(tv)
            if tv['signature']!=c['signature'] or tv['texts']!=queries:raise ValueError('查询特征漂移')
        else:
            start=time.perf_counter();vectors=[];chunks_record=[]
            for text in queries:
                chunks=p.text_chunks(tok(text,add_special_tokens=False,truncation=False)['input_ids'],tok.bos_token_id,tok.eos_token_id,77);maximum=max(map(len,chunks))
                pad=[x+[tok.pad_token_id]*(maximum-len(x)) for x in chunks];mask=[[1]*len(x)+[0]*(maximum-len(x)) for x in chunks]
                with torch.inference_mode():v=clip.get_text_features(input_ids=torch.tensor(pad,device='cuda'),attention_mask=torch.tensor(mask,device='cuda'),return_dict=True)
                if not isinstance(v,torch.Tensor):v=v.pooler_output
                if v.shape!=(len(chunks),768) or not torch.isfinite(v).all() or (v.float().norm(dim=-1)<=1e-8).any():raise ValueError('文本特征非法')
                v=v.float();v=(v/v.norm(dim=-1,keepdim=True)).mean(0)
                if v.norm()<=1e-8:raise ValueError('文本均值零向量')
                vectors.append((v/v.norm()).cpu().numpy().astype('float16').astype(float).tolist());chunks_record.append(chunks)
            tv=p.seal(dict(signature=c['signature'],texts=queries,vectors=vectors,token_chunks=chunks_record,wall_s=time.perf_counter()-start));p.write_once(tp,tv)
        vectors=np.asarray(tv['vectors'],dtype='float32');norm=np.linalg.norm(vectors,axis=1,keepdims=True)
        if vectors.shape!=(len(queries),768) or not np.isfinite(vectors).all() or (norm<=1e-8).any():raise ValueError('查询缓存非法')
        ids,f=coarse(c,idx);windows=old.propose(ids,f,vectors/norm,fps,count,conditions);wp=OUT/'windows'/f'{idx}.json'
        p.write_once(wp,p.seal(dict(signature=c['signature'],index=idx,queries=queries,windows=windows)))
        saved[str(wp)]=p.sha(wp);saved[str(tp)]=p.sha(tp)
        reports=[]
        for w in tqdm(windows,desc='局部片段 '+idx,unit='段',leave=False,dynamic_ncols=True):reports.append(inspect('local'+str(w['id']),sorted(set(globals_+w['indices'])),[w]))
        jointids=old.combined_frames(count,windows);joint=inspect('joint',jointids,windows,reports)
        units=k.evidence_units(globals_,windows,fps)
        u=p.seal(dict(signature=c['signature'],index=idx,windows=windows,global_indices=globals_,joint_indices=jointids,codes=joint['conditions'],joint_observation=joint['observation'],
            observation_flags=allflags,structural_issues=k.structural_issues(joint['conditions'],units,windows,fps,count),files=saved,semantically_verified=False))
        p.append(OUT/'units.jsonl',u);c['units'][idx]=u;bar.update(1);bar.set_postfix(index=idx);del vr
    print('完成；请summarize审阅。本轮未生成最终QA答案。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','run','summarize','check-smoke']);ap.add_argument('stage',choices=['smoke3','diag30']);args=ap.parse_args()
    print('SPACE',p.disk_guard());c=context(args.action=='init');print('冻结30题；已完成',len(c['units']))
    if args.action=='check-smoke':print(json.dumps(k.smoke_gate(c['units']),ensure_ascii=False));return
    if args.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认才执行')
        with (p.OLD/'cache/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(context(),args.stage)
    elif args.action=='summarize':
        ids=list(k.SMOKE) if args.stage=='smoke3' else c['protocol']['indices']
        if not set(ids)<=c['units'].keys():raise ValueError('阶段未完整，禁止读取人工定位')
        from condition_clips_v2_eval import summarize
        summarize(c,args.stage)

if __name__=='__main__':main()
