"""用户手动启动的GPU实现；导入本模块不加载模型或解码视频。"""
import json,time,math
from dataclasses import asdict
from pathlib import Path
from air_protocol import (ROOT,MODEL,CLIP,ARMS,digest,sha,seal,unseal,write_once,append,disk_guard,
    query,prompt,ANALYSIS,text_chunks,parse_answer,validate_unit)
from air_selection import select_frames,AIRConfig,uniform_indices,parse_score
from air_mage_adapter import selected_video_inputs,image_inputs,assert_inputs_equal

def validate_selection_cache(obj,row,hit,signature):
    unseal(obj)
    if (obj['signature']!=signature or obj['index']!=row['index'] or obj['total_frames']!=hit['count']
            or obj['fps']!=hit['fps'] or obj['config']!=asdict(AIRConfig())):raise ValueError('选择断点视频/规则身份不符')
    final=obj['final_indices'];rounds=obj['rounds'];seen=[];positives=[];invalid=[]
    if not final or len(final)>32 or final!=sorted(set(final)) or any(type(i) is not int or not 0<=i<hit['count'] for i in final):raise ValueError('最终选择断点索引非法')
    if len(rounds)>6:raise ValueError('迭代断点超预算')
    for number,r in enumerate(rounds,1):
        if r['iteration']!=number or len(r['candidates'])>12:raise ValueError('单轮断点非法')
        for i in r['candidates']:
            if i in seen or not 0<=i<hit['count']:raise ValueError('重复/越界评分')
            value=r['score_records'][str(i)];score=parse_score(value['raw_text'])
            if value['score']!=score:raise ValueError('评分轨迹解析不符')
            seen.append(i)
            if score is None:invalid.append(i)
            elif score>3:positives.append(i)
    if sorted(seen)!=obj['analyzed_indices'] or sorted(invalid)!=obj['invalid_score_indices']:raise ValueError('评分轨迹清单不符')
    if obj['fallback_reason']:
        if positives or final!=uniform_indices(hit['count'],32):raise ValueError('回退掩盖有效证据或改变基线')
    elif not set(final).issubset(positives):raise ValueError('最终输入并非已验证正分帧')
    return obj

def run_gpu(args,ctx):
    import os,random
    os.environ['HF_MODULES_CACHE']=str(ROOT/'hf_modules')
    import numpy as np
    import torch,transformers,sklearn,scipy,joblib,threadpoolctl,narwhals,cloudpickle
    from transformers import AutoProcessor,AutoModelForCausalLM,CLIPModel,CLIPProcessor
    from decord import VideoReader,cpu
    from PIL import Image
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for part in [processor.video_processor,processor.image_processor]:part.max_pixels=150000;part.min_pixels=3136
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    clip=CLIPModel.from_pretrained(str(CLIP),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    clip_processor=CLIPProcessor.from_pretrained(str(CLIP),local_files_only=True)
    if int(clip.config.projection_dim)!=768:raise ValueError('本地CLIP投影维度不是已登记的768')
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,joblib=joblib.__version__,threadpoolctl=threadpoolctl.__version__,
        narwhals=narwhals.__version__,cloudpickle=cloudpickle.__version__,gpu=torch.cuda.get_device_name(0),mage_dtype=str(model.dtype),clip_dtype=str(clip.dtype))
    write_once(ctx['store']/'runtime.json',runtime)
    store=ctx['store'];tokenizer=processor.tokenizer
    class Capture:
        def __init__(self,base):object.__setattr__(self,'base',base);object.__setattr__(self,'last',None)
        def __getattr__(self,n):return getattr(self.base,n)
        def __setattr__(self,n,v):
            if n in ['base','last']:object.__setattr__(self,n,v)
            else:setattr(self.base,n,v)
        def __call__(self,*a,**kw):
            value=self.base(*a,**kw);object.__setattr__(self,'last',value);return value
    capture=Capture(processor.video_processor);processor.video_processor=capture
    video_cache={}
    def check_space():disk_guard(args.min_free,args.budget)
    def get_video(row):
        v=row['video_id']
        # 只保留一个视频解码器和CLIP矩阵，避免积累长视频内存。
        if v not in video_cache:
            video_cache.clear();vr=VideoReader(row['video_path'],ctx=cpu(0));fps=float(vr.get_avg_fps());n=len(vr)
            if not math.isfinite(fps) or fps<=0 or n<32:raise ValueError('视频fps或帧数异常，不猜测时间轴')
            video_cache[v]=dict(vr=vr,fps=fps,count=n,coarse=None)
            write_once(store/'video_metadata'/f'{v}.json',dict(signature=ctx['signature'],video_id=v,fps=fps,frame_count=n))
        return video_cache[v]
    def raw_frames(hit,indices):return hit['vr'].get_batch([int(i) for i in indices]).asnumpy()
    def save_feature(path,indices,features,meta,cost):
        # 固定FP16持久化后再计算相似度，保证首次与断点回放数值一致。
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('特征文件已存在，不覆盖')
        with path.open('xb') as out:
            np.savez(out,indices=np.asarray(indices,dtype=np.int64),features=np.asarray(features,dtype=np.float16),
                meta=np.asarray(json.dumps(meta,sort_keys=True)),wall_s=np.asarray(float(cost)))
            out.flush();os.fsync(out.fileno())
    def read_feature(path,expected):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['meta'].item()));inds=data['indices'].tolist();features=data['features'].astype(np.float32);cost=float(data['wall_s'])
        if meta!=expected or features.shape!=(len(inds),768) or not np.isfinite(features).all() or not math.isfinite(cost) or cost<0:
            raise ValueError(f'CLIP缓存身份/数值不符，保留:{path}')
        norms=np.linalg.norm(features,axis=1,keepdims=True)
        if np.any(norms<=1e-8):raise ValueError('CLIP向量为零')
        return inds,features/norms,cost
    def encode_images(hit,indices):
        arrays=raw_frames(hit,indices);values=clip_processor(images=[Image.fromarray(a) for a in arrays],return_tensors='pt')
        values={k:v.to(device='cuda',dtype=clip.dtype) if v.is_floating_point() else v.to('cuda') for k,v in values.items()}
        with torch.inference_mode():features=clip.get_image_features(**values,return_dict=True)
        if not isinstance(features,torch.Tensor):features=features.pooler_output
        if features.shape!=(len(indices),768) or not torch.isfinite(features).all() or torch.any(features.float().norm(dim=-1)<=1e-8):raise ValueError('CLIP视觉投影数值/维度非法')
        features=features.float();features=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        out=features.cpu().numpy().astype(np.float16);del values,features,arrays
        return out
    def coarse(row,hit):
        if hit['coarse'] is not None:return hit['coarse']
        # 真正1fps时间格点，不用int(fps)步长导致采样率偏移。
        indices=sorted(set([int(round(t*hit['fps'])) for t in np.arange(0,hit['count']/hit['fps'],1.0) if int(round(t*hit['fps']))<hit['count']]+[hit['count']-1]))
        path=store/'clip_features'/row['video_id']/'coarse.npz';meta=dict(signature=ctx['signature'],video_id=row['video_id'],kind='coarse_1fps',indices_sha256=digest(indices))
        if not path.exists():
            check_space();start=time.perf_counter();features=[]
            for pos in tqdm(range(0,len(indices),24),desc=f'CLIP {row["video_id"]}',unit='批',leave=False,dynamic_ncols=True):
                check_space();features.append(encode_images(hit,indices[pos:pos+24]))
            save_feature(path,indices,np.concatenate(features),meta,time.perf_counter()-start)
        saved,features,_=read_feature(path,meta)
        if saved!=indices:raise ValueError('CLIP粗采样索引漂移')
        hit['coarse']=(indices,features);return hit['coarse']
    def text_vector(row):
        start=time.perf_counter();text=query(row);tok=clip_processor.tokenizer
        ids=tok(text,add_special_tokens=False,truncation=False)['input_ids']
        chunks=text_chunks(ids,int(tok.bos_token_id),int(tok.eos_token_id),77)
        meta=dict(signature=ctx['signature'],index=row['index'],query_sha256=digest(text),raw_content_tokens=len(ids),
            bos_id=int(tok.bos_token_id),eos_id=int(tok.eos_token_id),chunks=chunks,truncated=False,aggregation='mean unit chunk vectors then unit normalization')
        write_once(store/'clip_queries'/f'{row["index"]}.json',meta)
        feature_path=store/'clip_text_features'/f'{row["index"]}.npz'
        if feature_path.exists():
            saved,features,_=read_feature(feature_path,meta)
            if saved!=[0]:raise ValueError('CLIP问题向量缓存索引不符')
            return features[0]
        maximum=max(map(len,chunks));padded=[x+[int(tok.pad_token_id)]*(maximum-len(x)) for x in chunks]
        mask=[[1]*len(x)+[0]*(maximum-len(x)) for x in chunks]
        with torch.inference_mode():
            vectors=clip.get_text_features(input_ids=torch.tensor(padded,device='cuda'),attention_mask=torch.tensor(mask,device='cuda'),return_dict=True)
        if not isinstance(vectors,torch.Tensor):vectors=vectors.pooler_output
        if vectors.shape!=(len(chunks),768) or not torch.isfinite(vectors).all() or torch.any(vectors.float().norm(dim=-1)<=1e-8):raise ValueError('CLIP文本投影数值/维度非法')
        vectors=vectors.float();vectors=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-8);pooled=vectors.mean(0);pooled=pooled/pooled.norm().clamp_min(1e-8)
        result=pooled.cpu().numpy();del vectors,pooled
        if not np.isfinite(result).all():raise ValueError('CLIP文本特征非法')
        save_feature(feature_path,[0],result[None,:],meta,time.perf_counter()-start)
        return read_feature(feature_path,meta)[1][0]
    def native(row,text):
        chat=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True)
        return processor(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)
    def selected(row,hit,indices,text):
        frames=raw_frames(hit,indices)
        with selected_video_inputs(processor,frames,indices,hit['fps'],hit['count'],video_path=row['video_path']) as adapter:
            result=native(row,text);seconds=list(adapter.seconds)
            if adapter.calls!=1:raise ValueError('选帧适配器调用数异常')
        return result,seconds
    def visual(actual):
        grid=actual['image_grid_thw'].tolist();n=sum(t*h*w for t,h,w in grid)//int(processor.spatial_merge_size)**2
        if int((actual['input_ids']==tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())!=n:raise ValueError('视觉占位token不符')
        return dict(grid_thw=grid,pv_shape=list(actual['pixel_values'].shape),n_tok=n,
            patch_time_range=[float(actual['patch_positions'][:,0].min()),float(actual['patch_positions'][:,0].max())])
    def generate(actual,max_new):
        gpu={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in actual.items()}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode():out=model.generate(**gpu,max_new_tokens=max_new,do_sample=False)
        raw=tokenizer.decode(out[0,actual['input_ids'].shape[1]:],skip_special_tokens=True).strip();generated=int(out.shape[1]-actual['input_ids'].shape[1]);torch.cuda.synchronize()
        seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
        del gpu,out
        return raw,seconds,peak,generated
    def selection(row,hit):
        path=store/'selections'/f'{row["index"]}.json'
        if path.exists():
            obj=json.loads(path.read_text());unseal(obj)
            return validate_selection_cache(obj,row,hit,ctx['signature'])
        indices,features=coarse(row,hit);tv=text_vector(row);similarities=(features@tv).astype(float).tolist();lookup={i:j for j,i in enumerate(indices)}
        def similarity_frames(requested):
            result={}
            for idx in requested:
                if idx in lookup:vector=features[lookup[idx]]
                else:
                    p=store/'clip_features'/row['video_id']/f'extra_{idx}.npz';meta=dict(signature=ctx['signature'],video_id=row['video_id'],kind='extra',index=int(idx))
                    if not p.exists():
                        check_space();start=time.perf_counter();vec=encode_images(hit,[idx]);save_feature(p,[idx],vec,meta,time.perf_counter()-start)
                    saved,vec,_=read_feature(p,meta)
                    if saved!=[idx]:raise ValueError('新增帧缓存索引漂移')
                    vector=vec[0]
                result[int(idx)]=float(vector@tv)
            return result
        def score_frames(requested):
            result={}
            for idx in tqdm(requested,desc=f'逐帧评分 Q{row["index"]}',unit='帧',leave=False,dynamic_ncols=True):
                check_space();p=store/'scores'/row['index']/f'{idx}.json';text=ANALYSIS.format(query=query(row))
                if p.exists():
                    u=json.loads(p.read_text());unseal(u)
                    if (u['signature']!=ctx['signature'] or u['index']!=row['index'] or u['frame_index']!=idx
                            or u['prompt_sha256']!=digest(text) or u['score']!=parse_score(u['raw_text'])):raise ValueError('逐帧评分断点身份漂移')
                else:
                    before=time.perf_counter();frame=Image.fromarray(raw_frames(hit,[idx])[0]);actual=image_inputs(processor,frame,text);score_prep=time.perf_counter()-before
                    raw,elapsed,peak,generated=generate(actual,96)
                    u=seal(dict(signature=ctx['signature'],index=row['index'],frame_index=int(idx),prompt_sha256=digest(text),
                        input_sha256=digest(actual['input_ids'].tolist()),input_tokens=int(actual['input_ids'].shape[1]),visual=visual(actual),generated_tokens=generated,
                        raw_text=raw,score=parse_score(raw),inference_s=elapsed,preprocess_s=score_prep,peak_gib=peak))
                    write_once(p,u);del actual,frame
                result[int(idx)]=dict(score=u['score'],raw_text=u['raw_text'])
            return result
        def trace(r):write_once(store/'rounds'/row['index']/f'{r["iteration"]:02d}.json',dict(signature=ctx['signature'],index=row['index'],trace=r))
        result=select_frames(hit['count'],hit['fps'],indices,similarities,score_frames,similarity_frames,AIRConfig(),on_round=trace)
        result=seal(dict(signature=ctx['signature'],index=row['index'],**result));validate_selection_cache(result,row,hit,ctx['signature']);write_once(path,result);return result
    with tqdm(total=len(ctx['expected']),initial=len(ctx['expected']&set(ctx['units'])),desc='Mage A.I.R. '+args.stage,unit='答案',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx=row['index']
            if all(f'{idx}:{a}' in ctx['units'] for a in ARMS):continue
            check_space();hit=get_video(row);start=time.perf_counter()
            actual=native(row,prompt(row));inds=[int(i) for i in actual['patch_positions'][:,0].unique_consecutive().tolist()]
            seconds=list(capture.last['frame_timestamps'][0]);capture.last=None
            expected=uniform_indices(hit['count'],32)
            if inds!=expected:raise ValueError(f'原生均匀采样索引不符：count={hit["count"]},fps={hit["fps"]},expected={expected},actual={inds}')
            adapted,adapt_seconds=selected(row,hit,inds,prompt(row));assert_inputs_equal(actual,adapted)
            if seconds!=adapt_seconds:raise ValueError('原生uniform秒数未复现')
            write_once(store/'native_audits'/f'{idx}.json',dict(signature=ctx['signature'],index=idx,input_sha256=digest(actual['input_ids'].tolist()),indices=inds,seconds=seconds,tensor_equal=True))
            del adapted;prep=time.perf_counter()-start
            for arm in ARMS:
                k=f'{idx}:{arm}'
                if k in ctx['units']:
                    if ctx['units'][k]['input_sha256']!=digest(actual['input_ids'].tolist()):raise ValueError('恢复基线输入漂移')
                    continue
                check_space();sel=None;reuse=None
                if arm=='uniform32':use=actual;chosen=inds;secs=seconds;origin='generated';raw,elapsed,peak,generated=generate(use,32);calls=1
                else:
                    sel=selection(row,hit);chosen=[int(i) for i in sel['final_indices']]
                    if sel['fallback_reason']:
                        if chosen!=inds:raise ValueError('无证据回退不是原均匀索引')
                        base=ctx['units'][f'{idx}:uniform32'];use=actual;secs=seconds;origin='fallback_uniform';calls=0
                        raw=base['raw_text'];elapsed=peak=0.;generated=None;reuse=base['record_sha256']
                    else:
                        before=time.perf_counter();use,secs=selected(row,hit,chosen,prompt(row));prep+=time.perf_counter()-before
                        origin='generated';calls=1;raw,elapsed,peak,generated=generate(use,32)
                u=seal(dict(signature=ctx['signature'],index=idx,video_id=row['video_id'],arm=arm,
                    raw_text=raw,pred=parse_answer(raw),origin=origin,new_calls=calls,reuse_record=reuse,generated_tokens=generated,
                    prompt_sha256=digest(prompt(row)),input_sha256=digest(use['input_ids'].tolist()),input_tokens=int(use['input_ids'].shape[1]),
                    visual=visual(use),indices=chosen,seconds=secs,native_uniform_equal=True,
                    selection_sha256=sha(store/'selections'/f'{idx}.json') if sel else None,
                    inference_s=elapsed,peak_gib=peak,preprocess_s=prep))
                validate_unit(ctx,u,ctx['units']);append(store/'units.jsonl',u);ctx['units'][k]=u;bar.update(1);prep=0.
                bar.set_postfix(index=idx,arm=arm,frames=len(chosen),answer=u['pred'] or 'INVALID')
                print(f'落盘 {k}: {u["pred"] or "INVALID"}，帧={len(chosen)}，来源={origin}',flush=True)
                if arm=='air_cap32' and use is not actual:del use
            del actual
    print('阶段完成；请summarize检查。不会自动进入下一阶段。',flush=True)
