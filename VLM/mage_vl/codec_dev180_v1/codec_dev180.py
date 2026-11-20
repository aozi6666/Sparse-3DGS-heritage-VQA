"""Codec字幕dev180固定开发配对；缓存只读、不改patch、不训练、不下载。

13312完整来源为完整证据候选；32字幕锚点/2048为小对照。不是等文本预算实验。
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
sys.path[:0]=[str(PROJECT/'VQA/mage_vl/practical'),str(PROJECT/'VQA/mage_vl/subtitle_ablation_v1')]
import subtitle_ablation as common
from practical_common import ROOT,MODEL,SLOT,signature,sha,prompt,parse_answer
from subtitle_common import subtitle_source,subtitle_prompt
from continuous_common import continuous_evidence

OUT=ROOT/'22_codec_dev180_v1'
ARMS=('codec_no_sub','codec_complete13312','codec_anchor32_2048')
CODEC=dict(engine='hevc',target_canvas=32,patch=16,group_size=32,images_per_group=4,spatial_mask_mode='off',max_pixels=150000)
atomic=common.atomic_json
require=common.require


def guard():
    free=shutil.disk_usage('/root/autodl-tmp').free/1024**3
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and not f.is_symlink()) if OUT.exists() else 0
    require(free>=10 and size<=128*1024**2,'磁盘不足10GiB或本轮超过128MiB，停止，不清理')
    return dict(free_gib=round(free,3),experiment_mib=round(size/1024**2,3))


def cache_name(video):
    raw=f'{video}|eng=hevc|tc=32|gs=32|ipg=4|patch=16|mp=150000|mask=off'
    return Path(video).stem+'_'+hashlib.md5(raw.encode()).hexdigest()


def validate_meta(meta,video):
    require(meta.get('video')==video,'Codec元数据视频身份不符')
    fps=meta.get('fps');count=meta.get('total_frames')
    require(isinstance(fps,(int,float)) and math.isfinite(fps) and fps>0,'缺少真实FPS，不接受原生30fps回退')
    require(type(count) is int and count>0,'缺少真实帧数')
    cfg=meta.get('config',{})
    for k,v in dict(patch=16,group_size=32,images_per_group=4,max_pixels=150000,min_group_frames=8,max_group_frames=64).items():
        require(cfg.get(k)==v,'Codec缓存参数不符:'+k)
    require(cfg.get('avoid_keyframes') is True and cfg.get('grouping_mode')=='readiness','Codec采样/分组不同')
    require(cfg.get('num_sampled_frames')==min(256,count),'Codec来源采样预算不同')
    require(meta.get('metadata',{}).get('fps')==fps and meta.get('metadata',{}).get('total_frames')==count,'Codec两处时间元数据不同')
    return float(fps),count


def anchor_subset(ids,cap=32):
    require(ids and ids==sorted(set(ids)) and all(type(i) is int and i>=0 for i in ids),'来源帧号必须非空、去重、递增')
    require(cap>=2,'字幕锚点上限非法')
    if len(ids)<=cap:return list(ids)
    return [ids[round(j*(len(ids)-1)/(cap-1))] for j in range(cap)]


def align(ids,fps,cues,source,count_tokens):
    require(ids==sorted(set(ids)) and ids and math.isfinite(fps) and fps>0,'对齐时间非法')
    chosen={};hits=[]
    for i in ids:
        us=round(i/fps*1_000_000)
        hit=[c for c in cues if c['text'] and c['start_ms']*1000<=us<c['end_ms']*1000]
        hits.append([c['cue_id'] for c in hit]);chosen.update({c['cue_id']:c for c in hit})
    selected=sorted(chosen.values(),key=lambda c:(c['start_ms'],c['end_ms'],c['cue_id']))
    text='\n'.join(c['text'] for c in selected)
    status='used' if selected else ('no_hit' if source['status']=='available' else source['status'])
    return dict(source=source,frame_indices=ids,frame_seconds=[i/fps for i in ids],frame_cue_ids=hits,selected_cues=selected,text=text,text_tokens=count_tokens(text) if text else 0,status=status)


def evidence(ids,fps,cues,source,count):
    result={}
    for arm,anchors,budget in [('codec_complete13312',ids,13312),('codec_anchor32_2048',anchor_subset(ids),2048)]:
        original=align(anchors,fps,cues,source,count)
        full=continuous_evidence(original,cues,count,budget)
        result[arm]=dict(aligned=original,continuous=full,subtitle_anchors=anchors,budget=budget)
    return result


def init():
    """只读旧缓存数组/字幕，未解码视频、未加载模型。"""
    import numpy as np
    from tokenizers import Tokenizer
    previous=common.context('full714');b=previous['bundle'];rows=[];videos={};sources={}
    def pin(f):sources[str(f)]=sha(f)
    pin(common.OUT/'protocol.json');pin(common.OUT/'bundle.json')
    gate=ROOT/'21_codec_subtitles_v1/results/summary.json'
    require(json.loads(gate.read_text())['audit']['passed'],'先通过原9题Codec冒烟');pin(gate)
    budget_path=OUT/'preflight/complete_text_budget.json';pin(budget_path)
    prepared=json.loads(budget_path.read_text());require(prepared['budget']==13312,'完整字幕预算预检不符')
    ref=previous['protocol']['runtime']
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'));count=lambda x:len(tokenizer.encode(x,add_special_tokens=False).ids)
    journal=ROOT/'cache/ballots/86f337f035388253a22e9dcb7d7a6977005aae4ab1273fa75313735ba5ab2c16/units.jsonl';pin(journal)
    old={}
    for line in journal.read_text().splitlines(keepends=True):
        require(line.endswith('\n'),'旧Codec记录残行')
        u=json.loads(line)
        if u['backend']=='codec' and u['permutation_id']==0:
            require(u['index'] not in old,'旧Codec基线重复');old[u['index']]=u
    development=[r for r in b['rows'] if r['stage']=='dev180']
    require(len(development)==180,'开发清单不符')
    for video_id in sorted({r['video_id'] for r in development}):
        first=next(r for r in development if r['video_id']==video_id)
        chosen=[r for r in b['rows'] if r['video_id']==first['video_id']]
        require(len(chosen)==3 and all(r['index'] in old for r in chosen),'缺少三题原生Codec基线')
        rows.extend(chosen);video=first['video_path'];name=cache_name(video)
        hits=[]
        for root in [ROOT/'cache/codec',PROJECT/'VQA/mage_vl/outputs/r2v_codec_cache']:
            hits.extend(root.rglob(name+'/meta.json'))
        require(len(hits)==1,'缓存缺失或重复，停止，不自动生成/替换:'+name)
        fp=hits[0];meta=json.loads(fp.read_text());fps,n=validate_meta(meta,video)
        arr=np.load(fp.parent/'src_patch_position.npy',allow_pickle=False)
        require(arr.ndim==2 and arr.shape[1]==3 and arr.dtype.kind in 'iu','来源patch数组形状/类型非法')
        ids=np.unique(arr[arr[:,0]>=0,0]).astype(int).tolist()
        require(ids and ids[-1]<n,'Codec帧号越界')
        source,cues=subtitle_source(first['video_id'])
        if source['sha256'] is not None:pin(Path(source['path']))
        ee=evidence(ids,fps,cues,source,count)
        require(not ee['codec_complete13312']['continuous']['omitted_neighbor_ids'],'完整证据仍省略邻句，停止')
        files=[fp,fp.parent/'src_patch_position.npy']
        canvas=meta.get('canvas_files') or [f.name for f in sorted(fp.parent.glob('canvas_*.jpg'))]
        require(canvas and all(Path(name).name==name for name in canvas),'画布路径非法')
        files.extend(fp.parent/name for name in canvas)
        for f in files:
            require(f.is_file() and not f.is_symlink(),'画布缺失/符号链接，停止');pin(f)
        v=dict(video_path=video,cache_dir=str(fp.parent),fps=fps,total_frames=n,source_indices=ids,evidence=ee,
            cache_files_sha256={str(f):sha(f) for f in files},canvas_files=canvas,
            source_video_identity=first['video_identity'],note='字幕锚点不是视觉选帧；视觉patch完全不变。')
        videos[first['video_id']]=v
    require(len(rows)==180 and len(videos)==60,'只能固定dev180/60视频')
    refs={r['index']:dict(frames_continuous2048=b['old'][r['index']]['continuous'],codec_old=old[r['index']],
        codec_old_visual={k:old[r['index']][k] for k in ['grid_thw','pv_shape','patch_time_range','n_tok']}) for r in rows}
    for r in rows:require(refs[r['index']]['codec_old']['prompt_sha256']==signature(prompt(r,0)),'原生Codec提示词不同')
    for path,h in previous['protocol']['source_sha256'].items():sources[path]=h
    for f in Path(__file__).parent.glob('*.py'):pin(f)
    pin(Path(__file__).with_name('run_codec_dev180.sh'))
    protocol=dict(version='codec_subtitles_dev180_v1',probe_indices=prepared['probe_indices'],codec=CODEC,arms=list(ARMS),runtime=ref,model=previous['protocol']['model'],source_sha256=sources,
        primary='all retained nonpadding patch source times; neighbor1+timestamp; subtitle cap13312',
        control='uniform up to32 anchors from actual retained source times; same neighbor rule; cap2048; never change visual patches',
        policy='Fixed180 seen-development evaluation, NOT accuracy-gain proof. No semantic selection. 13312 vs2048 changes anchor coverage and text budget jointly; not isolated budget ablation. Frames reference has different visual and subtitle evidence. Cached preprocessing has nonzero original cost; not deployment timing.')
    sig=signature(protocol)
    atomic(OUT/'protocol.json',protocol)
    atomic(OUT/'bundle.json',common.sealed(dict(signature=sig,rows=rows,videos=videos,references=refs)))
    report=dict(signature=sig,indices=[r['index'] for r in rows],videos={v:dict(retained_source_times=len(x['source_indices']),canvas_count=len(x['canvas_files']),
        variants={a:dict(anchor_count=len(e['subtitle_anchors']),subtitle_tokens=e['continuous']['text_tokens'],omitted_neighbors=len(e['continuous']['omitted_neighbor_ids'])) for a,e in x['evidence'].items()}) for v,x in videos.items()},new_visual_cache_bytes=0,QA_calls=0)
    atomic(OUT/'budget_preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def context():
    guard();p=json.loads((OUT/'protocol.json').read_text());sig=signature(p)
    for f,h in p['source_sha256'].items():require(sha(f)==h,'来源变化，停止:'+f)
    for name,expected in p['model']['weights'].items():
        st=(MODEL/name).stat();require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==expected,'模型权重变化')
    b=common.unseal(json.loads((OUT/'bundle.json').read_text()));require(b['signature']==sig,'数据包身份不符')
    for v in b['videos'].values():
        st=Path(v['video_path']).stat();require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==v['source_video_identity'],'视频来源变化')
    units={}
    for f in (OUT/'units').glob('*.json'):
        u=json.loads(f.read_text());validate_unit(sig,b,u);k=(u['index'],u['arm']);require(k not in units,'重复断点');units[k]=u
    for k,u in units.items():
        audit=json.loads((OUT/'input_audits'/f'{u["index"]}.json').read_text());common.unseal(audit)
        require(audit['signature']==sig and all(u[n]==audit['inputs'][u['arm']][n] for n in ['tensor_sha256','input_sha256','input_tokens','visual']),'生成单元与CPU输入审计不同')
        if u['origin']=='reuse_current_exact':
            ref=units.get((u['index'],u['reuse']['arm']))
            require(ref is not None and ref['record_sha256']==u['reuse']['record_sha256'] and ref['raw_text']==u['raw_text'] and ref['tensor_sha256']==u['tensor_sha256'],'同题复用来源不同')
    return dict(protocol=p,signature=sig,bundle=b,units=units)


def texts(row,v):
    return dict(codec_no_sub=prompt(row,0),**{a:subtitle_prompt(row,e['continuous']['text']) for a,e in v['evidence'].items()})


def validate_unit(sig,b,u):
    common.unseal(u);row=next((r for r in b['rows'] if r['index']==u['index']),None)
    require(row is not None and u['arm'] in ARMS and u['signature']==sig,'单元身份不符')
    v=b['videos'][row['video_id']]
    require(u['prompt_sha256']==signature(texts(row,v)[u['arm']]),'单元提示不同')
    require(u['pred']==parse_answer(u['raw_text']) and u['native_equal'] is True,'解析/原生输入核对失败')
    require(u['origin'] in ['generated','reuse_current_exact'],'未知来源')
    if u['origin']=='generated':require(u['new_calls']==1 and 1<=u['generated_tokens']<=32,'生成计数非法')
    else:require(u['new_calls']==0 and u['generated_tokens'] is None and u['inference_s']==0 and u['reuse']['tensor_equal'],'复用非法')
    require(u['visual']==b['references'][row['index']]['codec_old_visual'],'Codec视觉布局改变')
    require(u['subtitle_tokens']==(0 if u['arm']=='codec_no_sub' else v['evidence'][u['arm']]['continuous']['text_tokens']),'文本长度非法')
    for n in ['inference_s','peak_gib']:require(math.isfinite(u[n]) and u[n]>=0,'成本非法')


@contextmanager
def readonly_codec(proc,b):
    """只接管原生缓存读取，禁止底层重建/删除缓存；模型及原生patch处理不变。"""
    prefix=type(proc).__module__.rsplit('.',1)[0]
    mod=importlib.import_module(prefix+'.codec_video_processing_mage_vl')
    original=mod.process_codec_video;loaded={}
    def read(video,cfg):
        v=next((v for v in b['videos'].values() if v['video_path']==video),None)
        require(v is not None,'未冻结的Codec视频')
        require(str(mod._cache_dir_for(video,cfg))==v['cache_dir'],'原生Codec缓存键不同')
        for k,value in CODEC.items():require(getattr(cfg,k)==value,'原生Codec参数不同:'+k)
        if video not in loaded:
            for f,h in v['cache_files_sha256'].items():require(sha(f)==h,'只读旧缓存改变')
            loaded[video]=mod._load_codec_result(Path(v['cache_dir']))
        require(loaded[video]['fps']==v['fps'],'Codec FPS发生变化')
        return loaded[video]
    mod.process_codec_video=read
    try:yield
    finally:mod.process_codec_video=original


class Inputs:
    def __init__(self,c):
        from transformers import AutoProcessor
        self.c=c;self.proc=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
        self.proc.image_processor.max_pixels=150000;self.proc.image_processor.min_pixels=3136
        self.current=None;self.cache=None
    def native(self,row,text):
        proc=self.proc;v=self.c['bundle']['videos'][row['video_id']]
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True)
        return proc(text=[chat],videos=[row['video_path']],video_backend='codec',max_pixels=150000,codec_config=dict(CODEC,cache_root=Path(v['cache_dir']).parent),return_tensors='pt',padding=True)
    def make(self,row):
        import torch
        from practical_vote import CaptureTokenizer
        c=self.c;v=c['bundle']['videos'][row['video_id']];p=self.proc;tok=p.tokenizer
        if row['video_id']!=self.current:
            cap=CaptureTokenizer(tok);p.tokenizer=cap
            try:val=self.native(row,SLOT)
            finally:p.tokenizer=tok
            require(cap.text is not None and len(cap.text)==1 and cap.text[0].count(SLOT)==1,'Codec插槽捕获失败')
            actual=sorted(set(int(i) for i in val['patch_positions'][:,0].tolist() if i>=0))
            require(actual==v['source_indices'],'实际来源时间与字幕锚点来源不同')
            require(int((val['input_ids']==tok.convert_tokens_to_ids('<|image_pad|>')).sum())==common.visual_info(val,p.spatial_merge_size)['n_tok'],'视觉token展开不符')
            self.cache=dict(template=cap.text[0],visual={k:t for k,t in val.items() if k not in ['input_ids','attention_mask']})
            self.current=row['video_id']
        result={};visual=self.cache['visual']
        for arm,text in texts(row,v).items():
            enc=tok([self.cache['template'].replace(SLOT,text)],padding=True,return_tensors='pt')
            packed=dict(visual,**enc);direct=self.native(row,text)
            require(common.equal_inputs(packed,direct,torch),'Codec原生展开与插槽不同')
            cfg=json.loads((MODEL/'config.json').read_text())
            limit=cfg['text_config']['max_position_embeddings']
            require(int(packed['input_ids'].shape[1])+32<=limit,'超出模型配置上下文长度')
            vi=common.visual_info(packed,p.spatial_merge_size)
            require(vi==c['bundle']['references'][row['index']]['codec_old_visual'],'原生Codec视觉未复现')
            if arm=='codec_no_sub':require(signature(packed['input_ids'].tolist())==c['bundle']['references'][row['index']]['codec_old']['prefix_sha256'],'原生Codec无字幕输入未复现')
            else:require(len(tok.encode(v['evidence'][arm]['continuous']['text'],add_special_tokens=False))==v['evidence'][arm]['continuous']['text_tokens'],'字幕token计数不同')
            result[arm]=packed
        require(all(common.tensor_digest({k:t for k,t in x.items() if k not in ['input_ids','attention_mask']})==common.tensor_digest(visual) for x in result.values()),'三路视觉Tensor不同')
        return result


def input_audit(c):
    from tqdm import tqdm
    engine=Inputs(c)
    with readonly_codec(engine.proc,c['bundle']),tqdm(total=len(c['bundle']['rows']),desc='Codec CPU输入审计',unit='题',dynamic_ncols=True) as bar:
        for row in c['bundle']['rows']:
            path=OUT/'input_audits'/f'{row["index"]}.json'
            if path.exists():
                u=json.loads(path.read_text());common.unseal(u);require(u['signature']==c['signature'],'审计断点来源不符');bar.update(1);continue
            guard();values=engine.make(row)
            u=common.sealed(dict(signature=c['signature'],index=row['index'],passed=True,inputs={a:dict(tensor_sha256=common.tensor_digest(x),input_sha256=signature(x['input_ids'].tolist()),input_tokens=int(x['input_ids'].shape[1]),visual=common.visual_info(x,engine.proc.spatial_merge_size)) for a,x in values.items()}))
            atomic(path,u);bar.update(1)
    print('所选开发题原生Codec输入审计完成；无问答/模型权重。')


def run(c):
    require(os.environ.get('PRACTICAL_STAGE_CONFIRMED')=='1','由用户设置PRACTICAL_STAGE_CONFIRMED=1')
    if not c.get('probe_mode'):
        gate=json.loads((OUT/'resource_gate.json').read_text())
        require(gate['signature']==c['signature'] and gate['passed'],'先通过最长输入显存探测')
    expected={(r['index'],a) for r in c['bundle']['rows'] for a in ARMS}
    require(not any(u['baseline_answer_equal'] is False for u in c['units'].values()),'已有控制答案未复现，先诊断，不通过续跑跳过')
    if expected<=c['units'].keys():print('所选答案完整，跳过模型。');return
    import random,numpy as np,torch,transformers
    require(torch.cuda.is_available(),'需要GPU；不运行CPU模型推理')
    for row in c['bundle']['rows']:
        u=json.loads((OUT/'input_audits'/f'{row["index"]}.json').read_text());common.unseal(u)
        require(u['signature']==c['signature'] and u['passed'],'先完成CPU输入审计')
    reference=c['protocol']['runtime'];rt=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype='torch.bfloat16')
    require(rt==reference,'与原基线硬件/环境不同，先核查:'+str(rt))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    from transformers import AutoModelForCausalLM
    from tqdm import tqdm
    engine=Inputs(c);model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0}).eval()
    require(str(model.dtype)==rt['dtype'],'精度不同');atomic(OUT/'runtime.json',rt)
    with readonly_codec(engine.proc,c['bundle']),tqdm(total=len(expected),initial=len(c['units']),desc='Mage Codec字幕 dev180',unit='答案',dynamic_ncols=True) as bar:
        for row in c['bundle']['rows']:
            idx=row['index']
            if all((idx,a) in c['units'] for a in ARMS):continue
            guard();values=engine.make(row);audit=json.loads((OUT/'input_audits'/f'{idx}.json').read_text())
            for a,x in values.items():require(common.tensor_digest(x)==audit['inputs'][a]['tensor_sha256'],'CPU审计后输入漂移')
            for arm in ARMS:
                if (idx,arm) in c['units']:continue
                x=values[arm];raw=None;proof=None;origin='generated';calls=1;generated=None;elapsed=peak=0.
                for other in ARMS:
                    old=c['units'].get((idx,other))
                    if old and common.equal_inputs(x,values[other],torch):
                        raw=old['raw_text'];proof=dict(arm=other,record_sha256=old['record_sha256'],tensor_equal=True);origin='reuse_current_exact';calls=0;break
                if calls:
                    gpu={k:t.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else t.to(model.device) for k,t in x.items()}
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    with torch.inference_mode():ans=model.generate(**gpu,max_new_tokens=32,do_sample=False)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
                    generated=int(ans.shape[1]-x['input_ids'].shape[1]);raw=engine.proc.tokenizer.decode(ans[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip();del gpu,ans
                v=c['bundle']['videos'][row['video_id']]
                baseline_equal=(parse_answer(raw)==c['bundle']['references'][idx]['codec_old']['semantic_pred']) if arm=='codec_no_sub' else None
                u=common.sealed(dict(signature=c['signature'],index=idx,video_id=row['video_id'],arm=arm,pred=parse_answer(raw),raw_text=raw,origin=origin,reuse=proof,new_calls=calls,generated_tokens=generated,
                    prompt_sha256=signature(texts(row,v)[arm]),input_sha256=signature(x['input_ids'].tolist()),tensor_sha256=common.tensor_digest(x),visual=audit['inputs'][arm]['visual'],input_tokens=int(x['input_ids'].shape[1]),
                    subtitle_tokens=0 if arm=='codec_no_sub' else v['evidence'][arm]['continuous']['text_tokens'],native_equal=True,inference_s=elapsed,peak_gib=peak,baseline_answer_equal=baseline_equal))
                validate_unit(c['signature'],c['bundle'],u);atomic(OUT/'units'/f'{idx}_{arm}.json',u);c['units'][(idx,arm)]=u;bar.update(1);bar.set_postfix(index=idx,arm=arm)
                print('落盘',idx,arm,u['pred'] or 'INVALID',origin,flush=True)
                require(baseline_equal is not False,'新无字幕控制答案未复现；保留结果，先诊断，不继续')


def summarize(c):
    rows=c['bundle']['rows'];require(len(c['units'])==3*len(rows),'先完成全部逻辑答案，不读取标签')
    labels_path=ROOT/'01_manifests/evaluation_labels.json';lock=json.loads((ROOT/'01_manifests/dataset_lock.json').read_text())
    require(sha(labels_path)==lock['labels_sha256'],'标签改变');labels=json.loads(labels_path.read_text())
    from practical_analyze import comparison
    names=list(ARMS)+['frames_continuous2048'];pred={a:{r['index']:c['units'][(r['index'],a)]['pred'] for r in rows} for a in ARMS}
    pred['frames_continuous2048']={r['index']:c['bundle']['references'][r['index']]['frames_continuous2048']['pred'] for r in rows}
    metrics=[];paired=[];details=[];cost=[]
    for duration,rs in [('overall',rows)]+[(d,[r for r in rows if r['duration']==d]) for d in ['short','medium','long'] if any(r['duration']==d for r in rows)]:
        for name in names:
            correct=sum(pred[name][r['index']]==labels[r['index']] for r in rs)
            metrics.append(dict(duration=duration,method=name,N=len(rs),correct=correct,accuracy_pct=100*correct/len(rs)))
        for name,base in [('codec_complete13312','codec_no_sub'),('codec_anchor32_2048','codec_no_sub'),('codec_complete13312','codec_anchor32_2048'),('codec_complete13312','frames_continuous2048')]:
            paired.append(dict(duration=duration,method=name,baseline=base,**comparison(rs,pred[name],pred[base],labels)))
    for r in rows:
        idx=r['index'];details.append(dict(index=idx,video_id=r['video_id'],duration=r['duration'],gt=labels[idx],**{a:pred[a][idx] for a in names}))
        for a in ARMS:
            u=c['units'][(idx,a)];cost.append(dict(index=idx,duration=r['duration'],method=a,input_tokens=u['input_tokens'],subtitle_tokens=u['subtitle_tokens'],visual_tokens=u['visual']['n_tok'],new_calls=u['new_calls'],new_inference_s=u['inference_s'],peak_gib=u['peak_gib']))
    invalid=sum(u['pred'] not in list('ABCD') for u in c['units'].values());matches=all(c['units'][(r['index'],'codec_no_sub')]['baseline_answer_equal'] for r in rows)
    report=dict(signature=c['signature'],N=len(rows),audit=dict(passed=invalid==0 and matches,invalid=invalid,old_codec_answer_parity=matches,complete_units=3*len(rows)),metrics=metrics,paired=paired,
        cost=dict(new_calls=sum(u['new_calls'] for u in c['units'].values()),reused=sum(not u['new_calls'] for u in c['units'].values()),inference_s=sum(u['inference_s'] for u in c['units'].values()),peak_gib=max(u['peak_gib'] for u in c['units'].values())),
        interpretation=c['protocol']['policy'],scope='Development180/60 videos, not independent final testing. Unified complete-text budget13312 is a new frozen protocol, not old6144. Text evidence and total tokens differ; no deployment timing.')
    atomic(OUT/'results/summary.json',report);atomic(OUT/'results/predictions.json',details);atomic(OUT/'results/token_cost.json',cost)
    common.csv_once(OUT/'results/predictions.csv',details);common.csv_once(OUT/'results/metrics.csv',metrics);common.csv_once(OUT/'results/token_cost.csv',cost)
    common.csv_once(OUT/'results/paired_comparisons.csv',[{**r,'video_cluster_bootstrap_95ci_pp':json.dumps(r['video_cluster_bootstrap_95ci_pp'])} for r in paired])
    print(json.dumps(report,ensure_ascii=False,indent=2));print('汇总:',OUT/'results/summary.json')
    require(report['audit']['passed'],'审计失败，保留结果，不自动扩样')


def resource_probe(c):
    # 只按输入长度选出的最长视频，不读取答案；探测答案直接计入开发阶段。
    ids=set(c['protocol']['probe_indices'])
    selected=dict(c,bundle=dict(c['bundle'],rows=[r for r in c['bundle']['rows'] if r['index'] in ids]),units={k:u for k,u in c['units'].items() if k[0] in ids},probe_mode=True)
    run(selected)
    require(len(selected['units'])==9,'最长视频探测尚未完整')
    import torch
    peak=max(u['peak_gib'] for u in selected['units'].values())
    total=torch.cuda.get_device_properties(0).total_memory/1024**3
    passed=peak<total*.9 and all(u['pred'] in list('ABCD') for u in selected['units'].values())
    gate=dict(signature=c['signature'],indices=sorted(ids),peak_gib=peak,total_device_gib=total,passed=passed,note='Largest CPU-estimated visual+subtitle input; other questions may differ slightly. Not accuracy selection. OOM stops and preserves checkpoints.')
    atomic(OUT/'resource_gate.json',gate)
    print(json.dumps(gate,ensure_ascii=False,indent=2))
    require(passed,'显存余量或回答合法性不足，停止，不扩开发集')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','audit-inputs','probe','run','summarize']);args=ap.parse_args()
    require(subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()=='feat/air-visual-evidence','分支必须为feat/air-visual-evidence')
    print('SPACE',guard(),flush=True);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'operation.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='init':init();return
        c=context()
        if args.action=='preflight':print('固定dev180；已完成',len(c['units']),'/540；不加载模型。')
        elif args.action=='audit-inputs':input_audit(c)
        elif args.action=='summarize':summarize(c)
        else:
            with (ROOT/'cache/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.action=='probe':resource_probe(c)
                else:run(c)


if __name__=='__main__':main()
