"""Codec字幕714覆盖性配对，固定13312/2048；缓存只读、不改patch、不训练、不下载。

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
sys.path[:0]=[str(PROJECT/'VLM/mage_vl/practical'),str(PROJECT/'VLM/mage_vl/subtitle_ablation_v1')]
import subtitle_ablation as common
from practical_common import ROOT,MODEL,SLOT,signature,sha,prompt,parse_answer
from subtitle_common import subtitle_source,subtitle_prompt
from continuous_common import continuous_evidence

OUT=ROOT/'23_codec_full714_v1'
DEV_OUT=ROOT/'22_codec_dev180_v1'
sys.path.insert(0,str(PROJECT/'VLM/mage_vl/codec_dev180_v1'))

def context_development():
    import codec_dev180
    return codec_dev180.context()

ARMS=('codec_no_sub','codec_complete13312','codec_anchor32_2048')
CODEC=dict(engine='hevc',target_canvas=32,patch=16,group_size=32,images_per_group=4,spatial_mask_mode='off',max_pixels=150000)
atomic=common.atomic_json
require=common.require


def guard():
    free=shutil.disk_usage('/root/autodl-tmp').free/1024**3
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and not f.is_symlink()) if OUT.exists() else 0
    require(free>=10 and size<=2*1024**3,'磁盘不足10GiB或本轮超过2GiB，停止，不清理')
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


def plan_context():
    p=common.unseal(json.loads((OUT/'plan.json').read_text()))
    require(p['signature']==signature(p['plan']),'全量计划损坏')
    for f,h in p['plan']['source_sha256'].items():require(sha(f)==h,'冻结来源变化:'+f)
    for r in p['rows']:
        st=Path(r['video_path']).stat()
        require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==r['video_identity'],'视频变化')
    return p


def init():
    import importlib.util
    previous=common.context('full714');b=previous['bundle']
    dev=context_development()
    require(json.loads((DEV_OUT/'results/summary.json').read_text())['audit']['passed'],'开发结果未通过')
    rows=sorted(b['rows'],key=lambda r:({'val90':0,'short219':1,'dev180':2,'test225':3}[r['stage']],int(r['index'])))
    require(len(rows)==714 and len({r['video_id'] for r in rows})==238,'本地全量范围不同')
    sources=dict(previous['protocol']['source_sha256'])
    for f in [common.OUT/'protocol.json',common.OUT/'bundle.json',DEV_OUT/'protocol.json',DEV_OUT/'bundle.json',DEV_OUT/'results/summary.json',DEV_OUT/'resource_gate.json']:
        sources[str(f)]=sha(f)
    for f in (DEV_OUT/'units').glob('*.json'):sources[str(f)]=sha(f)
    for f in Path(__file__).parent.glob('*.py'):sources[str(f)]=sha(f)
    shell=Path(__file__).with_name('run_codec_full714.sh');sources[str(shell)]=sha(shell)
    pkg=importlib.util.find_spec('compressed_video_preinfer')
    require(pkg is not None,'现有Codec工具缺失，不自动安装')
    for f in Path(pkg.origin).parent.rglob('*.py'):sources[str(f)]=sha(f)
    for f in [Path('/root/autodl-tmp/Mage_ViT/mage/bin/cv-preinfer'),Path('/root/autodl-tmp/VLMEvalKit/ffmpeg_linux/ffmpeg'),Path('/root/autodl-tmp/VLMEvalKit/ffmpeg_linux/ffprobe')]:
        require(f.is_file(),'现有工具缺失:'+str(f));sources[str(f)]=sha(f)
    existing={}
    for root in [ROOT/'cache/codec',PROJECT/'VLM/mage_vl/outputs/r2v_codec_cache']:
        for fp in root.rglob('meta.json'):existing.setdefault(fp.parent.name,[]).append(str(fp.parent))
    sources_srt={};catalog={};refs={}
    for row in rows:
        video=row['video_id'];src,_=subtitle_source(video);sources_srt[video]=src
        if src['sha256'] is not None:sources[src['path']]=src['sha256']
        hits=existing.get(cache_name(row['video_path']),[]);require(len(hits)<=1,'重复旧缓存，停止')
        catalog[video]=hits[0] if hits else None
        dr=dev['bundle']['references'].get(row['index'])
        refs[row['index']]=dict(frames_continuous2048=b['old'][row['index']]['continuous'],codec_old=dr['codec_old'] if dr else None,codec_old_visual=dr['codec_old_visual'] if dr else None)
    plan=dict(version='codec_full714_fixed13312_v1',codec=CODEC,arms=list(ARMS),runtime=previous['protocol']['runtime'],model=previous['protocol']['model'],source_sha256=sources,
        policy='Fixed13312 complete-text primary and32-anchor2048 control. No skipping/retuning. Missing caches built by USER CPU command. Any budget overflow blocks ALL GPU work. Reuse verified dev180; val90 audit gate then coverage714 without score selection. Both val/test have historical use; no new independent test.',
        new_cache_root=str(OUT/'cache/codec'),roles=previous['protocol']['roles'])
    atomic(OUT/'plan.json',common.sealed(dict(signature=signature(plan),plan=plan,rows=rows,catalog=catalog,srt_sources=sources_srt,references=refs)))
    print('714题计划冻结；已有缓存',sum(bool(v) for v in catalog.values()),'缺失',sum(not v for v in catalog.values()),'；不加载模型。')


def build_cache(row,plan):
    """只写新目录；旧缓存完全不传给生成器，不删除任何已有目录。"""
    require(os.environ.get('PRACTICAL_STAGE_CONFIRMED')=='1','用户设置PRACTICAL_STAGE_CONFIRMED=1后CPU补缓存')
    import uuid
    root=OUT/'cache/codec';root.mkdir(parents=True,exist_ok=True)
    target=root/cache_name(row['video_path'])
    if target.exists():
        require((target/'meta.json').is_file() and (target/'src_patch_position.npy').is_file(),'新缓存不完整，保留现场，不重建')
        return target
    work=OUT/'cache/build_attempts'/(row['video_id']+'_'+uuid.uuid4().hex)
    work.parent.mkdir(parents=True,exist_ok=True)
    cmd=['/root/autodl-tmp/Mage_ViT/mage/bin/cv-preinfer','--video',row['video_path'],'--out_dir',str(work),
        '--num_sampled_frames','256','--grouping_mode','readiness','--group_size','32','--images_per_group','4','--patch','16','--max_pixels','150000',
        '--readiness_sum_threshold','0','--min_group_frames','8','--max_group_frames','64','--avoid_keyframes','--canvas_format','jpg']
    log=OUT/'logs'/('cache_'+row['video_id']+'_'+uuid.uuid4().hex+'.log');start=time.perf_counter()
    with log.open('x') as out:
        rc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=7200).returncode
    atomic(OUT/'cache/attempt_records'/(work.name+'.json'),dict(command=cmd,returncode=rc,wall_s=time.perf_counter()-start,log=str(log),directory=str(work)))
    require(rc==0,'CPU Codec生成失败，保留日志和目录:'+str(log))
    require((work/'meta.json').is_file() and (work/'src_patch_position.npy').is_file(),'新缓存输出不完整，停止')
    require(not target.exists(),'新缓存目标已出现，停止，不覆盖')
    work.rename(target)
    return target


def prepare_cache():
    import numpy as np
    from tokenizers import Tokenizer
    from tqdm import tqdm
    pp=plan_context();tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'));count=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
    unique={r['video_id']:r for r in pp['rows']};videos={};problems=[]
    with tqdm(total=238,desc='CPU Codec缓存及预算',unit='视频',dynamic_ncols=True) as bar:
        for video,row in unique.items():
            guard();saved=OUT/'video_records'/f'{video}.json'
            if saved.exists():
                record=json.loads(saved.read_text());common.unseal(record);require(record['plan_signature']==pp['signature'],'视频断点签名不同')
                for f,h in record['video']['cache_files_sha256'].items():require(sha(f)==h,'缓存断点内容变化')
                videos[video]=record['video'];bar.update(1);continue
            directory=Path(pp['catalog'][video]) if pp['catalog'][video] else build_cache(row,pp)
            fp=directory/'meta.json';meta=json.loads(fp.read_text());fps,n=validate_meta(meta,row['video_path'])
            arr=np.load(directory/'src_patch_position.npy',allow_pickle=False)
            require(arr.ndim==2 and arr.shape[1]==3 and arr.dtype.kind in 'iu','patch数组非法')
            ids=np.unique(arr[arr[:,0]>=0,0]).astype(int).tolist();require(ids and ids[-1]<n,'来源索引越界')
            src,cues=subtitle_source(video);require(src==pp['srt_sources'][video],'字幕来源改变')
            full=continuous_evidence(align(ids,fps,cues,src,count),cues,count,100000)
            try:
                require(full['text_tokens']<=13312,'完整字幕超过冻结13312预算')
                ee=evidence(ids,fps,cues,src,count)
                require(not ee['codec_complete13312']['continuous']['omitted_neighbor_ids'],'主路线未保留完整邻句')
            except ValueError as exc:
                problems.append(dict(video=video,stage=row['stage'],full_tokens=full['text_tokens'],error=str(exc)));bar.update(1);continue
            canvas=meta.get('canvas_files') or [f.name for f in sorted(directory.glob('canvas_*.jpg'))]
            require(canvas and all(Path(x).name==x for x in canvas),'画布路径非法')
            files=[fp,directory/'src_patch_position.npy']+[directory/f for f in canvas]
            require(all(f.is_file() and not f.is_symlink() for f in files),'缓存缺失/符号链接')
            v=dict(video_path=row['video_path'],cache_dir=str(directory),fps=fps,total_frames=n,source_indices=ids,evidence=ee,
                cache_files_sha256={str(f):sha(f) for f in files},canvas_files=canvas,source_video_identity=row['video_identity'])
            atomic(saved,common.sealed(dict(plan_signature=pp['signature'],video=v)))
            videos[video]=v;bar.update(1);bar.set_postfix(stage=row['stage'])
    attempt=OUT/'preflight'/('budget_'+str(time.time_ns())+'.json')
    atomic(attempt,dict(plan_signature=pp['signature'],checked_videos=238,ready=len(videos),blocked=problems,QA_calls=0))
    require(not problems and len(videos)==238,'预算/完整性未通过；全部题保留，GPU未启动。见:'+str(attempt))
    sources=dict(pp['plan']['source_sha256'])
    sources[str(OUT/'plan.json')]=sha(OUT/'plan.json')
    for v in videos.values():sources.update(v['cache_files_sha256'])
    worst=max(pp['rows'],key=lambda r:videos[r['video_id']]['evidence']['codec_complete13312']['continuous']['text_tokens']+len(videos[r['video_id']]['source_indices']))['video_id']
    # 用实际画布像素/4补上视觉token，选择最大的文本＋视觉输入，而非按答案挑题。
    def visual_count(v):
        ar=np.load(Path(v['cache_dir'])/'src_patch_position.npy',allow_pickle=False)
        return int((ar[:,0]>=0).sum())//4
    worst=max(videos,key=lambda v:videos[v]['evidence']['codec_complete13312']['continuous']['text_tokens']+visual_count(videos[v]))
    protocol=dict(pp['plan'],source_sha256=sources,probe_indices=[r['index'] for r in pp['rows'] if r['video_id']==worst],plan_signature=pp['signature'])
    sig=signature(protocol)
    atomic(OUT/'protocol.json',protocol)
    atomic(OUT/'bundle.json',common.sealed(dict(signature=sig,rows=pp['rows'],videos=videos,references=pp['references'])))
    atomic(OUT/'preflight/ready.json',dict(signature=sig,questions=714,videos=238,max_subtitle_tokens=max(v['evidence']['codec_complete13312']['continuous']['text_tokens'] for v in videos.values()),probe_indices=protocol['probe_indices']))
    print('238视频缓存、来源、冻结预算全部通过；GPU尚未启动。')


def context():
    guard();p=json.loads((OUT/'protocol.json').read_text());sig=signature(p)
    for f,h in p['source_sha256'].items():require(sha(f)==h,'来源变化:'+f)
    for name,expected in p['model']['weights'].items():
        st=(MODEL/name).stat();require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==expected,'权重变化')
    b=common.unseal(json.loads((OUT/'bundle.json').read_text()));require(b['signature']==sig,'数据包签名不同')
    for row in b['rows']:
        st=Path(row['video_path']).stat();require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==row['video_identity'],'视频变化')
    units={}
    for f in (OUT/'units').glob('*.json'):
        u=json.loads(f.read_text());validate_unit(sig,b,u)
        aud=json.loads((OUT/'input_audits'/f'{u["index"]}.json').read_text());common.unseal(aud)
        require(aud['signature']==sig and all(u[k]==aud['inputs'][u['arm']][k] for k in ['tensor_sha256','input_sha256','input_tokens','visual']),'输入审计不符')
        key=(u['index'],u['arm']);require(key not in units,'重复断点');units[key]=u
    for (idx,arm),u in units.items():
        if u['origin']=='reuse_current_exact':
            ref=units.get((idx,u['reuse']['arm']))
            require(ref is not None and ref['record_sha256']==u['reuse']['record_sha256'] and ref['raw_text']==u['raw_text'] and ref['tensor_sha256']==u['tensor_sha256'],'同题复用不符')
        elif u['origin']=='reuse_prior_development':
            src=Path(u['reuse']['source_path']);require(sha(src)==u['reuse']['source_file_sha256'],'开发复用来源改变')
            old=json.loads(src.read_text());require(old['record_sha256']==u['reuse']['source_record_sha256'] and old['raw_text']==u['raw_text'] and old['tensor_sha256']==u['tensor_sha256'],'开发答案/输入改变')
    return dict(protocol=p,signature=sig,bundle=b,units=units,scope='full714')


def seed_development(c):
    old=context_development()
    for row in c['bundle']['rows']:
        if row['stage']!='dev180':continue
        idx=row['index'];audit=json.loads((OUT/'input_audits'/f'{idx}.json').read_text());common.unseal(audit)
        require(audit['signature']==c['signature'],'开发复用须先CPU审计')
        for arm in ARMS:
            if (idx,arm) in c['units']:continue
            src=DEV_OUT/'units'/f'{idx}_{arm}.json';u=old['units'][(idx,arm)];require(u['tensor_sha256']==audit['inputs'][arm]['tensor_sha256'],'开发复用Tensor不同')
            current=dict(common.unseal(u),signature=c['signature'],stage='dev180',origin='reuse_prior_development',new_calls=0,generated_tokens=None,inference_s=0.,peak_gib=0.,historical_peak_gib=u['peak_gib'],historical_inference_s=u['inference_s'],
                reuse=dict(source_path=str(src),source_file_sha256=sha(src),source_record_sha256=u['record_sha256'],tensor_equal=True))
            current=common.sealed(current);validate_unit(c['signature'],c['bundle'],current)
            atomic(OUT/'units'/f'{idx}_{arm}.json',current);c['units'][(idx,arm)]=current
    print('已核对并复用开发180题/540单元，不新增生成。')


def scoped(c,scope):
    selected=c['bundle']['rows'] if scope=='full714' else [r for r in c['bundle']['rows'] if r['stage']==scope]
    ids={r['index'] for r in selected}
    return dict(c,bundle=dict(c['bundle'],rows=selected),units={k:u for k,u in c['units'].items() if k[0] in ids},scope=scope)



def texts(row,v):
    return dict(codec_no_sub=prompt(row,0),**{a:subtitle_prompt(row,e['continuous']['text']) for a,e in v['evidence'].items()})


def validate_unit(sig,b,u):
    common.unseal(u);row=next((r for r in b['rows'] if r['index']==u['index']),None)
    require(row is not None and u['arm'] in ARMS and u['signature']==sig,'单元身份不符')
    v=b['videos'][row['video_id']]
    require(u['prompt_sha256']==signature(texts(row,v)[u['arm']]),'单元提示不同')
    require(u['pred']==parse_answer(u['raw_text']) and u['native_equal'] is True,'解析/原生输入核对失败')
    require(u['origin'] in ['generated','reuse_current_exact','reuse_prior_development'],'未知来源')
    if u['origin']=='generated':require(u['new_calls']==1 and 1<=u['generated_tokens']<=32,'生成计数非法')
    else:require(u['new_calls']==0 and u['generated_tokens'] is None and u['inference_s']==0 and u['reuse']['tensor_equal'],'复用非法')
    expected=b['references'][row['index']]['codec_old_visual']
    if expected is not None:require(u['visual']==expected,'Codec视觉布局改变')
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
            loaded.clear()  # 仅释放内存；从不删除任何缓存文件。
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
            ref=c['bundle']['references'][row['index']]
            if ref['codec_old_visual'] is not None:require(vi==ref['codec_old_visual'],'原生Codec视觉未复现')
            if arm=='codec_no_sub':
                if ref['codec_old'] is not None:require(signature(packed['input_ids'].tolist())==ref['codec_old']['prefix_sha256'],'原生Codec无字幕输入未复现')
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
    with readonly_codec(engine.proc,c['bundle']),tqdm(total=len(expected),initial=len(c['units']),desc='Mage Codec字幕 '+c.get('scope','full714'),unit='答案',dynamic_ncols=True) as bar:
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
                ref=c['bundle']['references'][idx]['codec_old']
                baseline_equal=(parse_answer(raw)==ref['semantic_pred']) if arm=='codec_no_sub' and ref is not None else None
                u=common.sealed(dict(signature=c['signature'],index=idx,video_id=row['video_id'],arm=arm,pred=parse_answer(raw),raw_text=raw,origin=origin,reuse=proof,new_calls=calls,generated_tokens=generated,
                    prompt_sha256=signature(texts(row,v)[arm]),input_sha256=signature(x['input_ids'].tolist()),tensor_sha256=common.tensor_digest(x),visual=audit['inputs'][arm]['visual'],input_tokens=int(x['input_ids'].shape[1]),
                    subtitle_tokens=0 if arm=='codec_no_sub' else v['evidence'][arm]['continuous']['text_tokens'],native_equal=True,inference_s=elapsed,peak_gib=peak,baseline_answer_equal=baseline_equal))
                validate_unit(c['signature'],c['bundle'],u);atomic(OUT/'units'/f'{idx}_{arm}.json',u);c['units'][(idx,arm)]=u;bar.update(1);bar.set_postfix(index=idx,arm=arm)
                print('落盘',idx,arm,u['pred'] or 'INVALID',origin,flush=True)
                require(baseline_equal is not False,'新无字幕控制答案未复现；保留结果，先诊断，不继续')



def resource_probe(c):
    # 只按输入长度选出的最长视频，不读取答案；探测答案直接计入开发阶段。
    ids=set(c['protocol']['probe_indices'])
    selected=dict(c,bundle=dict(c['bundle'],rows=[r for r in c['bundle']['rows'] if r['index'] in ids]),units={k:u for k,u in c['units'].items() if k[0] in ids},probe_mode=True)
    run(selected)
    require(len(selected['units'])==9,'最长视频探测尚未完整')
    import torch
    peak=max(max(u['peak_gib'],u.get('historical_peak_gib',0)) for u in selected['units'].values())
    total=torch.cuda.get_device_properties(0).total_memory/1024**3
    passed=peak<total*.9 and all(u['pred'] in list('ABCD') for u in selected['units'].values())
    gate=dict(signature=c['signature'],indices=sorted(ids),peak_gib=peak,total_device_gib=total,passed=passed,note='Largest CPU-estimated visual+subtitle input; other questions may differ slightly. Not accuracy selection. OOM stops and preserves checkpoints.')
    atomic(OUT/'resource_gate.json',gate)
    print(json.dumps(gate,ensure_ascii=False,indent=2))
    require(passed,'显存余量或回答合法性不足，停止，不扩开发集')



def summarize(c):
    rows=c['bundle']['rows'];expected={(r['index'],a) for r in rows for a in ARMS};require(expected<=c['units'].keys(),'先完成所选范围，不读取标签')
    labels_path=ROOT/'01_manifests/evaluation_labels.json';lock=json.loads((ROOT/'01_manifests/dataset_lock.json').read_text())
    require(sha(labels_path)==lock['labels_sha256'],'标签改变');labels=json.loads(labels_path.read_text())
    from practical_analyze import comparison
    names=list(ARMS)+['frames_continuous2048'];pred={a:{r['index']:c['units'][(r['index'],a)]['pred'] for r in rows} for a in ARMS}
    pred['frames_continuous2048']={r['index']:c['bundle']['references'][r['index']]['frames_continuous2048']['pred'] for r in rows}
    metrics=[];paired=[];details=[];cost=[]
    groups=[(c['scope'],rows)]
    if c['scope']=='full714':groups += [(s,[r for r in rows if r['stage']==s]) for s in ['short219','dev180','val90','test225'] if any(r['stage']==s for r in rows)]
    for stage,rr in groups:
        for duration,rs in [('overall',rr)]+[(d,[r for r in rr if r['duration']==d]) for d in ['short','medium','long'] if any(r['duration']==d for r in rr)]:
            for name in names:
                correct=sum(pred[name][r['index']]==labels[r['index']] for r in rs)
                metrics.append(dict(stage=stage,duration=duration,method=name,N=len(rs),correct=correct,accuracy_pct=100*correct/len(rs)))
            for name,base in [('codec_complete13312','codec_no_sub'),('codec_anchor32_2048','codec_no_sub'),('codec_complete13312','codec_anchor32_2048'),('codec_complete13312','frames_continuous2048')]:
                paired.append(dict(stage=stage,duration=duration,method=name,baseline=base,**comparison(rs,pred[name],pred[base],labels)))
    for r in rows:
        idx=r['index'];details.append(dict(index=idx,video_id=r['video_id'],stage=r['stage'],role=r['role'],duration=r['duration'],gt=labels[idx],**{a:pred[a][idx] for a in names}))
        for a in ARMS:
            u=c['units'][(idx,a)];cost.append(dict(index=idx,stage=r['stage'],duration=r['duration'],method=a,input_tokens=u['input_tokens'],subtitle_tokens=u['subtitle_tokens'],visual_tokens=u['visual']['n_tok'],origin=u['origin'],new_calls=u['new_calls'],new_inference_s=u['inference_s'],peak_gib=u['peak_gib']))
    selected=[c['units'][key] for key in expected];invalid=sum(u['pred'] not in list('ABCD') for u in selected)
    controls=[u['baseline_answer_equal'] for u in selected if u['arm']=='codec_no_sub' and u['baseline_answer_equal'] is not None]
    matches=all(controls)
    report=dict(signature=c['signature'],scope=c['scope'],N=len(rows),videos=len({r['video_id'] for r in rows}),audit=dict(passed=invalid==0 and matches,invalid=invalid,historical_controls_checked=len(controls),old_codec_answer_parity=matches if controls else None,complete_units=len(selected)),metrics=metrics,paired=paired,
        cost=dict(new_calls=sum(u['new_calls'] for u in selected),reused=sum(not u['new_calls'] for u in selected),inference_s=sum(u['inference_s'] for u in selected),peak_gib=max(u['peak_gib'] for u in selected)),
        interpretation=c['protocol']['policy'],scope_note='Local714 coverage includes seen development and historically diagnosed val/test. Not official2700 or new independent final testing. No deployment timing; text/visual budgets differ from Frames reference.')
    dest=OUT/'results'/c['scope'];atomic(dest/'summary.json',report);atomic(dest/'predictions.json',details);atomic(dest/'token_cost.json',cost)
    common.csv_once(dest/'predictions.csv',details);common.csv_once(dest/'metrics.csv',metrics);common.csv_once(dest/'token_cost.csv',cost)
    common.csv_once(dest/'paired_comparisons.csv',[{**r,'video_cluster_bootstrap_95ci_pp':json.dumps(r['video_cluster_bootstrap_95ci_pp'])} for r in paired])
    print(json.dumps(dict(N=len(rows),scope=c['scope'],audit=report['audit'],cost=report['cost']),ensure_ascii=False,indent=2));print('汇总:',dest/'summary.json')
    require(report['audit']['passed'],'审计失败，保留结果，不继续')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','prepare-cache','preflight','audit-inputs','seed-dev','probe','run-val','run-full','summarize-val','summarize-full']);args=ap.parse_args()
    require(subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()=='feat/air-visual-evidence','分支必须为feat/air-visual-evidence')
    print('SPACE',guard(),flush=True);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'operation.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='init':init();return
        if args.action=='prepare-cache':prepare_cache();return
        if not (OUT/'protocol.json').exists():
            plan_context();print('计划已冻结；先由用户运行prepare-cache，通过缓存/预算门禁再启动GPU。');return
        c=context()
        if args.action=='preflight':print('全量714，已完成',len(c['units']),'/2142；不加载模型。')
        elif args.action=='audit-inputs':input_audit(c)
        elif args.action=='seed-dev':seed_development(c)
        elif args.action.startswith('summarize-'):summarize(scoped(c,'val90' if args.action.endswith('val') else 'full714'))
        else:
            with (ROOT/'cache/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if args.action=='probe':resource_probe(c)
                elif args.action=='run-val':run(scoped(c,'val90'))
                else:
                    gate=json.loads((OUT/'results/val90/summary.json').read_text())
                    require(gate['signature']==c['signature'] and gate['audit']['passed'],'先完成val90数据审计；不按得分选择是否继续')
                    run(c)


if __name__=='__main__':main()
