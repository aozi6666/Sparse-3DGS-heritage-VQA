#!/usr/bin/env python3
"""Mage 均衡换序实验；init/preflight 不导入模型，run 才使用 GPU。"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid

from practical_common import (PROJECT,ROOT,MODEL,DATA,OLD_CACHE,LETTERS,SLOT,STAGES,
    sha,signature,write_once,append_line,load_dataset,make_splits,stage_rows,
    jobs_for,ballot_key,load_ballots,prompt,remap,softmax,parse_answer)


def git(*args):
    return subprocess.check_output(['git','-C',str(PROJECT),*args],text=True).strip()


def disk_guard(root,min_free,budget):
    free=shutil.disk_usage('/root/autodl-tmp').free/1024**3
    if free<min_free:
        raise RuntimeError(f'剩余 {free:.2f} GiB，小于保留值 {min_free}，停止且不清理')
    total=0
    if root.exists():
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
            total+=sum((Path(folder)/n).stat().st_size for n in files if not (Path(folder)/n).is_symlink())
    if total>budget*1024**3:
        raise RuntimeError(f'新实验目录超过 {budget} GiB 预算，停止供审查，不清理')
    return dict(free_gib=round(free,3),experiment_mib=round(total/1024**2,3))


def source_audit():
    specs=[('A.I.R.','https://github.com/UCF-AIR/A.I.R.',
            ['air/model.py','air/Process_Utils/process_messages.py'], 'MIT'),
           ('VCD','https://github.com/DAMO-NLP-SG/VCD',
            ['vcd_utils/vcd_sample.py','vcd_utils/vcd_add_noise.py'],'Apache-2.0'),
           ('VideoHV-Agent','https://github.com/Haorane/VideoHV-Agent',
            ['VideoHV-Agent/video_hv/verifier.py','VideoHV-Agent/video_hv/pipelines/egoschema_openai/prompts.py'],
            'README declares Apache-2.0; inspect repository license before copying code')]
    result=[]
    for name,url,files,license_note in specs:
        folder=Path('/root/autodl-tmp/references')/name
        for file in files:
            if not (folder/file).is_file():
                raise FileNotFoundError(folder/file)
        try:
            commit=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        except subprocess.CalledProcessError:
            commit=None
        result.append(dict(name=name,url=url,path=str(folder),commit=commit,license_note=license_note,
            file_sha256={f:sha(folder/f) for f in files},license_sha256=sha(folder/'LICENSE') if (folder/'LICENSE').exists() else None,
            use='reference only; no third-party runtime code imported or copied'))
    return result


def model_identity():
    names=['config.json','preprocessor_config.json','video_preprocessor_config.json',
           'processing_mage_vl.py','codec_video_processing_mage_vl.py','video_processing_mage_vl.py',
           'modeling_mage_vl.py','configuration_mage_vl.py','generation_config.json',
           'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja',
           'added_tokens.json','special_tokens_map.json','model.safetensors.index.json']
    for n in names:
        if not (MODEL/n).is_file():
            raise FileNotFoundError(MODEL/n)
    mapping=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for n in sorted(set(mapping.values())):
        p=MODEL/n
        if not p.is_file() or p.stat().st_size==0:
            raise FileNotFoundError(p)
        weights[n]=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    # 不重复读 9.9G 权重：记录分片清单/大小/mtime，并哈希代码与 tokenizer。
    return dict(path=str(MODEL),files={n:sha(MODEL/n) for n in names},weights=weights)


def initialize(args):
    if git('branch','--show-current')!='vlm':
        raise RuntimeError('必须在 vlm；不自动切分支')
    print(disk_guard(args.root,args.min_free,args.budget))
    rows,labels=load_dataset()
    splits=make_splits(rows)
    manifest=args.root/'01_manifests'
    write_once(manifest/'inputs.json',dict(rows=rows))
    write_once(manifest/'evaluation_labels.json',labels)
    write_once(manifest/'splits.json',splits)
    # 在 CPU 准备阶段剥离历史文件中的真值；GPU 运行只读无标签的审计摘录。
    from practical_common import read_lines
    old=PROJECT/'VQA/eval_results/r2v_mage/px150000_f32/phase2_signals_v1/short'
    previous=json.loads((old/'signal_config_lock.json').read_text())
    identity=model_identity()
    for name,value in previous['source_config']['model_sources_sha256'].items():
        if identity['files'][name]!=value:
            raise ValueError(f'旧冒烟来源的模型已漂移: {name}')
    for name,value in previous['tokenizer_files_sha256'].items():
        if identity['files'].get(name)!=value:
            raise ValueError(f'旧冒烟来源的 tokenizer 已漂移: {name}')
    if identity['weights']!=previous['source_config']['weights']:
        raise ValueError('旧权重分片身份不一致')
    smoke_indices={r['index'] for r in rows if r['duration']=='short'}
    legacy={r['index']:{b:dict(pred=r['phase1'][b+'_pred'],grid=r[b]['grid_thw'],shape=r[b]['pv_shape'],
                             time=r[b]['patch_time_range']) for b in ['codec','frames']}
            for r in read_lines(old/'signal_short.jsonl') if r['index'] in smoke_indices}
    write_once(manifest/'legacy_smoke_reference.json',legacy)
    write_once(manifest/'dataset_lock.json',dict(tsv_sha256=sha(DATA/'Video-MME.tsv'),
        inputs_sha256=sha(manifest/'inputs.json'),splits_sha256=sha(manifest/'splits.json'),
        labels_sha256=sha(manifest/'evaluation_labels.json'),seed=0,
        legacy_smoke_sha256=sha(manifest/'legacy_smoke_reference.json'),
        note='short is previously used; medium/long split is video-disjoint; test remains unopened',
        videos={r['video_path']:dict(bytes=Path(r['video_path']).stat().st_size,
                    mtime_ns=Path(r['video_path']).stat().st_mtime_ns) for r in rows}))
    write_once(args.root/'00_sources/audit.json',source_audit())
    for stage in STAGES:
        selected=stage_rows(args.root,stage)
        print(stage,len(selected),'题',len({r['video_id'] for r in selected}),'视频')


def context(args):
    if git('branch','--show-current')!='vlm':
        raise RuntimeError('必须在 vlm')
    print('SPACE',disk_guard(args.root,args.min_free,args.budget),flush=True)
    m=args.root/'01_manifests'
    lock=json.loads((m/'dataset_lock.json').read_text())
    if sha(DATA/'Video-MME.tsv')!=lock['tsv_sha256'] or sha(m/'inputs.json')!=lock['inputs_sha256'] or sha(m/'splits.json')!=lock['splits_sha256']:
        raise ValueError('数据或划分漂移，不自动更新清单')
    if sha(m/'legacy_smoke_reference.json')!=lock['legacy_smoke_sha256']:
        raise ValueError('历史冒烟审计摘录漂移')
    rows=stage_rows(args.root,'full')
    for path,expected in lock['videos'].items():
        p=Path(path)
        if not p.is_file() or dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=expected:
            raise ValueError(f'视频漂移: {path}')
    folder=Path(__file__).parent
    protocol=dict(version='practical_vote_v1',model=model_identity(),
        tsv_sha256=lock['tsv_sha256'],inputs_sha256=lock['inputs_sha256'],splits_sha256=lock['splits_sha256'],
        legacy_smoke_sha256=lock['legacy_smoke_sha256'],
        prompt=__import__('practical_common').PROMPT,
        frames=args.frames,max_pixels=150000,min_pixels=3136,
        codec=dict(engine='hevc',target_canvas=32,patch=16,group_size=32,images_per_group=4,spatial_mask_mode='off',max_pixels=150000),
        seed=0,do_sample=False,max_new_tokens=32,score='raw first-position logits; score aggregation only when first token is parsed answer; vote uses original free-generation parser',
        permutations='four cyclic rotations; semantic majority; ties prefer identity',
        scripts={n:sha(folder/n) for n in ['practical_common.py','practical_vote.py','practical_analyze.py']})
    sig=signature(protocol)
    tag=f'px150000_f{args.frames}_frames4_codec{4 if args.codec_vote else 1}'
    store=args.root/'cache/ballots'/sig
    selected=stage_rows(args.root,args.stage)
    chosen=None
    if args.stage in ['test','full']:
        frozen=json.loads((m/f'frozen_{tag}.json').read_text())
        if frozen['signature']!=sig:
            raise ValueError('冻结配置与当前代码/模型/预算不符')
        chosen=frozen['method']
        if args.stage=='full':
            summary=args.root/STAGES['test']/tag/'summary.json'
            if not summary.exists() or json.loads(summary.read_text())['signature']!=sig:
                raise ValueError('先完成独立测试并汇总，才允许运行 full')
    units=load_ballots(store/'units.jsonl',sig,rows)
    expected={ballot_key(r['index'],b,k) for r in selected for b,k in jobs_for(r,args.codec_vote,chosen)}
    return dict(protocol=protocol,signature=sig,tag=tag,store=store,rows=selected,all_rows=rows,
                units=units,expected=expected,chosen=chosen,stage_dir=args.root/STAGES[args.stage]/tag)


def prerequisites(args,ctx):
    for binary in [os.environ.get('CV_PREINFER_BIN','cv-preinfer'),'ffmpeg','ffprobe']:
        if not shutil.which(binary):
            raise FileNotFoundError(f'缺少可执行文件: {binary}')
    import importlib.util
    for name in ['torch','transformers','tqdm']:
        if importlib.util.find_spec(name) is None:
            raise ImportError(f'环境缺少 {name}；不自动安装')
    if args.stage!='smoke':
        p=args.root/STAGES['smoke']/ctx['tag']/'summary.json'
        audit=json.loads(p.read_text())
        if audit['signature']!=ctx['signature'] or not audit['audit']['smoke_pass']:
            raise ValueError('先对当前配置完成 smoke 并 summarize，检查通过后再进入下一阶段')
    if args.stage=='val':
        paths=[args.root/STAGES[s]/ctx['tag']/'summary.json' for s in ['dev30','dev60']]
        if not any(p.exists() and json.loads(p.read_text())['signature']==ctx['signature'] for p in paths):
            raise ValueError('先完成开发池并汇总，不能直接运行 val')


def cache_for(video,codec,root):
    """只读复用旧缓存；新的预推理缓存严格落到新实验根目录内。"""
    import hashlib
    tag=signature(codec)[:16]
    raw=(f'{video}|eng={codec["engine"]}|tc={codec["target_canvas"]}|gs={codec["group_size"]}'
         f'|ipg={codec["images_per_group"]}|patch={codec["patch"]}|mp={codec["max_pixels"]}|mask={codec["spatial_mask_mode"]}')
    name=f'{Path(video).stem}_{hashlib.md5(raw.encode()).hexdigest()}'
    for parent in [OLD_CACHE/tag,root/'cache/codec'/tag]:
        target=parent/name
        if target.exists():
            if not all((target/n).is_file() for n in ['meta.json','src_patch_position.npy']):
                raise ValueError(f'缓存残缺，拒绝让底层重建或删除: {target}')
            return parent
    parent=root/'cache/codec'/tag
    parent.mkdir(parents=True,exist_ok=True)
    # 先前中断留下的临时目录不自动清除，也不允许绕过检查静默增长。
    if any(p.is_dir() and p.name.startswith('.tmp_') for p in parent.iterdir()):
        raise ValueError(f'发现未完成的缓存临时目录，保留并等待审查: {parent}')
    return parent


class CaptureTokenizer:
    """仅捕获处理器最后 tokenize 的文本；不修改 tokenizer 或模型代码。"""
    def __init__(self,tokenizer):
        self.base=tokenizer; self.text=None
    def __getattr__(self,name):
        return getattr(self.base,name)
    def __call__(self,text,*args,**kwargs):
        self.text=list(text) if isinstance(text,list) else [text]
        return self.base(text,*args,**kwargs)


@contextmanager
def session_log(path):
    class Tee:
        def __init__(self,stream,file):
            self.stream,self.file=stream,file
        def write(self,s):
            self.file.write(s); self.file.flush()
            return self.stream.write(s)
        def flush(self):
            self.file.flush(); self.stream.flush()
        def isatty(self):
            return self.stream.isatty()
        def __getattr__(self,name):
            return getattr(self.stream,name)
    stdout,stderr=sys.stdout,sys.stderr
    with path.open('x',encoding='utf-8') as log:
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            yield
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout,sys.stderr=stdout,stderr


def run(args,ctx):
    prerequisites(args,ctx)
    root=args.root
    (root/'cache').mkdir(parents=True,exist_ok=True)
    # 不同配置也不能在本实验中同时占 GPU；外部任务仍需用户自行管理。
    with (root/'cache/gpu.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # 获取锁后再次读取，避免等待前的快照过期。
        ctx=context(args)
        ctx['store'].mkdir(parents=True,exist_ok=True)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        ctx['stage_dir'].mkdir(parents=True,exist_ok=True)
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],
            indices=[r['index'] for r in ctx['rows']],stage=args.stage,chosen=ctx['chosen'],codec_vote=args.codec_vote,
            store=str(ctx['store']),protocol=ctx['protocol']))
        missing=ctx['expected']-set(ctx['units'])
        if not missing:
            print('全部必要单元已落盘，跳过模型加载；请 summarize。')
            return
        if args.stage!='smoke' and os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':
            raise RuntimeError('当前阶段须由用户设置 PRACTICAL_STAGE_CONFIRMED=1')
        log=ctx['stage_dir']/('run_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.log')
        with session_log(log):
            run_gpu(args,ctx)


def run_gpu(args,ctx):
    import random
    import time
    # 必须在导入 Transformers 之前指定动态源码缓存位置，避免新文件散落到主目录。
    os.environ['HF_MODULES_CACHE']=str(args.root/'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoProcessor
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    processor.video_processor.max_pixels=150000
    processor.video_processor.min_pixels=3136
    processor.image_processor.max_pixels=150000
    processor.image_processor.min_pixels=3136
    tokenizer=processor.tokenizer
    probes={}
    for c in LETTERS:
        group=[]
        for text in [c,' '+c]:
            ids=tokenizer.encode(text,add_special_tokens=False)
            if len(ids)!=1 or tokenizer.decode(ids,clean_up_tokenization_spaces=False)!=text:
                raise ValueError(f'候选不是精确单 token: {text!r}')
            group.append(ids[0])
        probes[c]=group
    if len({x for group in probes.values() for x in group})!=8:
        raise ValueError('字母 token 重叠')
    write_once(ctx['store']/'tokens.json',probes)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,
                local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0))
    write_once(ctx['store']/'runtime.json',runtime)
    print('RUNTIME',runtime,'signature',ctx['signature'],flush=True)
    # 冒烟只读取旧答案用于输入/协议审计；从不将旧 gt 放进推理输入或选择器。
    legacy={}
    if args.stage=='smoke':
        legacy=json.loads((args.root/'01_manifests/legacy_smoke_reference.json').read_text())
        if any(r['index'] not in legacy for r in ctx['rows']):
            raise ValueError('缺少冒烟对应的原始协议审计来源')
    codec=ctx['protocol']['codec']
    cached={}  # 每个后端只缓存最后一个视频的 CPU 预处理，避免跨全数据积累 tensor。

    def original_text(question):
        return processor.apply_chat_template([{'role':'user','content':[
            {'type':'video'},{'type':'text','text':question}]}],tokenize=False,add_generation_prompt=True)

    def process(video,backend,question):
        if backend=='codec':
            os.environ['ONLINE_CODEC_CACHE_DIR']=str(cache_for(video,codec,args.root))
        return processor(text=[original_text(question)],videos=[video],video_backend=backend,
            num_frames=args.frames,max_pixels=150000,codec_config=codec,return_tensors='pt',padding=True)

    def prepared(row,backend):
        hit=cached.get(backend)
        if hit and hit['video']==row['video_path']:
            return hit,0.0
        start=time.perf_counter()
        capture=CaptureTokenizer(tokenizer)
        processor.tokenizer=capture
        try:
            values=process(row['video_path'],backend,SLOT)
        finally:
            processor.tokenizer=tokenizer
        if capture.text is None or len(capture.text)!=1 or capture.text[0].count(SLOT)!=1:
            raise ValueError('无法安全定位文本插槽，停止而不猜测替换位置')
        visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']}
        hit=dict(video=row['video_path'],text=capture.text[0],visual=visual)
        cached[backend]=hit
        return hit,time.perf_counter()-start

    expected=ctx['expected']; units=ctx['units']
    with (ctx['store']/'units.jsonl').open('a',encoding='utf-8') as out, \
         tqdm(total=len(ctx['rows']),initial=sum(all(ballot_key(r['index'],b,k) in units for b,k in jobs_for(r,args.codec_vote,ctx['chosen'])) for r in ctx['rows']),
              desc=f'Mage 换序 {args.stage}',unit='题',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            jobs=jobs_for(row,args.codec_vote,ctx['chosen'])
            if all(ballot_key(row['index'],b,k) in units for b,k in jobs):
                continue
            for backend,k in jobs:
                key=ballot_key(row['index'],backend,k)
                if key in units:
                    continue
                disk_guard(args.root,args.min_free,args.budget)
                bar.set_postfix(index=row['index'],backend=backend,perm=k,units=f'{len(expected & set(units))}/{len(expected)}')
                hit,prep_time=prepared(row,backend)
                question=prompt(row,k)
                text=hit['text'].replace(SLOT,question)
                encoding=tokenizer([text],padding=True,return_tensors='pt')
                inputs=dict(hit['visual'],input_ids=encoding['input_ids'],attention_mask=encoding['attention_mask'])
                sms=int(processor.spatial_merge_size)
                grid=inputs['image_grid_thw'].tolist()
                n_raw=sum(t*h*w for t,h,w in grid)
                if n_raw%(sms*sms):
                    raise ValueError('grid token 数无法整除 merge²')
                n_tok=n_raw//(sms*sms)
                image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
                if int((inputs['input_ids']==image_id).sum())!=n_tok:
                    raise ValueError('视觉 token 占位数量与视觉网格不一致')
                audit=dict(direct_input_equal=None,legacy_equal=None)
                if args.stage=='smoke' and k==0:
                    # 关键运行审计：缓存插槽路径必须与原 Mage 完整处理路径逐 tensor 相等。
                    direct=process(row['video_path'],backend,question)
                    same=set(direct)==set(inputs) and all(torch.equal(direct[a],inputs[a]) for a in direct)
                    if not same:
                        raise ValueError('复用视觉后的输入与原生处理路径不一致')
                    audit['direct_input_equal']=True
                    del direct
                prefix_hash=signature(inputs['input_ids'].tolist())
                shape=list(inputs['pixel_values'].shape)
                positions=inputs['patch_positions']
                time_range=[float(positions[:,0].min()),float(positions[:,0].max())]
                gpu_inputs={a:(v.to(device=model.device,dtype=model.dtype) if a=='pixel_values' else
                        v.to(model.device)) for a,v in inputs.items()}
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                with torch.inference_mode():
                    result=model.generate(**gpu_inputs,max_new_tokens=32,do_sample=False,
                        return_dict_in_generate=True,output_logits=True)
                if not result.logits:
                    raise ValueError('没有取得 next-token 位置')
                logits=result.logits[0][0].float()
                prefix_len=inputs['input_ids'].shape[1]
                token_id=int(result.sequences[0,prefix_len])
                if token_id!=int(logits.argmax()):
                    raise ValueError('生成 token 不等于原始 logits argmax，停止检查后处理')
                first_letter=next((c for c,ids in probes.items() if token_id in ids),'')
                raw_text=tokenizer.decode(result.sequences[0,prefix_len:],skip_special_tokens=True).strip()
                letter=parse_answer(raw_text)
                raw_logits=logits[[probes[c][0] for c in LETTERS]].cpu().tolist()
                p=softmax(raw_logits)
                semantic=LETTERS[(LETTERS.index(letter)+k)%4] if letter else ''
                torch.cuda.synchronize()
                latency=time.perf_counter()-start
                if args.stage=='smoke' and k==0 and (backend=='codec' or args.frames==32):
                    old=legacy[row['index']][backend]
                    audit['legacy_equal']=(semantic==old['pred'] and grid==old['grid'] and shape==old['shape'] and time_range==old['time'])
                    if not audit['legacy_equal']:
                        raise ValueError(f'恒等排列未复现原协议: {row["index"]}/{backend}')
                record=dict(signature=ctx['signature'],index=row['index'],video_id=row['video_id'],
                    backend=backend,permutation_id=k,prompt_sha256=signature(question),
                    display_pred=letter,semantic_pred=semantic,plain_logits=raw_logits,semantic_p=remap(p,k),
                    raw_text=raw_text,first_letter=first_letter,score_eligible=bool(first_letter and first_letter==letter),
                    generated_token_id=token_id,generated_text=tokenizer.decode([token_id],clean_up_tokenization_spaces=False),
                    prefix_sha256=prefix_hash,grid_thw=grid,pv_shape=shape,patch_time_range=time_range,
                    n_tok=n_tok,preprocess_s=prep_time,inference_s=latency,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    frames_max_pixels=processor.video_processor.max_pixels,frames_min_pixels=processor.video_processor.min_pixels,
                    audit=audit,origin_stage=args.stage)
                append_line(out,record); units[key]=record
                print(f'落盘 {key}: 显示字母={letter or "INVALID"}, 原选项={semantic or "INVALID"}, token={n_tok}, 推理={latency:.2f}s',flush=True)
                del result,logits,gpu_inputs
            bar.update(1)
    print(f'完成 {args.stage}；运行 summarize 检查结果。',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','preflight','run'])
    p.add_argument('--stage',choices=list(STAGES),default='smoke')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--frames',type=int,choices=[32,48,64],default=32)
    p.add_argument('--codec-vote',action='store_true')
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.root.resolve()!=ROOT or args.min_free<1 or args.budget<=0:
        p.error('固定结果根目录，磁盘保留至少 1 GiB，目录预算须为正数')
    if args.action=='init':
        initialize(args); return
    ctx=context(args)
    prerequisites(args,ctx)
    print(f'只读预检通过：{len(ctx["rows"])} 题 / {len(ctx["expected"])} 单元，已有 {len(ctx["expected"] & set(ctx["units"]))}；{ctx["tag"]}')
    if args.action=='run':
        run(args,ctx)


if __name__=='__main__':
    main()
