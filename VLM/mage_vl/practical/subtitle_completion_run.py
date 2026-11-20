#!/usr/bin/env python3
"""仅budget3072/short219；准备不加载模型，run由用户启动。"""
import argparse
import fcntl
import json
import os
import time

from practical_common import ROOT, MODEL, SLOT, signature, write_once, append_line, parse_answer
from practical_vote import CaptureTokenizer, disk_guard
from subtitle_common import CaptureVideoProcessor, make_alignment, sealed
from continuous_common import continuous_evidence
from subtitle_completion_common import context,key,query,old_source,validate_record,prepare_short


def equal_inputs(left,right,torch):
    return set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left)


def preflight(args,ctx):
    import importlib.util
    for name in ['torch','transformers','tokenizers','tqdm','decord']:
        if importlib.util.find_spec(name) is None: raise ImportError(f'缺少{name}，不自动安装')
    disk_guard(ROOT,args.min_free,args.budget)
    print(f'{args.scope}: {len(ctx["units"])}/{len(ctx["expected"])}答案已落盘，文本预算{ctx["tokens"]}',flush=True)
    if args.scope=='budget': print('仅24题新生成；156题重新构造两路输入逐tensor校验后复用。',flush=True)
    else: print('short219三路配对；缺字幕回退，旧单元仅在实际输入身份通过后复用。',flush=True)


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1': raise RuntimeError('用户设置PRACTICAL_STAGE_CONFIRMED=1后启动')
    with (ROOT/'cache/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); ctx=context(args)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],scope=args.scope,
            indices=[r['index'] for r in ctx['rows']],arms=list(ctx['arms']),store=str(ctx['store'])))
        if ctx['expected'].issubset(ctx['units']): print('全部已完成，跳过模型加载；请summarize。'); return
        run_gpu(args,ctx)


def run_gpu(args,ctx):
    import random
    os.environ['HF_MODULES_CACHE']=str(ROOT/'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoProcessor,AutoModelForCausalLM
    from tqdm import tqdm
    if not torch.cuda.is_available(): raise RuntimeError('CUDA不可用')
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for part in [processor.video_processor,processor.image_processor]: part.max_pixels=150000; part.min_pixels=3136
    tokenizer=processor.tokenizer; capture_video=CaptureVideoProcessor(processor.video_processor); processor.video_processor=capture_video
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,
        torch_dtype='auto',device_map={'':0}).eval()
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype))
    reference=json.loads((ctx['parent']['store']/'runtime.json').read_text())
    if runtime!=reference: raise ValueError('运行环境/硬件/精度与2048开发阶段不同')
    write_once(ctx['store']/'runtime.json',runtime)
    cached=None

    def native(row,text):
        chat=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True)
        return processor(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)

    def inputs(hit,text):
        enc=tokenizer([hit['template'].replace(SLOT,text)],padding=True,return_tensors='pt')
        return dict(hit['visual'],input_ids=enc['input_ids'],attention_mask=enc['attention_mask'])

    def prepare(row):
        nonlocal cached
        if cached is not None and cached['video_id']==row['video_id']: return cached
        capture=CaptureTokenizer(tokenizer); processor.tokenizer=capture
        try: values=native(row,SLOT)
        finally: processor.tokenizer=tokenizer
        if capture.text is None or len(capture.text)!=1 or capture.text[0].count(SLOT)!=1: raise ValueError('模板插槽异常')
        if capture_video.timestamps is None or len(capture_video.timestamps)!=1: raise ValueError('真实采样秒数缺失')
        indices=[int(i) for i in values['patch_positions'][:,0].unique_consecutive().tolist()]; seconds=capture_video.timestamps[0]
        video=row['video_id']
        if ctx['scope']=='budget':
            evidence=ctx['evidence'][video]
            if indices!=evidence['aligned']['frame_indices'] or seconds!=evidence['aligned']['frame_seconds']:
                raise ValueError('预算对照实际采样漂移')
        else:
            original=make_alignment(video,ctx['sources']['sources'][video],ctx['cues'][video],indices,list(seconds),ctx['count_tokens'])
            evidence=dict(aligned=original,continuous=continuous_evidence(original,ctx['cues'][video],ctx['count_tokens'],ctx['tokens']))
            write_once(ctx['store']/'alignments'/f'{video}.json',dict(signature=ctx['signature'],evidence=evidence,evidence_sha256=signature(evidence)))
        hit=dict(video_id=video,template=capture.text[0],visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']},evidence=evidence)
        texts={query(row,arm,evidence) for arm in ctx['arms']}
        if ctx['scope']=='budget':
            old_e=dict(aligned=evidence['aligned'],continuous=ctx['parent']['evidence'][video])
            texts.add(query(row,'continuous',old_e))
        for text in sorted(texts):
            expected=inputs(hit,text); direct=native(row,text)
            if not equal_inputs(expected,direct,torch) or capture_video.timestamps[0]!=seconds:
                raise ValueError('插槽构造与原生实际输入/时间不一致')
            del expected,direct
        cached=hit; del values
        return hit

    def generate(actual):
        gpu={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in actual.items()}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode(): output=model.generate(**gpu,max_new_tokens=32,do_sample=False)
        raw=tokenizer.decode(output[0,actual['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        generated=int(output.shape[1]-actual['input_ids'].shape[1]); torch.cuda.synchronize()
        elapsed=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()/1024**3
        del gpu,output
        return raw,generated,elapsed,peak

    with (ctx['store']/'units.jsonl').open('a',encoding='utf-8') as out, \
            tqdm(total=len(ctx['expected']),initial=len(ctx['units']),desc=f'Mage 字幕收尾 {args.scope}',unit='答案',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx=row['index']
            if all(key(idx,arm) in ctx['units'] for arm in ctx['arms']): continue
            disk_guard(ROOT,args.min_free,args.budget); start=time.perf_counter(); hit=prepare(row); e=hit['evidence']
            prep=time.perf_counter()-start
            for arm in ctx['arms']:
                k=key(idx,arm)
                if k in ctx['units']: continue
                disk_guard(ROOT,args.min_free,args.budget)
                text=query(row,arm,e); actual=inputs(hit,text); input_hash=signature(actual['input_ids'].tolist())
                visual=dict(grid_thw=actual['image_grid_thw'].tolist(),pv_shape=list(actual['pixel_values'].shape),
                    patch_time_range=[float(actual['patch_positions'][:,0].min()),float(actual['patch_positions'][:,0].max())],
                    n_tok=sum(t*h*w for t,h,w in actual['image_grid_thw'].tolist())//int(processor.spatial_merge_size)**2)
                if int((actual['input_ids']==tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())!=visual['n_tok']:
                    raise ValueError('视觉占位token不符')
                source=old_source(ctx,idx,arm); proof=None; checks=dict(native_input_equal=True)
                if ctx['scope']=='budget':
                    old_e=dict(aligned=e['aligned'],continuous=ctx['parent']['evidence'][row['video_id']])
                    previous=inputs(hit,query(row,'continuous',old_e)); previous_hash=signature(previous['input_ids'].tolist())
                    if previous_hash!=source['input_sha256'] or visual!=source['visual'] or signature(query(row,'continuous',old_e))!=source['prompt_sha256']:
                        raise ValueError('旧2048实际输入未复现，不允许继续')
                    checks.update(old_2048_input_equal=True,old_2048_input_sha256=previous_hash)
                    if idx not in ctx['changed'] and not equal_inputs(actual,previous,torch):
                        raise ValueError('计划复用题实际tensor不同；不擅自扩大新推理范围')
                    del previous
                reuse=source is not None and source['prompt_sha256']==signature(text) and source['input_sha256']==input_hash and source['visual']==visual
                if ctx['scope']=='budget' and (reuse==(idx in ctx['changed'])):
                    raise ValueError('预算变化题与实际输入身份不一致')
                if reuse:
                    # 新旧Prompt逐字相同，当前原生/插槽路径逐tensor相同，且旧前缀和视觉布局身份一致。
                    source_text=inputs(hit,text)
                    if not equal_inputs(actual,source_text,torch): raise ValueError('复用tensor不一致')
                    raw=source['raw_text']; generated=None; elapsed=peak=0.0; calls=0; origin='reused_old'
                    proof=dict(source_signature=source['signature'],source_record_sha256=source['record_sha256'],source_path=source['path'],
                        input_sha256_equal=True,visual_equal=True,tensor_equal=True)
                    del source_text
                elif ctx['scope']=='short' and arm!='no_sub' and query(row,arm,e)==query(row,'no_sub',e):
                    base=ctx['units'][key(idx,'no_sub')]
                    if input_hash!=base['input_sha256'] or visual!=base['visual']: raise ValueError('缺字幕输入未逐字回退')
                    raw=base['raw_text']; generated=None; elapsed=peak=0.0; calls=0; origin='reused_current_no_sub'
                    proof=dict(no_sub_record_sha256=base['record_sha256'],tensor_equal=True)
                else:
                    if ctx['scope']=='budget' and idx not in ctx['changed']: raise ValueError('禁止为156题新增生成')
                    raw,generated,elapsed,peak=generate(actual); calls=1; origin='generated'
                tokens=0 if arm=='no_sub' else e['aligned']['text_tokens'] if arm=='aligned' else e['continuous']['text_tokens']
                record=sealed(dict(signature=ctx['signature'],index=idx,video_id=row['video_id'],arm=arm,origin=origin,
                    raw_text=raw,pred=parse_answer(raw),generated_tokens=generated,new_answer_calls=calls,reuse_proof=proof,
                    prompt_sha256=signature(text),input_sha256=input_hash,input_tokens=int(actual['input_ids'].shape[1]),
                    evidence_sha256=signature(e),visual=visual,subtitle_tokens=tokens,checks=checks,
                    preprocess_s=prep,inference_s=elapsed,peak_allocated_gib=peak))
                validate_record(ctx,row,e,record,ctx['units'])
                append_line(out,record); ctx['units'][k]=record; prep=0.0; bar.update(1)
                bar.set_postfix(index=idx,arm=arm,new=calls)
                print(f'落盘 {k}: {record["pred"] or "INVALID"}, {origin}, 字幕{tokens}token',flush=True)
                del actual
    print('阶段完成；请summarize。不会自动选择预算或进入short/test/full。',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['init','preflight','run','prepare-short'])
    p.add_argument('--scope',choices=['budget','short'],default='budget'); p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0: p.error('磁盘预算无效')
    if args.action=='prepare-short': prepare_short(args); return
    ctx=context(args,initialize=args.action=='init')
    if args.action=='init': print('CPU清单准备完成，未抽帧/加载模型/推理。',ctx['signature']); return
    preflight(args,ctx)
    if args.action=='run': run(args,ctx)


if __name__=='__main__': main()
