#!/usr/bin/env python3
"""Mage 连续补句+时间标记；init/preflight不加载模型。"""
import argparse
import fcntl
import json
import os
import time

from practical_common import ROOT, MODEL, SLOT, signature, write_once, append_line, prompt, parse_answer
from practical_vote import CaptureTokenizer, disk_guard
from subtitle_common import CaptureVideoProcessor, subtitle_prompt, sealed
from continuous_common import context, RESULTS, TAG, STAGES


def preflight(args,ctx):
    if args.stage!='smoke':
        s=json.loads((RESULTS/'smoke'/TAG/'summary.json').read_text())
        if s['signature']!=ctx['signature'] or not s['audit']['passed']:
            raise ValueError('先完成本配置smoke并summarize')
    print(f'连续补句 {args.stage}: {len(ctx["expected"] & set(ctx["units"]))}/{len(ctx["rows"])}题已落盘',flush=True)


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':
        raise RuntimeError('请由用户设置PRACTICAL_STAGE_CONFIRMED=1')
    with (ROOT/'cache/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=context(args)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],stage=args.stage,
            indices=[r['index'] for r in ctx['rows']],store=str(ctx['store'])))
        if ctx['expected'].issubset(ctx['units']):
            print('全部已完成，跳过模型加载；请summarize。'); return
        run_gpu(args,ctx)


def run_gpu(args,ctx):
    import random
    os.environ['HF_MODULES_CACHE']=str(ROOT/'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoProcessor,AutoModelForCausalLM
    from tqdm import tqdm
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for part in [processor.video_processor,processor.image_processor]:
        part.max_pixels=150000; part.min_pixels=3136
    tokenizer=processor.tokenizer; capture_video=CaptureVideoProcessor(processor.video_processor)
    processor.video_processor=capture_video
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,
        torch_dtype='auto',device_map={'':0}).eval()
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype))
    if runtime!=json.loads((ctx['old']['store']/'runtime.json').read_text()):
        raise ValueError('环境/硬件/精度与开发阶段不同')
    write_once(ctx['store']/'runtime.json',runtime)
    cached=None

    def native(row,query):
        chat=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':query}]}],
            tokenize=False,add_generation_prompt=True)
        return processor(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=32,
            max_pixels=150000,return_tensors='pt',padding=True)

    def inputs(hit,query):
        text=tokenizer([hit['text'].replace(SLOT,query)],padding=True,return_tensors='pt')
        return dict(hit['visual'],input_ids=text['input_ids'],attention_mask=text['attention_mask'])

    def generate(values):
        actual={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in values.items()}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**actual,max_new_tokens=32,do_sample=False)
        raw=tokenizer.decode(output[0,values['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        generated=int(output.shape[1]-values['input_ids'].shape[1]); torch.cuda.synchronize()
        seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()/1024**3
        del output,actual
        return raw,generated,seconds,peak

    with (ctx['store']/'units.jsonl').open('a',encoding='utf-8') as out, \
            tqdm(total=len(ctx['rows']),initial=len(ctx['expected'] & set(ctx['units'])),desc=f'Mage 连续字幕 {args.stage}',unit='题',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx=row['index']
            if idx in ctx['units']: continue
            disk_guard(ROOT,args.min_free,args.budget); start=time.perf_counter(); e=ctx['evidence'][row['video_id']]
            if cached is None or cached['video_id']!=row['video_id']:
                capture=CaptureTokenizer(tokenizer); processor.tokenizer=capture
                try: values=native(row,SLOT)
                finally: processor.tokenizer=tokenizer
                if capture.text is None or len(capture.text)!=1 or capture.text[0].count(SLOT)!=1:
                    raise ValueError('模板插槽异常')
                indices=[int(i) for i in values['patch_positions'][:,0].unique_consecutive().tolist()]
                if indices!=e['original']['frame_indices'] or capture_video.timestamps[0]!=e['original']['frame_seconds']:
                    raise ValueError('真实采样帧号/秒数漂移')
                cached=dict(video_id=row['video_id'],text=capture.text[0],visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']})
                actual=inputs(cached,subtitle_prompt(row,e['text'])); direct=native(row,subtitle_prompt(row,e['text']))
                if set(actual)!=set(direct) or not all(torch.equal(actual[k],direct[k]) for k in actual):
                    raise ValueError('新Prompt未复现原生输入')
                del direct,actual,values
            plain=inputs(cached,prompt(row,0)); actual=inputs(cached,subtitle_prompt(row,e['text']))
            old=ctx['old']['base']['units'][f'{idx}:frames:p0']
            visual=dict(grid_thw=actual['image_grid_thw'].tolist(),pv_shape=list(actual['pixel_values'].shape),
                patch_time_range=[float(actual['patch_positions'][:,0].min()),float(actual['patch_positions'][:,0].max())],
                n_tok=sum(t*h*w for t,h,w in actual['image_grid_thw'].tolist())//int(processor.spatial_merge_size)**2)
            base_hash=signature(plain['input_ids'].tolist())
            if base_hash!=old['prefix_sha256'] or any(visual[k]!=old[k] for k in visual):
                raise ValueError('原视觉布局/无字幕输入未复现')
            if int((actual['input_ids']==tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())!=visual['n_tok']:
                raise ValueError('视觉token占位不符')
            prep=time.perf_counter()-start; control_s=0.0
            if args.stage=='smoke':
                raw,_,control_s,_=generate(plain)
                if parse_answer(raw)!=old['semantic_pred']:
                    raise ValueError('smoke无字幕控制未复现')
            if e['status']=='used':
                raw,generated,seconds,peak=generate(actual); calls=1
            else:
                raw=old['raw_text']; generated=None; seconds=peak=0.0; calls=0
            record=sealed(dict(signature=ctx['signature'],index=idx,video_id=row['video_id'],origin_stage=args.stage,
                pred=parse_answer(raw),raw_text=raw,status=e['status'],subtitle_tokens=e['text_tokens'],
                evidence_sha256=signature(e),prompt_sha256=signature(subtitle_prompt(row,e['text'])),
                input_sha256=signature(actual['input_ids'].tolist()),base_input_sha256=base_hash,
                input_tokens=int(actual['input_ids'].shape[1]),visual=visual,native_input_equal=True,
                new_answer_calls=calls,generated_tokens=generated,inference_s=seconds,preprocess_s=prep,
                control_s=control_s,peak_allocated_gib=peak))
            append_line(out,record); ctx['units'][idx]=record; bar.update(1)
            bar.set_postfix(index=idx,tokens=e['text_tokens'],added=len(e['added_cue_ids']))
            print(f'落盘 {idx}: {record["pred"] or "INVALID"}, 字幕{e["text_tokens"]}token，补{len(e["added_cue_ids"])}句',flush=True)
            del plain,actual


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['init','preflight','run'])
    p.add_argument('--stage',choices=STAGES,default='smoke'); p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0: p.error('磁盘预算无效')
    ctx=context(args,initialize=args.action=='init')
    if args.action=='init':
        print('CPU连续字幕准备完成，未抽帧或推理。',ctx['signature']); return
    preflight(args,ctx)
    if args.action=='run': run(args,ctx)


if __name__=='__main__': main()
