#!/usr/bin/env python3
"""Frames-32 + 原生采样时刻命中的SRT。init/preflight为CPU；仅run会加载模型。"""
import argparse
from copy import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from practical_common import (ROOT, MODEL, SLOT, LETTERS, sha, signature, write_once,
    append_line, stage_rows, prompt, parse_answer)
from practical_vote import context as baseline_context, disk_guard, CaptureTokenizer
from subtitle_common import (BASE_SIG, TAG, RESULTS, STAGES, MAX_SUB_TOKENS,
    SUB_HEADER, subtitle_source, subtitle_prompt, make_alignment,
    CaptureVideoProcessor, sealed, load_units)


def native_token_counter():
    # 加载的是已有tokenizer的CPU词表，不加载模型。避免增加pysubs2等依赖。
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    return lambda text: len(tokenizer.encode(text,add_special_tokens=False).ids)


def source_bundle(rows):
    sources={}; parsed={}
    for video_id in sorted({r['video_id'] for r in rows}):
        sources[video_id],parsed[video_id]=subtitle_source(video_id)
    return dict(version='native_aligned_srt_v1',sources=sources,
        indices=[r['index'] for r in rows],encoding='utf-8-sig, no encoding guessing',
        alignment='native processor frame_seconds; start <= time < end; all overlapping nonempty cues',
        deduplication='same cue ID once; identical text at different cue times is retained',
        max_subtitle_tokens=MAX_SUB_TOKENS,overflow='stop; never silently truncate',
        prompt_added_text=SUB_HEADER,
        no_subtitle='reuse unchanged Frames32 result; no GT-based fallback'),parsed


def make_context(args, initialize=False):
    a=copy(args); a.frames=32; a.codec_vote=False; a.root=ROOT
    base=baseline_context(a)
    if base['signature']!=BASE_SIG:
        raise ValueError('原32帧协议/模型/数据发生变化，拒绝混用基线')
    # 仅开放已用开发池和原smoke，不接触新路线的val/test/full推理。
    allowed=stage_rows(ROOT,'smoke')+stage_rows(ROOT,'dev60')
    if len({r['index'] for r in allowed})!=len(allowed):
        raise ValueError('smoke/dev60划分重叠')
    for row in allowed:
        if any(f'{row["index"]}:{backend}:p0' not in base['units'] for backend in ['frames','codec']):
            raise ValueError('开发池原32帧基线不完整，不自动补跑')
    bundle,parsed=source_bundle(allowed)
    manifest=RESULTS/'srt_manifest.json'
    if initialize:
        write_once(manifest,bundle)
    elif not manifest.exists():
        raise ValueError('请先运行 init（CPU字幕审计，不推理）')
    if json.loads(manifest.read_text())!=bundle:
        raise ValueError('字幕来源/协议漂移，停止，不覆盖旧清单')
    folder=Path(__file__).parent
    protocol=dict(version='practical_subtitles_v1',baseline_signature=BASE_SIG,
        baseline_protocol=base['protocol'],baseline_units_sha256=sha(base['store']/'units.jsonl'),
        srt_manifest_sha256=sha(manifest),subtitle_prompt_rule='Insert subtitle header and text before Select; else exact old prompt',
        frames=32,max_pixels=150000,min_pixels=3136,max_new_tokens=32,do_sample=False,seed=0,
        answer='old free-generation parser; no confidence selection, no option permutation',
        scope='aligned with-subtitle configuration; additional text modality, not a pure visual algorithm',
        scripts={n:sha(folder/n) for n in ['subtitle_common.py','practical_subtitles.py','subtitle_analyze.py']})
    sig=signature(protocol); store=ROOT/'cache/subtitles'/sig
    selected=stage_rows(ROOT,args.stage)
    count_tokens=native_token_counter()
    evidence_cache={}

    def evidence_loader(video_id):
        if video_id in evidence_cache:
            return evidence_cache[video_id]
        obj=json.loads((store/'alignments'/f'{video_id}.json').read_text())
        if obj['signature']!=sig:
            raise ValueError('字幕对齐文件签名不符')
        evidence=obj['evidence']
        if video_id not in bundle['sources']:
            raise ValueError('字幕对齐文件越界')
        expected=make_alignment(video_id,bundle['sources'][video_id],parsed[video_id],
            evidence['frame_indices'],evidence['frame_seconds'],count_tokens)
        if obj['evidence_sha256']!=signature(evidence) or evidence!=expected:
            raise ValueError('字幕对齐文件内容漂移/时间匹配不一致')
        evidence_cache[video_id]=evidence
        return evidence

    units=load_units(store/'units.jsonl',sig,allowed,base['units'],evidence_loader)
    return dict(signature=sig,protocol=protocol,store=store,stage_dir=RESULTS/args.stage/TAG,
        rows=selected,allowed=allowed,expected={r['index'] for r in selected},units=units,
        base=base,bundle=bundle,cues=parsed,evidence_loader=evidence_loader,count_tokens=count_tokens)


def preflight(args,ctx,gate=True):
    for name in ['torch','transformers','tqdm','tokenizers','numpy','decord','cv2']:
        if importlib.util.find_spec(name) is None:
            raise ImportError(f'环境缺少 {name}，不自动安装')
    if not gate:
        return
    previous=[] if args.stage=='smoke' else ['smoke']
    if args.stage=='dev60':
        previous.append('dev30')
    for stage in previous:
        summary=json.loads((RESULTS/stage/TAG/'summary.json').read_text())
        needed={r['index'] for r in stage_rows(ROOT,stage)}
        if (summary['signature']!=ctx['signature'] or summary['stage']!=stage
                or summary['N']!=len(needed) or not summary['audit']['passed']
                or not needed.issubset(ctx['units'])):
            raise ValueError(f'需先完成同配置 {stage} 并通过汇总审计')


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':
        raise RuntimeError('推理由用户设置 PRACTICAL_STAGE_CONFIRMED=1 后启动')
    (ROOT/'cache').mkdir(parents=True,exist_ok=True)
    with (ROOT/'cache/gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=make_context(args)
        ctx['store'].mkdir(parents=True,exist_ok=True)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],stage=args.stage,
            indices=[r['index'] for r in ctx['rows']],store=str(ctx['store']),protocol=ctx['protocol']))
        if ctx['expected'].issubset(ctx['units']):
            print('全部单元已完整落盘，跳过模型加载；请 summarize。',flush=True)
            return
        run_gpu(args,ctx)


def run_gpu(args,ctx):
    import random
    # 所有新动态源码缓存也放到指定实验根目录，不复制权重或视觉tensor。
    os.environ['HF_MODULES_CACHE']=str(ROOT/'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用')
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    processor.video_processor.max_pixels=150000; processor.video_processor.min_pixels=3136
    processor.image_processor.max_pixels=150000; processor.image_processor.min_pixels=3136
    tokenizer=processor.tokenizer
    capture_video=CaptureVideoProcessor(processor.video_processor)
    processor.video_processor=capture_video
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,
                local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype))
    write_once(ctx['store']/'runtime.json',runtime)
    print('RUNTIME',runtime,'signature',ctx['signature'],flush=True)
    cached=None

    def chat(question):
        return processor.apply_chat_template([{'role':'user','content':[
            {'type':'video'},{'type':'text','text':question}]}],tokenize=False,add_generation_prompt=True)

    def native(video,question):
        return processor(text=[chat(question)],videos=[video],video_backend='frames',
            num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)

    def prepare(row):
        nonlocal cached
        if cached and cached['video_id']==row['video_id']:
            return cached,0.0
        start=time.perf_counter()
        capture=CaptureTokenizer(tokenizer); processor.tokenizer=capture
        try:
            values=native(row['video_path'],SLOT)
        finally:
            processor.tokenizer=tokenizer
        if capture.text is None or len(capture.text)!=1 or capture.text[0].count(SLOT)!=1:
            raise ValueError('无法安全定位原生视频文本插槽')
        if capture_video.timestamps is None or len(capture_video.timestamps)!=1:
            raise ValueError('没有取得本次原生视频采样秒数')
        indices=[int(i) for i in values['patch_positions'][:,0].unique_consecutive().tolist()]
        evidence=make_alignment(row['video_id'],ctx['bundle']['sources'][row['video_id']],
            ctx['cues'][row['video_id']],indices,list(capture_video.timestamps[0]),ctx['count_tokens'])
        # 再与本次真正使用的HF tokenizer交叉核对，避免CPU词表计数和运行时不一致。
        if len(tokenizer.encode(evidence['text'],add_special_tokens=False))!=evidence['text_tokens']:
            raise ValueError('字幕token预算计数不一致')
        write_once(ctx['store']/'alignments'/f'{row["video_id"]}.json',dict(signature=ctx['signature'],
            evidence=evidence,evidence_sha256=signature(evidence)))
        visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']}
        cached=dict(video_id=row['video_id'],text=capture.text[0],visual=visual,evidence=evidence)
        return cached,time.perf_counter()-start

    def inputs_for(hit,question):
        encoding=tokenizer([hit['text'].replace(SLOT,question)],padding=True,return_tensors='pt')
        return dict(hit['visual'],input_ids=encoding['input_ids'],attention_mask=encoding['attention_mask'])

    def generate(inputs):
        gpu_inputs={k:(v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device))
                    for k,v in inputs.items()}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**gpu_inputs,max_new_tokens=32,do_sample=False)
        raw=tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        new_tokens=int(generated.shape[1]-inputs['input_ids'].shape[1])
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()/1024**3
        del generated,gpu_inputs
        return raw,parse_answer(raw),seconds,peak,new_tokens

    units=ctx['units']
    initial=len(ctx['expected'] & set(units))
    with (ctx['store']/'units.jsonl').open('a',encoding='utf-8') as out, \
         tqdm(total=len(ctx['rows']),initial=initial,desc=f'Mage 对齐字幕 {args.stage}',unit='题',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx=row['index']
            if idx in units:
                continue
            disk_guard(ROOT,args.min_free,args.budget)
            hit,prep_s=prepare(row); evidence=hit['evidence']
            old=ctx['base']['units'][f'{idx}:frames:p0']
            plain=inputs_for(hit,prompt(row,0))
            query=subtitle_prompt(row,evidence['text']); actual=inputs_for(hit,query)
            visual=dict(grid_thw=actual['image_grid_thw'].tolist(),pv_shape=list(actual['pixel_values'].shape),
                patch_time_range=[float(actual['patch_positions'][:,0].min()),float(actual['patch_positions'][:,0].max())],
                n_tok=sum(t*h*w for t,h,w in actual['image_grid_thw'].tolist())//int(processor.spatial_merge_size)**2)
            if any(visual[k]!=old[k] for k in visual):
                raise ValueError(f'新路线视觉布局未复现原Frames32: {idx}')
            base_hash=signature(plain['input_ids'].tolist())
            if base_hash!=old['prefix_sha256']:
                raise ValueError('原顺序无字幕输入签名漂移')
            image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
            if int((actual['input_ids']==image_id).sum())!=visual['n_tok']:
                raise ValueError('视觉占位token数改变')
            checks=dict(visual_unchanged=True,base_prefix_sha256=base_hash,
                        native_input_equal=None,no_sub_answer_equal=None)
            control_s=0.0; control_calls=0
            if args.stage=='smoke':
                # 字幕可以纠错；只要求无字幕控制复现，不要求带字幕答案仍等于旧答案。
                direct=native(row['video_path'],query)
                same=set(direct)==set(actual) and all(torch.equal(direct[k],actual[k]) for k in direct)
                if not same or capture_video.timestamps[0]!=evidence['frame_seconds']:
                    raise ValueError('插槽复用路径与原生字幕输入/采样时间不一致')
                checks['native_input_equal']=True
                del direct
                _,control_pred,control_s,_,_=generate(plain); control_calls=1
                if control_pred!=old['semantic_pred']:
                    raise ValueError('无字幕控制答案未复现原基线')
                checks['no_sub_answer_equal']=True
            if evidence['status']=='used':
                raw,pred,answer_s,peak,generated_tokens=generate(actual)
                origin='generated_with_subtitle'; calls=1
            else:
                # 只按输入可用性回退，不读取GT，不先看“答错了没有”。
                raw=old['raw_text']; pred=old['semantic_pred']; answer_s=peak=0.0
                generated_tokens=None; origin='reused_frames32_'+evidence['status']; calls=0
            record=sealed(dict(signature=ctx['signature'],index=idx,video_id=row['video_id'],
                origin_stage=args.stage,origin=origin,pred=pred,raw_text=raw,subtitle_status=evidence['status'],
                evidence_sha256=signature(evidence),prompt_sha256=signature(query),
                input_sha256=signature(actual['input_ids'].tolist()),input_tokens=int(actual['input_ids'].shape[1]),
                subtitle_tokens=evidence['text_tokens'],generated_tokens=generated_tokens,visual=visual,checks=checks,
                cost=dict(new_answer_calls=calls,extra_smoke_control_calls=control_calls,answer_s=answer_s,
                    control_s=control_s,preprocess_s=prep_s,peak_allocated_gib=peak)))
            append_line(out,record); units[idx]=record
            bar.set_postfix(index=idx,sub=evidence['status'],completed=f'{len(ctx["expected"] & set(units))}/{len(ctx["expected"])}')
            bar.update(1)
            print(f'落盘 {idx}: 答案={pred or "INVALID"}, 字幕={evidence["status"]}, '
                  f'命中{len(evidence["selected_cues"])}段/{evidence["text_tokens"]}token, 新生成={calls}次',flush=True)
            del actual,plain
    print('完成；请运行 summarize。',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','preflight','run'])
    p.add_argument('--stage',choices=STAGES,default='smoke')
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:
        p.error('保留至少1GiB空间，目录预算必须为正')
    ctx=make_context(args,initialize=args.action=='init')
    if args.action=='init':
        available=sum(v['status']=='available' for v in ctx['bundle']['sources'].values())
        print('CPU字幕清单已锁定:',RESULTS/'srt_manifest.json',
              f'可用字幕视频{available}/{len(ctx["bundle"]["sources"])}；未抽帧或推理')
        return
    preflight(args,ctx)
    print(f'Frames32+对齐字幕 {args.stage}: {len(ctx["rows"])}题，'
          f'{len(ctx["expected"] & set(ctx["units"]))}/{len(ctx["expected"])}单元已完成；{ctx["signature"]}',flush=True)
    if args.action=='run':
        run(args,ctx)
    else:
        print('只读预检通过；未导入Torch、未加载模型、未抽帧或推理')


if __name__=='__main__':
    main()
