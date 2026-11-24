#!/usr/bin/env python3
"""独立 val90 的 Frames32 无/有字幕配对；只有 run 加载 GPU，不读取答案标签。"""
import argparse
from copy import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import time

from practical_common import (ROOT, MODEL, SLOT, LETTERS, signature, sha,
    write_once, append_line, read_lines, stage_rows, prompt, parse_answer)
from practical_vote import disk_guard, CaptureTokenizer
from practical_subtitles import make_context as development_context, native_token_counter, source_bundle
from subtitle_common import (TAG, MAX_SUB_TOKENS, subtitle_prompt, make_alignment,
    CaptureVideoProcessor, sealed)

RESULTS = ROOT/'10_subtitle_holdout'
ARMS = ('frames32_no_sub', 'frames32_aligned_sub')
DEV_SIG = 'd0595146a0a8ace17ca37b906c2030fbabeefface296824dd8206a3305484d4e'


def unit_key(index, arm):
    return f'{index}:{arm}'


def validate_partition(rows, development, testing):
    """只按输入清单查划分；不根据留出答案挑样本。"""
    if len(rows)!=90 or len({r['index'] for r in rows})!=90 or len({r['video_id'] for r in rows})!=30:
        raise ValueError('val 必须为原划分的90题/30视频')
    groups=[{r['video_id'] for r in items} for items in [rows,development,testing]]
    if any(groups[i] & groups[j] for i,j in [(0,1),(0,2),(1,2)]):
        raise ValueError('开发/验证/测试视频重叠')


def alignment_loader(store, sig, sources, cues, count_tokens):
    cached={}
    def load(video_id):
        if video_id not in cached:
            obj=json.loads((store/'alignments'/f'{video_id}.json').read_text())
            evidence=obj['evidence']
            expected=make_alignment(video_id,sources[video_id],cues[video_id],
                evidence['frame_indices'],evidence['frame_seconds'],count_tokens)
            if (obj['signature']!=sig or evidence!=expected or obj['evidence_sha256']!=signature(evidence)
                    or obj['native_inputs_equal']!={'plain':True,'subtitle':True}):
                raise ValueError('采样、字幕对齐或原生输入审计发生漂移')
            cached[video_id]=obj
        return cached[video_id]
    return load


def validate_record(record, sig, row, alignment, existing):
    """每个答案单独恢复；字幕单元必须引用同题已落盘无字幕单元。"""
    item=dict(record); digest=item.pop('record_sha256',None)
    if digest!=signature(item):
        raise ValueError('单元记录校验失败')
    arm=item['arm']; idx=row['index']; evidence=alignment['evidence']
    key=unit_key(idx,arm)
    if arm not in ARMS or key in existing or item['signature']!=sig or item['index']!=idx or item['video_id']!=row['video_id']:
        raise ValueError('重复单元、配置或样本身份不符')
    query=prompt(row,0) if arm==ARMS[0] else subtitle_prompt(row,evidence['text'])
    if (item['origin_stage']!='val' or item['pred'] not in ['',*LETTERS]
            or item['pred']!=parse_answer(item['raw_text']) or item['prompt_sha256']!=signature(query)
            or item['evidence_sha256']!=signature(evidence) or item['subtitle_status']!=evidence['status']
            or item['subtitle_tokens']!=(evidence['text_tokens'] if arm==ARMS[1] else 0)
            or item['native_inputs_equal']!=alignment['native_inputs_equal']
            or item['visual_shared'] is not True):
        raise ValueError('输出、提示词、证据或审计字段不一致')
    visual=item['visual']
    if (len(visual['grid_thw'])!=32 or any(len(g)!=3 or g[0]!=1 or min(g)<=0 for g in visual['grid_thw'])
            or type(visual['n_tok']) is not int or visual['n_tok']<=0
            or visual['patch_time_range']!=[float(evidence['frame_indices'][0]),float(evidence['frame_indices'][-1])]
            or type(item['input_tokens']) is not int or item['input_tokens']<=visual['n_tok']
            or not all(isinstance(item[k],str) and len(item[k])==64 for k in ['input_sha256','base_input_sha256'])):
        raise ValueError('视觉布局、时间位置或输入token不合法')
    cost=item['cost']
    if any(not math.isfinite(cost[k]) or cost[k]<0 for k in ['inference_s','preprocess_s','peak_allocated_gib']):
        raise ValueError('成本字段非法')
    is_fallback=arm==ARMS[1] and evidence['status']!='used'
    if type(cost['new_answer_calls']) is not int or cost['new_answer_calls']!=int(not is_fallback):
        raise ValueError('新增调用计数不符')
    if is_fallback:
        if item['origin']!='reused_current_no_sub' or item['generated_tokens'] is not None:
            raise ValueError('缺字幕来源不符')
        if cost['inference_s']!=0 or cost['peak_allocated_gib']!=0:
            raise ValueError('复用单元不能记一次新生成成本')
    elif (item['origin']!='generated' or type(item['generated_tokens']) is not int
            or not 1<=item['generated_tokens']<=32):
        raise ValueError('生成来源或长度不符')
    if arm==ARMS[0]:
        if item['base_input_sha256']!=item['input_sha256'] or item['baseline_record_sha256'] is not None:
            raise ValueError('无字幕输入身份错误')
    else:
        base=existing.get(unit_key(idx,ARMS[0]))
        if base is None:
            raise ValueError('字幕单元缺少先落盘的配对无字幕基线')
        if (item['baseline_record_sha256']!=base['record_sha256'] or visual!=base['visual']
                or item['base_input_sha256']!=base['input_sha256']):
            raise ValueError('两路视觉或基线来源不一致')
        if is_fallback and any(item[k]!=base[k] for k in ['pred','raw_text','input_sha256','input_tokens']):
            raise ValueError('缺字幕没有逐字回退同题原始输入/回答')
    return key


def load_units(path, sig, rows, evidence_loader):
    lookup={r['index']:r for r in rows}; found={}
    for item in read_lines(path):
        if item.get('index') not in lookup:
            raise ValueError('验证单元越界，不混入开发/测试数据')
        row=lookup[item['index']]
        key=validate_record(item,sig,row,evidence_loader(row['video_id']),found)
        found[key]=item
    return found


def complete(ctx):
    return ctx['expected'].issubset(ctx['units'])


def make_context(args, initialize=False):
    # 原入口只读检查旧数据/模型/脚本身份；不编辑旧脚本，不改变开发签名。
    previous=copy(args); previous.stage='dev60'
    dev=development_context(previous)
    if dev['signature']!=DEV_SIG:
        raise ValueError('已完成开发协议发生变化，不进入留出评测')
    dev_files={s:{n:dev['stage_dir'].parent.parent/s/TAG/n for n in ['summary.json','predictions.json']}
               for s in ['smoke','dev30','dev60']}
    # stage_dir 是 09_subtitles/dev60/TAG；只读取已完成结果，决不读取val标签。
    for stage,files in dev_files.items():
        summary=json.loads(files['summary.json'].read_text())
        expected=stage_rows(ROOT,stage)
        if summary['signature']!=DEV_SIG or summary['N']!=len(expected) or not summary['audit']['passed']:
            raise ValueError(f'{stage} 开发/接口审计未通过')
    if not {r['index'] for r in dev['rows']}.issubset(dev['units']):
        raise ValueError('开发字幕原始单元不完整')
    rows=stage_rows(ROOT,'val')
    validate_partition(rows,stage_rows(ROOT,'dev60')+stage_rows(ROOT,'smoke'),stage_rows(ROOT,'test'))
    bundle,cues=source_bundle(rows)
    folder=Path(__file__).parent
    protocol=dict(version='aligned_subtitle_holdout_val_v1',candidate='Frames32 + native aligned subtitles',
        development_signature=DEV_SIG,development_protocol=dev['protocol'],
        development_artifacts={s:{n:sha(p) for n,p in files.items()} for s,files in dev_files.items()},
        development_runtime=json.loads((dev['store']/'runtime.json').read_text()),
        stage='val',indices=[r['index'] for r in rows],sources=bundle,
        frames=32,max_pixels=150000,min_pixels=3136,max_subtitle_tokens=MAX_SUB_TOKENS,
        seed=0,max_new_tokens=32,do_sample=False,arms=list(ARMS),
        baseline='generate current no-sub result first; same visual tensors; reuse only for input-based fallback',
        test_policy='val only; no test/full entry; no per-question or duration-based answer routing',
        native_audit='both actual prompts checked tensor-for-tensor once per pending video; no extra generation',
        scripts={n:sha(folder/n) for n in ['subtitle_holdout.py','subtitle_holdout_analyze.py']})
    candidate=RESULTS/'candidate_val_v1.json'
    if initialize:
        # 保存开发成果快照和来源校验；原结果一字不改。
        archive=dict(version='subtitle_dev60_archive_v1',signature=DEV_SIG,
            artifacts=protocol['development_artifacts'],
            summary=json.loads(dev_files['dev60']['summary.json'].read_text()),
            predictions=json.loads(dev_files['dev60']['predictions.json'].read_text()))
        write_once(ROOT/'07_archive/subtitles_dev60_v1.json',archive)
        write_once(candidate,protocol)
    if not candidate.exists() or json.loads(candidate.read_text())!=protocol:
        raise ValueError('先 init；候选登记后禁止改配置/源码/字幕，冲突不覆盖')
    sig=signature(protocol); store=ROOT/'cache/subtitle_holdout'/sig
    loader=alignment_loader(store,sig,bundle['sources'],cues,native_token_counter())
    units=load_units(store/'units.jsonl',sig,rows,loader)
    return dict(signature=sig,protocol=protocol,rows=rows,store=store,units=units,
        expected={unit_key(r['index'],arm) for r in rows for arm in ARMS},
        bundle=bundle,cues=cues,evidence_loader=loader,count_tokens=native_token_counter(),
        stage_dir=RESULTS/'val90'/TAG)


def preflight(args,ctx):
    for name in ['torch','transformers','tqdm','tokenizers','numpy','decord','cv2']:
        if importlib.util.find_spec(name) is None:
            raise ImportError(f'缺少 {name}，不自动安装')
    disk_guard(ROOT,args.min_free,args.budget)
    print(f'val90: {len(ctx["rows"])}题/30视频，已落盘 {len(ctx["units"])}/180 个配对答案；'
          f'签名 {ctx["signature"]}',flush=True)


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':
        raise RuntimeError('由用户设置 PRACTICAL_STAGE_CONFIRMED=1 后启动')
    with (ROOT/'cache/gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=make_context(args)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],stage='val',
            indices=[r['index'] for r in ctx['rows']],store=str(ctx['store']),arms=list(ARMS)))
        if complete(ctx):
            print('全部配对答案已落盘，跳过模型加载；请 summarize。',flush=True)
            return
        run_gpu(args,ctx)


def run_gpu(args,ctx):
    import random
    os.environ['HF_MODULES_CACHE']=str(ROOT/'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForCausalLM
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for part in [processor.video_processor,processor.image_processor]:
        part.max_pixels=150000; part.min_pixels=3136
    tokenizer=processor.tokenizer
    capture_video=CaptureVideoProcessor(processor.video_processor)
    processor.video_processor=capture_video
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,
        local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype))
    if runtime!=ctx['protocol']['development_runtime']:
        raise ValueError('GPU/环境/精度与开发阶段不同，停止以免混用')
    write_once(ctx['store']/'runtime.json',runtime)
    print('RUNTIME',runtime,flush=True)
    cached=None

    def native(video,query):
        text=processor.apply_chat_template([{'role':'user','content':[
            {'type':'video'},{'type':'text','text':query}]}],tokenize=False,add_generation_prompt=True)
        return processor(text=[text],videos=[video],video_backend='frames',num_frames=32,
            max_pixels=150000,return_tensors='pt',padding=True)

    def inputs_for(hit,query):
        text=tokenizer([hit['text'].replace(SLOT,query)],padding=True,return_tensors='pt')
        return dict(hit['visual'],input_ids=text['input_ids'],attention_mask=text['attention_mask'])

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
            raise ValueError('原生视频模板插槽异常')
        if capture_video.timestamps is None or len(capture_video.timestamps)!=1:
            raise ValueError('原生采样秒数缺失')
        indices=[int(i) for i in values['patch_positions'][:,0].unique_consecutive().tolist()]
        evidence=make_alignment(row['video_id'],ctx['bundle']['sources'][row['video_id']],
            ctx['cues'][row['video_id']],indices,list(capture_video.timestamps[0]),ctx['count_tokens'])
        if len(tokenizer.encode(evidence['text'],add_special_tokens=False))!=evidence['text_tokens']:
            raise ValueError('HF 与 CPU 字幕token计数不一致')
        hit=dict(video_id=row['video_id'],text=capture.text[0],evidence=evidence,
            visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']})
        # 两个实际Prompt各与原生处理逐tensor比较，只增加预处理、不额外生成答案。
        queries={'plain':prompt(row,0),'subtitle':subtitle_prompt(row,evidence['text'])}
        for name,query in queries.items():
            expected=inputs_for(hit,query); direct=native(row['video_path'],query)
            if (set(direct)!=set(expected) or not all(torch.equal(direct[k],expected[k]) for k in direct)
                    or capture_video.timestamps[0]!=evidence['frame_seconds']):
                raise ValueError(f'{name} 实际输入未复现原生处理/采样时刻')
            del direct,expected
        obj=dict(signature=ctx['signature'],evidence=evidence,evidence_sha256=signature(evidence),
            native_inputs_equal={'plain':True,'subtitle':True})
        write_once(ctx['store']/'alignments'/f'{row["video_id"]}.json',obj)
        cached=hit
        return hit,time.perf_counter()-start

    def generate(inputs):
        actual={k:(v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device))
                for k,v in inputs.items()}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**actual,max_new_tokens=32,do_sample=False)
        length=int(output.shape[1]-inputs['input_ids'].shape[1])
        raw=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        torch.cuda.synchronize()
        cost=dict(inference_s=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        del output,actual
        return raw,length,cost

    units=ctx['units']
    with (ctx['store']/'units.jsonl').open('a',encoding='utf-8') as out, \
            tqdm(total=180,initial=len(units),desc='Mage 字幕 val90 配对',unit='答案',dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx=row['index']
            if all(unit_key(idx,arm) in units for arm in ARMS):
                continue
            disk_guard(ROOT,args.min_free,args.budget)
            hit,prep_s=prepare(row); evidence=hit['evidence']
            plain=inputs_for(hit,prompt(row,0)); base_hash=signature(plain['input_ids'].tolist())
            for arm in ARMS:
                key=unit_key(idx,arm)
                if key in units:
                    # 中断在一题的两次生成之间时，重新解码也必须复现已保存的无字幕输入。
                    if units[key]['input_sha256']!=base_hash:
                        raise ValueError('恢复时无字幕输入漂移')
                    continue
                disk_guard(ROOT,args.min_free,args.budget)
                query=prompt(row,0) if arm==ARMS[0] else subtitle_prompt(row,evidence['text'])
                inputs=plain if arm==ARMS[0] else inputs_for(hit,query)
                visual=dict(grid_thw=inputs['image_grid_thw'].tolist(),pv_shape=list(inputs['pixel_values'].shape),
                    patch_time_range=[float(inputs['patch_positions'][:,0].min()),float(inputs['patch_positions'][:,0].max())],
                    n_tok=sum(t*h*w for t,h,w in inputs['image_grid_thw'].tolist())//int(processor.spatial_merge_size)**2)
                image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
                if int((inputs['input_ids']==image_id).sum())!=visual['n_tok']:
                    raise ValueError('视觉占位token数与网格不符')
                base=units.get(unit_key(idx,ARMS[0]))
                if arm==ARMS[1] and evidence['status']!='used':
                    raw=base['raw_text']; generated=None
                    cost=dict(inference_s=0.0,peak_allocated_gib=0.0,new_answer_calls=0)
                    origin='reused_current_no_sub'
                else:
                    raw,generated,cost=generate(inputs); cost['new_answer_calls']=1; origin='generated'
                cost['preprocess_s']=prep_s; prep_s=0.0
                record=sealed(dict(signature=ctx['signature'],index=idx,video_id=row['video_id'],arm=arm,
                    origin_stage='val',origin=origin,pred=parse_answer(raw),raw_text=raw,generated_tokens=generated,
                    subtitle_status=evidence['status'],evidence_sha256=signature(evidence),
                    subtitle_tokens=evidence['text_tokens'] if arm==ARMS[1] else 0,
                    prompt_sha256=signature(query),input_sha256=signature(inputs['input_ids'].tolist()),
                    input_tokens=int(inputs['input_ids'].shape[1]),base_input_sha256=base_hash,
                    baseline_record_sha256=base['record_sha256'] if arm==ARMS[1] else None,
                    visual=visual,visual_shared=True,native_inputs_equal={'plain':True,'subtitle':True},cost=cost))
                alignment=ctx['evidence_loader'](row['video_id'])
                validate_record(record,ctx['signature'],row,alignment,units)
                append_line(out,record); units[key]=record
                bar.set_postfix(index=idx,arm=arm,sub=evidence['status']); bar.update(1)
                print(f'落盘 {key}: 答案={record["pred"] or "INVALID"}, '
                    f'视觉token={visual["n_tok"]}, 字幕token={record["subtitle_tokens"]}, '
                    f'新生成={cost["new_answer_calls"]}次',flush=True)
                del inputs
            del plain
    print('val90配对完成；请 summarize 后检查结果。不会自动进入测试或全量。',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','preflight','run'])
    p.add_argument('--stage',choices=['val'],default='val')
    p.add_argument('--min-free',type=float,default=10)
    p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:
        p.error('保留至少1GiB，实验目录预算为正')
    ctx=make_context(args,initialize=args.action=='init')
    preflight(args,ctx)
    if args.action=='run':
        run(args,ctx)
    else:
        print('CPU准备/预检通过；未导入Torch、未抽帧、未加载模型或推理。',flush=True)


if __name__=='__main__':
    main()
