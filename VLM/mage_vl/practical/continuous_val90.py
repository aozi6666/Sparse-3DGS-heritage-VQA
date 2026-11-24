#!/usr/bin/env python3
"""冻结2048连续字幕的val90补充验证；不开放test/full，不调用API。"""
import argparse
from copy import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

from practical_common import ROOT,MODEL,SLOT,LETTERS,signature,sha,write_once,read_lines,append_line,parse_answer
from practical_vote import disk_guard,CaptureTokenizer
from subtitle_common import CaptureVideoProcessor,make_alignment,sealed
from continuous_common import continuous_evidence
from subtitle_completion_common import key,query,context as completion_context,load_alignment
from subtitle_completion_run import equal_inputs
from subtitle_holdout import make_context as old_context,ARMS as OLD_ARMS,unit_key
from practical_analyze import comparison

RESULTS=ROOT/'15_continuous_val90'
ARMS=('no_sub','aligned','continuous')


def old_source(ctx,idx,arm):
    if arm=='continuous': return None
    previous=ctx['old']; old_arm=OLD_ARMS[ARMS.index(arm)]
    u=previous['units'][unit_key(idx,old_arm)]
    return dict(raw_text=u['raw_text'],pred=u['pred'],input_sha256=u['input_sha256'],
        prompt_sha256=u['prompt_sha256'],visual=u['visual'],signature=previous['signature'],
        record_sha256=u['record_sha256'],path=str(previous['store']/'units.jsonl'))


def validate_record(ctx,row,e,record,existing):
    u=dict(record); digest=u.pop('record_sha256',None); idx=row['index']; arm=u['arm']; k=key(idx,arm)
    if (digest!=signature(u) or u['signature']!=ctx['signature'] or u['index']!=idx
            or u['video_id']!=row['video_id'] or arm not in ARMS or k in existing):
        raise ValueError('val90单元损坏/重复/身份不符')
    if (u['prompt_sha256']!=signature(query(row,arm,e)) or u['evidence_sha256']!=signature(e)
            or u['pred']!=parse_answer(u['raw_text']) or u['pred'] not in ['',*LETTERS]
            or u['checks']['native_input_equal'] is not True): raise ValueError('输入或答案不符')
    expected_tokens=0 if arm=='no_sub' else e['aligned']['text_tokens'] if arm=='aligned' else e['continuous']['text_tokens']
    base=ctx['old']['units'][unit_key(idx,OLD_ARMS[0])]
    if (u['subtitle_tokens']!=expected_tokens or u['visual']!=base['visual']
            or type(u['input_tokens']) is not int or u['input_tokens']<=u['visual']['n_tok']
            or not isinstance(u['input_sha256'],str) or len(u['input_sha256'])!=64):
        raise ValueError('字幕/视觉布局/输入长度不符')
    if any(not math.isfinite(u[n]) or u[n]<0 for n in ['inference_s','preprocess_s','peak_allocated_gib']):
        raise ValueError('成本非法')
    source=old_source(ctx,idx,arm)
    if source is not None:
        proof=dict(source_signature=source['signature'],source_record_sha256=source['record_sha256'],
            source_path=source['path'],input_sha256_equal=True,visual_equal=True,tensor_equal=True)
        if (u['origin']!='reused_old' or u['reuse_proof']!=proof or u['new_answer_calls']!=0
                or u['generated_tokens'] is not None or any(u[n]!=source[n] for n in ['raw_text','pred','input_sha256','prompt_sha256','visual'])):
            raise ValueError('旧基线只能验证输入后复用，禁止重新生成')
    elif query(row,arm,e)==query(row,'no_sub',e):
        b=existing.get(key(idx,'no_sub'))
        if (b is None or u['origin']!='reused_current_no_sub' or u['new_answer_calls']!=0
                or u['generated_tokens'] is not None
                or u['reuse_proof']!=dict(no_sub_record_sha256=b['record_sha256'],tensor_equal=True)
                or any(u[n]!=b[n] for n in ['raw_text','pred','input_sha256','input_tokens'])):
            raise ValueError('缺字幕必须回退同题无字幕输入/回答')
    elif (u['origin']!='generated' or u['new_answer_calls']!=1 or u['reuse_proof'] is not None
            or type(u['generated_tokens']) is not int or not 1<=u['generated_tokens']<=32):
        raise ValueError('连续字幕生成来源不符')
    if u['origin']!='generated' and (u['inference_s']!=0 or u['peak_allocated_gib']!=0):
        raise ValueError('复用不能记录新生成开销')
    if arm!='no_sub':
        b=existing.get(key(idx,'no_sub'))
        if b is None or b['visual']!=u['visual']: raise ValueError('缺少同题基线')
    return k


def context(args,initialize=False):
    disk_guard(ROOT,args.min_free,args.budget)
    # 冻结选择和short219源码身份继续核验；不读取test答案或按题调规则。
    chosen=completion_context(SimpleNamespace(scope='short',min_free=args.min_free,budget=args.budget))
    if chosen['tokens']!=2048 or not chosen['expected'].issubset(chosen['units']):
        raise ValueError('必须先完成冻结2048的short219')
    a=copy(args); a.stage='val'; old=old_context(a)
    if not old['expected'].issubset(old['units']): raise ValueError('旧val90基线不完整')
    summary=old['stage_dir']/'summary.json'; s=json.loads(summary.read_text())
    if not s['audit']['passed'] or s['N']!=90: raise ValueError('旧val审计失败')
    evidence={}
    for row in old['rows']:
        v=row['video_id']; original=old['evidence_loader'](v)['evidence']
        evidence[v]=dict(aligned=original,continuous=continuous_evidence(original,old['cues'][v],old['count_tokens'],2048))
    protocol=dict(version='continuous_val90_fixed2048_v1',tokens=2048,arms=list(ARMS),
        frozen_development_protocol=chosen['protocol'],old_protocol=old['protocol'],old_signature=old['signature'],
        old_summary_sha256=sha(summary),old_journal_sha256=sha(old['store']/'units.jsonl'),
        indices=[r['index'] for r in old['rows']],evidence_sha256={v:signature(e) for v,e in evidence.items()},
        source_sha256=sha(Path(__file__)),policy='Previously diagnosed val90; supplementary evaluation,not independent test. Reuse180 old answers only after input identity verification;generate continuous arm only;noAgent/API/test/full.')
    candidate=RESULTS/'candidate_v1.json'
    if initialize: write_once(candidate,protocol)
    if not candidate.exists() or json.loads(candidate.read_text())!=protocol: raise ValueError('先init；源码/数据漂移不覆盖')
    sig=signature(protocol); store=ROOT/'cache/continuous_val90'/sig
    ctx=dict(scope='val',tokens=2048,signature=sig,protocol=protocol,old=old,
        parent=dict(store=old['store']),store=store,stage_dir=RESULTS/'val90_t2048',rows=old['rows'],
        arms=ARMS,evidence=evidence,sources=old['bundle'],cues=old['cues'],count_tokens=old['count_tokens'],
        units={},expected={key(r['index'],arm) for r in old['rows'] for arm in ARMS})
    for v,e in evidence.items():
        p=store/'alignments'/f'{v}.json'; obj=dict(signature=sig,evidence=e,evidence_sha256=signature(e))
        if initialize: write_once(p,obj)
        if not p.exists() or json.loads(p.read_text())!=obj: raise ValueError('证据清单漂移')
    lookup={r['index']:r for r in old['rows']}
    for u in read_lines(store/'units.jsonl'):
        if u['index'] not in lookup: raise ValueError('样本越界')
        r=lookup[u['index']]; k=validate_record(ctx,r,evidence[r['video_id']],u,ctx['units']); ctx['units'][k]=u
    return ctx


def preflight(args,ctx):
    disk_guard(ROOT,args.min_free,args.budget)
    for n in ['torch','transformers','tqdm','decord','tokenizers','numpy']:
        if importlib.util.find_spec(n) is None: raise ImportError(n)
    print(f'val90固定2048：{len(ctx["units"])}/270逻辑答案；180个旧基线只验证复用，最多90次连续字幕生成。',flush=True)


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1': raise RuntimeError('由用户设置PRACTICAL_STAGE_CONFIRMED=1')
    with (ROOT/'cache/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); ctx=context(args)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],store=str(ctx['store']),arms=list(ARMS)))
        if ctx['expected'].issubset(ctx['units']): print('全部完成，跳过模型加载；请summarize。'); return
        run_gpu(args,ctx)


def summarize(ctx):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完成270单元，不读取标签')
    p=ROOT/'01_manifests/evaluation_labels.json'; lock=json.loads((p.parent/'dataset_lock.json').read_text())
    if sha(p)!=lock['labels_sha256']: raise ValueError('标签漂移')
    labels=json.loads(p.read_text()); rows=ctx['rows']; units=ctx['units']
    def group(items):
        if not items: return dict(N=0,methods={})
        preds={arm:{r['index']:units[key(r['index'],arm)]['pred'] for r in items} for arm in ARMS}
        metrics={arm:dict(correct=sum(preds[arm][r['index']]==labels[r['index']] for r in items),accuracy=sum(preds[arm][r['index']]==labels[r['index']] for r in items)/len(items)) for arm in ARMS}
        metrics['continuous']['vs']={arm:comparison(items,preds['continuous'],preds[arm],labels) for arm in ARMS[:2]}
        return dict(N=len(items),videos=len({r['video_id'] for r in items}),methods=metrics)
    invalid=sum(u['pred'] not in LETTERS or not u['pred'] for u in units.values())
    report=dict(signature=ctx['signature'],N=90,tokens=2048,interpretation=ctx['protocol']['policy'],
        audit=dict(passed=invalid==0,invalid=invalid,complete_units=len(units)),overall=group(rows),
        by_duration={d:group([r for r in rows if r['duration']==d]) for d in sorted({r['duration'] for r in rows})},
        subtitle_video_status={},budget_omitted_neighbor_videos=sum(bool(e['continuous']['omitted_neighbor_ids']) for e in ctx['evidence'].values()),cost={})
    for e in ctx['evidence'].values():
        status=e['aligned']['status']; report['subtitle_video_status'][status]=report['subtitle_video_status'].get(status,0)+1
    for arm in ARMS:
        items=[units[key(r['index'],arm)] for r in rows]
        report['cost'][arm]=dict(new_answer_calls=sum(u['new_answer_calls'] for u in items),reused_answers=sum(u['new_answer_calls']==0 for u in items),subtitle_tokens=sum(u['subtitle_tokens'] for u in items),input_tokens=sum(u['input_tokens'] for u in items),logical_visual_tokens=sum(u['visual']['n_tok'] for u in items),new_inference_s=sum(u['inference_s'] for u in items),peak_new_generation_gib=max(u['peak_allocated_gib'] for u in items))
    report['cost']['note']='New generation cost only; shared preprocessing includes native-input audits,not a deployment speed benchmark.'
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],predictions={a:units[key(r['index'],a)]['pred'] for a in ARMS},record_sha256={a:units[key(r['index'],a)]['record_sha256'] for a in ARMS}) for r in rows]
    for n,value in [('summary.json',report),('predictions.json',details)]: write_once(ctx['stage_dir']/n,value)
    write_once(ROOT/'07_archive/continuous_val90_fixed2048_v1.json',dict(summary=report,predictions=details,protocol=ctx['protocol'],journal_sha256=sha(ctx['store']/'units.jsonl'),source_sha256={n:sha(ctx['stage_dir']/n) for n in ['summary.json','predictions.json']}))
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('汇总路径:',ctx['stage_dir']/'summary.json')
    if not report['audit']['passed']: raise ValueError('非法答案保留为错误；审计失败，不自动重跑')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['init','preflight','run','summarize'])
    p.add_argument('--min-free',type=float,default=10); p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0: p.error('磁盘预算无效')
    ctx=context(args,initialize=args.action=='init')
    if args.action=='summarize': summarize(ctx)
    elif args.action=='run': run(args,ctx)
    else: preflight(args,ctx); print('CPU准备完成；未抽帧、加载模型、推理或调用API。')



# 原生预处理/生成沿用已审计的completion实现；仅限定val来源与基线复用。
def run_gpu(args, ctx):
    import random
    os.environ['HF_MODULES_CACHE'] = str(ROOT / 'cache/hf_modules')
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForCausalLM
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用')
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    for part in [processor.video_processor, processor.image_processor]:
        part.max_pixels = 150000
        part.min_pixels = 3136
    tokenizer = processor.tokenizer
    capture_video = CaptureVideoProcessor(processor.video_processor)
    processor.video_processor = capture_video
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True, torch_dtype='auto', device_map={'': 0}).eval()
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0), dtype=str(model.dtype))
    reference = json.loads((ctx['parent']['store'] / 'runtime.json').read_text())
    if runtime != reference:
        raise ValueError('运行环境/硬件/精度与2048开发阶段不同')
    write_once(ctx['store'] / 'runtime.json', runtime)
    cached = None

    def native(row, text):
        chat = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text}]}], tokenize=False, add_generation_prompt=True)
        return processor(text=[chat], videos=[row['video_path']], video_backend='frames', num_frames=32, max_pixels=150000, return_tensors='pt', padding=True)

    def inputs(hit, text):
        enc = tokenizer([hit['template'].replace(SLOT, text)], padding=True, return_tensors='pt')
        return dict(hit['visual'], input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])

    def prepare(row):
        nonlocal cached
        if cached is not None and cached['video_id'] == row['video_id']:
            return cached
        capture = CaptureTokenizer(tokenizer)
        processor.tokenizer = capture
        try:
            values = native(row, SLOT)
        finally:
            processor.tokenizer = tokenizer
        if capture.text is None or len(capture.text) != 1 or capture.text[0].count(SLOT) != 1:
            raise ValueError('模板插槽异常')
        if capture_video.timestamps is None or len(capture_video.timestamps) != 1:
            raise ValueError('真实采样秒数缺失')
        indices = [int(i) for i in values['patch_positions'][:, 0].unique_consecutive().tolist()]
        seconds = capture_video.timestamps[0]
        video = row['video_id']
        original = make_alignment(video, ctx['sources']['sources'][video], ctx['cues'][video], indices, list(seconds), ctx['count_tokens'])
        evidence = dict(aligned=original, continuous=continuous_evidence(original, ctx['cues'][video], ctx['count_tokens'], ctx['tokens']))
        for part in evidence.values():
            if len(tokenizer.encode(part['text'], add_special_tokens=False)) != part['text_tokens']: raise ValueError('HF/CPU字幕token计数不一致')
        if evidence != ctx['evidence'][video]:
            raise ValueError('实际帧或字幕未复现旧val来源')
        write_once(ctx['store'] / 'alignments' / f'{video}.json', dict(signature=ctx['signature'], evidence=evidence, evidence_sha256=signature(evidence)))
        hit = dict(video_id=video, template=capture.text[0], visual={k: v for k, v in values.items() if k not in ['input_ids', 'attention_mask']}, evidence=evidence)
        texts = {query(row, arm, evidence) for arm in ctx['arms']}
        for text in sorted(texts):
            expected = inputs(hit, text)
            direct = native(row, text)
            if not equal_inputs(expected, direct, torch) or capture_video.timestamps[0] != seconds:
                raise ValueError('插槽构造与原生实际输入/时间不一致')
            del expected, direct
        cached = hit
        del values
        return hit

    def generate(actual):
        gpu = {k: v.to(device=model.device, dtype=model.dtype) if k == 'pixel_values' else v.to(model.device) for k, v in actual.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**gpu, max_new_tokens=32, do_sample=False)
        raw = tokenizer.decode(output[0, actual['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        generated = int(output.shape[1] - actual['input_ids'].shape[1])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        del gpu, output
        return (raw, generated, elapsed, peak)
    with (ctx['store'] / 'units.jsonl').open('a', encoding='utf-8') as out, tqdm(total=len(ctx['expected']), initial=len(ctx['units']), desc='Mage 连续字幕 val90', unit='答案', dynamic_ncols=True) as bar:
        for row in ctx['rows']:
            idx = row['index']
            if all((key(idx, arm) in ctx['units'] for arm in ctx['arms'])):
                continue
            disk_guard(ROOT, args.min_free, args.budget)
            start = time.perf_counter()
            hit = prepare(row)
            e = hit['evidence']
            prep = time.perf_counter() - start
            for arm in ctx['arms']:
                k = key(idx, arm)
                if k in ctx['units']:
                    continue
                disk_guard(ROOT, args.min_free, args.budget)
                text = query(row, arm, e)
                actual = inputs(hit, text)
                input_hash = signature(actual['input_ids'].tolist())
                visual = dict(grid_thw=actual['image_grid_thw'].tolist(), pv_shape=list(actual['pixel_values'].shape), patch_time_range=[float(actual['patch_positions'][:, 0].min()), float(actual['patch_positions'][:, 0].max())], n_tok=sum((t * h * w for t, h, w in actual['image_grid_thw'].tolist())) // int(processor.spatial_merge_size) ** 2)
                if int((actual['input_ids'] == tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum()) != visual['n_tok']:
                    raise ValueError('视觉占位token不符')
                source = old_source(ctx, idx, arm)
                proof = None
                checks = dict(native_input_equal=True)
                reuse = source is not None and source['prompt_sha256'] == signature(text) and (source['input_sha256'] == input_hash) and (source['visual'] == visual)
                if arm in ARMS[:2] and (not reuse):
                    raise ValueError('旧val基线输入无法复现；禁止重新生成基线')
                if reuse:
                    source_text = inputs(hit, text)
                    if not equal_inputs(actual, source_text, torch):
                        raise ValueError('复用tensor不一致')
                    raw = source['raw_text']
                    generated = None
                    elapsed = peak = 0.0
                    calls = 0
                    origin = 'reused_old'
                    proof = dict(source_signature=source['signature'], source_record_sha256=source['record_sha256'], source_path=source['path'], input_sha256_equal=True, visual_equal=True, tensor_equal=True)
                    del source_text
                elif arm != 'no_sub' and query(row, arm, e) == query(row, 'no_sub', e):
                    base = ctx['units'][key(idx, 'no_sub')]
                    if input_hash != base['input_sha256'] or visual != base['visual']:
                        raise ValueError('缺字幕输入未逐字回退')
                    raw = base['raw_text']
                    generated = None
                    elapsed = peak = 0.0
                    calls = 0
                    origin = 'reused_current_no_sub'
                    proof = dict(no_sub_record_sha256=base['record_sha256'], tensor_equal=True)
                else:
                    raw, generated, elapsed, peak = generate(actual)
                    calls = 1
                    origin = 'generated'
                tokens = 0 if arm == 'no_sub' else e['aligned']['text_tokens'] if arm == 'aligned' else e['continuous']['text_tokens']
                record = sealed(dict(signature=ctx['signature'], index=idx, video_id=row['video_id'], arm=arm, origin=origin, raw_text=raw, pred=parse_answer(raw), generated_tokens=generated, new_answer_calls=calls, reuse_proof=proof, prompt_sha256=signature(text), input_sha256=input_hash, input_tokens=int(actual['input_ids'].shape[1]), evidence_sha256=signature(e), visual=visual, subtitle_tokens=tokens, checks=checks, preprocess_s=prep, inference_s=elapsed, peak_allocated_gib=peak))
                validate_record(ctx, row, e, record, ctx['units'])
                append_line(out, record)
                ctx['units'][k] = record
                prep = 0.0
                bar.update(1)
                bar.set_postfix(index=idx, arm=arm, new=calls)
                print(f"落盘 {k}: {record['pred'] or 'INVALID'}, {origin}, 字幕{tokens}token", flush=True)
                del actual
    print('val90完成；请summarize。不会进入Agent/test/full。', flush=True)

if __name__=='__main__': main()
