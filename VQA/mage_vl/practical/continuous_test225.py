#!/usr/bin/env python3
"""固定2048字幕配置的test225最终评测；不开放全量，不调参，不调用API。"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time
from practical_common import ROOT,MODEL,SLOT,LETTERS,signature,sha,write_once,read_lines,append_line,parse_answer,stage_rows
from practical_vote import disk_guard,CaptureTokenizer
from practical_subtitles import source_bundle
from subtitle_common import CaptureVideoProcessor,make_alignment,sealed
from continuous_common import continuous_evidence
from subtitle_completion_common import key,query,load_alignment
from subtitle_completion_run import equal_inputs
from continuous_val90 import context as validation_context
from practical_analyze import comparison

RESULTS=ROOT/'16_continuous_test225'
ARMS=('no_sub','aligned','continuous')


def check_partition(rows):
    if len(rows)!=225 or len({r['index'] for r in rows})!=225 or len({r['video_id'] for r in rows})!=75:
        raise ValueError('只能运行原test225/75视频')
    videos={r['video_id'] for r in rows}
    others=stage_rows(ROOT,'dev60')+stage_rows(ROOT,'val')+stage_rows(ROOT,'smoke')
    if videos & {r['video_id'] for r in others}: raise ValueError('测试与开发/验证视频重叠')
    if any(r['duration'] not in ['medium','long'] for r in rows): raise ValueError('原test225应只含中长视频')


def validate_record(ctx,row,e,record,existing):
    u=dict(record); digest=u.pop('record_sha256',None); arm=u['arm']; k=key(row['index'],arm)
    if (digest!=signature(u) or u['signature']!=ctx['signature'] or u['index']!=row['index']
            or u['video_id']!=row['video_id'] or arm not in ARMS or k in existing):
        raise ValueError('测试记录损坏/重复/身份不符')
    tokens=0 if arm=='no_sub' else e['aligned']['text_tokens'] if arm=='aligned' else e['continuous']['text_tokens']
    visual=u['visual']; original=e['aligned']
    if (u['pred']!=parse_answer(u['raw_text']) or u['pred'] not in ['',*LETTERS]
            or u['prompt_sha256']!=signature(query(row,arm,e)) or u['evidence_sha256']!=signature(e)
            or u['subtitle_tokens']!=tokens or u['checks']['native_input_equal'] is not True
            or len(visual['grid_thw'])!=32 or any(len(g)!=3 or g[0]!=1 or min(g)<=0 for g in visual['grid_thw'])
            or visual['patch_time_range']!=[float(original['frame_indices'][0]),float(original['frame_indices'][-1])]
            or type(visual['n_tok']) is not int or visual['n_tok']<=0
            or type(u['input_tokens']) is not int or u['input_tokens']<=visual['n_tok']
            or not isinstance(u['input_sha256'],str) or len(u['input_sha256'])!=64):
        raise ValueError('输入/输出/字幕/视觉布局不符')
    if any(not math.isfinite(u[n]) or u[n]<0 for n in ['inference_s','preprocess_s','peak_allocated_gib']): raise ValueError('成本非法')
    fallback=arm!='no_sub' and query(row,arm,e)==query(row,'no_sub',e)
    if fallback:
        base=existing.get(key(row['index'],'no_sub'))
        if (base is None or u['origin']!='reused_current_no_sub' or u['new_answer_calls']!=0
                or u['generated_tokens'] is not None or u['inference_s']!=0 or u['peak_allocated_gib']!=0
                or u['reuse_proof']!=dict(no_sub_record_sha256=base['record_sha256'],tensor_equal=True)
                or any(u[n]!=base[n] for n in ['raw_text','pred','input_sha256','input_tokens','visual'])):
            raise ValueError('缺字幕必须逐字回退同题当前无字幕回答')
    elif (u['origin']!='generated' or u['new_answer_calls']!=1 or u['reuse_proof'] is not None
            or type(u['generated_tokens']) is not int or not 1<=u['generated_tokens']<=32):
        raise ValueError('测试不复用历史预测；有证据时三路独立生成')
    if arm!='no_sub':
        b=existing.get(key(row['index'],'no_sub'))
        if b is None or visual!=b['visual']: raise ValueError('三路视觉不一致或缺基线')
    return k


def context(args,initialize=False):
    disk_guard(ROOT,args.min_free,args.budget)
    # 仅检查此前冻结协议与完整性；不按验证得分或测试答案选择输入。
    previous=validation_context(SimpleNamespace(min_free=args.min_free,budget=args.budget))
    if not previous['expected'].issubset(previous['units']): raise ValueError('先完成固定val90')
    summary=previous['stage_dir']/'summary.json'
    if not json.loads(summary.read_text())['audit']['passed']: raise ValueError('val90审计失败')
    rows=stage_rows(ROOT,'test'); check_partition(rows)
    sources,cues=source_bundle(rows)
    protocol=dict(version='final_test225_fixed2048_v1',tokens=2048,arms=list(ARMS),
        frozen_validation_protocol=previous['protocol'],validation_summary_sha256=sha(summary),
        validation_journal_sha256=sha(previous['store']/'units.jsonl'),
        source_sha256=sha(Path(__file__)),sources=sources,indices=[r['index'] for r in rows],
        input_manifest_sha256={n:sha(ROOT/'01_manifests'/n) for n in ['inputs.json','splits.json','dataset_lock.json']},
        generation=dict(frames=32,max_pixels=150000,min_pixels=3136,seed=0,max_new_tokens=32,do_sample=False),
        policy='One fixed final evaluation on existing test225; prior historical benchmark use must be disclosed. No test-driven tuning,thresholds,per-duration routing,Agent/API/full. Fresh paired inference; no historical test predictions loaded.')
    candidate=RESULTS/'candidate_v1.json'
    if initialize: write_once(candidate,protocol)
    if not candidate.exists() or json.loads(candidate.read_text())!=protocol: raise ValueError('先init；冻结后不覆盖协议漂移')
    sig=signature(protocol); store=ROOT/'cache/continuous_test225'/sig
    ctx=dict(signature=sig,protocol=protocol,tokens=2048,scope='test',parent=dict(store=previous['store']),
        store=store,stage_dir=RESULTS/'test225_t2048',rows=rows,arms=ARMS,sources=sources,cues=cues,
        count_tokens=previous['count_tokens'],units={},expected={key(r['index'],a) for r in rows for a in ARMS})
    lookup={r['index']:r for r in rows}; evidence={}
    for u in read_lines(store/'units.jsonl'):
        if u['index'] not in lookup: raise ValueError('测试单元越界')
        row=lookup[u['index']]; v=row['video_id']
        if v not in evidence: evidence[v]=load_alignment(ctx,v)
        k=validate_record(ctx,row,evidence[v],u,ctx['units']); ctx['units'][k]=u
    return ctx


def preflight(args,ctx):
    disk_guard(ROOT,args.min_free,args.budget)
    for n in ['torch','transformers','decord','tokenizers','tqdm','numpy']:
        if importlib.util.find_spec(n) is None: raise ImportError(n)
    print(f'test225固定2048：{len(ctx["units"])}/675逻辑答案；新三路配对，缺字幕回退，不加载历史测试预测。',flush=True)


def run(args,ctx):
    preflight(args,ctx)
    if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1': raise RuntimeError('用户设置PRACTICAL_STAGE_CONFIRMED=1后启动')
    with (ROOT/'cache/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctx=context(args)
        write_once(ctx['store']/'protocol.json',ctx['protocol'])
        write_once(ctx['stage_dir']/'run_manifest.json',dict(signature=ctx['signature'],store=str(ctx['store']),arms=list(ARMS)))
        if ctx['expected'].issubset(ctx['units']): print('全部完成，跳过模型加载；请summarize。');return
        run_gpu(args,ctx)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','preflight','run','summarize'])
    p.add_argument('--min-free',type=float,default=10);p.add_argument('--budget',type=float,default=1)
    args=p.parse_args()
    if args.min_free<1 or args.budget<=0:p.error('磁盘预算无效')
    ctx=context(args,initialize=args.action=='init')
    if args.action=='run':run(args,ctx)
    elif args.action=='summarize':summarize(ctx)
    else:preflight(args,ctx);print('CPU准备完成；未导入Torch/Transformers，未抽帧或推理。')


# 保留已审计的原生处理与生成路径；测试三路重新配对生成。
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
    with (ctx['store'] / 'units.jsonl').open('a', encoding='utf-8') as out, tqdm(total=len(ctx['expected']), initial=len(ctx['units']), desc='Mage 固定字幕 test225', unit='答案', dynamic_ncols=True) as bar:
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
                proof = None
                checks = dict(native_input_equal=True)
                if arm != 'no_sub' and query(row, arm, e) == query(row, 'no_sub', e):
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
    print('test225完成；请summarize。不自动扩跑或更改配置。', flush=True)

def summarize(ctx):
    if not ctx['expected'].issubset(ctx['units']): raise ValueError('未完成675单元，不读取标签')
    p=ROOT/'01_manifests/evaluation_labels.json'; lock=json.loads((p.parent/'dataset_lock.json').read_text())
    if sha(p)!=lock['labels_sha256']: raise ValueError('标签漂移')
    labels=json.loads(p.read_text()); rows=ctx['rows']; units=ctx['units']
    evidence={r['video_id']:load_alignment(ctx,r['video_id']) for r in rows}
    def group(items):
        if not items: return dict(N=0,methods={})
        preds={arm:{r['index']:units[key(r['index'],arm)]['pred'] for r in items} for arm in ARMS}
        metrics={arm:dict(correct=sum(preds[arm][r['index']]==labels[r['index']] for r in items),invalid=sum(preds[arm][r['index']] not in list(LETTERS) for r in items),accuracy=sum(preds[arm][r['index']]==labels[r['index']] for r in items)/len(items)) for arm in ARMS}
        metrics['continuous']['vs']={arm:comparison(items,preds['continuous'],preds[arm],labels) for arm in ARMS[:2]}
        return dict(N=len(items),videos=len({r['video_id'] for r in items}),methods=metrics)
    invalid=sum(u['pred'] not in LETTERS or not u['pred'] for u in units.values())
    report=dict(signature=ctx['signature'],N=225,tokens=2048,interpretation=ctx['protocol']['policy'],
        audit=dict(passed=invalid==0,invalid=invalid,complete_units=len(units)),overall=group(rows),
        by_duration={d:group([r for r in rows if r['duration']==d]) for d in sorted({r['duration'] for r in rows})},
        subtitle_video_status={},budget_omitted_neighbor_videos=sum(bool(e['continuous']['omitted_neighbor_ids']) for e in evidence.values()),cost={})
    report['by_task_type']={t:group([r for r in rows if r['task_type']==t]) for t in sorted({r['task_type'] for r in rows})}
    report['groups']={status:group([r for r in rows if evidence[r['video_id']]['aligned']['status']==status]) for status in ['used','missing','empty','no_hit']}
    for e in evidence.values():
        status=e['aligned']['status']; report['subtitle_video_status'][status]=report['subtitle_video_status'].get(status,0)+1
    for arm in ARMS:
        items=[units[key(r['index'],arm)] for r in rows]
        report['cost'][arm]=dict(new_answer_calls=sum(u['new_answer_calls'] for u in items),reused_answers=sum(u['new_answer_calls']==0 for u in items),subtitle_tokens=sum(u['subtitle_tokens'] for u in items),input_tokens=sum(u['input_tokens'] for u in items),logical_visual_tokens=sum(u['visual']['n_tok'] for u in items),new_inference_s=sum(u['inference_s'] for u in items),peak_new_generation_gib=max(u['peak_allocated_gib'] for u in items))
    report['cost']['shared_preprocess_s']=sum(u['preprocess_s'] for u in units.values())
    report['cost']['note']='New generation cost only; shared preprocessing includes native-input audits,not a deployment speed benchmark.'
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],predictions={a:units[key(r['index'],a)]['pred'] for a in ARMS},record_sha256={a:units[key(r['index'],a)]['record_sha256'] for a in ARMS}) for r in rows]
    for n,value in [('summary.json',report),('predictions.json',details)]: write_once(ctx['stage_dir']/n,value)
    write_once(ROOT/'07_archive/continuous_test225_fixed2048_v1.json',dict(summary=report,predictions=details,protocol=ctx['protocol'],journal_sha256=sha(ctx['store']/'units.jsonl'),source_sha256={n:sha(ctx['stage_dir']/n) for n in ['summary.json','predictions.json']}))
    print(json.dumps(report,ensure_ascii=False,indent=2)); print('汇总路径:',ctx['stage_dir']/'summary.json')
    if not report['audit']['passed']: raise ValueError('非法答案保留为错误；审计失败，不自动重跑')

if __name__=='__main__':main()
