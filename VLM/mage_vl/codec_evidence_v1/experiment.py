"""CLI: CPU preparation now; GPU and downloads only via explicit user commands."""
import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import protocol as p


def checkpoint(path, signature, **expected):
    if not Path(path).exists():
        return None
    x = p.unseal(p.read(path))
    p.require(x['signature'] == signature, '断点协议不符：' + str(path))
    p.require(all(x.get(k) == v for k, v in expected.items()), '断点身份不符：' + str(path))
    return x


def cpu_gate(m):
    x = checkpoint(p.OUT / 'audits/cpu_mixed.json', m['signature'], passed=True)
    p.require(x is not None, '请先完成audit-cpu')


def probe_gate(m):
    cpu_gate(m)
    x = checkpoint(p.OUT / 'audits/gpu_probe.json', m['signature'], passed=True)
    p.require(x is not None, '请先由用户执行probe-gpu')
    return x


def parse_answer(text):
    # Historical practical_common.parse_answer semantics.
    x = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', text.upper())
    return x.group(1) if x else ''


def runtime(m):
    import torch
    import transformers
    expected = m['protocol']['runtime']
    actual = dict(torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(0), dtype='torch.bfloat16')
    p.require(actual == expected, 'Mage环境/硬件不同，停止核查：' + str(actual))
    return actual


def load_mage(m):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    rt = runtime(m)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(str(p.MODEL), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 0}).eval()
    module = sys.modules[type(model).__module__]
    p.require(p.sha(module.__file__) == p.sha(p.MODEL / 'modeling_mage_vl.py'), '实际加载模型源码不符')
    return model, rt


def generate(model, proc, values, max_tokens=32):
    import torch
    gpu = {k: v.to(device=model.device, dtype=model.dtype) if k == 'pixel_values' else v.to(model.device)
           for k, v in values.items()}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**gpu, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    raw = proc.tokenizer.decode(output[0, values['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    count = int(output.shape[1] - values['input_ids'].shape[1])
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del gpu, output
    return dict(raw_text=raw, pred=parse_answer(raw), generated_tokens=count, inference_s=elapsed, peak_gib=peak)


def probe(m):
    cpu_gate(m)
    if checkpoint(p.OUT / 'audits/gpu_probe.json', m['signature'], passed=True):
        print('GPU探测已完成，跳过模型。'); return
    p.gpu_guard()
    from mixed_inputs import Engine, frames
    from tqdm import tqdm
    rows = p.scope_rows(m, 'pilot60')
    controls = [next(r for r in rows if r['duration'] == d) for d in ('medium', 'long')]
    worst = max(m['rows'], key=lambda r: m['references'][r['index']]['input_tokens'])
    controls = list({r['index']: r for r in controls + [worst]}.values())
    engine = Engine(m)
    model, rt = load_mage(m)
    records = []
    for r in tqdm(controls, desc='Mage联合输入GPU探测', unit='题', dynamic_ncols=True):
        v = m['videos'][r['video_id']]
        old = p.baseline(m, r)
        for arm in ('A_control', 'B_probe'):
            path = p.OUT / 'probe_units' / (r['index'] + '_' + arm + '.json')
            saved = checkpoint(path, m['signature'], index=r['index'], arm=arm)
            if saved:
                records.append(saved); continue
            if arm == 'A_control':
                x, audit = engine.make(r)
                result = generate(model, engine.proc, x)
                p.require(result['pred'] == old['pred'], '纯Codec基线答案未复现，停止，不运行后续实验')
            else:
                ids = p.uniform(v['candidate_ids'])
                with frames(v, ids) as (images, rgb):
                    x, audit = engine.make(r, images, ids)
                result = generate(model, engine.proc, x)
                audit['rgb_sha256'] = rgb
            saved = p.seal(dict(signature=m['signature'], index=r['index'], arm=arm,
                input_audit=audit, **result))
            p.atomic(path, saved); records.append(p.unseal(saved))
            print(r['index'], arm, result['raw_text'], flush=True)
    # One-image joint description, only an interface diagnostic, not benchmark QA.
    r = controls[0]; v = m['videos'][r['video_id']]
    desc_path = p.OUT / 'probe_units/joint_description.json'
    description = checkpoint(desc_path, m['signature'])
    if description is None:
        fid = p.uniform(v['candidate_ids'])[0]
        with frames(v, [fid]) as (images, _):
            x, _, _ = engine.call(r, images, [fid], answer_text='Describe the video scene and the supplementary full frame. Mention both, briefly.')
        description = dict(signature=m['signature'], **generate(model, engine.proc, x, max_tokens=96))
        p.atomic(desc_path, p.seal(description))
    report = p.seal(dict(signature=m['signature'], passed=True, runtime=rt,
        control_indices=[r['index'] for r in controls], new_QA_calls=len(records)+1,
        max_input_tokens=max(x['input_audit']['input_tokens'] for x in records),
        peak_gib=max(x['peak_gib'] for x in records), joint_description=description['raw_text'],
        note='Structural audit + control parity + valid mixed generation. Description is for human review, not a proof of semantic grounding; no accuracy claim.'))
    p.atomic(p.OUT / 'audits/gpu_probe.json', report)
    print('探测通过；联合描述：', description['raw_text'])


def download_blip():
    """Explicit user action, one PyTorch checkpoint, no TensorFlow duplicate."""
    from huggingface_hub import HfApi, snapshot_download
    p.disk_guard()
    p.require(os.environ.get('CODEC_EVIDENCE_DOWNLOAD_CONFIRMED') == '1', '下载必须由用户显式设置CODEC_EVIDENCE_DOWNLOAD_CONFIRMED=1')
    p.BLIP.mkdir(parents=True, exist_ok=True)
    plan_path = p.BLIP / 'codec_evidence_download.json'
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    if plan_path.exists():
        plan = p.unseal(p.read(plan_path))
        p.require(plan['repo_id'] == 'Salesforce/blip-itm-large-coco', '下载目标身份不符')
    else:
        revision = HfApi(endpoint=endpoint).model_info('Salesforce/blip-itm-large-coco').sha
        plan = dict(repo_id='Salesforce/blip-itm-large-coco', revision=revision)
        p.atomic(plan_path, p.seal(plan))
    names = ['pytorch_model.bin', 'config.json', 'preprocessor_config.json', 'special_tokens_map.json',
             'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'README.md']
    snapshot_download(plan['repo_id'], revision=plan['revision'], local_dir=str(p.BLIP),
                      endpoint=endpoint, allow_patterns=names, max_workers=2)
    required = [x for x in names if x != 'README.md']
    p.require(all((p.BLIP / n).is_file() for n in required), 'BLIP文件缺失，原断点保留，可重跑下载')
    from tqdm import tqdm
    hashes = {n: p.sha(p.BLIP / n) for n in tqdm(required, desc='固定BLIP文件身份', unit='文件', dynamic_ncols=True)}
    p.atomic(p.BLIP / 'codec_evidence_ready.json', p.seal(dict(**plan,
        files_sha256=hashes, files_identity={n:p.identity(p.BLIP/n) for n in required})))
    print('BLIP已固定，仅PyTorch权重；目录：', p.BLIP)


def scorer_identity():
    ready = p.BLIP / 'codec_evidence_ready.json'
    p.require(ready.is_file(), '缺BLIP固定版本，先由用户运行download-blip')
    x = p.unseal(p.read(ready))
    for n, ident in x['files_identity'].items():
        p.require(p.identity(p.BLIP/n) == ident, 'BLIP文件身份改变：' + n)
    return x


def read_score(m, row, scorer=None):
    path = p.OUT / 'scores' / (row['index'] + '.json')
    x = checkpoint(path, m['signature'], index=row['index'], query_sha256=p.digest(p.query(row)))
    p.require(x is not None, '先完成BLIP打分：' + row['index'])
    v = m['videos'][row['video_id']]
    p.require(x['candidate_ids'] == v['candidate_ids'], '分数候选池不同')
    p.require(len(x['scores']) == len(x['rgb_sha256']) == len(v['candidate_ids']), '分数断点长度不符')
    p.require(all(math.isfinite(s) for s in x['scores']), '非法分数')
    if scorer is not None:
        p.require(x['scorer_sha256'] == p.digest(scorer), '分数来自不同BLIP模型')
    return x


def read_unit(m, row, arm, score_record):
    x = checkpoint(p.OUT/'units'/(row['index']+'_'+arm+'.json'), m['signature'], index=row['index'], arm=arm)
    if x is None:
        return None
    p.require(x['pred'] == parse_answer(x['raw_text']) and x['input_audit']['passed'], '答案/输入审计断点异常')
    if arm == 'A':
        old = p.baseline(m,row)
        p.require(x['raw_text'] == old['raw_text'] and x['input_audit']['tensor_sha256'] == old['tensor_sha256']
                  and x['new_calls'] == 0 and x['selected_ids'] == [], '旧基线复用身份错误')
    else:
        v = m['videos'][row['video_id']]
        selected = p.uniform(v['candidate_ids']) if arm == 'B' else p.semantic(v['candidate_ids'],score_record['scores'],v['total_frames'])
        p.require(x['selected_ids'] == selected and x['score_record_sha256'] == p.digest(score_record), '问答断点来源分数/选帧不同')
        p.require(x['rgb_sha256'] == [score_record['rgb_sha256'][v['candidate_ids'].index(i)] for i in selected], '问答断点RGB身份不同')
        p.require(x['input_audit']['base_tensor_sha256'] == m['references'][row['index']]['tensor_sha256'], '问答断点旧Codec身份不同')
    return x


def score(m, scope):
    probe_gate(m)
    ident = scorer_identity()
    rows = p.scope_rows(m, scope)
    if all((p.OUT/'scores'/(r['index']+'.json')).exists() for r in rows):
        for r in rows: read_score(m, r, ident)
        print('所选打分完整，跳过模型。'); return
    p.gpu_guard()
    import torch
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    from tqdm import tqdm
    from mixed_inputs import frames
    proc = BlipProcessor.from_pretrained(str(p.BLIP), local_files_only=True)
    model = BlipForImageTextRetrieval.from_pretrained(str(p.BLIP), local_files_only=True).to('cuda:0').eval()
    limit = min(model.config.text_config.max_position_embeddings, proc.tokenizer.model_max_length)
    lengths = {r['index']:len(proc.tokenizer.encode(p.query(r), add_special_tokens=True)) for r in m['rows']}
    p.require(max(lengths.values()) <= limit, '完整查询超过BLIP长度；停止不截断，请先统一修订文本协议')
    p.atomic(p.OUT/'scorer_protocol.json', p.seal(dict(signature=m['signature'], scorer=ident,
        dtype='torch.float32', batch_size=8, query_token_lengths=lengths, query_limit=limit,
        score='softmax(itm_score.float(),dim=-1)[:,1]', text_truncation=False)))
    groups = {}
    for r in rows: groups.setdefault(r['video_id'], []).append(r)
    total = sum(math.ceil(len(m['videos'][v]['candidate_ids'])/8) for v in groups)
    with tqdm(total=total, desc='BLIP打分（视频批断点）', unit='批', dynamic_ncols=True) as bar:
        for vid, group in groups.items():
            p.disk_guard()
            if all((p.OUT/'scores'/(r['index']+'.json')).exists() for r in group):
                for r in group: read_score(m,r,ident)
                bar.update(math.ceil(len(m['videos'][vid]['candidate_ids'])/8)); continue
            v = m['videos'][vid]; ids = v['candidate_ids']; chunks=[]
            for offset in range(0, len(ids), 8):
                batch_ids = ids[offset:offset+8]
                path = p.OUT/'score_chunks'/vid/(str(offset)+'.json')
                old = checkpoint(path,m['signature'],candidate_ids=batch_ids,scorer_sha256=p.digest(ident),
                                 query_sha256={r['index']:p.digest(p.query(r)) for r in group})
                if old is None:
                    start=time.perf_counter()
                    with frames(v,batch_ids) as (images,rgb):
                        result={}
                        for r in group:
                            inputs=proc(images=images,text=[p.query(r)]*len(images),padding=True,
                                        truncation=False,return_tensors='pt').to('cuda:0')
                            p.require(inputs['input_ids'].shape[1] <= limit,'BLIP输入超限')
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            begin=time.perf_counter()
                            with torch.inference_mode(): output=model(**inputs)
                            torch.cuda.synchronize()
                            values=output.itm_score.float().softmax(dim=-1)[:,1].cpu().tolist()
                            p.require(len(values)==len(batch_ids) and all(math.isfinite(x) for x in values),'ITM结果非法')
                            result[r['index']]=dict(scores=values,gpu_s=time.perf_counter()-begin,
                                peak_gib=torch.cuda.max_memory_allocated()/1024**3)
                            del inputs,output
                    old=dict(signature=m['signature'],candidate_ids=batch_ids,rgb_sha256=rgb,
                        scorer_sha256=p.digest(ident),query_sha256={r['index']:p.digest(p.query(r)) for r in group},
                        rows=result,wall_s=time.perf_counter()-start)
                    p.atomic(path,p.seal(old))
                chunks.append(old);bar.update(1);bar.set_postfix(video=vid)
            for r in group:
                scores=[s for c in chunks for s in c['rows'][r['index']]['scores']]
                record=dict(signature=m['signature'],index=r['index'],video_id=vid,
                    candidate_ids=ids,scores=scores,rgb_sha256=[h for c in chunks for h in c['rgb_sha256']],
                    query_sha256=p.digest(p.query(r)),scorer_sha256=p.digest(ident),
                    scoring_gpu_s=sum(c['rows'][r['index']]['gpu_s'] for c in chunks),
                    video_shared_scoring_wall_s=sum(c['wall_s'] for c in chunks),
                    selected_ids=p.semantic(ids,scores,v['total_frames']))
                p.atomic(p.OUT/'scores'/(r['index']+'.json'),p.seal(record))
    print('BLIP分数完成；没有生成新问答。')


def run(m, scope):
    probe_gate(m)
    ident = scorer_identity()
    from mixed_inputs import Engine, frames
    from tqdm import tqdm
    rows=p.scope_rows(m,scope)
    scores={r['index']:read_score(m,r,ident) for r in rows}
    for r in rows:
        for arm in ('A','B','C'):
            read_unit(m,r,arm,scores[r['index']])
    expected=[p.OUT/'units'/(r['index']+'_'+arm+'.json') for r in rows for arm in ('A','B','C')]
    if all(f.exists() for f in expected):
        print('所选答案完整，跳过模型；可运行summarize。');return
    p.gpu_guard()
    engine=Engine(m);model,rt=load_mage(m)
    with tqdm(total=len(expected),initial=sum(f.exists() for f in expected),desc='Mage A/B/C '+scope,unit='答案',dynamic_ncols=True) as bar:
        for r in rows:
            idx=r['index'];v=m['videos'][r['video_id']];s=scores[idx]
            if all((p.OUT/'units'/(idx+'_'+a+'.json')).exists() for a in ('A','B','C')): continue
            p.disk_guard(); base,base_audit=engine.make(r)
            old=p.baseline(m,r)
            a_path=p.OUT/'units'/(idx+'_A.json')
            if not a_path.exists():
                p.atomic(a_path,p.seal(dict(signature=m['signature'],index=idx,video_id=r['video_id'],arm='A',
                    origin='reuse_old_exact_input',raw_text=old['raw_text'],pred=old['pred'],new_calls=0,
                    old_source=m['references'][idx],input_audit=base_audit,selected_ids=[],rgb_sha256=[],
                    inference_s=0.,peak_gib=0.,generated_tokens=None)))
                bar.update(1)
            selections={'B':p.uniform(v['candidate_ids']),'C':p.semantic(v['candidate_ids'],s['scores'],v['total_frames'])}
            p.require(selections['C']==s['selected_ids'],'语义选帧断点规则漂移')
            prepared={}; completed={}
            for arm in ('B','C'):
                path=p.OUT/'units'/(idx+'_'+arm+'.json')
                saved=checkpoint(path,m['signature'],index=idx,arm=arm,
                    selected_ids=selections[arm],score_record_sha256=p.digest(s))
                if saved:
                    completed[arm]=saved;continue
                ids=selections[arm]
                with frames(v,ids) as (images,rgb):
                    p.require(rgb==[s['rgb_sha256'][v['candidate_ids'].index(i)] for i in ids],'评分与推理来源RGB不同')
                    x,audit=engine.make(r,images,ids)
                prepared[arm]=(x,audit)
                other='B' if arm=='C' else 'C'
                # Both arms use identical geometry; check against prior/current arm.
                ref_audit=completed[other]['input_audit'] if other in completed else (prepared[other][1] if other in prepared else None)
                if ref_audit:
                    p.require(audit['image_grids']==ref_audit['image_grids'] and audit['extra_visual_tokens']==ref_audit['extra_visual_tokens'],'B/C视觉预算不同')
                origin='generated';calls=1
                if other in completed and completed[other]['input_audit']['tensor_sha256']==audit['tensor_sha256']:
                    prev=completed[other]
                    result={k:prev[k] for k in ('raw_text','pred')}
                    result.update(inference_s=0.,peak_gib=0.,generated_tokens=None)
                    origin='reuse_same_question_exact_tensor';calls=0
                else:
                    result=generate(model,engine.proc,x)
                record=dict(signature=m['signature'],index=idx,video_id=r['video_id'],arm=arm,
                    origin=origin,new_calls=calls,selected_ids=ids,rgb_sha256=rgb,
                    score_record_sha256=p.digest(s),input_audit=audit,runtime=rt,**result)
                if calls==0: record['reuse_arm']=other
                p.atomic(path,p.seal(record));completed[arm]=record;bar.update(1)
                bar.set_postfix(index=idx,arm=arm,pred=result['pred'] or 'INVALID')
                del x
    print('当前范围A/B/C完成；执行summarize查看配对效果。')


def summarize(m,scope):
    import numpy as np
    rows=p.scope_rows(m,scope);units={};missing=[]
    for r in rows:
        score_record=read_score(m,r)
        for a in ('A','B','C'):
            x=read_unit(m,r,a,score_record)
            if x is None:missing.append((r['index'],a))
            else:units[(r['index'],a)]=x
    p.require(not missing,'范围尚未完整，保留断点，请续跑；缺失'+str(len(missing))+'单元')
    # GT is accessed only here, after immutable selection and prediction files exist.
    tsv=p.PROJECT/'VLM/datasets/VideoMME/Video-MME.tsv'
    labels={}
    with tsv.open(encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f,delimiter='\t'):
            p.require(x['index'] not in labels,'GT索引重复')
            labels[x['index']]=x['answer'].strip().upper()
    correct={a:[int(units[(r['index'],a)]['pred']==labels[r['index']]) for r in rows] for a in ('A','B','C')}
    vids=sorted({r['video_id'] for r in rows});rng=np.random.default_rng(20261109)
    paired={}
    for aa,bb in [('B','A'),('C','A'),('C','B')]:
        diff=np.asarray(correct[aa])-np.asarray(correct[bb])
        groups=[np.asarray([i for i,r in enumerate(rows) if r['video_id']==v]) for v in vids]
        sums=np.asarray([diff[g].sum() for g in groups]);sizes=np.asarray([len(g) for g in groups])
        pick=rng.integers(0,len(vids),size=(4000,len(vids)))
        boot=100*sums[pick].sum(axis=1)/sizes[pick].sum(axis=1)
        paired[aa+'-'+bb]=dict(rescue=int((diff==1).sum()),harm=int((diff==-1).sum()),net=int(diff.sum()),
            delta_pp=float(diff.mean()*100),video_cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    report=dict(signature=m['signature'],scope=scope,N=len(rows),videos=len(vids),
        metrics={a:dict(correct=sum(correct[a]),accuracy_pct=100*sum(correct[a])/len(rows),
            invalid=sum(units[(r['index'],a)]['pred'] not in 'ABCD' or not units[(r['index'],a)]['pred'] for r in rows)) for a in ('A','B','C')},
        paired=paired,new_QA_calls=sum(x['new_calls'] for x in units.values()),
        mage_generation_s=sum(x['inference_s'] for x in units.values()),peak_gib=max(x['peak_gib'] for x in units.values()),
        blip_gpu_s=sum(read_score(m,r)['scoring_gpu_s'] for r in rows),
        note='Seen development, not independent testing. Generation times exclude decode/load/audit; BLIP wall time shared per video, never multiply by 3. No rollout to remaining120 or full714.')
    path=p.OUT/'results'/(scope+'_'+p.digest(report)[:12]+'.json')
    p.atomic(path,p.seal(report));print(json.dumps(report,ensure_ascii=False,indent=2));print('报告：',path)


def status(m):
    counts={}
    for scope in ('pilot60','remaining120'):
        rows=p.scope_rows(m,scope)
        counts[scope]=dict(questions=len(rows),score_ready=sum((p.OUT/'scores'/(r['index']+'.json')).exists() for r in rows),
            units={a:sum((p.OUT/'units'/(r['index']+'_'+a+'.json')).exists() for r in rows) for a in ('A','B','C')})
    print(json.dumps(dict(branch=p.BRANCH,signature=m['signature'],disk=p.disk_guard(),
        cpu_audit=(p.OUT/'audits/cpu_mixed.json').exists(),gpu_probe=(p.OUT/'audits/gpu_probe.json').exists(),
        blip_ready=(p.BLIP/'codec_evidence_ready.json').exists(),scopes=counts),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','audit-cpu','status','download-blip','probe-gpu','score-gpu','run-gpu','summarize'])
    parser.add_argument('--scope',choices=['pilot60','remaining120'],default='pilot60')
    args=parser.parse_args()
    p.branch_guard();p.disk_guard()
    # Set before importing transformers; no shared remote-code cache mutation.
    os.environ['HF_MODULES_CACHE']=str(p.OUT/'hf_modules')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    if args.action!='download-blip':
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.action in ('init','audit-cpu','status','summarize','download-blip'):
        os.environ['CUDA_VISIBLE_DEVICES']=''
    with p.operation_lock():
        if args.action=='init':p.init();return
        if args.action=='download-blip':download_blip();return
        m=p.context()
        if args.action=='audit-cpu':
            from mixed_inputs import audit_cpu
            audit_cpu(m)
        elif args.action=='probe-gpu':probe(m)
        elif args.action=='score-gpu':score(m,args.scope)
        elif args.action=='run-gpu':run(m,args.scope)
        elif args.action=='summarize':summarize(m,args.scope)
        elif args.action=='status':status(m)


if __name__=='__main__':
    main()
