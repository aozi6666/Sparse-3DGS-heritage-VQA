"""字幕收尾消融：只删标记或邻句；不改视觉、预算规则、QA或模型。

旧结果只读。init不解码、不加载模型；GPU仅由用户显式run启动。
每题每路原子JSON断点，完整断点跳过，损坏/来源漂移停止且保留现场。
"""
import argparse
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'practical'))

from practical_common import ROOT, PROJECT, MODEL, SLOT, signature, sha, prompt, parse_answer
from subtitle_common import subtitle_prompt
from continuous_common import timestamp

OUT = ROOT/'20_subtitle_ablation_v1/frozen_r1'
ARMS = ('no_time', 'no_neighbors')
BASES = ('no_sub', 'aligned', 'continuous')
BRANCH = 'feat/air-visual-evidence'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atomic_json(path, obj):
    """先fsync临时文件，再原子建立最终名；不覆盖既有结果。"""
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text()) == obj, f'文件冲突，保留不覆盖:{path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.own_', delete=False) as f:
        temp = Path(f.name)
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        require(json.loads(path.read_text()) == obj, f'并发文件冲突:{path}')
    finally:
        temp.unlink()  # 仅清理由本函数创建的临时文件，不清理任何历史缓存。


def sealed(obj):
    return dict(obj, record_sha256=signature(obj))


def unseal(obj):
    v = dict(obj); digest = v.pop('record_sha256', None)
    require(digest == signature(v), '断点记录SHA不符，停止，不自动修复')
    return v


def guard():
    require(shutil.disk_usage('/root/autodl-tmp').free >= 10*1024**3, '磁盘不足10GiB，停止，不清理')
    size = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file() and not f.is_symlink()) if OUT.exists() else 0
    require(size <= 256*1024**2, '本轮目录超过256MiB，停止审查，不清理')
    return dict(free_gib=round(shutil.disk_usage('/root/autodl-tmp').free/1024**3, 3), experiment_mib=round(size/1024**2, 3))


def ablate(aligned, full, count):
    """直接编辑冻结文本块，不重新检索或把省下的token换成其他字幕。"""
    blocks = full['blocks']
    original = {c['cue_id']:c for c in aligned['selected_cues']}
    all_ids = [i for b in blocks for i in b['cue_ids']]
    require(len(all_ids) == len(set(all_ids)), '字幕块cue重复')
    require(set(all_ids) == set(original) | set(full['added_cue_ids']), '补句身份不符')
    # 不用正则替换正文，避免误删字幕中恰好长得像时间标记的文字。
    no_time = '\n'.join(b['text'] for b in blocks)
    anchor_blocks = []
    for b in blocks:
        kept = [original[i] for i in b['cue_ids'] if i in original]
        if not kept:
            continue
        anchor_blocks.append(dict(id='S'+str(kept[0]['cue_id']), cue_ids=[c['cue_id'] for c in kept],
            start_ms=min(c['start_ms'] for c in kept), end_ms=max(c['end_ms'] for c in kept),
            text=' '.join(c['text'] for c in kept)))
    no_neighbors = '\n'.join(f'{b["id"]} [{timestamp(b["start_ms"])} --> {timestamp(b["end_ms"])}] {b["text"]}' for b in anchor_blocks)
    require({i for b in anchor_blocks for i in b['cue_ids']} == set(original), '删邻句时丢失原命中cue')
    result = {}
    for arm, text, ids in [('no_time', no_time, all_ids), ('no_neighbors', no_neighbors, [i for b in anchor_blocks for i in b['cue_ids']])]:
        n = count(text) if text else 0
        require(n <= 2048, '删减后文本仍超2048；停止，不截断或调整预算')
        result[arm] = dict(text=text, text_tokens=n, cue_ids=ids)
    return result


def old_contexts():
    """只读适配历史记录，不调用仍限制vlm分支的旧执行context。"""
    from practical_common import stage_rows
    rows={r['index']:r for r in json.loads((ROOT/'01_manifests/inputs.json').read_text())['rows']}
    paths=dict(short219='14_subtitle_completion/short219_t2048',
        dev180='11_continuous_subtitles/dev60/f32_px150000_neighbor1_time_t2048_v1',
        val90='15_continuous_val90/val90_t2048',test225='16_continuous_test225/test225_t2048')
    split_names=dict(short219='short_dev',dev180='dev60',val90='val',test225='test')
    archive=json.loads((ROOT/'17_subtitle_final_report/20261103_v1/source_manifest.json').read_text())
    result={}
    for stage,rel in paths.items():
        dest=ROOT/rel;manifest=json.loads((dest/'run_manifest.json').read_text());store=Path(manifest['store'])
        require(store.is_relative_to(ROOT/'cache'),'旧缓存路径越界')
        require(manifest['signature']==archive['stages'][stage]['signature'],'历史阶段签名不符')
        selected=stage_rows(ROOT,split_names[stage]);ids={r['index'] for r in selected}
        require(len(ids)==archive['stages'][stage]['N'],'历史阶段数量不符')
        units={}
        for line in (store/'units.jsonl').read_text().splitlines(keepends=True):
            require(line.endswith('\n'),'旧JSONL残行，保留')
            u=json.loads(line);unseal(u)
            require(u['signature']==manifest['signature'],'旧单元签名不符')
            k=u['index'] if stage=='dev180' else f'{u["index"]}:{u["arm"]}'
            require(k not in units,'旧单元重复');units[k]=u
        expected=ids if stage=='dev180' else {f'{idx}:{arm}' for idx in ids for arm in BASES}
        require(expected<=units.keys(),'旧阶段不完整')
        c=dict(rows=selected,store=store,stage_dir=dest,protocol=json.loads((store/'protocol.json').read_text()),units=units,expected=expected,evidence={},source_files=[])
        for video in sorted({r['video_id'] for r in selected}):
            fp=store/('evidence' if stage=='dev180' else 'alignments')/f'{video}.json'
            obj=json.loads(fp.read_text());e=obj['evidence']
            require(obj['signature']==manifest['signature'] and obj['evidence_sha256']==signature(e),'旧字幕证据身份不符')
            c['evidence'][video]=e
        if stage=='dev180':
            op=c['protocol']['old_protocol'];base_store=ROOT/'cache/ballots'/op['baseline_signature'];aligned_store=ROOT/'cache/subtitles'/c['protocol']['old_signature']
            require(sha(base_store/'units.jsonl')==op['baseline_units_sha256'],'原始无字幕journal改变')
            require(signature(op)==c['protocol']['old_signature'],'原对齐协议签名不符')
            au={}
            for line in (aligned_store/'units.jsonl').read_text().splitlines(keepends=True):
                require(line.endswith('\n'),'原对齐JSONL残行');u=json.loads(line);unseal(u)
                require(u['signature']==c['protocol']['old_signature'] and u['index'] not in au,'原对齐单元身份不符');au[u['index']]=u
            bu={}
            for line in (base_store/'units.jsonl').read_text().splitlines(keepends=True):
                require(line.endswith('\n'),'无字幕JSONL残行');u=json.loads(line)
                if u['backend']=='frames' and u['permutation_id']==0:
                    require(u['signature']==op['baseline_signature'],'无字幕来源签名不符')
                    k=f'{u["index"]}:frames:p0';require(k not in bu,'无字幕单元重复');bu[k]=u
            require(ids<=au.keys() and all(f'{i}:frames:p0' in bu for i in ids),'开发三路缺失')
            c['old']=dict(units=au,base=dict(units=bu))
            c['source_files']=[base_store/'units.jsonl',aligned_store/'units.jsonl']
        result[stage]=c
    return result


def source_record(c, stage, idx, arm):
    if stage == 'dev180':
        if arm == 'no_sub':
            u = c['old']['base']['units'][f'{idx}:frames:p0']
            return dict(pred=u['semantic_pred'], raw_text=u['raw_text'], input_sha256=u['prefix_sha256'],
                prompt_sha256=u['prompt_sha256'], visual={k:u[k] for k in ['grid_thw','pv_shape','patch_time_range','n_tok']},
                input_tokens=None, subtitle_tokens=0, source_record_sha256=signature(u), inference_s=u['inference_s'])
        u = c['old']['units'][idx] if arm == 'aligned' else c['units'][idx]
    else:
        u = c['units'][f'{idx}:{arm}']
    unseal(u)
    return dict(pred=u['pred'], raw_text=u['raw_text'], input_sha256=u['input_sha256'], prompt_sha256=u['prompt_sha256'],
        visual=u['visual'], input_tokens=u['input_tokens'], subtitle_tokens=u['subtitle_tokens'],
        source_record_sha256=u['record_sha256'], inference_s=u.get('inference_s', u.get('cost', {}).get('answer_s', 0)))


def build_bundle():
    """复用旧context校验链；不读取答案标签，不导入Torch/模型。"""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    cs = old_contexts()
    sources = {}; rows = []; evidence = {}; old = {}; roles = {}; runtime = None
    def pin(path):
        sources[str(path)] = sha(path)
    archive = ROOT/'17_subtitle_final_report/20261103_v1/source_manifest.json'
    pin(archive)
    archived = json.loads(archive.read_text())
    for rel, info in archived['sources'].items():
        path = PROJECT/rel
        require(sha(path) == info['sha256'], '历史成果来源漂移:'+str(path)); pin(path)
    for stage, c in cs.items():
        require(c['expected'].issubset(c['units']), stage+'历史结果不完整')
        roles[stage] = archived['stages'][stage]['role']
        for f in [c['store']/'runtime.json', c['stage_dir']/'run_manifest.json', c['store']/'protocol.json']:
            pin(f)
        for f in c['source_files']:
            pin(f)
        rt = json.loads((c['store']/'runtime.json').read_text())
        require(runtime is None or runtime == rt, '四阶段旧运行环境不同'); runtime = rt
        for row in c['rows']:
            row = dict(row, stage=stage, role=roles[stage])
            idx = row['index']; video = row['video_id']
            require(idx not in old, '问题重复')
            video_stat=Path(row['video_path']).stat()
            row['video_identity']=dict(bytes=video_stat.st_size,mtime_ns=video_stat.st_mtime_ns)
            rows.append(row)
            if video not in evidence:
                if stage == 'dev180':
                    fp = c['store']/'evidence'/f'{video}.json'; full = c['evidence'][video]; aligned = full['original']
                else:
                    fp = c['store']/'alignments'/f'{video}.json'; e = c['evidence'][video]; full=e['continuous']; aligned=e['aligned']
                pin(fp)
                require(full['budget'] == 2048, '不是冻结2048配置')
                if aligned['source']['status'] == 'available':
                    pin(Path(aligned['source']['path']))
                changed = ablate(aligned, full, count)
                evidence[video] = dict(aligned=aligned, continuous=full, variants=changed, source_path=str(fp), source_sha256=sha(fp))
            e = evidence[video]
            old[idx] = {a:source_record(c, stage, idx, a) for a in BASES}
            for arm, u in old[idx].items():
                text = '' if arm=='no_sub' else e[arm]['text']
                require(u['prompt_sha256']==signature(subtitle_prompt(row,text)), '旧提示身份不同')
                require(parse_answer(u['raw_text'])==u['pred'], '旧答案解析不一致')
    require(len(rows)==714 and len(evidence)==238, '本地714题/238视频覆盖不符')
    by_video = {}
    for row in rows:
        require(row['video_id'] not in by_video or by_video[row['video_id']]==row['stage'], '视频跨阶段')
        by_video[row['video_id']]=row['stage']
    # 旧模型配置逐文件哈希＋权重大小/mtime；不加载或复制9GB权重。
    ident = cs['dev180']['protocol']['old_protocol']['baseline_protocol']['model']
    for name, h in ident['files'].items():
        require(sha(MODEL/name)==h, '原模型配置变更:'+name); pin(MODEL/name)
    for name, stat in ident['weights'].items():
        st=(MODEL/name).stat(); require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==stat, '权重身份变化:'+name)
    for f in [ROOT/'01_manifests/inputs.json', ROOT/'01_manifests/splits.json', ROOT/'01_manifests/dataset_lock.json']:
        pin(f)
    for f in Path(__file__).parent.glob('*.py'):
        pin(f)
    for f in (PROJECT/'VLM/mage_vl/practical').glob('*.py'):
        pin(f)
    pin(Path(__file__).with_name('run_subtitle_ablation.sh'))
    smoke_video = next(r['video_id'] for r in rows if r['stage']=='dev180' and evidence[r['video_id']]['continuous']['added_cue_ids'])
    smoke = [r['index'] for r in rows if r['video_id']==smoke_video]
    return dict(rows=rows,evidence=evidence,old=old,sources=sources,runtime=runtime,model=ident,smoke=smoke,roles=roles)


def initialize():
    guard(); bundle=build_bundle()
    protocol=dict(version='subtitle_ablation_v1',arms=list(ARMS),baselines=list(BASES),smoke=bundle['smoke'],
        rule='no_time removes block ID and time intervals only, keeping exact full block text/order/cues. no_neighbors removes only added cues inside frozen full blocks, retaining block grouping and updating IDs/time bounds to remaining anchor cues. No budget refill.',
        caveat='Two component removals, not a strict 2x2 factorial. Original aligned renderer differs. Time removal includes block IDs; no_neighbors updates coverage timestamps. Total input lengths differ. All714 is posthoc coverage, not official benchmark or a new independent test.',
        generation=dict(frames=32,max_pixels=150000,min_pixels=3136,max_new_tokens=32,seed=0,do_sample=False),
        runtime=bundle['runtime'],model=bundle['model'],source_sha256=bundle['sources'],roles=bundle['roles'])
    sig=signature(protocol)
    atomic_json(OUT/'protocol.json',protocol)
    atomic_json(OUT/'bundle.json',sealed(dict(signature=sig,**{k:v for k,v in bundle.items() if k not in ['sources','model','runtime']})))
    plan=[]
    for r in bundle['rows']:
        idx=r['index']; e=bundle['evidence'][r['video_id']]
        for arm in ARMS:
            text=subtitle_prompt(r,e['variants'][arm]['text'])
            matches=[b for b in BASES if text==subtitle_prompt(r,'' if b=='no_sub' else e[b]['text'])]
            plan.append(dict(index=idx,stage=r['stage'],arm=arm,subtitle_tokens=e['variants'][arm]['text_tokens'],identical_old_prompt=matches))
    atomic_json(OUT/'planned_work.json',dict(signature=sig,total=1428,max_new_calls=sum(not p['identical_old_prompt'] for p in plan),identical_prompt_units=sum(bool(p['identical_old_prompt']) for p in plan),items=plan))
    print('已冻结714题字幕消融；CPU准备不解码/加载模型。',sig)
    print('计划最大新增生成数:',sum(not p['identical_old_prompt'] for p in plan))


def context(scope):
    guard()
    p=json.loads((OUT/'protocol.json').read_text()); sig=signature(p)
    for f,h in p['source_sha256'].items():
        require(sha(f)==h,'来源漂移，保留已有断点:'+f)
    for name, expected in p['model']['weights'].items():
        st=(MODEL/name).stat(); require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==expected,'权重身份变更')
    b=unseal(json.loads((OUT/'bundle.json').read_text())); require(b['signature']==sig,'bundle身份不符')
    selected=b['rows'] if scope=='full714' else [r for r in b['rows'] if r['index'] in p['smoke']]
    lookup={r['index']:r for r in b['rows']}; units={}
    for f in (OUT/'units').glob('*.json'):
        u=json.loads(f.read_text()); validate_unit(sig,b,lookup,u); key=(u['index'],u['arm'])
        require(key not in units,'重复断点');units[key]=u
    for (idx,arm), u in units.items():
        if u['origin']=='reused_exact_current_input':
            other=units.get((idx,u['reuse_proof']['arm']))
            require(other is not None and other['record_sha256']==u['reuse_proof']['record_sha256'] and other['raw_text']==u['raw_text'] and other['tensor_sha256']==u['tensor_sha256'], '同题复用来源不符')
    return dict(protocol=p,signature=sig,bundle=b,rows=selected,units=units)


def validate_unit(sig, b, rows, u):
    unseal(u); idx=u['index'];arm=u['arm']
    require(idx in rows and arm in ARMS and u['signature']==sig,'断点范围/协议不符')
    r=rows[idx];e=b['evidence'][r['video_id']]
    require(u['video_id']==r['video_id'] and u['stage']==r['stage'],'断点身份不符')
    require(u['prompt_sha256']==signature(subtitle_prompt(r,e['variants'][arm]['text'])),'消融提示漂移')
    require(u['pred']==parse_answer(u['raw_text']) and u['pred'] in ['',*'ABCD'],'答案解析漂移')
    require(u['source_audit']['passed'] is True and set(u['source_audit']['baselines'])==set(BASES),'缺少三路输入身份核查')
    for a, audit in u['source_audit']['baselines'].items():
        source=b['old'][idx][a]
        require(audit['input_sha256']==source['input_sha256'] and audit['source_record_sha256']==source['source_record_sha256'],'三路来源不符')
    require(u['visual']==b['old'][idx]['continuous']['visual'],'视觉布局发生变化')
    require(u['subtitle_tokens']==e['variants'][arm]['text_tokens'] and 0<=u['subtitle_tokens']<=2048,'字幕预算不符')
    require(u['origin'] in ['generated','reused_exact_old_input','reused_exact_current_input'],'未知来源')
    if u['origin']=='generated':
        require(u['new_calls']==1 and type(u['generated_tokens']) is int and 1<=u['generated_tokens']<=32,'生成计数不符')
    else:
        require(u['new_calls']==0 and u['inference_s']==0 and u['generated_tokens'] is None,'复用重复计费')
        require(u['reuse_proof']['tensor_equal'] is True,'复用缺少Tensor一致性')
        if u['origin']=='reused_exact_old_input':
            source=b['old'][idx][u['reuse_proof']['arm']]
            require(u['raw_text']==source['raw_text'] and u['input_sha256']==source['input_sha256'],'复用答案/输入不同')
    require(u['native_equal'] is True,'原生提示展开检查失败')
    for n in ['inference_s','peak_gib']:
        require(isinstance(u[n],(int,float)) and math.isfinite(u[n]) and u[n]>=0,'成本非法')


class ReplayVideo:
    """每视频仅解码一次，其余原生Prompt核对重用相同CPU像素，不持久化Tensor。"""
    OWN={'base','path','output','decode_s'}
    def __init__(self,base,path):
        for k,v in dict(base=base,path=path,output=None,decode_s=0.).items():object.__setattr__(self,k,v)
    def __getattr__(self,n):return getattr(self.base,n)
    def __setattr__(self,n,v):
        if n in self.OWN:object.__setattr__(self,n,v)
        else:setattr(self.base,n,v)
    def __call__(self,videos,**kwargs):
        require(videos==[self.path],'视频缓存来源越界')
        if self.output is None:
            start=time.perf_counter();self.output=self.base(videos=videos,**kwargs);self.decode_s=time.perf_counter()-start
        return dict(self.output)


def visual_info(values, sms):
    return dict(grid_thw=values['image_grid_thw'].tolist(),pv_shape=list(values['pixel_values'].shape),
        patch_time_range=[float(values['patch_positions'][:,0].min()),float(values['patch_positions'][:,0].max())],
        n_tok=sum(t*h*w for t,h,w in values['image_grid_thw'].tolist())//sms**2)


def tensor_digest(values):
    import hashlib
    h=hashlib.sha256()
    for n,t in sorted(values.items()):
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(list(t.shape)).encode());h.update(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def equal_inputs(a,b,torch):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)


def run(c,scope):
    require(os.environ.get('PRACTICAL_STAGE_CONFIRMED')=='1','请由用户设置PRACTICAL_STAGE_CONFIRMED=1')
    if scope=='full714':
        gate=json.loads((OUT/'results/smoke3/summary.json').read_text())
        require(gate['signature']==c['signature'] and gate['audit']['passed'],'先完成本配置smoke3并汇总')
    expected={(r['index'],a) for r in c['rows'] for a in ARMS}
    if expected<=c['units'].keys():print('已完整，跳过模型加载。');return
    import random,numpy as np,torch,transformers
    require(torch.cuda.is_available(),'本步骤需要GPU；当前CPU可执行init/preflight/test-cpu，不启动CPU推理')
    from transformers import AutoModelForCausalLM,AutoProcessor
    from tqdm import tqdm
    from practical_vote import CaptureTokenizer
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    ref=c['protocol']['runtime']
    rt=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype=ref['dtype'])
    require(rt==ref,'运行版本/显卡与旧结果不同，先核查，不盲目复用:'+str(rt))
    proc=AutoProcessor.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.max_pixels=150000;part.min_pixels=3136
    # 显式固定bf16，不让当前设备/配置自动改变精度。
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0}).eval()
    require(str(model.dtype)==ref['dtype'],'模型精度不同')
    atomic_json(OUT/'runtime.json',rt)
    base_video=proc.video_processor;tokenizer=proc.tokenizer;current_video=None;cache=None
    def native(row,text):
        chat=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True)
        return proc(text=[chat],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)
    def packed(text):
        enc=tokenizer([cache['template'].replace(SLOT,text)],padding=True,return_tensors='pt')
        return dict(cache['visual'],input_ids=enc['input_ids'],attention_mask=enc['attention_mask'])
    with tqdm(total=len(expected),initial=len(expected&c['units'].keys()),desc='Mage 字幕消融 '+scope,unit='答案',dynamic_ncols=True) as bar:
        for row in c['rows']:
            idx=row['index']
            if all((idx,a) in c['units'] for a in ARMS):continue
            guard(); video=row['video_id'];e=c['bundle']['evidence'][video];start=time.perf_counter()
            if video!=current_video:
                st=Path(row['video_path']).stat()
                require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==row['video_identity'],'视频文件身份漂移')
                proc.video_processor=ReplayVideo(base_video,row['video_path'])
                cap=CaptureTokenizer(tokenizer);proc.tokenizer=cap
                try:values=native(row,SLOT)
                finally:proc.tokenizer=tokenizer
                require(cap.text is not None and len(cap.text)==1 and cap.text[0].count(SLOT)==1,'原生插槽捕获异常')
                ids=values['patch_positions'][:,0].unique_consecutive().tolist()
                seconds=proc.video_processor.output['frame_timestamps'][0]
                require(ids==e['aligned']['frame_indices'] and seconds==e['aligned']['frame_seconds'],'实际帧号/时间漂移')
                cache=dict(template=cap.text[0],visual={k:v for k,v in values.items() if k not in ['input_ids','attention_mask']})
                current_video=video
                audit_video=dict(signature=c['signature'],video_id=video,frame_indices=ids,frame_seconds=seconds,visual_sha256=tensor_digest(cache['visual']))
                atomic_json(OUT/'video_audits'/f'{video}.json',sealed(audit_video))
                atomic_json(OUT/'preprocess_attempts'/f'{video}_{uuid.uuid4().hex}.json',sealed(dict(**audit_video,decode_and_preprocess_s=proc.video_processor.decode_s)))
                del values
            audit=dict(passed=True,baselines={});baseline_inputs={}
            for arm in BASES:
                text=subtitle_prompt(row,'' if arm=='no_sub' else e[arm]['text'])
                values=packed(text);direct=native(row,text)
                require(equal_inputs(values,direct,torch),'旧Prompt插槽与原生输入不一致')
                old=c['bundle']['old'][idx][arm];h=signature(values['input_ids'].tolist());vi=visual_info(values,proc.spatial_merge_size)
                require(h==old['input_sha256'] and vi==old['visual'],'旧三路输入身份未复现:'+idx+':'+arm)
                if old['input_tokens'] is not None:require(int(values['input_ids'].shape[1])==old['input_tokens'],'旧输入长度不同')
                audit['baselines'][arm]=dict(input_sha256=h,input_tokens=int(values['input_ids'].shape[1]),source_record_sha256=old['source_record_sha256'],tensor_sha256=tensor_digest(values))
                baseline_inputs[arm]=values;del direct
            prep=time.perf_counter()-start
            for arm in ARMS:
                if (idx,arm) in c['units']:continue
                guard();variant=e['variants'][arm]
                text=subtitle_prompt(row,variant['text']);values=packed(text);direct=native(row,text)
                require(equal_inputs(values,direct,torch),'新Prompt与原生输入不一致');del direct
                require(len(tokenizer.encode(variant['text'],add_special_tokens=False))==variant['text_tokens'],'文本token计数不同')
                matches=[b for b in BASES if equal_inputs(values,baseline_inputs[b],torch)]
                vi=visual_info(values,proc.spatial_merge_size);proof=None;raw=None;origin='generated';n=1;generated=None;elapsed=peak=0.
                if matches:
                    b=matches[0];raw=c['bundle']['old'][idx][b]['raw_text'];n=0;origin='reused_exact_old_input';proof=dict(arm=b,tensor_equal=True)
                else:
                    # 同题另一消融输入确实完全相同时才复用；不按预测置信度选择。
                    for other in ARMS:
                        oldunit=c['units'].get((idx,other))
                        if oldunit and equal_inputs(values,packed(subtitle_prompt(row,e['variants'][other]['text'])),torch):
                            raw=oldunit['raw_text'];n=0;origin='reused_exact_current_input';proof=dict(arm=other,record_sha256=oldunit['record_sha256'],tensor_equal=True);break
                if n:
                    gpu={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in values.items()}
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    with torch.inference_mode():ans=model.generate(**gpu,max_new_tokens=32,do_sample=False)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
                    generated=int(ans.shape[1]-values['input_ids'].shape[1]);raw=tokenizer.decode(ans[0,values['input_ids'].shape[1]:],skip_special_tokens=True).strip();del gpu,ans
                u=sealed(dict(signature=c['signature'],index=idx,stage=row['stage'],video_id=video,arm=arm,origin=origin,raw_text=raw,pred=parse_answer(raw),new_calls=n,generated_tokens=generated,reuse_proof=proof,
                    prompt_sha256=signature(text),input_sha256=signature(values['input_ids'].tolist()),tensor_sha256=tensor_digest(values),input_tokens=int(values['input_ids'].shape[1]),
                    subtitle_tokens=variant['text_tokens'],visual=vi,native_equal=True,source_audit=audit,preprocess_s=prep if arm==ARMS[0] else 0.,inference_s=elapsed,peak_gib=peak))
                validate_unit(c['signature'],c['bundle'],{idx:row},u)
                atomic_json(OUT/'units'/f'{idx}_{arm}.json',u);c['units'][(idx,arm)]=u;bar.update(1);bar.set_postfix(index=idx,arm=arm,new=n)
                print('落盘',idx,arm,u['pred'] or 'INVALID',origin,flush=True);del values
            del baseline_inputs
    proc.video_processor=base_video


def csv_once(path, rows):
    import io
    f=io.StringIO();fields=list(rows[0])
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    text=f.getvalue();path=Path(path)
    if path.exists():require(path.read_bytes()==text.encode(),'CSV冲突，不覆盖')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',newline='') as out:out.write(text)


def summarize(c,scope):
    expected={(r['index'],a) for r in c['rows'] for a in ARMS}
    require(expected<=c['units'].keys(),'尚未完整，先续跑，不读取标签')
    lock=json.loads((ROOT/'01_manifests/dataset_lock.json').read_text());labels_path=ROOT/'01_manifests/evaluation_labels.json'
    require(sha(labels_path)==lock['labels_sha256'],'标签身份漂移')
    labels=json.loads(labels_path.read_text())
    from practical_analyze import comparison
    b=c['bundle'];rows=c['rows'];names=list(BASES)+list(ARMS)
    pp={a:{r['index']:b['old'][r['index']][a]['pred'] for r in rows} for a in BASES}
    pp.update({a:{r['index']:c['units'][(r['index'],a)]['pred'] for r in rows} for a in ARMS})
    metrics=[];pairs=[];details=[];cost=[]
    groups=[('all714_coverage' if scope=='full714' else 'smoke3',rows)]
    groups += [(s,[r for r in rows if r['stage']==s]) for s in b['roles'] if any(r['stage']==s for r in rows)]
    for stage,rr in groups:
        for duration,rs in [('overall',rr)]+[(d,[r for r in rr if r['duration']==d]) for d in ['short','medium','long'] if any(r['duration']==d for r in rr)]:
            for a in names:
                good=sum(pp[a][r['index']]==labels[r['index']] for r in rs)
                metrics.append(dict(stage=stage,duration=duration,N=len(rs),videos=len({r['video_id'] for r in rs}),method=a,correct=good,accuracy_pct=100*good/len(rs),invalid=sum(pp[a][r['index']] not in list('ABCD') for r in rs)))
            for a,base in [('no_time','continuous'),('no_neighbors','continuous'),('continuous','aligned'),('continuous','no_sub')]:
                m=comparison(rs,pp[a],pp[base],labels)
                pairs.append(dict(stage=stage,duration=duration,method=a,baseline=base,N=len(rs),**m))
    for r in rows:
        idx=r['index'];details.append(dict(index=idx,video_id=r['video_id'],stage=r['stage'],role=r['role'],duration=r['duration'],task_type=r['task_type'],gt=labels[idx],**{a:pp[a][idx] for a in names}))
        for a in names:
            u=c['units'][(idx,a)] if a in ARMS else b['old'][idx][a]
            tokens=u.get('input_tokens')
            if tokens is None:tokens=c['units'][(idx,ARMS[0])]['source_audit']['baselines'][a]['input_tokens']
            cost.append(dict(index=idx,stage=r['stage'],duration=r['duration'],method=a,input_tokens=tokens,subtitle_tokens=u['subtitle_tokens'],visual_tokens=u['visual']['n_tok'],new_calls=u['new_calls'] if a in ARMS else 0,new_inference_s=u['inference_s'] if a in ARMS else 0.,source_record_sha256=u.get('record_sha256',u.get('source_record_sha256'))))
    selected=[c['units'][k] for k in expected]
    invalid=sum(u['pred'] not in list('ABCD') for u in selected)
    report=dict(signature=c['signature'],scope=scope,N=len(rows),audit=dict(passed=invalid==0,invalid_new=invalid,source_input_checks=True,complete_units=len(selected)),metrics=metrics,paired=pairs,
        new_cost=dict(calls=sum(u['new_calls'] for u in selected),reused=sum(not u['new_calls'] for u in selected),inference_s=sum(u['inference_s'] for u in selected),peak_gib=max(u['peak_gib'] for u in selected)),
        interpretation=c['protocol']['caveat'],timing_note='New generation cost excludes model loading, resumed idle time and historical preprocessing. Cached source audits add work; not a deployment-speed or Table5 comparison.',reference_stage_roles=b['roles'])
    dest=OUT/'results'/scope
    atomic_json(dest/'summary.json',report);atomic_json(dest/'predictions.json',details);atomic_json(dest/'token_cost.json',cost)
    csv_once(dest/'predictions.csv',details);csv_once(dest/'metrics.csv',metrics);csv_once(dest/'token_cost.csv',cost)
    csv_once(dest/'paired_comparisons.csv',[{**r,'video_cluster_bootstrap_95ci_pp':json.dumps(r['video_cluster_bootstrap_95ci_pp'])} for r in pairs])
    print(json.dumps(dict(N=len(rows),audit=report['audit'],new_cost=report['new_cost']),ensure_ascii=False,indent=2));print('汇总:',dest/'summary.json')
    require(not invalid,'非法回答已保留并按错计，先诊断，不自动重跑或扩样')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','run','summarize']);ap.add_argument('scope',choices=['smoke3','full714'],nargs='?',default='smoke3');args=ap.parse_args()
    require(subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip()==BRANCH,'必须在'+BRANCH)
    print('SPACE',guard(),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'operation.lock').open('a') as op:
        fcntl.flock(op,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='init':initialize();return
        c=context(args.scope)
        if args.action=='run':
            with (ROOT/'cache/gpu.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);run(c,args.scope)
        elif args.action=='summarize':summarize(c,args.scope)
        else:
            needed={(r['index'],a) for r in c['rows'] for a in ARMS}
            print(args.scope,':',len(needed&c['units'].keys()),'/',len(needed),'逻辑答案；不加载模型。')


if __name__=='__main__':main()
