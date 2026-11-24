"""A.I.R.-Mage首版协议与断点；CPU导入，不读答案、不导入模型。"""
import hashlib,json,math,os,shutil,subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_sources'))
from frozen_sources import protocol_script_hash, SOURCES

PROJECT=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
OLD=PROJECT/'VLM/eval_results/practical_mage_v1'
ROOT=OLD/'18_air_frames_v1'
MODEL=PROJECT/'VLM/mage_vl/models'
CLIP=PROJECT/'VLM/models/CLIP-ViT-Large-patch14'
REFERENCE=SOURCES/'A.I.R.'
BRANCH='feat/air-visual-evidence'
REVISION='native_uniform_fix1'
ARMS=('uniform32','air_cap32')
ANSWER_HEADER=('These are the frames of a video. Select the best answer to the following '
 'multiple-choice question based on the video. Respond with only the letter '
 '(A, B, C, or D) of the correct option.\n')
ANALYSIS=('You are an expert visual reasoner. Assess how useful this image is as evidence for the following question and all its options.\n'
 '{query}\nRate relevance from 1 to 5: 1 unrelated; 2 minor contextual hints; 3 preparatory or follow-up context; '
 '4 clear but incomplete evidence; 5 sufficient and unambiguous evidence. Do not guess the answer from language alone. '
 'Return exactly two lines, with the score first:\nScore: <1-5>\nReasoning: <one brief sentence>')

def sha(path):
    path = Path(path)
    if path.resolve() == Path(__file__).resolve():
        return protocol_script_hash(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write_once(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'已有文件冲突，保留不覆盖:{path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def seal(value):return dict(value,record_sha256=digest(value))
def unseal(value):
    v=dict(value);h=v.pop('record_sha256',None)
    if h!=digest(v):raise ValueError('断点记录哈希损坏，保留现场')
    return v
def read_journal(path):
    if not path.exists():return []
    rows=[]
    with path.open() as f:
        for line in f:
            if not line.endswith('\n'):raise ValueError('断点残行，保留现场，不自动截断')
            row=json.loads(line);unseal(row);rows.append(row)
    return rows
def append(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def disk_guard(min_free=10,budget=1):
    free=shutil.disk_usage('/root/autodl-tmp').free/1024**3
    total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()) if ROOT.exists() else 0
    if free<min_free or total>budget*1024**3:raise RuntimeError(f'空间保护停止，不清理：free={free:.2f}GiB，AIR={total/1024**2:.1f}MiB')
    return dict(free_gib=round(free,3),air_mib=round(total/1024**2,3))
def query(row):return row['question']+'\n'+'\n'.join(c+'.'+s for c,s in zip('ABCD',row['bodies']))
def prompt(row):return ANSWER_HEADER+query(row)
def parse_answer(raw):
    import re
    s=raw.strip().upper();m=re.fullmatch(r'[\s\(\[]*([ABCD])[\s\)\]\.]*',s)
    if m:return m.group(1)
    m=re.search(r'(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([ABCD])\b',s)
    return m.group(1) if m else ''
def text_chunks(ids,bos,eos,max_length=77):
    if max_length<3 or not ids:raise ValueError('CLIP文本为空或长度非法')
    chunks=[[bos,*ids[i:i+max_length-2],eos] for i in range(0,len(ids),max_length-2)]
    if [n for c in chunks for n in c[1:-1]]!=ids:raise ValueError('CLIP完整query token未保持')
    return chunks
def model_identity(folder,weight_names=None):
    configs={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in ['.json','.py','.jinja','.txt']}
    weights={}
    for p in sorted(folder.iterdir()):
        if p.is_file() and (p.suffix in ['.safetensors','.bin']) and (weight_names is None or p.name in weight_names):
            st=p.stat();weights[p.name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    if not configs or not weights or any(x['bytes']==0 for x in weights.values()):raise FileNotFoundError(f'缺本地模型:{folder}')
    return dict(path=str(folder),files=configs,weights=weights)
def load_rows():
    rows=json.loads((OLD/'01_manifests/inputs.json').read_text())['rows']
    if any('gt' in r or 'answer' in r for r in rows):raise ValueError('推理清单含真值')
    if any(not isinstance(r.get('bodies'),list) or len(r['bodies'])!=4 or any(not isinstance(s,str) for s in r['bodies']) for r in rows):raise ValueError('必须保留完整四选项')
    split=json.loads((OLD/'01_manifests/splits.json').read_text());dev=set(split['dev']);dev30=set(split['dev30'])
    pool=[r for r in rows if r['video_id'] in dev]
    if len(pool)!=180 or len({r['video_id'] for r in pool})!=60:raise ValueError('旧开发池变化')
    # smoke属于已有dev30，不抽取新数据，也不按正确答案选题。
    selected=[r for r in pool if r['video_id'] in dev30]
    if len(selected)!=90:raise ValueError('原dev30应为90题')
    first=selected[0]['video_id'];smoke=[r for r in selected if r['video_id']==first]
    if len(smoke)!=3:raise ValueError('smoke应为同视频3题')
    return pool,dict(smoke3=smoke,dev30=selected)
def validate_unit(ctx,u,existing):
    unseal(u);idx=u['index'];arm=u['arm'];key=f'{idx}:{arm}'
    row=ctx['lookup'].get(idx)
    if row is None or arm not in ARMS or key in existing or u['signature']!=ctx['signature'] or u['video_id']!=row['video_id']:raise ValueError('配对单元身份不符或重复')
    if u['prompt_sha256']!=digest(prompt(row)) or u['pred']!=parse_answer(u['raw_text']):raise ValueError('提示词或解析漂移')
    inds=u['indices'];seconds=u['seconds'];v=u['visual']
    if (not inds or len(inds)>32 or inds!=sorted(set(inds)) or any(type(n) is not int or n<0 for n in inds)
            or len(seconds)!=len(inds) or any(not math.isfinite(s) or s<0 for s in seconds)
            or any(a>=b for a,b in zip(seconds,seconds[1:]))
            or len(v['grid_thw'])!=len(inds) or v['n_tok']<=0 or any(len(g)!=3 or g[0]!=1 or min(g)<=0 for g in v['grid_thw'])
            or u['native_uniform_equal'] is not True or type(u['input_tokens']) is not int or u['input_tokens']<=v['n_tok']
            or not isinstance(u['input_sha256'],str) or len(u['input_sha256'])!=64):raise ValueError('采样/位置/视觉审计不符')
    metadata=json.loads((ctx['store']/'video_metadata'/f'{row["video_id"]}.json').read_text())
    count=metadata['frame_count'];fps=metadata['fps']
    if metadata['signature']!=ctx['signature'] or metadata['video_id']!=row['video_id'] or type(count) is not int or count<32 or not math.isfinite(fps) or fps<=0:
        raise ValueError('来源视频元数据不符')
    if (inds[-1]>=count or any(abs(s-i/fps)>1.1e-6 for i,s in zip(inds,seconds))
            or v['patch_time_range']!=[float(inds[0]),float(inds[-1])]):raise ValueError('真实时间轴或patch位置不符')
    for n in ['inference_s','peak_gib','preprocess_s']:
        if not math.isfinite(u[n]) or u[n]<0:raise ValueError('成本非法')
    if arm=='uniform32':
        from air_selection import uniform_indices
        audit=json.loads((ctx['store']/'native_audits'/f'{idx}.json').read_text())
        if (inds!=uniform_indices(count,32) or audit!=dict(signature=ctx['signature'],index=idx,input_sha256=u['input_sha256'],indices=inds,seconds=seconds,tensor_equal=True)
                or u['origin']!='generated' or u['new_calls']!=1 or u['reuse_record'] is not None or u['selection_sha256'] is not None):raise ValueError('基线必须本轮原生均匀生成')
    else:
        base=existing.get(f'{idx}:uniform32')
        if base is None:raise ValueError('缺少同题已落盘基线')
        sel=ctx['store']/'selections'/f'{idx}.json'
        obj=json.loads(sel.read_text());unseal(obj)
        if obj['signature']!=ctx['signature'] or obj['index']!=idx or obj['final_indices']!=inds or u['selection_sha256']!=sha(sel):raise ValueError('选择轨迹漂移')
        if obj['total_frames']!=count or obj['fps']!=fps or obj['config']!=ctx['protocol']['selection']:raise ValueError('选择轨迹来源或规则不符')
        if obj['fallback_reason']:
            if (u['origin']!='fallback_uniform' or u['new_calls']!=0 or u['reuse_record']!=base['record_sha256']
                    or any(u[k]!=base[k] for k in ['raw_text','pred','input_sha256','visual','indices','seconds'])
                    or u['inference_s']!=0 or u['peak_gib']!=0):raise ValueError('无证据回退必须逐字复用本题基线')
        elif u['origin']!='generated' or u['new_calls']!=1 or u['reuse_record'] is not None:raise ValueError('选择后生成来源不符')
    if u['new_calls']:
        if type(u['generated_tokens']) is not int or not 1<=u['generated_tokens']<=32:raise ValueError('QA生成长度不符')
    elif u['generated_tokens'] is not None:raise ValueError('回退不应新增生成token')
    return key
def context(stage,initialize=False):
    from air_selection import AIRConfig
    if stage not in ['smoke3','dev30']:raise ValueError('仅开放smoke3/dev30')
    branch=subprocess.check_output(['git','-C',str(PROJECT),'branch','--show-current'],text=True).strip()
    if branch!=BRANCH:raise RuntimeError(f'必须在{BRANCH}，保留改动并停止')
    pool,stages=load_rows();folder=Path(__file__).parent
    video={}
    for r in pool:
        p=Path(r['video_path']);st=p.stat();video[r['video_id']]=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    cfg=AIRConfig()
    import dataclasses
    protocol=dict(version='air_mage_frames_v1_'+REVISION,uniform_sampling='native torch.linspace CPU then round().long(), not numpy truncation',selection=dataclasses.asdict(cfg),pool_indices=[r['index'] for r in pool],
        sources={n:sha(OLD/'01_manifests'/n) for n in ['inputs.json','splits.json','dataset_lock.json']},
        videos=video,mage=model_identity(MODEL),clip=model_identity(CLIP,['model.safetensors']),
        reference_commit='400fb52f1cfbf64debc6408cd215cfa375fd9797',
        reference_sha256={n:sha(REFERENCE/n) for n in ['air/model.py','air/Process_Utils/process_messages.py','LICENSE']},
        scripts={n:protocol_script_hash(folder/n) for n in ['air_protocol.py','air_selection.py','air_mage_adapter.py','air_runtime.py','air_run.py','air_analyze.py']},
        prompts=dict(answer=ANSWER_HEADER,analysis=ANALYSIS),generation=dict(seed=0,answer_max_new=32,analysis_max_new=96,greedy=True,max_pixels=150000,min_pixels=3136),
        coarse_fps=1.0,clip_feature_storage='fp16 persisted then float32 renormalized for every first/replay similarity',
        clip_query='full question+ABCD, content-token chunks<=75, normalize each text vector, mean then renormalize; no truncation',
        adaptation='HFCLIP ViT-L14 instead of EVA; Mage analysis+answer; cap32, C12,I6; positives only, zero-positive uniform fallback; no QueryIP/subtitles/Codec/API; developer coverage only')
    candidate=ROOT/f'candidate_{REVISION}.json'
    if initialize:write_once(candidate,protocol)
    if not candidate.exists() or json.loads(candidate.read_text())!=protocol:raise ValueError('先init；配置/来源/源码变化不覆盖旧候选')
    sig=digest(protocol);store=ROOT/'cache'/sig;lookup={r['index']:r for r in pool}
    ctx=dict(signature=sig,protocol=protocol,store=store,stage_dir=ROOT/REVISION/stage,rows=stages[stage],lookup=lookup,units={})
    ctx['expected']={f'{r["index"]}:{a}' for r in ctx['rows'] for a in ARMS}
    for u in read_journal(store/'units.jsonl'):
        k=validate_unit(ctx,u,ctx['units']);ctx['units'][k]=u
    return ctx
