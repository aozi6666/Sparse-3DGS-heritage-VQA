"""先核验30题旧输入，再最多7次Mage QA；逐题原子断点，推理阶段禁止读标签。"""
import argparse
from copy import deepcopy
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import air_protocol as p
import air_hybrid as hashing
import condition_clips_qa30 as backend
import condition_clips_noobs_qa30 as noobs
import leaf_probe_run as selector
from leaf_probe import write_once
import leaf_qa30_protocol as k

OUT=p.ROOT/k.VERSION
CODE=['leaf_qa30_protocol.py','leaf_qa30_run.py','leaf_qa30_analyze.py','run_leaf_qa30.sh',
      'air_protocol.py','air_hybrid.py','air_mage_adapter.py','condition_clips_qa30.py','condition_clips_noobs_qa30.py','leaf_probe.py']
ACCESS={'may_score':False,'labels_allowed':False}


def access_guard():
    def hook(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:raise RuntimeError('本地QA诊断禁止联网')
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            if path.name in ['.env','deepseek.env'] or path.name.startswith('.env.'):raise RuntimeError('本轮不读取API凭据')
            if path.name=='evaluation_labels.json' and not ACCESS['labels_allowed']:raise RuntimeError('完整预测前/推理阶段禁止读取标签')
            if 'oracle_evidence_diag_v1' in str(path):raise RuntimeError('禁止人工证据位置输入')
    sys.addaudithook(hook)


def dependencies():
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','torchvision','Pillow']}
    spec=importlib.util.find_spec('decord')
    if spec is None or spec.origin is None:raise ValueError('缺少既有decord，不自动安装')
    folder=Path(spec.origin).parent
    versions['decord_files']={n:p.sha(folder/n) for n in ['__init__.py','video_reader.py','libdecord.so'] if (folder/n).is_file()}
    if not versions['decord_files']:raise ValueError('缺少decord身份文件')
    return versions


def frozen_locations(parent):
    result={};files={}
    for i in parent['protocol']['indices']:
        path=selector.OUT/'locations'/f'{i}.json';u=json.loads(path.read_text());p.unseal(u)
        group=next(g for g,items in parent['protocol']['groups'].items() if i in items)
        if u['signature']!=parent['signature'] or u['index']!=i or u['group']!=group or u['original_noobs_indices']!=parent['plans'][i]['indices']:raise ValueError('冻结定位身份异常')
        files[str(path)]=p.sha(path)
        if u['result'] is not None:
            z=u['result'];ids=sorted(set(z['anchors']+[f for w in z['windows'] for f in w['indices']]))
            if ids!=z['indices'] or len(z['windows'])!=6 or any(len(w['indices'])!=4 for w in z['windows']):raise ValueError('冻结窗口/帧序列异常')
            curve=selector.OUT/'curves'/f'{i}.npz'
            if p.sha(curve)!=u['curves_sha256']:raise ValueError('定位曲线来源漂移')
            files[str(curve)]=p.sha(curve)
            for text,sha in u['text_files'].items():
                tp=selector.text_path(text)
                if p.sha(tp)!=sha:raise ValueError('定位文本向量来源漂移')
                files[str(tp)]=sha
        result[i]=u
    return result,files


def context(initialize=False):
    if subprocess.check_output(['git','-C',str(p.PROJECT),'branch','--show-current'],text=True).strip()!=p.BRANCH:raise ValueError('分支不符，保留现场')
    parent=selector.context();src=noobs.context()
    if len(src['units'])!=30 or parent['protocol']['indices']!=src['protocol']['indices']:raise ValueError('旧QA或定位30题范围不完整')
    locs,files=frozen_locations(parent);ids=parent['protocol']['indices'];groups=parent['protocol']['groups']
    if '484' not in groups['candidate']:raise ValueError('冻结候选必须保留484，不按已见失败剔除')
    plans=k.build_plans(ids,groups,locs,src['plans'],parent['meta'])
    for i in ids:
        old=src['units'][i];base=src['bases'][i]
        if old['indices']!=plans[i]['old_noobs_indices'] or old['prompt_sha']!=p.digest(p.prompt(src['rows'][i])) or old['pred']!=p.parse_answer(old['raw_text']):raise ValueError('旧noobs输入/文本/解析不符')
        if old['pred'] not in list('ABCD') or base['pred'] not in list('ABCD'):raise ValueError('旧预测有非法答案，不能静默复用')
    expected=dict(seed=0,max_new_tokens=32,greedy=True,max_pixels=150000,min_pixels=3136)
    if src['protocol']['generation']!=expected:raise ValueError('旧生成配置不符，不改原QA配置')
    native_path=p.ROOT/'candidate_native_uniform_fix1.json';native=json.loads(native_path.read_text());mage=p.model_identity(p.MODEL)
    if native['mage']!=mage:raise ValueError('Mage模型/处理器身份漂移')
    label_lock=p.OLD/'01_manifests/dataset_lock.json'
    for path in [selector.OUT/'candidate.json',selector.OUT/'results/summary.json',selector.OUT/'results/review_packet.json',
                 selector.OUT/'results/assistant_visual_review_v1.json',noobs.OUT/'candidate.json',noobs.OUT/'units.jsonl',
                 backend.OUT/'runtime.json',native_path,label_lock]:files[str(path)]=p.sha(path)
    # 旧答案仅用于明确授权的复用/基线，不进入QA提示；源选择已冻结。
    files.update(parent['protocol']['source_files'])
    dep=dependencies();runtime=json.loads((backend.OUT/'runtime.json').read_text())
    if any(dep[n]!=runtime[n] for n in ['torch','transformers','numpy']):raise ValueError('与历史QA依赖版本不同，先诊断，不自动重跑')
    folder=Path(__file__).parent;analysis_file=p.PROJECT/'VQA/mage_vl/practical/practical_analyze.py'
    protocol=dict(version=k.VERSION,indices=ids,groups=groups,plans=plans,parent_signature=parent['signature'],source_noobs_signature=src['signature'],
                  source_noobs_records={i:u['record_sha256'] for i,u in src['units'].items()},source_uniform_records={i:u['record_sha256'] for i,u in src['bases'].items()},
                  original_QA_prompt_sha={i:p.digest(p.prompt(src['rows'][i])) for i in ids},generation=expected,mage=mage,
                  dependencies=dep,historical_QA_runtime=runtime,source_files=files,
                  code={str(folder/n):p.sha(folder/n) for n in CODE},analysis_code_sha256=p.sha(analysis_file),
                  policy='Seen development30; frozen7 changed-pixel QA, anchor4+fallback19 exact noobs reuse after FULL tensor audit. '
                         'Not general automatic routing. No query/observation/subtitle/API/Codec/GT in QA. '
                         'Keep484 and frozen <=32-frame plans. Summarize only after all30 audits/predictions complete.')
    if initialize:write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；代码/来源/配置变化不覆盖旧协议')
    c=dict(signature=p.digest(protocol),protocol=protocol,plans=plans,rows=src['rows'],noobs=src,parent=parent,audits={},generations={},units={})
    for folder in ['audits','generations','records']:
        found={path.stem for path in (OUT/folder).glob('*.json')}
        allowed={i for i in ids if plans[i]['generate']} if folder=='generations' else set(ids)
        if not found<=allowed:raise ValueError('结果目录有越界题号或非候选生成，保留现场')
    for i in ids:
        ap=audit_path(i)
        if ap.exists():c['audits'][i]=k.validate_audit(c,json.loads(ap.read_text()))
    for i in ids:
        gp=generation_path(i)
        if gp.exists():c['generations'][i]=k.validate_generation(c,json.loads(gp.read_text()))
    for i in ids:
        up=unit_path(i)
        if up.exists():c['units'][i]=k.validate_unit(c,json.loads(up.read_text()),c['generations'].get(i))
    return c


def audit_path(i):return OUT/'audits'/f'{i}.json'
def generation_path(i):return OUT/'generations'/f'{i}.json'
def unit_path(i):return OUT/'records'/f'{i}.json'


def save_failure(c,i,phase,error,observed=None):
    # 失败另存，不能伪装为有效审计/预测，也不能自动换旧答案掩盖候选失败。
    info=None
    if isinstance(observed,dict):
        allowed=['input','indices','raw_text','generated_token_ids','native_equal']
        info={name:observed[name] for name in allowed if name in observed}
        try:json.dumps(info,allow_nan=False)
        except (TypeError,ValueError):info='observed fields not JSON serializable'
    path=OUT/'failures'/f'{phase}_{i}_{uuid.uuid4().hex}.json'
    write_once(path,p.seal(dict(signature=c['signature'],index=i,phase=phase,error_type=type(error).__name__,error=str(error)[:3000],
                                source_noobs_record_sha=c['noobs']['units'][i]['record_sha256'],
                                expected_source_input={n:c['noobs']['units'][i][n] for n in k.FP_FIELDS},observed=info,
                                usable_checkpoint=False,unix_s=time.time())))
    return path


def load_processor():
    from transformers import AutoProcessor
    proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [proc.video_processor,proc.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    return proc


def input_fingerprint(original,use,proc,indices):
    if use['patch_positions'][:,0].unique_consecutive().tolist()!=indices:raise ValueError('实际patch原帧位置与冻结输入不符')
    grid=use['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//proc.spatial_merge_size**2
    fp=dict(tensor_sha=hashing.tensor_sha(use),baseline_tensor_sha=hashing.tensor_sha(original),input_tokens=int(use['input_ids'].shape[1]),visual_tokens=visual,grid_thw=grid)
    k.validate_input(fp,indices)
    return fp


def prepare(c,i,proc,plan):
    from decord import VideoReader,cpu
    # 与旧QA后端构造参数一致；不额外改变解码线程策略。
    vr=VideoReader(c['rows'][i]['video_path'],ctx=cpu(0))
    if len(vr)!=plan['count'] or abs(float(vr.get_avg_fps())-plan['fps'])>1e-8:raise ValueError('视频fps/帧数身份漂移')
    # 唯一文本由旧backend.input_pair内部p.prompt(row)构造；不传入观察/关系/旧答案。
    original,use=backend.input_pair(proc,vr,c['rows'][i],plan,c['noobs']['bases'][i])
    fp=input_fingerprint(original,use,proc,plan['indices'])
    if fp['baseline_tensor_sha']!=c['noobs']['units'][i]['baseline_tensor_sha']:raise ValueError('原生uniform完整输入与历史SHA不符')
    return original,use,fp


class SourceInputAuditor:
    def __init__(self,c):self.proc=load_processor()
    def inspect(self,c,i):
        start=time.perf_counter();plan=dict(c['plans'][i],indices=c['plans'][i]['old_noobs_indices'])
        original,use,fp=prepare(c,i,self.proc,plan);elapsed=time.perf_counter()-start
        del original,use
        return p.seal(dict(signature=c['signature'],index=i,indices=plan['indices'],**k.source_refs(c,i),
                           prompt_sha=p.digest(p.prompt(c['rows'][i])),input=fp,native_equal=True,preprocess_s=elapsed))


def audit_inputs(c,auditor_factory=None):
    from tqdm import tqdm
    auditor=None
    with tqdm(total=30,initial=len(c['audits']),desc='源输入身份审计（不生成）',unit='题',dynamic_ncols=True) as bar:
        for i in c['protocol']['indices']:
            if i in c['audits']:continue
            p.disk_guard()
            if auditor is None:auditor=(auditor_factory or SourceInputAuditor)(c)
            audit=None
            try:audit=auditor.inspect(c,i);k.validate_audit(c,audit)
            except Exception as exc:
                path=save_failure(c,i,'source_audit',exc,audit)
                raise RuntimeError(f'源输入审计失败 {i}；未接受该输入，详情：{path}') from None
            write_once(audit_path(i),audit);c['audits'][i]=audit;bar.update(1);bar.set_postfix(index=i)
    print('30题旧uniform/noobs完整输入SHA核验完成；没有生成答案。')


class MageBackend:
    def __init__(self,c):
        import random,numpy as np,torch,transformers
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():raise RuntimeError('CUDA不可用，不安装或自动切换模型')
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        self.proc=load_processor();self.torch=torch
        self.model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(self.model.dtype))
        if runtime['dtype']!=c['protocol']['historical_QA_runtime']['dtype']:raise ValueError('推理dtype与历史QA不同')
        write_once(OUT/'runtime.json',runtime)
    def generate(self,c,i):
        torch=self.torch;start=time.perf_counter();original,use,fp=prepare(c,i,self.proc,c['plans'][i]);prep=time.perf_counter()-start
        if fp['baseline_tensor_sha']!=c['audits'][i]['input']['baseline_tensor_sha']:raise ValueError('生成前原生uniform与源输入审计不符')
        gpu={n:t.to(device=self.model.device,dtype=self.model.dtype) if n=='pixel_values' else t.to(self.model.device) for n,t in use.items()}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode():answer=self.model.generate(**gpu,max_new_tokens=32,do_sample=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**3
        tokens=answer[0,use['input_ids'].shape[1]:].tolist();raw=self.proc.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        gen=p.seal(dict(signature=c['signature'],index=i,indices=c['plans'][i]['indices'],**k.source_refs(c,i),audit_record_sha=c['audits'][i]['record_sha256'],
                        prompt_sha=p.digest(p.prompt(c['rows'][i])),input=fp,native_equal=True,generated_token_ids=tokens,raw_text=raw,pred=p.parse_answer(raw),
                        preprocess_s=prep,inference_s=elapsed,peak_gib=peak))
        del answer,gpu,original,use
        return gen


def materialize(c,i):
    unit=k.make_unit(c,i,c['generations'].get(i));write_once(unit_path(i),unit);c['units'][i]=unit


def run(c,backend_factory=None):
    k.require_audits(c)
    # 先完成纯复用或已落盘生成，保证完整时不加载处理器/模型、不重复生成。
    for i in c['protocol']['indices']:
        if i not in c['units'] and (not c['plans'][i]['generate'] or i in c['generations']):materialize(c,i)
    from tqdm import tqdm
    model=None
    with tqdm(total=30,initial=len(c['units']),desc='冻结7题QA＋23题复用',unit='题',dynamic_ncols=True) as bar:
        for i in c['protocol']['indices']:
            if i in c['units']:continue
            if not c['plans'][i]['generate']:raise ValueError('非候选题意外进入生成')
            p.disk_guard()
            gen=None
            try:
                if model is None:model=(backend_factory or MageBackend)(c)
                gen=model.generate(c,i);k.validate_generation(c,gen)
            except Exception as exc:
                path=save_failure(c,i,'candidate_QA',exc,gen)
                raise RuntimeError(f'候选QA失败 {i}；不自动改用旧答，详情：{path}') from None
            # 原生成缓存必须先于逐题记录提交；中断后按缓存补记录，不重抽答案。
            write_once(generation_path(i),gen);c['generations'][i]=gen
            materialize(c,i);bar.update(1);bar.set_postfix(index=i,pred=gen['pred'] or 'INVALID')
    print('30题完成；仅冻结7候选有新生成，其余按审计复用。请summarize，不自动扩样。')


def preflight(c):
    print(json.dumps(dict(N=30,groups={g:len(ids) for g,ids in c['protocol']['groups'].items()},planned_new_generations=sum(x['generate'] for x in c['plans'].values()),
                          planned_reused_answers=sum(not x['generate'] for x in c['plans'].values()),completed_source_audits=len(c['audits']),completed_predictions=len(c['units']),
                          pending_new_generations=sum(x['generate'] and i not in c['generations'] for i,x in c['plans'].items()),
                          candidates={i:dict(old_frames=len(x['old_noobs_indices']),new_frames=len(x['indices'])) for i,x in c['plans'].items() if x['group']=='candidate'},
                          labels_read=False,note='Metadata/source preflight only. Full video/input tensor audit is a separate user-executed stage.'),ensure_ascii=False,indent=2))


def read_labels(c):
    k.require_complete(c)
    if not ACCESS['may_score']:raise ValueError('只有summarize动作可以读取标签')
    lock_path=p.OLD/'01_manifests/dataset_lock.json';lock=json.loads(lock_path.read_text());path=lock_path.parent/'evaluation_labels.json'
    ACCESS['labels_allowed']=True
    try:
        if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签身份漂移')
        labels=json.loads(path.read_text())
    finally:ACCESS['labels_allowed']=False
    if any(i not in labels or labels[i] not in list('ABCD') for i in c['protocol']['indices']):raise ValueError('30题标签缺失或非法，不丢题')
    return labels,lock['labels_sha256']


def main():
    access_guard();ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['init','preflight','audit-inputs','run','summarize']);args=ap.parse_args();ACCESS['may_score']=args.action=='summarize'
    print('SPACE',p.disk_guard(),flush=True);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=context(args.action=='init')
        if args.action in ['init','preflight']:preflight(c)
        elif args.action=='audit-inputs':
            if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户终端确认后设置PRACTICAL_STAGE_CONFIRMED=1')
            audit_inputs(c)
        elif args.action=='run':
            if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户终端确认后设置PRACTICAL_STAGE_CONFIRMED=1')
            with (p.OLD/'cache/gpu.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);run(c)
        else:
            from leaf_qa30_analyze import summarize
            summarize(c)


if __name__=='__main__':
    # analyze模块必须复用本进程的标签访问闸门，避免脚本/模块双实例状态分裂。
    sys.modules['leaf_qa30_run']=sys.modules[__name__]
    try:main()
    except (OSError,RuntimeError,ValueError) as exc:print('停止：'+str(exc),file=sys.stderr);sys.exit(1)
