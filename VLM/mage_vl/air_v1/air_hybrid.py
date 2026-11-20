"""单独的g16+l16候选；旧AIR代码与结果保持原位，便于回退。"""
import argparse,fcntl,json,math,os,time
from pathlib import Path
import air_protocol as p
from air_selection import parse_score,uniform_indices
from air_mage_adapter import selected_video_inputs,assert_inputs_equal

VERSION='hybrid_g16l16_v1'
OUT=p.ROOT/VERSION
ARMS=('uniform32','air_cap32','hybrid32')

def choose(base,scores,similarity,fps):
    """不读真值；全局16、跨2秒时间簇轮选局部最多16、不足补uniform。"""
    if len(base)!=32 or base!=sorted(set(base)) or not math.isfinite(fps) or fps<=0:raise ValueError('基线/fps非法')
    anchors=[base[i] for i in uniform_indices(32,16)]
    pool=sorted(i for i,s in scores.items() if s in [4,5] and i not in anchors)
    if any(i<0 or not math.isfinite(similarity[i]) for i in pool):raise ValueError('局部候选非法')
    clusters=[]
    for i in pool:
        if not clusters or (i-clusters[-1][-1])/fps>2:clusters.append([i])
        else:clusters[-1].append(i)
    clusters.sort(key=lambda g:(-max(scores[i] for i in g),-max(similarity[i] for i in g),g[0]))
    local=[];taken=[[] for g in clusters]
    while len(local)<16:
        advanced=False
        for j,g in enumerate(clusters):
            available=[i for i in g if i not in taken[j]]
            if not available:continue
            # 先取代表；以后优先增加时间跨度，而非持续堆同一瞬间。
            def rank(i):
                distance=min(abs(i-k) for k in taken[j]) if taken[j] else 0
                return (-distance,-scores[i],-similarity[i],i)
            i=min(available,key=rank);local.append(i);taken[j].append(i);advanced=True
            if len(local)==16:break
        if not advanced:break
    used=set(anchors+local);fill=[]
    for i in base:
        if len(used)==32:break
        if i not in used:fill.append(i);used.add(i)
    final=sorted(used)
    if len(final)!=32 or not set(anchors).issubset(final):raise AssertionError('必须恰好32并保留全局锚点')
    return dict(indices=final,anchors=anchors,local=local,uniform_fill=fill,clusters=clusters)

def validate_record(c,u,seen):
    p.unseal(u);idx=u['index'];plan=c['plans'].get(idx)
    if plan is None or idx in seen or u['signature']!=c['signature'] or u['plan_sha256']!=p.digest(plan):raise ValueError('断点身份/重复非法')
    if u['indices']!=plan['indices'] or u['pred']!=p.parse_answer(u['raw_text']) or u['native_equal'] is not True:raise ValueError('断点输入/答案不符')
    if len(u['tensor_sha256'])!=64 or u['input_tokens']<=u['visual_tokens'] or u['visual_tokens']<=0:raise ValueError('断点审计不符')
    for k in ['inference_s','preprocess_s','peak_gib']:
        if not math.isfinite(u[k]) or u[k]<0:raise ValueError('断点成本非法')
    source=plan['reuse_arm']
    if source:
        old=c['source']['units'][f'{idx}:{source}']
        if u['origin']!='reused_'+source or u['source_record']!=old['record_sha256'] or u['raw_text']!=old['raw_text'] or u['new_calls']!=0 or u['inference_s']!=0 or u['generated_tokens']!=0:raise ValueError('复用来源不符')
    elif u['origin']!='generated' or u['source_record'] is not None or u['new_calls']!=1 or not 1<=u['generated_tokens']<=32:raise ValueError('新增生成来源不符')

def context(init=False):
    src=p.context('dev30')
    if not src['expected'].issubset(src['units']):raise ValueError('原90题未完整，禁止补跑旧结果')
    plans={};files={};invalid=0
    for row in src['rows']:
        idx=row['index'];sp=src['store']/'selections'/f'{idx}.json';sel=json.loads(sp.read_text());p.unseal(sel);files[str(sp)]=p.sha(sp)
        scores={}
        for frame in sel['analyzed_indices']:
            path=src['store']/'scores'/idx/f'{frame}.json';u=json.loads(path.read_text());p.unseal(u)
            if (u['signature']!=src['signature'] or u['index']!=idx or u['frame_index']!=frame or u['score']!=parse_score(u['raw_text']) or u['prompt_sha256']!=p.digest(p.ANALYSIS.format(query=p.query(row)))):raise ValueError('历史评分来源非法')
            records=[r['score_records'][str(frame)] for r in sel['rounds'] if str(frame) in r['score_records']]
            if records!=[dict(score=u['score'],raw_text=u['raw_text'])]:raise ValueError('评分轨迹不匹配')
            scores[frame]=u['score'];invalid+=u['score'] is None;files[str(path)]=p.sha(path)
        base=src['units'][f'{idx}:uniform32'];v=choose(base['indices'],scores,{int(i):s for i,s in sel['observed_similarity'].items()},sel['fps'])
        v.update(index=idx,video_id=row['video_id'],fps=sel['fps'],frame_count=sel['total_frames'],reuse_arm=None)
        for arm in ['uniform32','air_cap32']:
            if v['indices']==src['units'][f'{idx}:{arm}']['indices']:v['reuse_arm']=arm;break
        plans[idx]=v
    summary=src['stage_dir']/'summary.json';files[str(summary)]=p.sha(summary)
    scripts={n:p.sha(Path(__file__).parent/n) for n in ['air_hybrid.py','run_air_hybrid.sh']}
    protocol=dict(version=VERSION,source_signature=src['signature'],source_files=files,source_answers={k:u['record_sha256'] for k,u in sorted(src['units'].items())},scripts=scripts,plans=plans,
        rule='16 native-uniform32 anchors + <=16 positive local, consecutive gap<=2s clusters, score/sim/frame ranked round-robin; temporal-farthest within cluster; uniform fill to32',
        inherited_invalid_scores=invalid,scope='fixed existing dev30,posthoc development; no subtitles/API/new scoring')
    if init:p.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；新候选不覆盖旧身份')
    c=dict(source=src,plans=plans,signature=p.digest(protocol),protocol=protocol,units={})
    for u in p.read_journal(OUT/'units.jsonl'):validate_record(c,u,c['units']);c['units'][u['index']]=u
    return c

def tensor_sha(actual):
    import hashlib,torch
    h=hashlib.sha256()
    for k in sorted(actual):
        t=actual[k].detach().cpu().contiguous()
        h.update(p.digest([k,str(t.dtype),list(t.shape)]).encode());h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(c):
    if len(c['units'])==90:print('90题已完成，跳过模型加载；请summarize');return
    # 只有用户run进入这里；不加载CLIP，也不重新生成逐帧评分。
    import random,numpy as np,torch,transformers
    from transformers import AutoProcessor,AutoModelForCausalLM
    from decord import VideoReader,cpu
    from tqdm import tqdm
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True)
    for part in [processor.video_processor,processor.image_processor]:part.min_pixels=3136;part.max_pixels=150000
    model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
    p.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype)))
    cached={}
    with tqdm(total=90,initial=len(c['units']),desc='Mage 全局16+局部16 dev30',unit='题',dynamic_ncols=True) as bar:
        for row in c['source']['rows']:
            idx=row['index']
            if idx in c['units']:continue
            p.disk_guard();start=time.perf_counter();plan=c['plans'][idx]
            if row['video_id'] not in cached:
                cached.clear();vr=VideoReader(row['video_path'],ctx=cpu(0));cached[row['video_id']]=vr
            vr=cached[row['video_id']]
            if len(vr)!=plan['frame_count'] or abs(float(vr.get_avg_fps())-plan['fps'])>1e-8:raise ValueError('视频元数据漂移')
            text=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':p.prompt(row)}]}],tokenize=False,add_generation_prompt=True)
            def native():return processor(text=[text],videos=[row['video_path']],video_backend='frames',num_frames=32,max_pixels=150000,return_tensors='pt',padding=True)
            def selected(inds):
                rgb=vr.get_batch(inds).asnumpy()
                with selected_video_inputs(processor,rgb,inds,plan['fps'],plan['frame_count'],video_path=row['video_path']):return native()
            # 每题均匀原生与适配逐tensor回归，不关闭旧输入验收门槛。
            base=c['source']['units'][f'{idx}:uniform32'];actual=native();adapt=selected(base['indices']);assert_inputs_equal(actual,adapt);del adapt
            if p.digest(actual['input_ids'].tolist())!=base['input_sha256']:raise ValueError('历史基线输入身份不符')
            if actual['image_grid_thw'].tolist()!=base['visual']['grid_thw']:raise ValueError('历史基线grid漂移')
            use=actual if plan['indices']==base['indices'] else selected(plan['indices'])
            if use['patch_positions'][:,0].unique_consecutive().tolist()!=plan['indices']:raise ValueError('新帧号未保持')
            grid=use['image_grid_thw'].tolist();visual=sum(t*h*w for t,h,w in grid)//processor.spatial_merge_size**2
            ids_sha=p.digest(use['input_ids'].tolist());th=tensor_sha(use);prep=time.perf_counter()-start
            arm=plan['reuse_arm'];source_record=None;elapsed=peak=0.;generated=0
            if arm:
                old=c['source']['units'][f'{idx}:{arm}']
                if ids_sha!=old['input_sha256'] or grid!=old['visual']['grid_thw'] or visual!=old['visual']['n_tok']:raise ValueError('复用输入身份不符')
                raw=old['raw_text'];origin='reused_'+arm;source_record=old['record_sha256'];calls=0
            else:
                gpu={k:v.to(device=model.device,dtype=model.dtype) if k=='pixel_values' else v.to(model.device) for k,v in use.items()}
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
                with torch.inference_mode():answer=model.generate(**gpu,max_new_tokens=32,do_sample=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-before;peak=torch.cuda.max_memory_allocated()/1024**3
                generated=int(answer.shape[1]-use['input_ids'].shape[1]);raw=processor.tokenizer.decode(answer[0,use['input_ids'].shape[1]:],skip_special_tokens=True).strip();origin='generated';calls=1
                del gpu,answer
            u=p.seal(dict(signature=c['signature'],index=idx,plan_sha256=p.digest(plan),indices=plan['indices'],raw_text=raw,pred=p.parse_answer(raw),origin=origin,source_record=source_record,new_calls=calls,generated_tokens=generated,input_tokens=int(use['input_ids'].shape[1]),visual_tokens=visual,input_ids_sha256=ids_sha,tensor_sha256=th,native_equal=True,inference_s=elapsed,peak_gib=peak,preprocess_s=prep))
            validate_record(c,u,c['units']);p.append(OUT/'units.jsonl',u);c['units'][idx]=u;bar.update(1);bar.set_postfix(index=idx,origin=origin,answer=u['pred'] or 'INVALID')
            print('落盘',idx,origin,u['pred'],flush=True);del use,actual

def summarize(c):
    if len(c['units'])!=90:raise ValueError('未完成，不读取GT')
    source=c['source'];path=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((path.parent/'dataset_lock.json').read_text())
    if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签身份变化')
    labels=json.loads(path.read_text());rows=source['rows']
    import sys
    sys.path.insert(0,str(p.PROJECT/'VLM/mage_vl/practical'))
    from practical_analyze import comparison
    preds={a:{r['index']:(c['units'][r['index']] if a=='hybrid32' else source['units'][f'{r["index"]}:{a}'])['pred'] for r in rows} for a in ARMS}
    diagnostic=set('265 378 450 452 459 486 704 324 379 418 428 484 594 600'.split())
    def group(items):
        if not items:return dict(N=0)
        return dict(N=len(items),methods={a:dict(correct=sum(preds[a][r['index']]==labels[r['index']] for r in items),accuracy=sum(preds[a][r['index']]==labels[r['index']] for r in items)/len(items),invalid=sum(preds[a][r['index']] not in list('ABCD') for r in items)) for a in ARMS},versus_uniform=comparison(items,preds['hybrid32'],preds['uniform32'],labels),versus_old_air=comparison(items,preds['hybrid32'],preds['air_cap32'],labels))
    units=list(c['units'].values());bad=sum(u['pred'] not in list('ABCD') for u in units)
    report=dict(signature=c['signature'],N=90,audit=dict(passed=not bad,invalid_answers=bad,source_invalid_scores=c['protocol']['inherited_invalid_scores'],source_warning='Old audit remains failed; malformed scores explicitly excluded from local representatives, no new scoring'),overall=group(rows),by_duration={d:group([r for r in rows if r['duration']==d]) for d in ['medium','long']},diagnostic14=group([r for r in rows if r['index'] in diagnostic]),other76=group([r for r in rows if r['index'] not in diagnostic]),cost=dict(new_QA_calls=sum(u['new_calls'] for u in units),reused_answers=sum(not u['new_calls'] for u in units),QA_inference_s=sum(u['inference_s'] for u in units),preprocess_s=sum(u['preprocess_s'] for u in units),visual_tokens=sum(u['visual_tokens'] for u in units),input_tokens=sum(u['input_tokens'] for u in units),peak_gib=max(u['peak_gib'] for u in units)),historical_selector_cost=json.loads((source['stage_dir']/'summary.json').read_text())['cost'],note='Posthoc development,not independent test. Cache reuse saves this rerun,not zero deployment selector cost.')
    details=[dict(index=r['index'],gt=labels[r['index']],predictions={a:preds[a][r['index']] for a in ARMS},record_sha256=c['units'][r['index']]['record_sha256']) for r in rows]
    p.write_once(OUT/'dev30/predictions.json',details);p.write_once(OUT/'dev30/summary.json',report);p.write_once(OUT/'dev30/units_snapshot.json',units)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('汇总路径:',OUT/'dev30/summary.json')
    if bad:raise ValueError('非法答案保留，请诊断，不自动重跑')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','preflight','run','summarize']);args=parser.parse_args()
    print('SPACE',p.disk_guard());c=context(args.action=='init')
    print('固定90题；可复用',sum(bool(x['reuse_arm']) for x in c['plans'].values()),'最多新增',sum(not x['reuse_arm'] for x in c['plans'].values()),'已完成',len(c['units']))
    if args.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认后才run')
        with (p.OLD/'cache/gpu.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);c=context();run(c)
    elif args.action=='summarize':summarize(c)
    elif args.action=='init':print('CPU登记完成，未运行模型')

if __name__=='__main__':main()
