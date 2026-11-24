"""固定30题定位诊断入口：7候选、4锚点探测、19原样回退；没有QA或API动作。"""
import argparse
import io
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

import air_protocol as p
import condition_text30_reaudit_run as text_source
import leaf_probe as k

OUT=p.ROOT/k.VERSION
CODE=['leaf_probe.py','leaf_probe_run.py','leaf_probe_media.py','run_leaf_probe.sh','air_protocol.py','air_selection.py']


def access_guard():
    # 模型/视频仅允许本地run阶段读取；全程禁止API、密钥、GT、旧QA及人工位置。
    def hook(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:raise RuntimeError('本地定位禁止联网')
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            if path.name in ['.env','deepseek.env','evaluation_labels.json','predictions.json','units.jsonl'] or path.name.startswith('.env.') or 'oracle_evidence_diag_v1' in str(path):raise RuntimeError('禁止密钥/标签/旧QA/人工定位输入')
    sys.addaudithook(hook)


def cached_text(path,signature,clip):
    tv=json.loads(path.read_text());p.unseal(tv)
    if tv['signature']!=signature or len(tv['texts'])!=len(tv['vectors']) or len(tv['texts'])!=len(tv['token_chunks']):raise ValueError('旧文本缓存来源/行数异常')
    k.unit_rows(tv['vectors'])
    return tv


def context(init=False):
    if subprocess.check_output(['git','-C',str(p.PROJECT),'branch','--show-current'],text=True).strip()!=p.BRANCH:raise ValueError('分支不符')
    parent=text_source.context()
    if len(parent['units'])!=30:raise ValueError('先完成30题缓存离线重审')
    scope_path=text_source.OUT/'results/assistant_localization_readiness_v1.json'
    scope=json.loads(scope_path.read_text());packet=text_source.OUT/'results/review_packet.json'
    if scope['source_sha256']!=p.sha(packet):raise ValueError('诊断范围与审阅包不符')
    groups={name:scope[key] for name,key in [('candidate','visual_localization_candidates'),('anchor_only','visual_anchor_probe_only'),('fallback','retain_original_fallback')]}
    ids=parent['protocol']['indices'];joined=sum(groups.values(),[])
    if len(joined)!=30 or len(set(joined))!=30 or set(joined)!=set(ids) or [len(v) for v in groups.values()]!=[7,4,19]:raise ValueError('7/4/19范围重复或变化')
    pool,stages=p.load_rows();allrows={r['index']:r for r in pool};rows={i:allrows[i] for i in ids}
    first={r['index'] for r in stages['dev30']}
    native_path=p.ROOT/'candidate_native_uniform_fix1.json';native=json.loads(native_path.read_text());sig=p.digest(native)
    clip=p.model_identity(p.CLIP,['model.safetensors'])
    if native['clip']!=clip:raise ValueError('CLIP模型身份漂移')
    noobs_path=p.ROOT/'condition_clips_noobs_qa30/candidate.json';noobs=json.loads(noobs_path.read_text())
    if noobs['indices']!=ids:raise ValueError('原noobs30题身份不符')
    v2_path=p.ROOT/'condition_clips_v2/candidate.json';v2=json.loads(v2_path.read_text())
    if p.digest(v2)!=noobs['source_v2_signature'] or v2['clip']!=clip:raise ValueError('文本特征原编码器身份异常')
    files={str(path):p.sha(path) for path in [scope_path,packet,native_path,noobs_path,v2_path]}
    files.update(parent['protocol']['source_files'])
    meta={};features={};plans={};queries={};catalog={};video_stats={}
    for i,row in rows.items():
        if p.query(row)!=p.query(parent['source']['rows'][i]):raise ValueError('原问题/选项漂移')
        store=p.ROOT/('cache' if i in first else 'extension90_v1/source_cache')/sig
        mp=store/'video_metadata'/f'{row["video_id"]}.json';fp=store/'clip_features'/row['video_id']/'coarse.npz'
        m=json.loads(mp.read_text());st=Path(row['video_path']).stat()
        if m['signature']!=sig or m['video_id']!=row['video_id'] or native['videos'][row['video_id']]!=dict(path=row['video_path'],bytes=st.st_size,mtime_ns=st.st_mtime_ns):raise ValueError('视频/元数据漂移')
        k.time_bounds(m['fps'],m['frame_count'],None)
        plan=noobs['plans'][i]
        if plan['count']!=m['frame_count'] or plan['fps']!=m['fps'] or plan['indices']!=sorted(set(plan['indices'])) or not 1<=len(plan['indices'])<=32 or any(type(x)is not int or not 0<=x<m['frame_count'] for x in plan['indices']):raise ValueError('原回退帧异常')
        if v2['files'].get(str(mp))!=p.sha(mp) or noobs['files'].get(str(fp))!=p.sha(fp):raise ValueError('历史元数据/图像特征哈希漂移')
        meta[i]=m;features[i]=str(fp);plans[i]=plan;files[str(mp)]=p.sha(mp);files[str(fp)]=p.sha(fp)
        video_stats[row['video_id']]=native['videos'][row['video_id']]
        if i not in groups['fallback']:
            unit=parent['units'][i]
            if unit['parsed'] is None:raise ValueError('活动诊断题无合法缓存查询')
            queries[i]=unit['parsed']['queries']
            if [q['event_id'] for q in queries[i]]!=[e['event_id'] for e in parent['source']['fixed'][i]['events']]:raise ValueError('事件ID/顺序不符')
        tp=p.ROOT/'condition_clips_v2/text_features'/f'{i}.json'
        if noobs['files'].get(str(tp))!=p.sha(tp):raise ValueError('旧文本来源哈希漂移')
        tv=cached_text(tp,noobs['source_v2_signature'],clip);files[str(tp)]=p.sha(tp)
        for n,text in enumerate(tv['texts']):
            # 仅精确字符串与锁定CLIP编码协议复用，绝不模糊匹配“差不多”的查询。
            catalog.setdefault(text,dict(path=str(tp),sha256=p.sha(tp),row=n,vector=tv['vectors'][n],token_chunks=tv['token_chunks'][n],historical_file_wall_s=tv['wall_s']))
    flat=list(dict.fromkeys(q['text'] for i in ids if i in queries for q in queries[i]))
    reused={text:catalog[text] for text in flat if text in catalog}
    folder=Path(__file__).parent
    protocol=dict(version=k.VERSION,indices=ids,groups=groups,scope_sha256=p.sha(scope_path),source_signature=parent['signature'],image_signature=sig,
                  source_records={i:u['record_sha256'] for i,u in parent['units'].items()},source_files=files,video_stats=video_stats,
                  queries=queries,clip=clip,reusable_text=reused,texts=flat,
                  code={name:p.sha(folder/name) for name in CODE},
                  selection=dict(global_anchors=8,windows=6,seconds=12,frames_per_window=4,max_frames=32,raw_hits_per_leaf=3),
                  embedding=dict(dtype='float16',dimension=768,max_chunk_tokens=77,aggregation='mean unit chunk vectors then unit normalization; persist FP16 then renormalize',local_only=True),
                  policy='Fixed seen-development diagnosis, scope manually reviewed by question type, NOT general learned routing. '
                         'Frozen queries only. No GT/manual ranges/API/Mage/subtitles/QA. Peaks, numeric order and shared windows are NOT semantic evidence certification. '
                         'Anchor-only and fallback QA inputs remain unchanged; no automatic QA selection commit.')
    if init:k.write_once(OUT/'candidate.json',protocol)
    if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；来源/代码漂移不覆盖')
    c=dict(signature=p.digest(protocol),protocol=protocol,parent=parent,rows=rows,meta=meta,features=features,plans=plans)
    return c


def feature(c,i):
    path=Path(c['features'][i])
    if p.sha(path)!=c['protocol']['source_files'][str(path)]:raise ValueError('图像缓存漂移')
    with np.load(path,allow_pickle=False) as z:
        indices=z['indices'].tolist();f=z['features'].astype(np.float32);info=json.loads(str(z['meta'].item()));cost=float(z['wall_s'])
    if info!=dict(signature=c['protocol']['image_signature'],video_id=c['rows'][i]['video_id'],kind='coarse_1fps',indices_sha256=p.digest(indices)):raise ValueError('图像特征元数据异常')
    m=c['meta'][i]
    if indices!=sorted(set(indices)) or any(type(x)is not int or not 0<=x<m['frame_count'] for x in indices) or len(indices)!=len(f) or not math.isfinite(cost) or cost<0:raise ValueError('图像帧/成本非法')
    return indices,k.unit_rows(f),cost


def text_path(text):return OUT/'text_features'/(p.digest(text)+'.json')


def read_text(c,text):
    path=text_path(text);u=json.loads(path.read_text());p.unseal(u)
    if u['signature']!=c['signature'] or u['text']!=text:raise ValueError('文本断点身份漂移')
    k.unit_rows([u['vector']])
    if not u['token_chunks'] or any(not 3<=len(x)<=77 for x in u['token_chunks']):raise ValueError('文本chunk非法')
    old=c['protocol']['reusable_text'].get(text)
    if old:
        if u['origin']!='reused_exact' or u['vector']!=old['vector'] or u['token_chunks']!=old['token_chunks'] or u['source']!={n:old[n] for n in ['path','sha256','row','historical_file_wall_s']}:raise ValueError('旧文本复用来源异常')
    elif u['origin']!='new_local_clip':raise ValueError('新文本来源非法')
    return u


def encode(c):
    from tqdm import tqdm
    texts=c['protocol']['texts'];done={t for t in texts if text_path(t).exists()}
    for t in done:read_text(c,t)
    model=tok=torch=None
    with tqdm(total=len(texts),initial=len(done),desc='CLIP叶查询编码',unit='查询',dynamic_ncols=True) as bar:
        for text in texts:
            if text in done:continue
            p.disk_guard();old=c['protocol']['reusable_text'].get(text)
            if old:
                unit=dict(signature=c['signature'],text=text,origin='reused_exact',vector=old['vector'],token_chunks=old['token_chunks'],encoding_wall_s=0.,source={n:old[n] for n in ['path','sha256','row','historical_file_wall_s']})
            else:
                if model is None:
                    import torch,transformers
                    from transformers import CLIPModel,CLIPProcessor
                    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用，不自动改设备/安装')
                    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
                    model=CLIPModel.from_pretrained(str(p.CLIP),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
                    tok=CLIPProcessor.from_pretrained(str(p.CLIP),local_files_only=True).tokenizer
                    k.write_once(OUT/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype)))
                ids=tok(text,add_special_tokens=False,truncation=False)['input_ids'];chunks=p.text_chunks(ids,tok.bos_token_id,tok.eos_token_id,77);width=max(map(len,chunks))
                pad=[x+[tok.pad_token_id]*(width-len(x)) for x in chunks];mask=[[1]*len(x)+[0]*(width-len(x)) for x in chunks]
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():v=model.get_text_features(input_ids=torch.tensor(pad,device='cuda'),attention_mask=torch.tensor(mask,device='cuda'),return_dict=True)
                if not isinstance(v,torch.Tensor):v=v.pooler_output
                if v.shape!=(len(chunks),768) or not torch.isfinite(v).all() or (v.float().norm(dim=-1)<=1e-8).any():raise ValueError('文本编码数值异常')
                v=v.float();v=(v/v.norm(dim=-1,keepdim=True)).mean(0)
                if v.norm()<=1e-8:raise ValueError('文本均值向量为零')
                vector=(v/v.norm()).cpu().numpy().astype(np.float16).astype(float).tolist();torch.cuda.synchronize()
                unit=dict(signature=c['signature'],text=text,origin='new_local_clip',vector=vector,token_chunks=chunks,encoding_wall_s=time.perf_counter()-start,source=None)
            k.write_once(text_path(text),p.seal(unit));read_text(c,text);bar.update(1)


def calculate(c,i):
    group=next(g for g,ids in c['protocol']['groups'].items() if i in ids);m=c['meta'][i]
    base=dict(signature=c['signature'],index=i,group=group,original_noobs_indices=c['plans'][i]['indices'],
              original_qa_selection_unchanged=True,QA_selection_committed=False)
    if group=='fallback':return dict(**base,status='retained_fallback',result=None,queries=[],text_files={},feature_file=None)
    ids,f,cost=feature(c,i);queries=c['protocol']['queries'][i];texts=[read_text(c,q['text']) for q in queries]
    fixed=c['parent']['source']['fixed'][i];bounds=(fixed.get('time_range') or {}).get('bounds')
    from air_selection import uniform_indices
    low,high=k.time_bounds(m['fps'],m['frame_count'],bounds);lo=math.ceil(low*m['fps']-1e-8);hi=min(m['frame_count']-1,math.floor(high*m['fps']+1e-8))
    anchors=[lo+n for n in uniform_indices(hi-lo+1,8)]
    common=dict(queries=queries,text_files={q['text']:p.sha(text_path(q['text'])) for q in queries},feature_file=dict(path=c['features'][i],sha256=p.sha(c['features'][i]),historical_image_encoding_wall_s=cost))
    try:
        result,raw,relative=k.locate(ids,f,[t['vector'] for t in texts],m['fps'],m['frame_count'],[q['event_id'] for q in queries],fixed['given_edges'],fixed['relation'],anchors,bounds)
        # 只保存曲线，不复制图像特征；每个叶事件的原始和相对分数可完整复查。
        curve=OUT/'curves'/f'{i}.npz';arrays=dict(indices=np.asarray(ids,dtype=np.int64),cosine=raw,relative=relative)
        if curve.exists():
            with np.load(curve,allow_pickle=False) as old:
                if set(old.files)!=set(arrays) or any(not np.array_equal(old[n],x) for n,x in arrays.items()):raise ValueError('曲线断点漂移')
        else:
            out=io.BytesIO();np.savez_compressed(out,**arrays);k.atomic_bytes(curve,out.getvalue())
        return dict(**base,**common,status='localized_candidates_only',result=result,curves_sha256=p.sha(curve))
    except k.Unavailable as exc:
        return dict(**base,**common,status='retrieval_unavailable_keep_fallback',result=None,reason=str(exc))


def read_location(c,i):
    path=OUT/'locations'/f'{i}.json';u=json.loads(path.read_text());p.unseal(u)
    if u!=p.seal(calculate(c,i)):raise ValueError('定位断点回放漂移：'+i)
    return u


def locate(c):
    from tqdm import tqdm
    ids=c['protocol']['indices'];done={i for i in ids if (OUT/'locations'/f'{i}.json').exists()}
    for i in done:read_location(c,i)
    with tqdm(total=30,initial=len(done),desc='固定30题叶事件定位/回退',unit='题',dynamic_ncols=True) as bar:
        for i in ids:
            if i in done:continue
            p.disk_guard();k.write_once(OUT/'locations'/f'{i}.json',p.seal(calculate(c,i)));bar.update(1)


def preflight(c):
    # 纯CPU核对所有图像缓存，不解码、不读取模型权重；图像缓存不符即停。
    for i in c['protocol']['indices']:feature(c,i)
    texts=c['protocol']['texts'];cached=sum(text_path(t).exists() for t in texts)
    for t in texts:
        if text_path(t).exists():read_text(c,t)
    print(json.dumps(dict(N=30,groups={g:len(v) for g,v in c['protocol']['groups'].items()},unique_queries=len(texts),exact_old_text_reuse=len(c['protocol']['reusable_text']),completed_texts=cached,missing_new_texts=sum(not text_path(t).exists() and t not in c['protocol']['reusable_text'] for t in texts),new_image_encoding=0,QA=0),ensure_ascii=False,indent=2))


def main():
    access_guard()
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['init','preflight','encode','locate','render','summarize']);args=ap.parse_args()
    print('SPACE',p.disk_guard(),flush=True);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=context(args.action=='init')
        if args.action in ['init','preflight']:preflight(c)
        elif args.action=='encode':
            if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户终端确认后设置PRACTICAL_STAGE_CONFIRMED=1')
            with (p.OLD/'cache/gpu.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);encode(c)
        elif args.action=='locate':locate(c)
        else:
            import leaf_probe_media as media
            if args.action=='render':media.render(c)
            else:media.summarize(c)


if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError) as exc:print('停止：'+str(exc),file=sys.stderr);sys.exit(1)
