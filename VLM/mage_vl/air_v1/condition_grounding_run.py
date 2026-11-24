"""纯文本解析→CLIP叶定位→独立QA；自动路径不读标注/人工位置。"""
import argparse,fcntl,json,os,time,subprocess,math
from pathlib import Path
import numpy as np
import air_protocol as p
import condition_grounding as k
import condition_clips as old

OUT=p.ROOT/k.VERSION
CODE=['condition_grounding.py','condition_grounding_run.py','condition_grounding_qa.py','run_condition_grounding.sh']

def context(init=False):
 if subprocess.check_output(['git','-C',str(p.PROJECT),'branch','--show-current'],text=True).strip()!=p.BRANCH:raise ValueError('分支不符')
 source_path=p.ROOT/'condition_clips_noobs_qa30/candidate.json';source=json.loads(source_path.read_text());ids=source['indices']
 pool,stages=p.load_rows();lookup={r['index']:r for r in pool};rows={i:lookup[i] for i in ids}
 base_path=p.ROOT/'candidate_native_uniform_fix1.json';base=json.loads(base_path.read_text());sig=p.digest(base)
 if p.model_identity(p.MODEL)!=base['mage'] or p.model_identity(p.CLIP,['model.safetensors'])!=base['clip']:raise ValueError('模型身份漂移')
 for name,sha in base['sources'].items():
  if p.sha(p.OLD/'01_manifests'/name)!=sha:raise ValueError('输入清单漂移')
 first={r['index'] for r in stages['dev30']};features={};meta={};files={}
 for i,row in rows.items():
  st=Path(row['video_path']).stat()
  if base['videos'][row['video_id']]!=dict(path=row['video_path'],bytes=st.st_size,mtime_ns=st.st_mtime_ns):raise ValueError('视频身份漂移')
  store=p.ROOT/('cache' if i in first else 'extension90_v1/source_cache')/sig
  mp=store/'video_metadata'/f'{row["video_id"]}.json';fp=store/'clip_features'/row['video_id']/'coarse.npz';meta[i]=json.loads(mp.read_text());features[i]=str(fp)
  if meta[i]['signature']!=sig or meta[i]['video_id']!=row['video_id']:raise ValueError('元数据来源异常')
  files[str(mp)]=p.sha(mp);files[str(fp)]=p.sha(fp)
 folder=Path(__file__).parent
 protocol=dict(version=k.VERSION,indices=ids,parse5=list(k.PARSE5),questions={i:p.query(r) for i,r in rows.items()},source_candidate_sha=p.sha(source_path),source_signature=sig,source_files=files,
  mage=base['mage'],clip=base['clip'],code={n:p.sha(folder/n) for n in CODE},parser=dict(system=k.SYSTEM,max_new_tokens=512,max_attempts=2,greedy=True,seed=0),
  budget=dict(global_frames=8,clips=6,clip_seconds=12,clip_frames=4,max_frames=32,max_pixels=150000,min_pixels=3136,qa_max_new_tokens=32),
  policy='Only question+all options to parser; no GT/manual ranges/old answers. Independent/known_seq/unknown_order/spatial only; unsupported retrieval falls back exactly to noobs. HiMu-inspired limited adaptation,not reproduction; no semantic certification.')
 if init:p.write_once(OUT/'candidate.json',protocol)
 if not (OUT/'candidate.json').exists() or json.loads((OUT/'candidate.json').read_text())!=protocol:raise ValueError('先init；来源/代码漂移不覆盖')
 c=dict(signature=p.digest(protocol),protocol=protocol,rows=rows,meta=meta,features=features,oldplans=source['plans'],parses={})
 for i in ids:
  path=OUT/'parses'/f'{i}.json'
  if not path.exists():continue
  u=json.loads(path.read_text());p.unseal(u)
  if u['signature']!=c['signature'] or u['index']!=i:raise ValueError('解析身份漂移')
  d,error=k.parse(u['raw_text'],rows[i])
  if u['parsed']!=d or u['error']!=error:raise ValueError('解析回放变化')
  for fn,sha in u['attempt_files'].items():
   if not Path(fn).resolve().is_relative_to(OUT.resolve()) or p.sha(fn)!=sha:raise ValueError('解析原始调用漂移')
  c['parses'][i]=u
 return c

def ready(c,ids):
 if not set(ids)<=c['parses'].keys():raise ValueError('先完成解析阶段')
 bad=[i for i in ids if c['parses'][i]['parsed'] is None]
 if bad:raise ValueError('解析失败保留，先诊断：'+','.join(bad))

def parse_stage(c,stage):
 ids=list(k.PARSE5) if stage=='parse5' else c['protocol']['indices']
 if stage=='parse30':
  ready(c,k.PARSE5)
  if os.environ.get('PRACTICAL_PARSE_REVIEWED')!='1':raise ValueError('先审阅parse5再设置PRACTICAL_PARSE_REVIEWED=1')
 if all(i in c['parses'] for i in ids):print('解析已落盘，跳过模型加载');return
 import torch,transformers,random
 from transformers import AutoProcessor,AutoModelForCausalLM
 from tqdm import tqdm
 if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 proc=AutoProcessor.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True);tok=proc.tokenizer
 model=AutoModelForCausalLM.from_pretrained(str(p.MODEL),trust_remote_code=True,local_files_only=True,torch_dtype='auto',device_map={'':0}).eval()
 p.write_once(OUT/'parser_runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype=str(model.dtype)))
 with tqdm(total=len(ids),initial=sum(i in c['parses'] for i in ids),desc='纯文本条件解析 '+stage,unit='题',dynamic_ncols=True) as bar:
  for i in ids:
   if i in c['parses']:continue
   p.disk_guard();row=c['rows'][i];text=k.parser_prompt(row);saved={};record=None
   for attempt in range(2):
    path=OUT/'parse_calls'/i/f'{attempt}.json'
    chat=proc.apply_chat_template([{'role':'system','content':k.SYSTEM},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    # 仅tokenizer编码纯文本；没有video/image占位或视觉张量。
    inputs=tok([chat],return_tensors='pt',padding=True)
    if set(inputs)-{'input_ids','attention_mask','token_type_ids'} or '<|vision_start|>' in chat:raise ValueError('解析输入含视觉占位')
    if path.exists():
     record=json.loads(path.read_text());p.unseal(record)
     if record['signature']!=c['signature'] or record['index']!=i or record['attempt']!=attempt or record['prompt_sha']!=p.digest(chat) or record['input_ids_sha']!=p.digest(inputs['input_ids'].tolist()):raise ValueError('解析调用断点漂移')
     parsed,error=k.parse(record['raw_text'],row)
     if record['parsed']!=parsed or record['error']!=error:raise ValueError('解析调用重放变化')
    else:
     gpu={n:t.to(model.device) for n,t in inputs.items()};torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
     with torch.inference_mode():answer=model.generate(**gpu,max_new_tokens=512,do_sample=False)
     torch.cuda.synchronize();elapsed=time.perf_counter()-start;n=int(answer.shape[1]-inputs['input_ids'].shape[1]);raw=tok.decode(answer[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip();parsed,error=k.parse(raw,row)
     record=p.seal(dict(signature=c['signature'],index=i,attempt=attempt,prompt_sha=p.digest(chat),input_ids_sha=p.digest(inputs['input_ids'].tolist()),input_tokens=int(inputs['input_ids'].shape[1]),generated_tokens=n,at_cap=n==512,raw_text=raw,parsed=parsed,error=error,inference_s=elapsed,peak_gib=torch.cuda.max_memory_allocated()/1024**3,pixel_values_supplied=False))
     p.write_once(path,record);del gpu,answer
    saved[str(path)]=p.sha(path)
    if record['parsed'] is not None:break
    text=k.parser_prompt(row)+'\nFORMAT REPAIR: Previous parse failed: '+str(record['error'])+'. Return only the schema with short fields. Do not answer.'
   result=p.seal(dict(signature=c['signature'],index=i,raw_text=record['raw_text'],parsed=record['parsed'],error=record['error'],attempt_files=saved))
   p.write_once(OUT/'parses'/f'{i}.json',result);c['parses'][i]=result;bar.update(1);bar.set_postfix(index=i,valid=record['parsed'] is not None)
 print('解析完成；请summarize-parse审阅。schema合法不等于语义正确。')

def summarize_parse(c,stage):
 ids=list(k.PARSE5) if stage=='parse5' else c['protocol']['indices']
 if not set(ids)<=c['parses'].keys():raise ValueError('解析未完整')
 details=[];calls=[]
 for i in ids:
  u=c['parses'][i];d=u['parsed'];details.append(dict(index=i,question=c['rows'][i]['question'],options=c['rows'][i]['bodies'],parse=d,error=u['error'],raw_text=u['raw_text'],fallback=k.fallback_reason(d) if d else 'invalid_parse',semantic_review='pending_not_certified'))
  for fn in u['attempt_files']:calls.append(json.loads(Path(fn).read_text()))
 from collections import Counter
 report=dict(signature=c['signature'],stage=stage,N=len(ids),schema_valid=sum(d['parse'] is not None for d in details),evidence=dict(Counter((d['parse'] or {}).get('evidence','invalid') for d in details)),relations=dict(Counter((d['parse'] or {}).get('relation','invalid') for d in details)),unsupported_or_invalid={d['index']:d['fallback'] for d in details if d['fallback']},semantic_pass=None,cost=dict(calls=len(calls),generated_tokens=sum(x['generated_tokens'] for x in calls),inference_s=sum(x['inference_s'] for x in calls),at_cap=sum(x['at_cap'] for x in calls)),note='Text-only parsing,zero video evidence,zero answers; structural checks are not semantic certification.')
 p.write_once(OUT/'parse_review'/stage/'summary.json',report);p.write_once(OUT/'parse_review'/stage/'review_packet.json',details)
 print(json.dumps(report,ensure_ascii=False,indent=2));print('解析审阅:',OUT/'parse_review'/stage/'review_packet.json');ready(c,ids)

def ground(c):
 ready(c,c['protocol']['indices'])
 if os.environ.get('PRACTICAL_PARSE_REVIEWED')!='1':raise ValueError('先审阅parse30再确认定位阶段')
 import torch,transformers
 from transformers import CLIPModel,CLIPProcessor
 from tqdm import tqdm
 clip=tok=None
 for i in tqdm(c['protocol']['indices'],desc='条件化叶事件定位',unit='题',dynamic_ncols=True):
  path=OUT/'grounding'/f'{i}.json'
  if path.exists():continue
  p.disk_guard();d=c['parses'][i]['parsed'];base=c['oldplans'][i];m=c['meta'][i];reason=k.fallback_reason(d);result=None;text_calls=0;wall=0.;queries=k.searches(d)
  if reason is None:
   fp=Path(c['features'][i]);tp=OUT/'leaf_features'/f'{i}.json'
   if tp.exists():
    tv=json.loads(tp.read_text());p.unseal(tv)
    if tv['signature']!=c['signature'] or tv['texts']!=queries:raise ValueError('叶查询缓存漂移')
   else:
    if clip is None:
     if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
     clip=CLIPModel.from_pretrained(str(p.CLIP),local_files_only=True,torch_dtype=torch.float16).to('cuda').eval();tok=CLIPProcessor.from_pretrained(str(p.CLIP),local_files_only=True).tokenizer
    start=time.perf_counter();vecs=[];chunks_record=[]
    for text in queries:
     chunks=p.text_chunks(tok(text,add_special_tokens=False,truncation=False)['input_ids'],tok.bos_token_id,tok.eos_token_id,77);maximum=max(map(len,chunks));pad=[x+[tok.pad_token_id]*(maximum-len(x)) for x in chunks];mask=[[1]*len(x)+[0]*(maximum-len(x)) for x in chunks]
     with torch.inference_mode():v=clip.get_text_features(input_ids=torch.tensor(pad,device='cuda'),attention_mask=torch.tensor(mask,device='cuda'),return_dict=True)
     if not isinstance(v,torch.Tensor):v=v.pooler_output
     if v.shape!=(len(chunks),768) or not torch.isfinite(v).all() or (v.float().norm(dim=-1)<=1e-8).any():raise ValueError('叶查询向量异常')
     v=v.float();v=(v/v.norm(dim=-1,keepdim=True)).mean(0)
     if v.norm()<=1e-8:raise ValueError('叶均值零向量')
     vecs.append((v/v.norm()).cpu().numpy().astype('float16').astype(float).tolist());chunks_record.append(chunks)
    torch.cuda.synchronize();tv=p.seal(dict(signature=c['signature'],texts=queries,vectors=vecs,token_chunks=chunks_record,wall_s=time.perf_counter()-start));p.write_once(tp,tv)
   vectors=np.asarray(tv['vectors'],dtype='float32');norm=np.linalg.norm(vectors,axis=1,keepdims=True)
   if not np.isfinite(vectors).all() or (norm<=1e-8).any():raise ValueError('缓存向量异常')
   vectors/=norm;text_calls=len(queries);wall=tv['wall_s']
   with np.load(fp,allow_pickle=False) as z:
    ids=z['indices'].tolist();f=z['features'].astype('float32');fm=json.loads(str(z['meta'].item()))
   if fm!=dict(signature=c['protocol']['source_signature'],video_id=c['rows'][i]['video_id'],kind='coarse_1fps',indices_sha256=p.digest(ids)):raise ValueError('共享图像特征身份漂移')
   fn=np.linalg.norm(f,axis=1,keepdims=True)
   if f.shape!=(len(ids),768) or not np.isfinite(f).all() or (fn<=1e-8).any():raise ValueError('共享图像特征异常')
   try:result=k.select_windows(ids,f/fn,vectors,m['fps'],m['frame_count'],d)
   except ValueError as exc:reason='retrieval_fallback:'+str(exc)
  if result is None:result=dict(indices=base['indices'],windows=base['windows'],assigned_leaf_windows={},leaf_peaks=[],condition_satisfied=False,note='Exact frozen noobs input fallback,not a claim of solved condition')
  u=p.seal(dict(signature=c['signature'],index=i,parse_sha=p.sha(OUT/'parses'/f'{i}.json'),feature_sha=p.sha(c['features'][i]),queries=queries,result=result,fallback_reason=reason,same_as_noobs=result['indices']==base['indices'],text_encodings=text_calls,text_wall_s=wall))
  p.write_once(path,u)
 print('定位完成；先ground-report审阅，未运行QA。')

def load_ground(c):
 result={}
 for i in c['protocol']['indices']:
  path=OUT/'grounding'/f'{i}.json'
  if not path.exists():raise ValueError('先完成全部定位')
  u=json.loads(path.read_text());p.unseal(u)
  if u['signature']!=c['signature'] or u['index']!=i or u['parse_sha']!=p.sha(OUT/'parses'/f'{i}.json') or u['feature_sha']!=p.sha(c['features'][i]):raise ValueError('定位来源漂移')
  r=u['result'];ids=r['indices'];m=c['meta'][i]
  if ids!=sorted(set(ids)) or not 1<=len(ids)<=32 or any(type(x) is not int or not 0<=x<m['frame_count'] for x in ids):raise ValueError('定位帧预算异常')
  if u['fallback_reason']:
   if ids!=c['oldplans'][i]['indices'] or r['windows']!=c['oldplans'][i]['windows']:raise ValueError('回退非原输入')
  elif ids!=old.combined_frames(m['frame_count'],r['windows']) or len(r['windows'])!=6:raise ValueError('定位窗口/帧号不符')
  result[i]=u
 return result

def ground_report(c):
 gs=load_ground(c);details=[dict(index=i,question=c['rows'][i]['question'],parse=c['parses'][i]['parsed'],**{n:u[n] for n in ['queries','result','fallback_reason','same_as_noobs']},semantic_grounding_review='pending') for i,u in gs.items()]
 from collections import Counter
 report=dict(signature=c['signature'],N=30,changed=sum(not u['same_as_noobs'] for u in gs.values()),fallbacks=dict(Counter(u['fallback_reason'] for u in gs.values() if u['fallback_reason'])),text_encodings=sum(u['text_encodings'] for u in gs.values()),text_wall_s=sum(u['text_wall_s'] for u in gs.values()),new_QA=0,semantic_grounding_pass=None,note='Inspect leaf matches before QA. Assigned windows are retrieval candidates,not correct-evidence certification.')
 p.write_once(OUT/'ground_review/summary.json',report);p.write_once(OUT/'ground_review/review_packet.json',details);print(json.dumps(report,ensure_ascii=False,indent=2));print('定位审阅:',OUT/'ground_review/review_packet.json')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','preflight','parse5','parse30','summarize5','summarize30','ground','ground-report','qa','summarize-qa']);a=ap.parse_args();print('SPACE',p.disk_guard());c=context(a.action=='init');print('解析',len(c['parses']),'/30')
 if a.action in ['init','preflight']:return
 if a.action in ['summarize5','summarize30']:summarize_parse(c,'parse5' if a.action=='summarize5' else 'parse30');return
 if a.action=='ground-report':ground_report(c);return
 if a.action=='summarize-qa':
  from condition_grounding_qa import summarize
  summarize(c);return
 if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise ValueError('用户确认后运行')
 with (p.OLD/'cache/gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=context()
  if a.action in ['parse5','parse30']:parse_stage(c,a.action)
  elif a.action=='ground':ground(c)
  elif a.action=='qa':
   if os.environ.get('PRACTICAL_LOCALIZATION_REVIEWED')!='1':raise ValueError('先审阅叶定位再设置PRACTICAL_LOCALIZATION_REVIEWED=1')
   from condition_grounding_qa import run
   run(c)

if __name__=='__main__':main()
