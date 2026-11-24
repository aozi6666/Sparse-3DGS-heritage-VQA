"""条件解析契约与子事件窗口选择；只有相对检索分数，没有事实认证。"""
import json,math,re
import numpy as np
import condition_clips as old

VERSION='condition_grounding_v1'
PARSE5=('543','625','484','549','325')
SCHEMA={'evidence','events','relation','time_range','time_quote','unresolved'}
RELATIONS={'independent','known_sequence','unknown_order','spatial_order','first_event','final_event','count','absence'}
SUPPORTED={'independent','known_sequence','unknown_order','spatial_order'}
SYSTEM='You are a text-only video-search query compiler. You have NOT seen a video. Compile the question, never answer it or invent observations. Return only the requested compact JSON.'

def clean(s):return ' '.join(s.replace('\\n','\n').split())
def number(s):
 words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'twelve':12,'fifteen':15,'twenty':20,'thirty':30,'sixty':60}
 return words.get(s.lower(),float(s) if re.fullmatch(r'\d+(?:\.\d+)?',s) else None)
def explicit_range(question):
 q=question.lower();n=r'(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|sixty)';unit=r'(seconds?|minutes?|hours?)'
 m=re.search(r'\b(?:first|initial|opening)\s+'+n+r'\s+'+unit+r'\b',q)
 if m:
  scale=3600 if m[2].startswith('hour') else 60 if m[2].startswith('minute') else 1
  return dict(bounds=[0.,number(m[1])*scale],quote=question[m.start():m.end()])
 m=re.search(r'\b(?:from|between)\s+'+n+r'\s+(?:to|and)\s+'+n+r'\s+'+unit+r'\b',q)
 if m:
  scale=3600 if m[3].startswith('hour') else 60 if m[3].startswith('minute') else 1
  return dict(bounds=[number(m[1])*scale,number(m[2])*scale],quote=question[m.start():m.end()])
 return None

def asks_unknown_order(question):
 q=question.lower()
 return 'single scene' not in q and bool(re.search(r'in (?:which|what) order|sequence of events|sorts? the sequence|order are the following',q))

def parser_prompt(row):
 from air_protocol import query
 return ('Compile the following original question and ALL its answer options into a visual-search plan. The options are alternatives, not established facts. Do not select or paraphrase a correct answer.\n'+query(row)+
 '\nReturn ONE JSON object with exactly SIX keys. Field contract: '
 'evidence: choose ONE string from visual, mixed, speech, unknown. '
 'events: list of 1..4 objects, each with exactly target, action, question_quote (all strings). target is the bound object/person; action is its visible action/state or empty; question_quote is a literal supporting substring from the QUESTION. '
 'relation: choose ONE string from independent, known_sequence, unknown_order, spatial_order, first_event, final_event, count, absence. '
 'time_range: null or [start_seconds,end_seconds], two numbers derived ONLY from an explicit numeric phrase. '
 'time_quote: exact supporting phrase or empty string when range is null. '
 'unresolved: list of at most3 short strings. Choose actual enum values, never a pipe-separated list or a placeholder.'
 '\nUse 1 to 4 events. Each target/action <=80 characters. Keep the same entity bound to its action: a red-jersey man being interviewed is not merely a red jersey. Every question_quote must be a literal substring of the QUESTION, not an option. Do not add an outcome/cause learned from one answer option. '
 'If the question explicitly lists (a)-(d) events, include every listed event as a separate search, in their listed order, NOT inferred chronology. '
 'When asked to determine UNKNOWN chronological order, use unknown_order, never known_sequence. For ordering within a single scene use spatial_order. '
 'Use known_sequence only for before/after/then relations explicitly GIVEN by the question, list events chronologically. '
 'An explicit range such as first six minutes means time_range=[0,360] and time_quote="first six minutes". It does NOT mean first_event. Without explicit numeric bounds use null and empty time_quote; never invent 24 seconds. '
 'First magic / first to start means first_event, not a fixed numeric range; final outcome means final_event. Count/negation require count/absence. '
 'Mark speech or mixed if the requested fact depends on speech/dialogue/narration; do not invent a visual explanation. Unknown evidence is allowed. '
 'unresolved holds up to3 short limitations, not an answer. Do not output markdown, reasoning, or a letter answer.')

def parse(raw,row):
 try:
  s=raw.strip()
  if s.startswith('```') and s.endswith('```'):s='\n'.join(s.splitlines()[1:-1])
  d=json.loads(s)
  if not isinstance(d,dict) or set(d)!=SCHEMA:raise ValueError('schema keys')
  if d['evidence'] not in ['visual','mixed','speech','unknown'] or d['relation'] not in RELATIONS:raise ValueError('evidence/relation')
  if not isinstance(d['events'],list) or not 1<=len(d['events'])<=4:raise ValueError('events 1..4')
  for e in d['events']:
   if not isinstance(e,dict) or set(e)!={'target','action','question_quote'}:raise ValueError('event keys')
   for k,limit in [('target',80),('action',80),('question_quote',240)]:
    if not isinstance(e[k],str) or len(e[k])>limit or (k!='action' and not e[k].strip()):raise ValueError('event text length')
   if e['target'].strip().upper() in list('ABCD'):raise ValueError('answer letter is not target')
   if clean(e['question_quote']) not in clean(row['question']):raise ValueError('event quote not in question')
  qs=[clean(e['target']+' '+e['action']) for e in d['events']]
  if len(set(qs))!=len(qs):raise ValueError('duplicate searches')
  if not isinstance(d['unresolved'],list) or len(d['unresolved'])>3 or any(not isinstance(x,str) or len(x)>160 for x in d['unresolved']):raise ValueError('unresolved list')
  q=row['question'];expected=explicit_range(q)
  if not isinstance(d['time_quote'],str):raise ValueError('time quote type')
  if d['time_range'] is not None:
   r=d['time_range']
   if not isinstance(r,list) or len(r)!=2 or any(type(x) not in [int,float] or not math.isfinite(x) for x in r) or not 0<=r[0]<r[1]:raise ValueError('numeric range')
   if expected is None or any(abs(a-b)>1e-6 for a,b in zip(r,expected['bounds'])) or clean(expected['quote']).lower() not in clean(d['time_quote']).lower() or clean(d['time_quote']).lower() not in clean(q).lower():raise ValueError('range not supported by exact original phrase')
  elif expected is not None:raise ValueError('explicit numeric range was omitted')
  elif d['time_quote']!='':raise ValueError('time quote without range')
  residual=q.lower().replace(expected['quote'].lower(),'') if expected else q.lower()
  if expected and d['relation']=='first_event' and not re.search(r'\b(first|earliest)\b',residual):raise ValueError('numeric range confused with first event')
  if asks_unknown_order(q) and d['relation']!='unknown_order':raise ValueError('unknown chronology must not be imposed')
  if 'single scene' in q.lower() and re.search(r'\border\b',q.lower()) and d['relation']!='spatial_order':raise ValueError('spatial order confused with temporal order')
  if d['relation']=='known_sequence' and not re.search(r'\b(before|after|then)\b',q.lower()):raise ValueError('no explicitly given sequence')
  if d['relation']=='known_sequence' and len(d['events'])<2:raise ValueError('known sequence needs at least two events')
  if d['evidence']=='visual' and d['relation']=='unknown_order' and len(d['events'])<2:raise ValueError('unknown order needs distinct events or explicit unsupported evidence')
  items=old.explicit_items(q)
  if items and len(d['events'])!=len(items):raise ValueError('explicit subevent count not preserved')
  return d,None
 except (ValueError,TypeError,KeyError,AttributeError) as exc:return None,str(exc)

def fallback_reason(d):
 if d['evidence']!='visual':return 'unsupported_evidence_'+d['evidence']
 if d['relation'] not in SUPPORTED:return 'unsupported_relation_'+d['relation']
 return None

def searches(d):return [clean(e['target']+' '+e['action']) for e in d['events']]

def relative_scores(features,vectors):
 scores=features@vectors.T;lo=scores.min(0);span=np.ptp(scores,axis=0)
 if not np.isfinite(scores).all() or np.any(span<=1e-8):raise ValueError('flat or invalid leaf retrieval curve')
 return (scores-lo)/span

def temporal_gates(signals,seconds,relation):
 # HiMu启发的SEQ：真实时间严格递增时的过去/未来累积门控；不是概率或充分性认证。
 if len(seconds)!=signals.shape[0] or np.any(np.diff(seconds)<=0):raise ValueError('time axis not strictly increasing')
 if relation!='known_sequence':return signals.copy()
 out=signals.copy()
 for j in range(signals.shape[1]):
  for k in range(j):out[:,j]*=np.r_[0,np.maximum.accumulate(signals[:,k])[:-1]]
  for k in range(j+1,signals.shape[1]):out[:,j]*=np.r_[np.maximum.accumulate(signals[::-1,k])[::-1][1:],0]
 return out

def select_windows(indices,features,vectors,fps,count,d):
 if fallback_reason(d):raise ValueError('unsupported parser plan')
 if not math.isfinite(fps) or fps<=0 or count<32 or features.shape!=(len(indices),768) or vectors.shape!=(len(d['events']),768) or indices!=sorted(set(indices)) or any(type(i) is not int or not 0<=i<count for i in indices) or not np.isfinite(features).all() or not np.isfinite(vectors).all():raise ValueError('retrieval dimensions')
 sec=np.asarray(indices,dtype=float)/fps
 duration=(count-1)/fps;bounds=d['time_range'] or [0.,duration];low=max(0.,bounds[0]);high=min(duration,bounds[1])
 curves=relative_scores(features,vectors);curves[(sec<low)|(sec>high),:]=0
 curves=temporal_gates(curves,sec,d['relation'])
 if high-low<72-1e-6:raise ValueError('explicit range too short for six disjoint12s windows')
 eligible=np.where((sec>=low)&(sec<=high))[0]
 if not len(eligible):raise ValueError('no candidate in explicit range')
 windows=[];assigned={};peaks=[]
 def add(pos,leaf,origin):
  t=float(sec[pos]);start=max(low,min(t-6,high-12));end=start+12
  if any(min(end,w['end'])-max(start,w['start'])>1e-6 for w in windows):return False
  ix=np.linspace(round(start*fps),min(count-1,round(end*fps)),4).round().astype(int).tolist()
  if len(set(ix))!=4:return False
  windows.append(dict(id=len(windows),start=start,end=end,indices=ix,origin=origin,leaf=leaf));return True
 ranked=[sorted(eligible.tolist(),key=lambda i:(-float(curves[i,j]),indices[i])) for j in range(curves.shape[1])]
 # 每个必要子事件先分到一个窗口；最佳时间已在现有窗口内时允许共享，不伪称语义已覆盖。
 for j,rank in enumerate(ranked):
  if not rank or curves[rank[0],j]<=0:raise ValueError('no positive conditioned leaf curve')
  best=rank[0];peaks.append(dict(leaf=j,frame=indices[best],second=float(sec[best]),relative_score=float(curves[best,j])))
  shared=next((w['id'] for w in windows if w['start']<=sec[best]<=w['end']),None)
  if shared is not None:assigned[str(j)]=shared;continue
  for pos in rank:
   if add(pos,j,'leaf_required'):assigned[str(j)]=windows[-1]['id'];break
  if str(j) not in assigned:raise ValueError('leaf could not obtain window')
 pointers=[0]*len(ranked)
 while len(windows)<6:
  advanced=False
  for j,rank in enumerate(ranked):
   while pointers[j]<len(rank):
    pos=rank[pointers[j]];pointers[j]+=1
    if add(pos,j,'leaf_round_robin'):advanced=True;break
   if len(windows)==6:break
  if not advanced:break
 for t in np.linspace(low,high,128):
  if len(windows)==6:break
  # 几何补足只在给定范围；不新增特征，不按真值选时间。
  start=max(low,min(float(t)-6,high-12));end=start+12
  if any(min(end,w['end'])-max(start,w['start'])>1e-6 for w in windows):continue
  ix=np.linspace(round(start*fps),min(count-1,round(end*fps)),4).round().astype(int).tolist()
  if len(set(ix))==4:windows.append(dict(id=len(windows),start=start,end=end,indices=ix,origin='range_uniform_fill',leaf=None))
 if len(windows)!=6:raise ValueError('six disjoint windows unavailable')
 final=old.combined_frames(count,windows)
 if len(final)>32:raise ValueError('frame budget overflow')
 return dict(windows=windows,indices=final,assigned_leaf_windows=assigned,leaf_peaks=peaks,condition_satisfied=False,note='Relative CLIP evidence candidates only; leaf window assignment does not prove semantic grounding or all-condition sufficiency.')
