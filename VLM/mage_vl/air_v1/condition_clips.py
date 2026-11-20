"""纯CPU规则：只使用原问题/自动检索信号；不读人工定位或答案。"""
import json,math,re
import numpy as np
from air_selection import uniform_indices

IDS=('379','418','484','626','265','450','486','704','325','454','625','655')
VERSION='condition_clips_v1'
MAX_NEW=384

def requirements(question):
    q=question.lower();result=['target_identity','candidate_distinction']
    # 同场拼图的排列不强行当成跨视频时序；仍要求多对象的顺序依据。
    if re.search(r'\b(first|earliest)\b',q):result.append('first_event')
    if re.search(r'\b(beginning|opening|early part)\b',q):result.append('beginning')
    if re.search(r'\b(which part|when in the video)\b',q):result.append('relative_position')
    if re.search(r'\b(final victory|finally|at the end|last|ending)\b',q):result.append('final_event')
    if re.search(r'\b(order|sequence|steps)\b',q):result.append('spatial_order' if 'single scene' in q else 'temporal_order')
    if re.search(r'\b(main idea|main focus|main content|topic|video about)\b',q):result.append('whole_video')
    if re.search(r'\b(how many|count|number of)\b',q):result.append('count_target_action')
    if re.search(r'\b(days|two|both|four|following items)\b',q) or len(explicit_items(question))>1:result.append('multiple_targets')
    if re.search(r'\b(after|before|when|why)\b',q):result.append('event_relation')
    return result

def explicit_items(question):
    return re.findall(r'(?m)^\([a-d]\)\s*(.+)',question)[:4]

def query_texts(original_query,question,report):
    # 原问题与所有选项始终保留；仅增加本轮自动观察的检索对象。
    targets=explicit_items(question)
    if report:targets.extend(report['targets'])
    return list(dict.fromkeys([original_query]+[x.strip() for x in targets if x.strip()]))[:5]

def propose(indices,features,vectors,fps,count,conditions):
    if (len(indices)!=len(features) or features.ndim!=2 or features.shape[1]!=768 or vectors.ndim!=2 or vectors.shape[1]!=768
        or not np.isfinite(features).all() or not np.isfinite(vectors).all() or fps<=0 or not math.isfinite(fps) or count<32):raise ValueError('检索信号身份/数值非法')
    duration=(count-1)/fps;length=min(12.,duration);windows=[]
    def add(start,origin):
        start=max(0.,min(float(start),duration-length));end=start+length
        if any(min(end,w['end'])-max(start,w['start'])>1e-6 for w in windows):return False
        ix=sorted(set(np.linspace(round(start*fps),min(count-1,round(end*fps)),4).round().astype(int).tolist()))
        if len(ix)!=4:return False
        windows.append(dict(id=len(windows),start=start,end=end,indices=ix,origin=origin));return True
    if 'first_event' in conditions or 'beginning' in conditions:
        add(0,'ordered_prefix');add(12,'ordered_prefix')
    elif 'final_event' in conditions:
        add(duration-12,'ordered_suffix');add(duration-24,'ordered_suffix')
    scores=features@vectors.T
    ranked=[sorted(range(len(indices)),key=lambda i:(-float(scores[i,j]),indices[i])) for j in range(len(vectors))]
    cursors=[0]*len(ranked)
    while len(windows)<6:
        advanced=False
        for j,rank in enumerate(ranked):
            while cursors[j]<len(rank):
                pos=rank[cursors[j]];cursors[j]+=1
                if add(indices[pos]/fps-length/2,f'query_{j}'):
                    advanced=True;break
            if len(windows)==6:break
        if not advanced:break
    for center in np.linspace(0,duration,64):
        if len(windows)==6:break
        add(float(center)-length/2,'uniform_fallback')
    if len(windows)!=6:raise ValueError('无法构成6个不重叠片段，停止而非重复窗口')
    return windows

def combined_frames(count,windows):
    anchors=uniform_indices(count,8)
    return sorted(set(anchors+[i for w in windows for i in w['indices']]))

def parse_report(raw,allowed,conditions):
    try:
        s=raw.strip()
        if s.startswith('```') and s.endswith('```'):
            s='\n'.join(s.splitlines()[1:-1])
        obj=json.loads(s)
        if set(obj)!= {'facts','met','missing','targets','claim'}:raise ValueError('schema keys')
        if obj['claim'] not in ['insufficient','partial','supported']:raise ValueError('claim')
        allowed=set(allowed)
        if not isinstance(obj['facts'],list) or len(obj['facts'])>8 or not isinstance(obj['met'],list) or len(obj['met'])>len(conditions):raise ValueError('lists')
        for f in obj['facts']:
            if set(f)!= {'frame','text'} or type(f['frame']) is not int or f['frame'] not in allowed or not isinstance(f['text'],str) or not 1<=len(f['text'])<=240:raise ValueError('unseen fact frame/text')
        seen=set()
        for m in obj['met']:
            if set(m)!= {'condition','frames'} or m['condition'] not in conditions or m['condition'] in seen:raise ValueError('condition')
            seen.add(m['condition']);ix=m['frames']
            if not isinstance(ix,list) or not ix or len(ix)!=len(set(ix)) or any(type(i) is not int or i not in allowed for i in ix):raise ValueError('unseen evidence frame')
        for k,limit in [('missing',8),('targets',4)]:
            if not isinstance(obj[k],list) or len(obj[k])>limit or any(not isinstance(x,str) or not 1<=len(x)<=160 for x in obj[k]):raise ValueError('text lists')
        if obj['claim']=='supported' and (obj['missing'] or seen!=set(conditions) or not obj['facts']):raise ValueError('supported without complete listed conditions')
        return obj,None
    except (ValueError,TypeError,KeyError,AttributeError) as exc:return None,str(exc)

def guard(report,conditions,windows,fps,count,required_items=0):
    """仅做引用/覆盖约束；绝不将模型自述包装为语义认证。"""
    if report is None:return dict(status='invalid',machine_verified=False,issues=['malformed report'])
    met={x['condition']:x['frames'] for x in report['met']};issues=[]
    if 'first_event' in met or 'beginning' in met:
        if not any(w['origin']=='ordered_prefix' for w in windows):issues.append('No ordered prefix search')
        refs=met.get('first_event',met.get('beginning',[]))
        if not any(i/fps<=24 for i in refs):issues.append('First/beginning claimed only from later frames')
    if 'final_event' in met and not any(i/fps>=(count-1)/fps-24 for i in met['final_event']):issues.append('Final claimed only from middle frames')
    if 'temporal_order' in met:
        touched={w['id'] for w in windows if set(w['indices'])&set(met['temporal_order'])}
        if len(touched)<max(2,required_items):issues.append('Temporal order lacks references across all required event windows')
    for k in ['event_relation','multiple_targets','count_target_action','spatial_order','relative_position']:
        if k in met and len(met[k])<2:issues.append(k+' lacks multiple-frame references')
    missing=set(conditions)-set(met)
    if missing:issues.append('Missing conditions: '+','.join(sorted(missing)))
    status='needs_more_evidence' if issues or report['claim']!='supported' else 'model_supported_unverified'
    return dict(status=status,machine_verified=False,issues=issues,
                limitation='Sampled coverage and cited frame IDs do not prove factual truth,absence of earlier events,or global completeness.')

def make_prompt(row,conditions,stage,allowed,global_indices,windows,local_reports=None):
    from air_protocol import query
    return ('Inspect only the supplied original video frames and their native timestamps. This is evidence inspection, not final multiple-choice answering.\n'
        +query(row)+'\nRequired conditions: '+json.dumps(conditions)+
        '\nStage: '+stage+'; frame identifiers (original video indices): '+json.dumps(allowed)+
        '\nGlobal overview identifiers: '+json.dumps(global_indices)+'\nCandidate windows: '+json.dumps(windows)+
        '\nPrior automatic local reports (may be wrong; check against pixels): '+json.dumps(local_reports or [],ensure_ascii=False)+
        '\nIdentify visible facts, conditions actually evidenced, and missing evidence. An object appearing does NOT prove first/final/order,whole-video topic,or action count. '
        'First requires earlier-event context; final requires ending evidence; order requires all relevant events and their times. '
        'No audio/subtitles were added: state insufficiency if speech is required. Never infer unseen evidence or simply select an answer. '
        'Return ONE compact JSON object only, at most4 short facts,4 short search targets. '
        'Schema: {"facts":[{"frame":123,"text":"visible fact"}],"met":[{"condition":"target_identity","frames":[123,456]}],'
        '"missing":["what is still unknown"],"targets":["short visual search phrase"],"claim":"insufficient|partial|supported"}. '
        'Use only listed frame integers and required condition names. supported is allowed only if ALL required conditions have cited evidence and missing is empty. '
        'For initial global overview, use partial/insufficient when sparse samples cannot establish timing. Keep facts and targets short.')
