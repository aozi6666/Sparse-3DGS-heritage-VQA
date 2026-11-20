"""v2纯CPU契约：结构合法不等于语义正确；不使用人工定位或GT。"""
import re,json,hashlib
import condition_clips as old

VERSION='condition_clips_v2'
CAP=64
SMOKE=('379','454','625')
CORE=('condition_clips_v2.py','condition_clips_v2_run.py','condition_clips_v2_eval.py','run_condition_clips_v2.sh')
MEANING={
 'target_identity':'The pictured object/person/event is the exact target asked about.',
 'candidate_distinction':'Visible evidence distinguishes the original answer options, rather than mere topic relevance.',
 'first_event':'This is the first relevant event; earlier relevant events have been checked, not just object presence.',
 'beginning':'The specified event is in the beginning of the video, not merely the first retrieved clip.',
 'relative_position':'The target event time is established relative to the full video duration.',
 'final_event':'The actual final outcome is observed after intermediate events, not an earlier temporary state.',
 'spatial_order':'All requested objects and their order in the SAME SCENE are visibly established.',
 'temporal_order':'ALL requested events are identified and their chronological order is established.',
 'whole_video':'Evidence covers the video sufficiently to establish its main topic, not one isolated scene.',
 'count_target_action':'The complete specified action/count interval is observed and distinct items/events can be counted.',
 'multiple_targets':'ALL requested targets/events are identified; evidence for only one is insufficient.',
 'event_relation':'The requested before/after/cause/event-result relation is visible with necessary context.'}

def select_rows(pool):
    lookup={r['index']:r for r in pool}
    if len(lookup)!=180 or not set(old.IDS)<=lookup.keys():raise ValueError('原开发池身份异常')
    chosen=[lookup[i] for i in old.IDS];used={r['video_id'] for r in chosen}
    # 固定哈希与时长分层，不按真值/旧预测/人工窗口选择新增题。
    rank=lambda r:hashlib.sha256(('condition_clips_v2_extra18:'+r['index']).encode()).hexdigest()
    for duration in ['medium','long']:
        added=0
        for r in sorted(pool,key=rank):
            if r['duration']==duration and r['video_id'] not in used:
                chosen.append(r);used.add(r['video_id']);added+=1
                if added==9:break
        if added!=9:raise ValueError('新增视频数量不足')
    if len(chosen)!=30 or len({r['index'] for r in chosen})!=30:raise ValueError('30题选择异常')
    return chosen

def atom_tokens(tokenizer):
    mapping={}
    for ch in 'UPEGNJ012345':
        ids=tokenizer.encode(ch,add_special_tokens=False)
        if len(ids)!=1 or ids[0] in tokenizer.all_special_ids or tokenizer.decode(ids)!=ch:raise ValueError('状态字符不是稳定的单token:'+ch)
        mapping[ch]=ids[0]
    if len(set(mapping.values()))!=len(mapping):raise ValueError('状态token碰撞')
    if tokenizer.eos_token_id is None:raise ValueError('缺少EOS')
    return mapping,int(tokenizer.eos_token_id)

def allowed_next(prefix,n_conditions,refs,atoms,eos):
    # 每条件固定两个token：U/N或P,E加本次可见单元编号。模型不生成JSON和原帧号。
    step=len(prefix)
    if step>=2*n_conditions:return [eos]
    if step%2==0:return [atoms[x] for x in 'UPE']
    if prefix[-1]==atoms['U']:return [atoms['N']]
    if prefix[-1] not in [atoms['P'],atoms['E']]:raise ValueError('非法前缀')
    return [atoms[x] for x in refs]

def parse_codes(ids,conditions,refs,atoms,eos):
    if len(ids)!=2*len(conditions)+1 or ids[-1]!=eos:raise ValueError('代码长度/EOS异常')
    rev={v:k for k,v in atoms.items()};result=[]
    for j,c in enumerate(conditions):
        s=rev.get(ids[2*j]);ref=rev.get(ids[2*j+1])
        if s not in ['U','P','E'] or (s=='U' and ref!='N') or (s!='U' and ref not in refs):raise ValueError('代码状态/引用异常')
        result.append(dict(condition=c,state=dict(U='unknown',P='partial',E='model_claimed_sufficient')[s],unit=None if ref=='N' else ref,semantically_verified=False))
    return result

def observation_flags(text,at_cap):
    s=text.strip();words=re.findall(r"[a-zA-Z']+",s.lower());grams=[tuple(words[i:i+5]) for i in range(max(0,len(words)-4))]
    from collections import Counter
    repeated=bool(grams and max(Counter(grams).values())>=3)
    placeholder=any(x in s.lower() for x in ['what is still unknown','short visual search phrase','visible fact here'])
    return dict(empty=not bool(s),at_token_cap=bool(at_cap),repetition_loop=repeated,placeholder=placeholder)

def evidence_units(global_ids,windows,fps):
    # 编号与实际原视频坐标唯一绑定；这些映射由程序生成，模型不能改写。
    result={'G':dict(role='sparse global overview; NOT continuous coverage',frames=global_ids,seconds=[round(i/fps,4) for i in global_ids])}
    for w in windows:result[str(w['id'])]=dict(role='local clip',frames=w['indices'],seconds=[round(i/fps,4) for i in w['indices']],start=w['start'],end=w['end'])
    if len(windows)>1:
        ids=sorted(set(global_ids+[i for w in windows for i in w['indices']]))
        result['J']=dict(role='joint set of all supplied windows and global frames; NOT proof of completeness',frames=ids,seconds=[round(i/fps,4) for i in ids])
    return result

def prompt(row,conditions,units,phase,observation=None,prior=None):
    import air_protocol as p
    base=('Inspect the supplied video frames at their original times. Do NOT answer the multiple choice question. Do NOT invent audio or subtitles.\n'
      +p.query(row)+'\nEvidence units (program mapping, frame indices are NOT seconds): '+json.dumps(units)+
      '\nRequired conditions in this exact order: '+json.dumps([{'name':c,'meaning':MEANING[c]} for c in conditions])+ 
      '\nEarlier automatic clip observations (may be wrong; verify pixels): '+json.dumps(prior or [],ensure_ascii=False))
    if phase=='observation':return base+'\nWrite ONE short sentence (at most 25 words) about visible evidence relevant to this question. State what is missing if necessary. No JSON, no lists, no numeric references, no answer letter, no repeated frame-by-frame captions.'
    refs=list(units)
    return base+'\nCurrent short observation (may be wrong): '+str(observation)+(
      '\nFor EACH required condition output exactly TWO single-character codes in the listed condition order. '
      'First code: U=unknown/not evidenced; P=partial evidence, condition still incomplete; E=all parts of this condition visibly established. '
      'Second code: N only after U; otherwise one evidence unit from '+','.join(refs)+'. '
      'Concatenate all pairs without spaces, headings, JSON or explanation. '
      'E is not allowed merely because a related object appears. First/final/order need the specified time context and all relevant events. '
      'If one evidence unit cannot support the condition, use U or P. Sparse global frames alone rarely establish complete event coverage.')

def structural_issues(codes,units,windows,fps,count):
    issues=[]
    for c in codes:
        if c['state']!='model_claimed_sufficient':continue
        name=c['condition'];ref=c['unit'];u=units[ref]
        if name in ['first_event','beginning'] and (ref=='G' or u.get('end',1e9)>24):issues.append(name+': no early local citation')
        if name=='final_event' and (ref=='G' or u.get('start',0)<(count-1)/fps-24):issues.append(name+': no ending local citation')
        if name in ['temporal_order','whole_video','multiple_targets','event_relation','count_target_action']:
            if ref!='J':issues.append(name+': citation does not identify joint cross-window evidence')
        if name=='temporal_order' and ref=='J' and len(windows)<2:issues.append('temporal_order: no multi-window coverage')
    return issues

def smoke_gate(units):
    if not set(SMOKE)<=units.keys():raise ValueError('先完成smoke3')
    flags=[x for idx in SMOKE for x in units[idx]['observation_flags']]
    if len(flags)!=24 or any(f['empty'] or f['repetition_loop'] or f['placeholder'] for f in flags):raise ValueError('smoke观察存在空值/循环/占位文字，保留现场先诊断')
    if sum(f['at_token_cap'] for f in flags)>6:raise ValueError('smoke观察截断比例超过25%，不扩到30题')
    return dict(passed=True,scope='protocol only, NOT semantic validation',observations=24,at_cap=sum(f['at_token_cap'] for f in flags))
