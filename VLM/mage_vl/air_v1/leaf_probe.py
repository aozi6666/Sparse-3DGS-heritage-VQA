"""仅诊断查询到画面：相对分数不是置信度，时间先后成立不是事件识别正确。"""
import math
import json
import os
from pathlib import Path
import tempfile
import numpy as np

VERSION = 'condition_leaf_localization_probe_v1'


def atomic_bytes(path,data):
    """只提交本函数新建的临时文件，不覆盖/删除既有缓存；中断后可校验复用。"""
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('已有文件内容冲突，保留现场')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.new_',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(name,path)
        except FileExistsError:
            if path.read_bytes()!=data:raise ValueError('并发文件冲突，保留既有文件')
    finally:os.unlink(name)  # 仅清理本次刚创建的临时文件，不清理任何已有数据。


def write_once(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('已有JSON冲突，保留现场')
        return
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


class Unavailable(ValueError):
    pass


def unit_rows(values, dimension=768):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=dimension or not np.isfinite(x).all():raise ValueError('特征维度/数值异常')
    n=np.linalg.norm(x,axis=1,keepdims=True)
    if (n<=1e-8).any():raise ValueError('零向量')
    return x/n


def time_bounds(fps,count,bounds):
    if not math.isfinite(fps) or fps<=0 or type(count) is not int or count<32:raise ValueError('视频时间元数据异常')
    duration=(count-1)/fps
    if bounds is None:return 0.,duration
    if len(bounds)!=2 or any(type(x) not in [int,float] or not math.isfinite(x) for x in bounds) or not 0<=bounds[0]<bounds[1]:raise ValueError('明确范围异常')
    low,high=max(0.,bounds[0]),min(duration,bounds[1])
    if low>=high:raise Unavailable('明确范围在视频之外')
    return low,high


def ordered_pair(curves,eligible,earlier,later):
    # 在严格递增的原帧轴上选总相对分数最大的已知先后对；不预设未知顺序。
    best_prior=None;best=None
    for pos in eligible:
        if best_prior is not None:
            score=float(curves[best_prior,earlier]+curves[pos,later])
            candidate=(score,-best_prior,-pos)
            if best is None or candidate>best[0]:best=(candidate,best_prior,pos)
        if best_prior is None or curves[pos,earlier]>curves[best_prior,earlier]:best_prior=pos
    if best is None:raise Unavailable('先后关系没有两个不同时间候选')
    return best[1],best[2]


def locate(indices,features,vectors,fps,count,event_ids,edges,relation,anchors,bounds=None):
    if (indices!=sorted(set(indices)) or not indices or any(type(x) is not int or not 0<=x<count for x in indices)
        or len(set(event_ids))!=len(event_ids) or not 1<=len(event_ids)<=4):raise ValueError('事件ID或原帧索引异常')
    f=unit_rows(features);v=unit_rows(vectors)
    if f.shape[0]!=len(indices) or v.shape[0]!=len(event_ids):raise ValueError('图像/查询行数不符')
    low,high=time_bounds(fps,count,bounds);seconds=np.asarray(indices,dtype=np.float64)/fps
    eligible=np.flatnonzero((seconds>=low)&(seconds<=high)).tolist()
    if not eligible:raise Unavailable('范围内无既有1fps候选')
    if len(anchors)!=8 or anchors!=sorted(set(anchors)) or any(not 0<=x<count or not low-1e-8<=x/fps<=high+1e-8 for x in anchors):raise ValueError('8个锚点越界/重复')
    raw=(f@v.T).astype(np.float32)
    active=raw[eligible];span=np.ptp(active,axis=0)
    flat=[event_ids[j] for j,s in enumerate(span) if s<=1e-8]
    if flat:raise Unavailable('无区分度查询:'+','.join(flat))
    relative=np.zeros_like(raw);relative[eligible]=(active-active.min(0))/span
    ranked=[sorted(eligible,key=lambda pos:(-float(relative[pos,j]),indices[pos])) for j in range(len(event_ids))]
    def interval(pos):
        t=float(seconds[pos]);start=max(low,min(t-6,high-12));return start,start+12
    def hit(pos,j):return dict(frame=indices[pos],second=float(seconds[pos]),cosine=float(raw[pos,j]),relative_score=float(relative[pos,j]))
    leaves=[]
    for j,rank in enumerate(ranked):
        hits=[];intervals=[]
        for pos in rank:
            a,b=interval(pos)
            if any(min(b,y)-max(a,x)>1e-8 for x,y in intervals):continue
            hits.append(hit(pos,j));intervals.append((a,b))
            if len(hits)==3:break
        leaves.append(dict(event_id=event_ids[j],raw_top3=hits))
    if high-low<72-1e-8:raise Unavailable('范围不足以容纳6个不重叠12秒窗口')
    windows=[];assigned={};chosen={}
    def add(start,origin):
        start=max(low,min(float(start),high-12));end=start+12
        if any(min(end,w['end'])-max(start,w['start'])>1e-8 for w in windows):return None
        lo,hi=math.ceil(low*fps-1e-8),min(count-1,math.floor(high*fps+1e-8))
        ix=np.clip(np.linspace(round(start*fps),round(end*fps),4).round().astype(np.int64),lo,hi).tolist()
        if len(set(ix))!=4:return None
        n=len(windows);windows.append(dict(id=n,start=start,end=end,indices=ix,origin=origin));return n
    numeric_edges=[]
    if edges:
        if relation!='known_before_after' or len(edges)!=1 or len(event_ids)!=2:raise ValueError('不支持的已知关系图；不猜测事件顺序')
        e=edges[0];early=event_ids.index(e['earlier']);late=event_ids.index(e['later'])
        a,b=ordered_pair(relative,eligible,early,late)
        chosen={early:a,late:b}
        if seconds[b]-seconds[a]<=12:
            w=add((seconds[a]+seconds[b])/2-6,'known_pair_shared')
            assigned={early:w,late:w}
        else:
            for j in [early,late]:assigned[j]=add(interval(chosen[j])[0],'known_pair')
        if any(w is None for w in assigned.values()):raise Unavailable('先后对不能形成预算内窗口')
        numeric_edges.append(dict(earlier=event_ids[early],later=event_ids[late],earlier_frame=indices[a],later_frame=indices[b],numeric_order=True,semantic_match=None))
    elif relation=='known_before_after':raise ValueError('已知先后关系丢失')
    for j,rank in enumerate(ranked):
        if j in assigned:continue
        best=rank[0];shared=next((w['id'] for w in windows if w['start']<=seconds[best]<=w['end']),None)
        if shared is not None:assigned[j]=shared;chosen[j]=best;continue
        for pos in rank:
            w=add(interval(pos)[0],'leaf_required')
            if w is not None:assigned[j]=w;chosen[j]=pos;break
        if j not in assigned:raise Unavailable('必要叶事件无可分配窗口')
    pointers=[0]*len(ranked)
    while len(windows)<6:
        advanced=False
        for j,rank in enumerate(ranked):
            while pointers[j]<len(rank):
                pos=rank[pointers[j]];pointers[j]+=1
                if add(interval(pos)[0],'leaf_round_robin') is not None:advanced=True;break
            if len(windows)==6:break
        if not advanced:break
    for start in np.arange(low,high-12+1e-7,12):
        if len(windows)==6:break
        add(start,'range_uniform_fill')
    if len(windows)!=6:raise Unavailable('6个不重叠窗口不足，保持原回退')
    proposal=sorted(set(anchors+[i for w in windows for i in w['indices']]))
    if len(proposal)>32:raise ValueError('预算超过32帧')
    for j,leaf in enumerate(leaves):
        leaf.update(assigned_window=assigned[j],assigned_hit=hit(chosen[j],j),
                    peak_sampled_in_proposal=indices[chosen[j]] in proposal,
                    nearest_proposal_gap_s=min(abs(i-indices[chosen[j]])/fps for i in proposal))
    joint=None
    if relation=='spatial_order':
        pos=max(eligible,key=lambda x:(float(relative[x].min()),-indices[x]))
        joint=dict(frame=indices[pos],second=float(seconds[pos]),min_leaf_relative_score=float(relative[pos].min()),
                   note='Same-frame retrieval intersection diagnostic only; not artwork recognition/spatial-order certification.')
    return dict(indices=proposal,anchors=anchors,windows=windows,leaves=leaves,numeric_edges=numeric_edges,joint_scene_probe=joint,
                bounds=[low,high],semantic_grounding_pass=None,QA_selection_committed=False),raw,relative
