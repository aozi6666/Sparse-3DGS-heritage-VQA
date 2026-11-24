"""小型画面审阅包；额外峰值缩略图仅供诊断，绝不当成32帧QA输入。"""
import hashlib
import html
import io
import json
import math
from pathlib import Path
import time
import numpy as np

import air_protocol as p
import leaf_probe_run as r
import leaf_probe as k


def preview_path(i):return r.OUT/'previews'/i/'record.json'


def read_preview(c,i,loc):
    u=json.loads(preview_path(i).read_text());p.unseal(u)
    if u['signature']!=c['signature'] or u['index']!=i or u['location_sha256']!=p.sha(r.OUT/'locations'/f'{i}.json'):raise ValueError('缩略图断点来源异常')
    for path,sha in u['images'].items():
        if not Path(path).resolve().is_relative_to((r.OUT/'previews'/i).resolve()) or p.sha(path)!=sha:raise ValueError('缩略图损坏/越界')
    expected=sorted(set(loc['original_noobs_indices']+loc['result']['indices']+[h['frame'] for e in loc['result']['leaves'] for h in e['raw_top3']]+([loc['result']['joint_scene_probe']['frame']] if loc['result']['joint_scene_probe'] else [])))
    if u['decoded_frame_indices']!=expected or not u['time_axis_pass']:raise ValueError('缩略图原帧/时间审计异常')
    return u


def make_board(path,rows,images,fps,header,cell_w=192):
    from PIL import Image,ImageDraw
    cell_h=140;cols=max(len(row) for row in rows);board=Image.new('RGB',(cols*cell_w,40+cell_h*len(rows)),(245,245,245));draw=ImageDraw.Draw(board)
    draw.text((8,8),header,fill='black')
    for y,row in enumerate(rows):
        for x,(frame,label) in enumerate(row):
            if frame is None:
                draw.text((x*cell_w+4,44+y*cell_h),label,fill='black');continue
            im=images[frame].copy();im.thumbnail((cell_w-4,106))
            px=x*cell_w+(cell_w-im.width)//2;py=42+y*cell_h
            board.paste(im,(px,py));draw.text((x*cell_w+3,py+108),label,fill='black')
    out=io.BytesIO();board.save(out,format='JPEG',quality=82,optimize=False)
    k.atomic_bytes(path,out.getvalue())


def render(c):
    from tqdm import tqdm
    active=[i for i in c['protocol']['indices'] if i not in c['protocol']['groups']['fallback']]
    locations={i:r.read_location(c,i) for i in c['protocol']['indices']}
    done={i for i in active if preview_path(i).exists()}
    for i in done:read_preview(c,i,locations[i])
    reader_class=None
    with tqdm(total=len(active),initial=len(done),desc='定位证据画面预览',unit='题',dynamic_ncols=True) as bar:
        for i in active:
            if i in done:continue
            loc=locations[i]
            if loc['result'] is None:bar.update(1);continue
            p.disk_guard();start=time.perf_counter()
            if reader_class is None:
                from decord import VideoReader,cpu
                from PIL import Image
                reader_class=VideoReader
            m=c['meta'][i];row=c['rows'][i];vr=reader_class(row['video_path'],ctx=cpu(0),num_threads=1)
            if len(vr)!=m['frame_count'] or abs(float(vr.get_avg_fps())-m['fps'])>1e-8:raise ValueError('解码视频时间元数据变化')
            result=loc['result'];peakframes=[h['frame'] for e in result['leaves'] for h in e['raw_top3']]
            if result['joint_scene_probe']:peakframes.append(result['joint_scene_probe']['frame'])
            all_ids=sorted(set(loc['original_noobs_indices']+result['indices']+peakframes))
            pixels=vr.get_batch(all_ids).asnumpy()
            if pixels.dtype!=np.uint8 or pixels.shape[0]!=len(all_ids):raise ValueError('解码帧异常')
            pts=np.asarray(vr.get_frame_timestamp(all_ids),dtype=float)
            expected=np.asarray(all_ids)/m['fps']
            if pts.shape!=(len(all_ids),2) or not np.isfinite(pts).all() or np.any(np.diff(pts[:,0])<=0):raise ValueError('实际PTS时间轴异常')
            drift=float(np.max(np.abs(pts[:,0]-expected)))
            if drift>max(.1,2/m['fps']):raise ValueError('frame/fps与PTS差异过大，停止预览，不猜测时间')
            images={frame:Image.fromarray(array) for frame,array in zip(all_ids,pixels)}
            rows=[]
            for name,inds in [('OLD',loc['original_noobs_indices']),('PROBE',result['indices'])]:
                for begin in range(0,len(inds),8):
                    rows.append([(frame,f'{name} F{frame} {frame/m["fps"]:.1f}s') for frame in inds[begin:begin+8]])
            comparison=r.OUT/'previews'/i/'comparison.jpg';peaks=r.OUT/'previews'/i/'peaks.jpg'
            make_board(comparison,rows,images,m['fps'],'OLD=noobs | PROBE=8 anchors + 6x4 clips; QA NOT RUN')
            prows=[]
            for leaf in result['leaves']:
                prows.append([(hit['frame'],f'{leaf["event_id"]} #{n+1} {hit["second"]:.1f}s {hit["cosine"]:.3f}') for n,hit in enumerate(leaf['raw_top3'])])
            if result['joint_scene_probe']:
                joint=result['joint_scene_probe'];prows.append([(joint['frame'],f'JOINT-PROBE {joint["second"]:.1f}s')])
            make_board(peaks,prows,images,m['fps'],'DIAGNOSTIC raw peaks; extra frames are NOT QA input',cell_w=320)
            u=p.seal(dict(signature=c['signature'],index=i,location_sha256=p.sha(r.OUT/'locations'/f'{i}.json'),images={str(path):p.sha(path) for path in [comparison,peaks]},decoded_frame_indices=all_ids,
                          pts_seconds=pts[:,0].tolist(),frame_over_fps_seconds=expected.tolist(),max_pts_drift_s=drift,time_axis_pass=True,
                          decoded_frames=len(all_ids),wall_s=time.perf_counter()-start,note='Preview decode and extra diagnostic peaks only; does not perform image feature encoding or QA.'))
            k.write_once(preview_path(i),u);bar.update(1)
            del vr,pixels,images


def summarize(c):
    from collections import Counter
    locs={i:r.read_location(c,i) for i in c['protocol']['indices']};previews={};details=[];sections=[];templates=[]
    for i,u in locs.items():
        result=u['result'];row=c['rows'][i];original=c['parent']['source']['fixed'][i]
        if result is not None:previews[i]=read_preview(c,i,u)
        details.append(dict(index=i,group=u['group'],question=row['question'],options=row['bodies'],status=u['status'],queries=u['queries'],original_constraints=original,
                            question_operators=c['parent']['protocol']['partitions'][i]['explicit_operators'],result=result,
                            original_noobs_indices=u['original_noobs_indices'],original_qa_selection_unchanged=True,
                            previews=previews.get(i),reason=u.get('reason'),semantic_review='pending'))
        text='<section><h2>'+html.escape(i+' / '+u['group'])+'</h2><p>'+html.escape(row['question'])+'</p><pre>'+html.escape(json.dumps(dict(queries=u['queries'],constraints=original,result=result,reason=u.get('reason')),ensure_ascii=False,indent=2))+'</pre>'
        if result is not None:
            text+=f'<img loading="lazy" src="../previews/{i}/peaks.jpg"><img loading="lazy" src="../previews/{i}/comparison.jpg">'
            for leaf in result['leaves']:templates.append(dict(index=i,event_id=leaf['event_id'],object_action_match=None,all_visible_conditions_match=None,required_time_relation_supported=None,best_visible_second=None,notes='',scope='diagnostic only; never automatic selector input'))
        text+='</section>';sections.append(text)
    texts=[r.read_text(c,t) for t in c['protocol']['texts']];hist={}
    for u in locs.values():
        if u.get('feature_file'):hist[u['feature_file']['path']]=u['feature_file']['historical_image_encoding_wall_s']
    states=dict(Counter(u['status'] for u in locs.values()))
    report=dict(signature=c['signature'],N=30,groups={g:len(ids) for g,ids in c['protocol']['groups'].items()},statuses=states,
                unique_leaf_queries=len(texts),new_text_encodings=sum(u['origin']=='new_local_clip' for u in texts),exact_text_reuse=sum(u['origin']=='reused_exact' for u in texts),
                constraints=dict(numeric_order_edges=[e for u in locs.values() if u['result'] for e in u['result']['numeric_edges']],unknown_order_not_imposed=True,
                                 peak_missing_from_proposal=sum(not leaf['peak_sampled_in_proposal'] for u in locs.values() if u['result'] for leaf in u['result']['leaves'])),
                cost=dict(new_text_encoding_wall_s=sum(u['encoding_wall_s'] for u in texts),historical_unique_video_image_encoding_wall_s=sum(hist.values()),
                          preview_decode_wall_s=sum(u['wall_s'] for u in previews.values()),preview_decoded_frames=sum(u['decoded_frames'] for u in previews.values()),
                          new_image_encoding=0,new_API_calls=0,Mage_generations=0,QA_calls=0),
                semantic_grounding_pass=None,accuracy_gain_evidence=False,
                note='Seen development diagnosis with manually frozen 7/4/19 scope; not general automatic routing or accuracy result. '
                     'Numerical temporal constraints are not semantic correctness. Contact sheets contain extra diagnostic frames; proposed input remains <=32. '
                     'Existing QA inputs/results unchanged. Costs include historical feature work; preprocessing/preview is not deployment speed benchmark.')
    r.OUT.joinpath('results').mkdir(parents=True,exist_ok=True)
    k.write_once(r.OUT/'results/summary.json',report);k.write_once(r.OUT/'results/review_packet.json',details);k.write_once(r.OUT/'results/manual_review_template.json',templates)
    page='<!doctype html><meta charset="utf-8"><title>查询到画面：定位诊断</title><style>body{font:16px sans-serif;max-width:1700px;margin:auto;padding:24px}img{max-width:100%;display:block;margin:16px 0}pre{white-space:pre-wrap}section{border-top:2px solid #888;margin:32px 0}h1{color:#20455b}</style><h1>查询→画面：固定开发30题定位诊断</h1><p>7题候选＋4题仅锚点探测＋19题原回退。先审阅对象、动作、限定条件与前后过程；高相似度、分配到窗口、满足数值先后均不能证明证据正确。没有运行QA。</p>'+''.join(sections)
    pagepath=r.OUT/'results/review.html'
    k.atomic_bytes(pagepath,page.encode())
    for path,sha in c['protocol']['source_files'].items():
        if p.sha(path)!=sha:raise ValueError('来源文件在诊断过程中发生变化')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('逐题审阅：',r.OUT/'results/review_packet.json');print('画面审阅页：',pagepath)
