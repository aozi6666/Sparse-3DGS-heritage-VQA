"""只在当前阶段全部完成后读旧标签；不根据GT修改选帧。"""
import json,sys
from collections import Counter
import numpy as np
from air_protocol import ROOT,OLD,PROJECT,ARMS,ANALYSIS,query,digest,sha,write_once,unseal
from air_selection import parse_score

def summarize(ctx):
    if not ctx['expected'].issubset(ctx['units']):raise ValueError('阶段未完成，不读取GT或发布部分分数')
    label_path=OLD/'01_manifests/evaluation_labels.json';lock=json.loads((label_path.parent/'dataset_lock.json').read_text())
    if sha(label_path)!=lock['labels_sha256']:raise ValueError('旧标签身份漂移')
    labels=json.loads(label_path.read_text());rows=ctx['rows'];units=ctx['units'];selections={};scores=[]
    for row in rows:
        idx=row['index'];s=json.loads((ctx['store']/'selections'/f'{idx}.json').read_text());unseal(s);selections[idx]=s
        for frame in s['analyzed_indices']:
            u=json.loads((ctx['store']/'scores'/idx/f'{frame}.json').read_text());unseal(u)
            if (u['signature']!=ctx['signature'] or u['index']!=idx or u['frame_index']!=frame or u['score']!=parse_score(u['raw_text'])
                    or u['prompt_sha256']!=digest(ANALYSIS.format(query=query(row)))
                    or type(u['generated_tokens']) is not int or not 1<=u['generated_tokens']<=96):raise ValueError('评分来源/解析漂移')
            matching=[r['score_records'][str(frame)] for r in s['rounds'] if str(frame) in r['score_records']]
            if matching!=[dict(score=u['score'],raw_text=u['raw_text'])]:raise ValueError('选择轨迹与原始评分记录不同')
            scores.append(u)
    def group(items):
        if not items:return dict(N=0,methods={})
        pred={a:{r['index']:units[f'{r["index"]}:{a}']['pred'] for r in items} for a in ARMS}
        methods={a:dict(correct=sum(pred[a][r['index']]==labels[r['index']] for r in items),accuracy=sum(pred[a][r['index']]==labels[r['index']] for r in items)/len(items),invalid=sum(pred[a][r['index']] not in list('ABCD') for r in items)) for a in ARMS}
        rescue=sum(pred[ARMS[1]][r['index']]==labels[r['index']] and pred[ARMS[0]][r['index']]!=labels[r['index']] for r in items)
        harm=sum(pred[ARMS[1]][r['index']]!=labels[r['index']] and pred[ARMS[0]][r['index']]==labels[r['index']] for r in items)
        vs=dict(rescue=rescue,harm=harm,net=rescue-harm,delta_pp=100*(rescue-harm)/len(items),video_cluster_bootstrap_95ci_pp=None,note='Single-video interface smoke; no significance/generalization claim.')
        if len({r['video_id'] for r in items})>1:
            sys.path.insert(0,str(PROJECT/'VQA/mage_vl/practical'))
            from practical_analyze import comparison
            vs=comparison(items,pred[ARMS[1]],pred[ARMS[0]],labels)
        return dict(N=len(items),videos=len({r['video_id'] for r in items}),methods=methods,paired=vs)
    invalid_answers=sum(units[k]['pred'] not in list('ABCD') for k in ctx['expected']);invalid_scores=sum(u['score'] is None for u in scores)
    errors=[]
    if invalid_answers:errors.append(f'{invalid_answers} invalid QA answers retained as errors')
    if invalid_scores:errors.append(f'{invalid_scores} malformed analysis scores retained as invalid,not neutral')
    report=dict(signature=ctx['signature'],stage=ctx['stage_dir'].name,N=len(rows),
        interpretation='Existing official Video-MME development videos; historical714 allseen. AIR-Mage adaptation with HFCLIP,not officialpaper reproduction. Same32frameCAP/perframepixelbudget,actualcounts differ. No subtitles/Codec/Agent/API.',
        audit=dict(passed=not errors,complete_units=len(ctx['expected']),invalid_answers=invalid_answers,invalid_scores=invalid_scores,errors=errors),
        overall=group(rows),by_duration={d:group([r for r in rows if r['duration']==d]) for d in sorted({r['duration'] for r in rows})},
        selection=dict(question_fallbacks=sum(bool(s['fallback_reason']) for s in selections.values()),actual_frames=[len(s['final_indices']) for s in selections.values()],rounds=[len(s['rounds']) for s in selections.values()],analysis_frames=len(scores)),cost={})
    for a in ARMS:
        uu=[units[f'{r["index"]}:{a}'] for r in rows]
        report['cost'][a]=dict(actual_frames=sum(len(u['indices']) for u in uu),logical_visual_tokens=sum(u['visual']['n_tok'] for u in uu),input_tokens=sum(u['input_tokens'] for u in uu),new_QA_calls=sum(u['new_calls'] for u in uu),QA_inference_s=sum(u['inference_s'] for u in uu),preprocess_s=sum(u['preprocess_s'] for u in uu),peak_gib=max(u['peak_gib'] for u in uu))
    report['cost']['analysis']=dict(calls=len(scores),input_tokens=sum(u['input_tokens'] for u in scores),visual_tokens=sum(u['visual']['n_tok'] for u in scores),generated_tokens=sum(u['generated_tokens'] for u in scores),inference_s=sum(u['inference_s'] for u in scores),preprocess_s=sum(u['preprocess_s'] for u in scores),peak_gib=max([u['peak_gib'] for u in scores] or [0.]))
    # 阶段只计本阶段实际用过的共享特征，不能被未来新增问题的缓存改变。
    clipfiles=[]
    for video in sorted({r['video_id'] for r in rows}):
        folder=ctx['store']/'clip_features'/video;coarse=folder/'coarse.npz'
        with np.load(coarse,allow_pickle=False) as data:coarse_indices=set(data['indices'].tolist())
        used={int(i) for r in rows if r['video_id']==video for i in selections[r['index']]['observed_similarity']}
        clipfiles.append(coarse)
        clipfiles.extend(folder/f'extra_{i}.npz' for i in sorted(used-coarse_indices))
    costs=[]
    for p in clipfiles:
        with np.load(p,allow_pickle=False) as data:costs.append(float(data['wall_s']))
    report['cost']['clip_features']=dict(shared_video_feature_files=len(costs),recorded_wall_s=sum(costs),note='Shared once per video in protocol; includes decode and featurepreprocess. Extra features may be reused by questions.')
    text_cost=[]
    for r in rows:
        with np.load(ctx['store']/'clip_text_features'/f'{r["index"]}.npz',allow_pickle=False) as data:text_cost.append(float(data['wall_s']))
    report['cost']['clip_text']=dict(query_feature_files=len(text_cost),recorded_wall_s=sum(text_cost))
    report['cost']['note']='Persisted measured costs across resumptions,not current command wall time. Include selector CLIP+analysis,not just final QA. Native uniform parity adds extra preprocessing; not deployment speed benchmark.'
    details=[dict(index=r['index'],video_id=r['video_id'],duration=r['duration'],gt=labels[r['index']],predictions={a:units[f'{r["index"]}:{a}']['pred'] for a in ARMS},record_sha256={a:units[f'{r["index"]}:{a}']['record_sha256'] for a in ARMS},fallback=selections[r['index']]['fallback_reason']) for r in rows]
    write_once(ctx['stage_dir']/'predictions.json',details);write_once(ctx['stage_dir']/'summary.json',report)
    # 全局journal会随扩阶段继续追加；用本阶段不可变快照归档，支持重复汇总。
    snapshot=ctx['stage_dir']/'units_snapshot.json'
    write_once(snapshot,[units[k] for k in sorted(ctx['expected'])])
    write_once(ROOT/'archive'/f'{ctx["stage_dir"].name}_{ctx["signature"]}.json',dict(protocol=ctx['protocol'],summary=report,predictions=details,
        stage_units_sha256=sha(snapshot),journal_scope='stage-only immutable snapshot; shared global journal is append-only',
        stage_units_path=str(snapshot)))
    print(json.dumps(report,ensure_ascii=False,indent=2));print('汇总路径:',ctx['stage_dir']/'summary.json')
    if errors:raise ValueError('审计未通过，保留结果，请先诊断，不自动重跑或扩样')
