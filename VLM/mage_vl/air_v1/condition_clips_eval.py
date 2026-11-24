"""仅在自动12题完成后读取人工定位，离线参照不影响自动选择。"""
import json,math
from collections import Counter
import condition_clips as k
import condition_clips_run as r
import air_protocol as p

def summarize(c):
    if len(c['units'])!=12:raise ValueError('未完整，不能读取人工时间参照')
    # 人工窗口只存在于这个离线评估模块；没有传回run/propose/prompt的路径。
    refroot=p.ROOT/'oracle_evidence_diag_v1'
    reviews=json.loads((refroot/'review_v2.json').read_text())
    input_ref=json.loads((refroot/'inputs_reviewed_v2.json').read_text())
    reference_ids={x['index']:x['frame_indices'] for x in input_ref['rows']}
    details=[];all_calls={};text_cost=0.
    for idx in k.IDS:
        u=c['units'][idx];meta=c['meta'][idx];fps=meta['fps'];windows=[(a,b) for a,b,_ in reviews[idx]['windows'] if b is not None]
        def hits(ids):return sum(any(a<=i/fps<=b for i in ids) for a,b in windows)
        calls=[]
        for path in u['files']:
            if '/calls/' in path:
                x=json.loads(open(path).read());p.unseal(x)
                if x['signature']!=c['signature'] or x['index']!=idx:raise ValueError('调用来源不符')
                all_calls[path]=x;calls.append(x)
        tv=json.loads((r.OUT/'text_features'/f'{idx}.json').read_text());p.unseal(tv);text_cost+=tv['wall_s']
        details.append(dict(index=idx,question=c['rows'][idx]['question'],evidence_status=reviews[idx]['status'],
            required_conditions=c['protocol']['conditions'][idx],auto_windows=u['windows'],auto_frame_ids=u['joint_indices'],
            reference_ranges=windows,reference_ranges_total=len(windows),reference_ranges_touched=hits(u['joint_indices']),global_only_ranges_touched=hits(u['global_indices']),
            reference_frame_nearest_s=sum(min(abs(i-j)/fps for j in u['joint_indices']) for i in reference_ids[idx])/32,
            joint_report=u['joint'],structural_guard=u['guard'],invalid_reports=u['invalid_reports'],raw_call_paths=[path for path in u['files'] if '/calls/' in path]))
    invalid=sum(u['invalid_reports'] for u in c['units'].values())
    def group(status):
        ds=[d for d in details if d['evidence_status']==status]
        return dict(N=len(ds),all_reference_ranges_touched=sum(d['reference_ranges_touched']==d['reference_ranges_total'] for d in ds),reference_ranges_total=sum(d['reference_ranges_total'] for d in ds),reference_ranges_touched=sum(d['reference_ranges_touched'] for d in ds),global_only_ranges_touched=sum(d['global_only_ranges_touched'] for d in ds))
    raw=list(all_calls.values());historical_coarse=[]
    import numpy as np
    for fp in sorted(set(c['features'].values())):
        with np.load(fp,allow_pickle=False) as z:historical_coarse.append(float(z['wall_s']))
    report=dict(signature=c['signature'],questions=12,logical_calls=96,audit=dict(passed=not invalid,invalid_logical_reports=invalid,structural_checks_only=True),
        evidence_status_groups={s:group(s) for s in ['explicit','uncertain','visual_insufficient']},
        model_claims=dict(Counter((u['joint'] or {}).get('claim','invalid') for u in c['units'].values())),
        guarded_statuses=dict(Counter(u['guard']['status'] for u in c['units'].values())),
        cost=dict(actual_calls=len(raw),repair_calls=sum(x['attempt']==1 for x in raw),inference_s=sum(x['inference_s'] for x in raw),preprocess_s=sum(x['preprocess_s'] for x in raw),visual_tokens=sum(x['visual_tokens'] for x in raw),input_tokens=sum(x['input_tokens'] for x in raw),generated_tokens=sum(x['generated_tokens'] for x in raw),peak_gib=max(x['peak_gib'] for x in raw),new_CLIP_text_wall_s=text_cost,historical_shared_coarse_image_wall_s=sum(historical_coarse)),
        cache_note='Existing CLIP video features reused. Historical image encoding has nonzero deployment cost; not a speed benchmark.',
        interpretation='Automatic bounded evidence-search feasibility on seen diagnostic12. Manual references read only after completion. Temporal range overlap is NOT semantic correctness/QA accuracy. Model supported claims are not machine-certified. No final answers generated.',
        reference_sha256={n:p.sha(refroot/n) for n in ['review_v2.json','inputs_reviewed_v2.json']})
    p.write_once(r.OUT/'results/review_packet.json',details);p.write_once(r.OUT/'results/summary.json',report);p.write_once(r.OUT/'results/units_snapshot.json',[c['units'][i] for i in k.IDS])
    print(json.dumps(report,ensure_ascii=False,indent=2));print('自动片段报告:',r.OUT/'results/summary.json');print('逐题事实/缺证据:',r.OUT/'results/review_packet.json')
    if invalid:raise ValueError('部分报告两次尝试后仍格式非法，结果已保存；先诊断，不自动重跑。')
