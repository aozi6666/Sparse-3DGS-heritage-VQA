"""只有阶段完整后才加载旧12题人工定位；新增18题不捏造证据标签。"""
import json
from collections import Counter
import numpy as np
import air_protocol as p
import condition_clips as old
import condition_clips_v2 as k
import condition_clips_v2_run as r

def summarize(c,stage):
    ids=list(k.SMOKE) if stage=='smoke3' else c['protocol']['indices']
    if not set(ids)<=c['units'].keys():raise ValueError('阶段未完整，禁止离线评估')
    root=p.ROOT/'oracle_evidence_diag_v1';rp=root/'review_v2.json';ref=json.loads(rp.read_text())
    details=[];allcalls={};textwall=0.;historical=0.;features=set()
    for idx in ids:
        u=c['units'][idx];row=c['rows'][idx];m=c['meta'][idx];fps=m['fps'];unitmap=k.evidence_units(u['global_indices'],u['windows'],fps)
        call_records=[]
        for path,sha in u['files'].items():
            if p.sha(path)!=sha:raise ValueError('断点漂移')
            if '/calls/' in path:
                d=json.loads(open(path).read());p.unseal(d);allcalls[path]=d;call_records.append(d)
        tv=json.loads((r.OUT/'text_features'/f'{idx}.json').read_text());p.unseal(tv);textwall+=tv['wall_s']
        if idx in old.IDS:
            ranges=[(a,b) for a,b,_ in ref[idx]['windows'] if b is not None]
            hits=lambda ix:sum(any(a<=i/fps<=b for i in ix) for a,b in ranges)
            comparison=dict(status=ref[idx]['status'],ranges=ranges,total=len(ranges),touched=hits(u['joint_indices']),global8_touched=hits(u['global_indices']))
        else:comparison=None
        details.append(dict(index=idx,video_id=row['video_id'],duration=row['duration'],question=row['question'],options=row['bodies'],
            cohort='prior_diagnostic12' if idx in old.IDS else 'fixed_extra18_no_evidence_labels',
            evidence_units=unitmap,windows=u['windows'],joint_observation=u['joint_observation'],condition_codes=u['codes'],
            structural_issues=u['structural_issues'],observation_flags=u['observation_flags'],
            reference_temporal_coverage=comparison,semantic_review=dict(status='not_reviewed',facts_correct=None,all_question_conditions_met=None,notes=''),
            calls=[str(path) for path in u['files'] if '/calls/' in path]))
        if c['features'][idx] not in features:
            features.add(c['features'][idx])
            with np.load(c['features'][idx],allow_pickle=False) as z:historical+=float(z['wall_s'])
    calls=list(allcalls.values());obs=[d for d in calls if d['kind']=='observation'];codes=[d for d in calls if d['kind']=='codes']
    allflags=[d['flags'] for d in obs];audit=Counter(x['state'] for idx in ids for x in c['units'][idx]['codes'])
    paired=[d['reference_temporal_coverage'] for d in details if d['reference_temporal_coverage'] and d['reference_temporal_coverage']['status']=='explicit']
    report=dict(signature=c['signature'],stage=stage,questions=len(ids),videos=len({c['rows'][i]['video_id'] for i in ids}),
        audit=dict(all_expected_calls_present=len(calls)==16*len(ids),code_protocol='constrained by construction, not an instruction-following score',
            observation_flags={key:sum(f[key] for f in allflags) for key in ['empty','at_token_cap','repetition_loop','placeholder']},
            semantic_pass=None,semantic_review_required=True),
        joint_condition_states=dict(audit),joint_structural_issues=sum(len(c['units'][i]['structural_issues']) for i in ids),
        explicit_prior_diagnostic_reference=dict(N=len(paired),ranges=sum(d['total'] for d in paired),touched=sum(d['touched'] for d in paired),global8_touched=sum(d['global8_touched'] for d in paired),all_ranges_touched=sum(d['total']==d['touched'] for d in paired)),
        cost=dict(actual_calls=len(calls),observation_calls=len(obs),constrained_code_calls=len(codes),repair_calls=0,inference_s=sum(d['inference_s'] for d in calls),preprocess_s=sum(d['preprocess_s'] for d in calls),visual_tokens=sum(d['visual_tokens'] for d in calls),input_tokens=sum(d['input_tokens'] for d in calls),generated_tokens=sum(d['generated_tokens'] for d in calls),peak_gib=max(d['peak_gib'] for d in calls),new_clip_text_wall_s=textwall,historical_shared_image_features_wall_s=historical),
        interpretation='Seen development evidence feasibility, NOT final QA/not an accuracy-gain result. Constrained codes ensure syntax and allowed-unit references only. E is an unverified model claim. Manual ranges only offline, temporal overlap is not semantic correctness. Extra18 have no manual evidence labels. Cached image encoding has nonzero deployment cost.',
        reference_sha256=p.sha(rp))
    destination=r.OUT/'results'/stage
    p.write_once(destination/'summary.json',report);p.write_once(destination/'review_packet.json',details);p.write_once(destination/'units_snapshot.json',[c['units'][i] for i in ids])
    print(json.dumps(report,ensure_ascii=False,indent=2));print('汇总:',destination/'summary.json');print('逐题语义审阅:',destination/'review_packet.json')
    if stage=='smoke3':print('工程门槛:',k.smoke_gate(c['units']))
