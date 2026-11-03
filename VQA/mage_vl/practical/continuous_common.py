"""连续字幕实验的CPU协议：无标签、固定邻接规则和文本预算。"""
from copy import copy
import json
from pathlib import Path

from practical_common import ROOT, signature, sha, write_once, stage_rows, read_lines, prompt, parse_answer
from practical_subtitles import make_context as old_context
from subtitle_common import sealed, subtitle_prompt

RESULTS=ROOT/'11_continuous_subtitles'
TAG='f32_px150000_neighbor1_time_t2048_v1'
STAGES=('smoke','dev30','dev60')
OLD_SIG='d0595146a0a8ace17ca37b906c2030fbabeefface296824dd8206a3305484d4e'


def timestamp(ms):
    seconds,millis=divmod(ms,1000); minutes,seconds=divmod(seconds,60); hours,minutes=divmod(minutes,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'


def continuous_evidence(original,cues,count_tokens,budget=2048):
    """每个命中cue最多补前后各一个非空cue；超预算时均匀保留邻句，绝不切断句子。"""
    useful=[c for c in cues if c['text']]
    positions={c['cue_id']:i for i,c in enumerate(useful)}
    anchors={positions[c['cue_id']] for c in original['selected_cues']}
    if not anchors:
        return dict(original=original,blocks=[],text='',text_tokens=0,status=original['status'],
            added_cue_ids=[],omitted_neighbor_ids=[],budget=budget)
    neighbors=sorted({j for i in anchors for j in [i-1,i+1] if 0<=j<len(useful)
        and max(0,useful[j]['start_ms']-useful[i]['end_ms'],useful[i]['start_ms']-useful[j]['end_ms'])<=2000}-anchors)

    def render(chosen):
        runs=[]
        for i in sorted(chosen):
            if not runs or i!=runs[-1][-1]+1 or useful[i]['start_ms']>useful[runs[-1][-1]]['end_ms']+2000:
                runs.append([i])
            else:
                runs[-1].append(i)
        blocks=[]
        for run in runs:
            selected=[useful[i] for i in run]
            blocks.append(dict(id='S'+str(selected[0]['cue_id']),cue_ids=[c['cue_id'] for c in selected],
                start_ms=min(c['start_ms'] for c in selected),end_ms=max(c['end_ms'] for c in selected),
                text=' '.join(c['text'] for c in selected)))
        text='\n'.join(f'{b["id"]} [{timestamp(b["start_ms"])} --> {timestamp(b["end_ms"])}] {b["text"]}' for b in blocks)
        return blocks,text,count_tokens(text)

    # 不以题目/选项/GT排序邻句；预算不足时沿整条时间轴均匀选取完整邻句。
    chosen=None
    for n in range(len(neighbors),-1,-1):
        subset=[] if n==0 else ([neighbors[len(neighbors)//2]] if n==1 else
            [neighbors[round(j*(len(neighbors)-1)/(n-1))] for j in range(n)])
        blocks,text,tokens=render(anchors|set(subset))
        if tokens<=budget:
            chosen=set(subset); break
    if chosen is None:
        raise ValueError('仅时间标记+原始命中句已超预算，停止，不截断原句')
    return dict(original=original,blocks=blocks,text=text,text_tokens=tokens,status='used',budget=budget,
        added_cue_ids=[useful[i]['cue_id'] for i in sorted(chosen)],
        omitted_neighbor_ids=[useful[i]['cue_id'] for i in neighbors if i not in chosen])


def load_records(path,sig,rows,evidence,baseline):
    lookup={r['index']:r for r in rows}; units={}
    for record in read_lines(path):
        obj=dict(record); digest=obj.pop('record_sha256',None); idx=obj['index']
        if digest!=signature(obj) or idx not in lookup or idx in units or obj['signature']!=sig:
            raise ValueError('连续字幕单元损坏/重复/来源不符')
        row=lookup[idx]; e=evidence[row['video_id']]; base=baseline[f'{idx}:frames:p0']
        if (obj['video_id']!=row['video_id'] or obj['evidence_sha256']!=signature(e)
                or obj['prompt_sha256']!=signature(subtitle_prompt(row,e['text']))
                or obj['pred']!=parse_answer(obj['raw_text']) or obj['pred'] not in ['',*'ABCD']
                or obj['subtitle_tokens']!=e['text_tokens'] or obj['status']!=e['status']
                or obj['base_input_sha256']!=base['prefix_sha256']
                or any(obj['visual'][k]!=base[k] for k in ['grid_thw','pv_shape','patch_time_range','n_tok'])
                or not obj['native_input_equal']):
            raise ValueError('连续字幕单元的输入/原始回答/视觉协议不符')
        if e['status']!='used' and (obj['raw_text']!=base['raw_text'] or obj['new_answer_calls']!=0):
            raise ValueError('缺字幕回退不是原基线')
        if obj['new_answer_calls']!=int(e['status']=='used'):
            raise ValueError('生成计数不符')
        units[idx]=record
    return units


def context(args,initialize=False):
    old_args=copy(args); old_args.stage='dev60'
    old=old_context(old_args)
    if old['signature']!=OLD_SIG:
        raise ValueError('旧字幕开发协议漂移')
    for stage in ['smoke','dev30','dev60']:
        s=json.loads((ROOT/'09_subtitles'/stage/'f32_px150000_aligned_sub_v1/summary.json').read_text())
        if not s['audit']['passed'] or s['signature']!=OLD_SIG:
            raise ValueError('旧字幕接口/开发结果未通过')
    folder=Path(__file__).parent
    protocol=dict(version='continuous_neighbor1_time_v1',old_protocol=old['protocol'],old_signature=OLD_SIG,
        old_dev60_summary_sha256=sha(ROOT/'09_subtitles/dev60/f32_px150000_aligned_sub_v1/summary.json'),
        rule='one previous and next nonempty cue per anchor with at most 2s gap; merge adjacent cues with at most 2s gap; timestamp blocks',
        budget=2048,overflow='retain all anchors; uniformly subsample whole neighbors; stop if anchors alone exceed budget',
        no_hit='exact original no-sub prompt/answer; never fill unrelated subtitles',
        inference=dict(frames=32,max_pixels=150000,min_pixels=3136,seed=0,max_new_tokens=32,do_sample=False),
        scope='development only; neighboring context is additional subtitle information, not standard frame-hit protocol',
        scripts={n:sha(folder/n) for n in ['continuous_common.py','continuous_mage.py','continuous_analyze.py']})
    sig=signature(protocol); store=ROOT/'cache/continuous_subtitles'/sig
    # 先确认所有视频都满足预算，再登记候选；初始化失败不留下半套候选配置。
    evidence={video:continuous_evidence(old['evidence_loader'](video),old['cues'][video],old['count_tokens'])
        for video in sorted({r['video_id'] for r in old['allowed']})}
    if initialize:
        write_once(RESULTS/'candidate_v1.json',protocol)
    if not (RESULTS/'candidate_v1.json').exists() or json.loads((RESULTS/'candidate_v1.json').read_text())!=protocol:
        raise ValueError('先init；配置漂移时不覆盖原候选')
    for video,e in evidence.items():
        path=store/'evidence'/f'{video}.json'
        obj=dict(signature=sig,evidence=e,evidence_sha256=signature(e))
        if initialize:
            write_once(path,obj)
        if not path.exists() or json.loads(path.read_text())!=obj:
            raise ValueError('连续字幕证据不存在或漂移，先init，不静默重建')
    units=load_records(store/'units.jsonl',sig,old['allowed'],evidence,old['base']['units'])
    rows=stage_rows(ROOT,args.stage)
    return dict(signature=sig,protocol=protocol,store=store,rows=rows,old=old,evidence=evidence,units=units,
        expected={r['index'] for r in rows},stage_dir=RESULTS/args.stage/TAG)
