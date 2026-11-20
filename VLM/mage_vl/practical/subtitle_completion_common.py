"""收尾字幕基线：3072预算对照和冻结后的short219；纯CPU协议，不读GT。"""
from copy import copy
import json
import math
from pathlib import Path

from practical_common import ROOT, LETTERS, signature, sha, write_once, read_lines, stage_rows, prompt, parse_answer
from practical_subtitles import source_bundle
from continuous_common import context as parent_context, continuous_evidence
from subtitle_common import subtitle_prompt, make_alignment

RESULTS=ROOT/'14_subtitle_completion'
PARENT_SIG='ae099c4c1b9cff19b52ce30454b87b6d64c6e0a92928d48e697f351af77c2027'
ARMS=('no_sub','aligned','continuous')


def prepare_short(args):
    a=copy(args); a.stage='dev60'; parent=parent_context(a)
    if parent['signature']!=PARENT_SIG: raise ValueError('原开发协议漂移')
    rows=stage_rows(ROOT,'short_dev')
    if len(rows)!=219 or len({r['video_id'] for r in rows})!=73: raise ValueError('短视频划分变化')
    sources,_=source_bundle(rows)
    write_once(RESULTS/'short219_sources.json',sources)
    print('short219源字幕CPU审计完成：',len(sources['sources']),'视频；未选预算、未抽帧或推理。')


def key(index,arm): return f'{index}:{arm}'


def query(row,arm,evidence):
    text='' if arm=='no_sub' else evidence['aligned']['text'] if arm=='aligned' else evidence['continuous']['text']
    return subtitle_prompt(row,text)


def load_alignment(ctx,video):
    obj=json.loads((ctx['store']/'alignments'/f'{video}.json').read_text())
    if obj['signature']!=ctx['signature'] or obj['evidence_sha256']!=signature(obj['evidence']):
        raise ValueError('短视频对齐证据签名不符')
    e=obj['evidence']; original=e['aligned']
    expected=make_alignment(video,ctx['sources']['sources'][video],ctx['cues'][video],
        original['frame_indices'],original['frame_seconds'],ctx['count_tokens'])
    if original!=expected or e['continuous']!=continuous_evidence(expected,ctx['cues'][video],ctx['count_tokens'],ctx['tokens']):
        raise ValueError('短视频对齐/连续字幕漂移')
    return e


def get_evidence(ctx,video):
    return ctx['evidence'][video] if ctx['scope']=='budget' else load_alignment(ctx,video)


def old_source(ctx,idx,arm):
    """仅提供身份可追溯的旧单元；不从历史219题混有GT的日志中取推理答案。"""
    parent=ctx['parent']; old=parent['old']
    if ctx['scope']=='budget' or arm=='continuous':
        item=parent['units'].get(idx)
        if item is None: return None
        return dict(record=item,raw_text=item['raw_text'],pred=item['pred'],input_sha256=item['input_sha256'],
            prompt_sha256=item['prompt_sha256'],visual=item['visual'],signature=parent['signature'],
            record_sha256=item['record_sha256'],path=str(parent['store']/'units.jsonl'))
    if arm=='aligned':
        item=old['units'].get(idx)
        if item is None: return None
        return dict(record=item,raw_text=item['raw_text'],pred=item['pred'],input_sha256=item['input_sha256'],
            prompt_sha256=item['prompt_sha256'],visual=item['visual'],signature=old['signature'],
            record_sha256=item['record_sha256'],path=str(old['store']/'units.jsonl'))
    item=old['base']['units'].get(f'{idx}:frames:p0')
    if item is None: return None
    return dict(record=item,raw_text=item['raw_text'],pred=item['semantic_pred'],input_sha256=item['prefix_sha256'],
        prompt_sha256=item['prompt_sha256'],visual={n:item[n] for n in ['grid_thw','pv_shape','patch_time_range','n_tok']},
        signature=old['base']['signature'],record_sha256=signature(item),path=str(old['base']['store']/'units.jsonl'))


def validate_record(ctx,row,e,record,existing):
    item=dict(record); digest=item.pop('record_sha256',None); idx=row['index']; arm=item['arm']; k=key(idx,arm)
    if (digest!=signature(item) or item['signature']!=ctx['signature'] or item['index']!=idx
            or item['video_id']!=row['video_id'] or arm not in ctx['arms'] or k in existing):
        raise ValueError('收尾单元损坏/重复/身份不符')
    tokens=0 if arm=='no_sub' else e['aligned']['text_tokens'] if arm=='aligned' else e['continuous']['text_tokens']
    if (item['pred']!=parse_answer(item['raw_text']) or item['pred'] not in ['',*LETTERS]
            or item['prompt_sha256']!=signature(query(row,arm,e)) or item['evidence_sha256']!=signature(e)
            or item['subtitle_tokens']!=tokens or item['checks']['native_input_equal'] is not True
            or not isinstance(item['input_sha256'],str) or len(item['input_sha256'])!=64
            or type(item['input_tokens']) is not int or item['input_tokens']<=item['visual']['n_tok']):
        raise ValueError('答案、输入、证据或原生检查不符')
    visual=item['visual']; original=e['aligned']
    if (len(visual['grid_thw'])!=32 or visual['patch_time_range']!=[float(original['frame_indices'][0]),float(original['frame_indices'][-1])]
            or type(visual['n_tok']) is not int or visual['n_tok']<=0):
        raise ValueError('视觉布局/时间位置非法')
    if any(not math.isfinite(item[n]) or item[n]<0 for n in ['inference_s','preprocess_s','peak_allocated_gib']):
        raise ValueError('成本数值非法')
    origin=item['origin']; source=old_source(ctx,idx,arm)
    if origin=='reused_old':
        if source is None or item['reuse_proof']!=dict(source_signature=source['signature'],source_record_sha256=source['record_sha256'],source_path=source['path'],
                input_sha256_equal=True,visual_equal=True,tensor_equal=True):
            raise ValueError('复用来源或逐tensor检查不符')
        if any(item[n]!=source[n] for n in ['raw_text','pred','input_sha256','visual','prompt_sha256']):
            raise ValueError('复用输入或旧答案被改变')
        if item['new_answer_calls']!=0 or item['generated_tokens'] is not None:
            raise ValueError('复用不应新增生成调用')
    elif origin=='reused_current_no_sub':
        base=existing.get(key(idx,'no_sub'))
        if base is None or arm=='no_sub' or query(row,arm,e)!=query(row,'no_sub',e):
            raise ValueError('无字幕回退没有同题相同输入')
        if any(item[n]!=base[n] for n in ['raw_text','pred','input_sha256','visual','input_tokens']):
            raise ValueError('缺字幕回退改变了回答或输入')
        if item['reuse_proof']!=dict(no_sub_record_sha256=base['record_sha256'],tensor_equal=True) or item['new_answer_calls']!=0:
            raise ValueError('本次无字幕回退引用不符')
    elif origin=='generated':
        if item['new_answer_calls']!=1 or type(item['generated_tokens']) is not int or not 1<=item['generated_tokens']<=32 or item['reuse_proof'] is not None:
            raise ValueError('生成来源或长度不符')
    else: raise ValueError('未知来源')
    if origin!='generated' and (item['inference_s']!=0 or item['peak_allocated_gib']!=0):
        raise ValueError('复用单元不应记新生成开销')
    if ctx['scope']=='budget':
        changed=idx in ctx['changed']
        if (origin=='generated')!=changed or item['checks']['old_2048_input_equal'] is not True:
            raise ValueError('预算对照只允许24题新生成，其他必须验证后复用')
        source=old_source(ctx,idx,'continuous')
        if item['checks']['old_2048_input_sha256']!=source['input_sha256'] or visual!=source['visual']:
            raise ValueError('旧2048输入或视觉布局未复现')
    elif arm!='no_sub':
        base=existing.get(key(idx,'no_sub'))
        if base is None or visual!=base['visual']: raise ValueError('短视频三路视觉不相同或缺少基线')
    return k


def context(args,initialize=False):
    a=copy(args); a.stage='dev60'; parent=parent_context(a)
    if parent['signature']!=PARENT_SIG or not {r['index'] for r in parent['rows']}.issubset(parent['units']):
        raise ValueError('原2048开发结果不完整或漂移')
    summary=parent['stage_dir']/'summary.json'; old_summary=json.loads(summary.read_text())
    if not old_summary['audit']['passed'] or old_summary['N']!=180: raise ValueError('原2048审计未通过')
    folder=Path(__file__).parent
    protocol=dict(version='subtitle_completion_v1',scope=args.scope,parent_protocol=parent['protocol'],parent_signature=PARENT_SIG,
        parent_summary_sha256=sha(summary),parent_journal_sha256=sha(parent['store']/'units.jsonl'),
        generation=dict(frames=32,max_pixels=150000,min_pixels=3136,do_sample=False,seed=0,max_new_tokens=32),
        scripts={n:sha(folder/n) for n in ['subtitle_completion_common.py','subtitle_completion_run.py','subtitle_completion_analyze.py']})
    count=parent['old']['count_tokens']; cues=parent['old']['cues']; evidence={}; changed=set(); sources=None
    if args.scope=='budget':
        tokens=3072; rows=parent['rows']; arms=('continuous',)
        for video in sorted({r['video_id'] for r in rows}):
            previous=parent['evidence'][video]
            current=continuous_evidence(previous['original'],cues[video],count,3072)
            if current['omitted_neighbor_ids']: raise ValueError('3072仍省略邻句，超出本次收尾范围')
            evidence[video]=dict(aligned=previous['original'],continuous=current)
            if current['text']!=previous['text']:
                changed.update(r['index'] for r in rows if r['video_id']==video)
        if len(changed)!=24 or len({r['video_id'] for r in rows if r['index'] in changed})!=8:
            raise ValueError('预算变化题不再是8视频/24题，停止审查')
        protocol.update(tokens=3072,indices=[r['index'] for r in rows],changed_indices=sorted(changed),
            evidence_sha256={v:signature(e) for v,e in evidence.items()},
            reuse='156: full 2048/3072 tensors equal plus saved old prefix/visual/source identity; 24: new generation only')
        candidate=RESULTS/'budget3072_candidate.json'; stage_dir=RESULTS/'budget3072_dev180'
    else:
        chosen=RESULTS/'selected_short_config.json'
        if not chosen.exists(): raise ValueError('先完成预算汇总，再显式 freeze-short --tokens 2048 或 3072')
        selected=json.loads(chosen.read_text()); budget_candidate=RESULTS/'budget3072_candidate.json'
        if selected['budget_candidate_sha256']!=sha(budget_candidate): raise ValueError('冻结预算来源不符')
        budget_result=RESULTS/'budget3072_dev180/summary.json'
        if selected['budget_summary_sha256']!=sha(budget_result) or not json.loads(budget_result.read_text())['audit']['passed']:
            raise ValueError('预算汇总来源变更或未通过')
        tokens=selected['tokens']
        if tokens not in [2048,3072]: raise ValueError('只允许一次固定的2048/3072选择')
        if selected['source_script_sha256']!=protocol['scripts']: raise ValueError('冻结后的收尾源码发生变化')
        rows=stage_rows(ROOT,'short_dev'); arms=ARMS
        if len(rows)!=219 or len({r['video_id'] for r in rows})!=73: raise ValueError('short划分应为219题/73视频')
        sources,cues=source_bundle(rows)
        prepared=RESULTS/'short219_sources.json'
        if selected['short_sources_sha256']!=sha(prepared) or json.loads(prepared.read_text())!=sources:
            raise ValueError('冻结后的短视频字幕来源发生变化')
        protocol.update(tokens=tokens,indices=[r['index'] for r in rows],sources=sources,selection_sha256=sha(chosen),
            plain='generate current control unless old six-question input identity is verified; no blind historical219 reuse',
            aligned='original frame-hit protocol, unchanged 2048 safety cap; no prefix truncation',
            continuous='selected fixed budget; same neighbor1+time block function as development')
        candidate=RESULTS/f'short219_t{tokens}_candidate.json'; stage_dir=RESULTS/f'short219_t{tokens}'
    protocol['arms']=list(arms)
    if initialize: write_once(candidate,protocol)
    if not candidate.exists() or json.loads(candidate.read_text())!=protocol: raise ValueError('先init；配置变化不覆盖旧清单')
    sig=signature(protocol); store=ROOT/'cache/subtitle_completion'/sig
    ctx=dict(scope=args.scope,tokens=tokens,parent=parent,protocol=protocol,signature=sig,store=store,
        stage_dir=stage_dir,rows=rows,arms=arms,changed=changed,evidence=evidence,cues=cues,sources=sources,
        count_tokens=count,units={},expected={key(r['index'],arm) for r in rows for arm in arms})
    if args.scope=='budget':
        for video,e in evidence.items():
            path=store/'evidence'/f'{video}.json'; obj=dict(signature=sig,evidence=e,evidence_sha256=signature(e))
            if initialize: write_once(path,obj)
            if not path.exists() or json.loads(path.read_text())!=obj: raise ValueError('预算证据清单不存在或漂移')
    lookup={r['index']:r for r in rows}; cache={}
    for record in read_lines(store/'units.jsonl'):
        if record.get('index') not in lookup: raise ValueError('收尾单元越界')
        row=lookup[record['index']]; video=row['video_id']
        if video not in cache: cache[video]=get_evidence(ctx,video)
        k=validate_record(ctx,row,cache[video],record,ctx['units']); ctx['units'][k]=record
    return ctx
