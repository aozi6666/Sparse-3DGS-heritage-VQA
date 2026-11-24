"""冻结7题QA契约：来源、输入与答案复用均显式验证；无模型/标签读取。"""
from copy import deepcopy
import math
import re

import air_protocol as p

VERSION='leaf_frozen7_qa30_v1'
FP_FIELDS=('tensor_sha','baseline_tensor_sha','input_tokens','visual_tokens','grid_thw')


def finite_cost(value):
    return type(value) in [int,float] and math.isfinite(value) and value>=0


def frame_ids(ids,count):
    if not isinstance(ids,list) or not 1<=len(ids)<=32 or ids!=sorted(set(ids)) or any(type(x)is not int or not 0<=x<count for x in ids):raise ValueError('冻结原帧序列异常')


def build_plans(indices,groups,locations,oldplans,meta):
    joined=sum(groups.values(),[])
    if set(groups)!={'candidate','anchor_only','fallback'} or [len(groups[n]) for n in ['candidate','anchor_only','fallback']]!=[7,4,19] or len(joined)!=30 or len(set(joined))!=30 or set(joined)!=set(indices):raise ValueError('固定7/4/19范围异常')
    plans={}
    for i in indices:
        group=next(g for g,items in groups.items() if i in items);m=meta[i];loc=locations[i];old=oldplans[i]
        if loc['index']!=i or loc['group']!=group or loc['original_noobs_indices']!=old['indices']:raise ValueError('定位记录/原输入来源异常')
        if old['fps']!=m['fps'] or old['count']!=m['frame_count']:raise ValueError('原noobs时间轴不符')
        chosen=deepcopy(old['indices'])
        if group=='candidate':
            if loc['status']!='localized_candidates_only' or loc['result'] is None:raise ValueError('7候选定位不完整，不自动删题或回退')
            chosen=deepcopy(loc['result']['indices'])
        frame_ids(chosen,m['frame_count']);frame_ids(old['indices'],m['frame_count'])
        changed=chosen!=old['indices']
        if group!='candidate' and changed:raise ValueError('23题复用范围被改动')
        plans[i]=dict(indices=chosen,old_noobs_indices=deepcopy(old['indices']),fps=m['fps'],count=m['frame_count'],group=group,
                      generate=group=='candidate' and changed,unchanged_from_noobs=not changed)
    return plans


def validate_input(fp,ids):
    if set(fp)!=set(FP_FIELDS):raise ValueError('输入指纹字段不完整')
    if any(not isinstance(fp[n],str) or not re.fullmatch('[0-9a-f]{64}',fp[n]) for n in ['tensor_sha','baseline_tensor_sha']):raise ValueError('输入Tensor哈希异常')
    grid=fp['grid_thw']
    if not isinstance(grid,list) or len(grid)!=len(ids) or any(not isinstance(g,list) or len(g)!=3 or g[0]!=1 or any(type(x)is not int or x<1 for x in g) for g in grid):raise ValueError('输入网格/帧数异常')
    if type(fp['visual_tokens'])is not int or fp['visual_tokens']<=0 or type(fp['input_tokens'])is not int or fp['input_tokens']<=fp['visual_tokens']:raise ValueError('输入token计数异常')


def source_refs(c,i):
    return dict(source_noobs_record_sha=c['noobs']['units'][i]['record_sha256'],
                source_uniform_record_sha=c['noobs']['bases'][i]['record_sha256'])


def validate_audit(c,u):
    p.unseal(u);i=u['index']
    if i not in c['rows'] or u['signature']!=c['signature']:raise ValueError('输入审计身份异常')
    old=c['noobs']['units'][i]
    if any(u[name]!=value for name,value in source_refs(c,i).items()):raise ValueError('审计来源记录不符')
    if u['indices']!=c['plans'][i]['old_noobs_indices'] or u['indices']!=old['indices'] or u['native_equal'] is not True:raise ValueError('审计原帧/原生适配不符')
    if u['prompt_sha']!=p.digest(p.prompt(c['rows'][i])) or u['prompt_sha']!=old['prompt_sha']:raise ValueError('审计QA提示词漂移')
    validate_input(u['input'],u['indices'])
    if u['input']!={n:old[n] for n in FP_FIELDS}:raise ValueError('重建的完整uniform/noobs输入Tensor与旧记录不一致')
    if not finite_cost(u['preprocess_s']):raise ValueError('审计成本异常')
    return u


def validate_generation(c,g):
    p.unseal(g);i=g['index']
    if i not in c['rows'] or g['signature']!=c['signature'] or not c['plans'][i]['generate']:raise ValueError('非7候选新输入禁止生成')
    audit=c['audits'].get(i)
    if audit is None:raise ValueError('缺少源输入审计')
    validate_audit(c,audit)
    if g['audit_record_sha']!=audit['record_sha256'] or any(g[name]!=value for name,value in source_refs(c,i).items()):raise ValueError('生成审计/来源链异常')
    if g['indices']!=c['plans'][i]['indices'] or g['prompt_sha']!=p.digest(p.prompt(c['rows'][i])) or g['native_equal'] is not True:raise ValueError('新输入帧/提示词/原生路径漂移')
    validate_input(g['input'],g['indices'])
    if g['input']['baseline_tensor_sha']!=audit['input']['baseline_tensor_sha']:raise ValueError('新生成原生uniform输入与历史审计不符')
    if g['input']['tensor_sha']==audit['input']['tensor_sha']:raise ValueError('帧已改变但仍使用原noobs输入')
    tokens=g['generated_token_ids']
    if not isinstance(tokens,list) or not 1<=len(tokens)<=32 or any(type(x)is not int or x<0 for x in tokens):raise ValueError('生成token长度/类型异常')
    if not isinstance(g['raw_text'],str) or g['pred']!=p.parse_answer(g['raw_text']):raise ValueError('生成答案解析漂移')
    if any(not finite_cost(g[n]) for n in ['preprocess_s','inference_s','peak_gib']):raise ValueError('生成成本异常')
    return g


def make_unit(c,i,generation=None):
    audit=c['audits'].get(i)
    if audit is None:raise ValueError('先完成输入审计')
    validate_audit(c,audit);plan=c['plans'][i];old=c['noobs']['units'][i]
    if plan['generate']:
        if generation is None:raise ValueError('缺少7候选原生成缓存')
        validate_generation(c,generation)
        if generation['index']!=i:raise ValueError('生成归属题号异常')
        origin='new_generation';data=generation;tokens=len(generation['generated_token_ids']);link=generation['record_sha256']
    else:
        if generation is not None:raise ValueError('复用题不允许接入新生成')
        origin='reused_noobs';data=dict(raw_text=old['raw_text'],pred=old['pred'],input=audit['input'],preprocess_s=0.,inference_s=0.,peak_gib=0.);tokens=0;link=None
    return p.seal(dict(signature=c['signature'],index=i,group=plan['group'],origin=origin,indices=plan['indices'],
                       **source_refs(c,i),audit_record_sha=audit['record_sha256'],generation_record_sha=link,
                       prompt_sha=p.digest(p.prompt(c['rows'][i])),input=deepcopy(data['input']),raw_text=data['raw_text'],pred=data['pred'],
                       native_equal=True,new_calls=int(plan['generate']),generated_tokens=tokens,
                       preprocess_s=data['preprocess_s'],inference_s=data['inference_s'],peak_gib=data['peak_gib']))


def validate_unit(c,u,generation=None):
    p.unseal(u)
    if u!=make_unit(c,u['index'],generation):raise ValueError('逐题记录与审计/原始生成/复用来源不符')
    return u


def require_audits(c):
    ids=c['protocol']['indices']
    if set(c['audits'])!=set(ids):raise ValueError('先完成audit-inputs的30题源输入身份审计')
    for a in c['audits'].values():validate_audit(c,a)


def require_complete(c):
    if len(c.get('units',{}))!=30:raise ValueError('30题预测未完整，不读标签或汇总')
    require_audits(c)
    if set(c['units'])!=set(c['protocol']['indices']):raise ValueError('30题预测范围异常')
    for i,u in c['units'].items():validate_unit(c,u,c['generations'].get(i))
