"""新增90题隔离编排；不修改已冻结AIR或g16+l16模块。"""
import argparse,fcntl,json,os,gc
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import air_protocol as p
import air_hybrid as h

EXT=p.ROOT/'extension90_v1'

def remaining(pool,first):
    ids={r['index'] for r in first};videos={r['video_id'] for r in first}
    rows=[r for r in pool if r['index'] not in ids]
    if len(rows)!=90 or len({r['index'] for r in rows})!=90 or len({r['video_id'] for r in rows})!=30 or {r['video_id'] for r in rows}&videos:raise ValueError('新增90题/30视频划分异常或重叠')
    return rows

def source_context(init=False):
    # 读取并验证原候选；只在独立source_cache写新增题，绝不追加旧账本。
    first=p.context('dev30');frozen=h.context()
    if len(frozen['units'])!=90:raise ValueError('前90题候选尚未完整')
    rows=remaining(p.load_rows()[0],first['rows']);folder=Path(__file__).parent
    manifest=dict(version='extension90_v1',source_signature=first['signature'],hybrid_signature=frozen['signature'],
        indices=[r['index'] for r in rows],videos=sorted({r['video_id'] for r in rows}),
        code={n:p.sha(folder/n) for n in ['air_extension.py','run_air_extension.sh','air_hybrid.py','run_air_hybrid.sh']},
        prior_summary_sha256=p.sha(h.OUT/'dev30/summary.json'),
        policy='Same frozen AIR+g16l16; disjoint remaining90; independent stores; invalid score formats excluded and reported,other audit failure stops')
    if init:p.write_once(EXT/'manifest.json',manifest)
    if not (EXT/'manifest.json').exists() or json.loads((EXT/'manifest.json').read_text())!=manifest:raise ValueError('先init；扩样来源/源码不覆盖')
    c=dict(first,rows=rows,store=EXT/'source_cache'/first['signature'],stage_dir=EXT/'source90',units={})
    c['expected']={f'{r["index"]}:{arm}' for r in rows for arm in p.ARMS};ids={r['index'] for r in rows}
    for u in p.read_journal(c['store']/'units.jsonl'):
        if u['index'] not in ids:raise ValueError('隔离账本含其他阶段题目')
        key=p.validate_unit(c,u,c['units']);c['units'][key]=u
    return c

@contextmanager
def hybrid_scope(source):
    """仅本进程重定向数据/输出，选择函数与问答逻辑逐字使用冻结代码。"""
    original_context=p.context;original_out=h.OUT
    p.context=lambda *a,**kw:source;h.OUT=EXT/'hybrid'
    try:yield
    finally:p.context=original_context;h.OUT=original_out

def source_summary(source):
    import air_analyze as a
    old=a.ROOT;a.ROOT=EXT/'source_archive'
    try:
        try:a.summarize(source)
        except ValueError:
            path=source['stage_dir']/'summary.json'
            if not path.exists():raise
            report=json.loads(path.read_text())
            # 沿用原混合候选政策：仅格式非法评分可排除，不掩盖其他错误。
            if report['audit']['invalid_answers'] or any('malformed analysis scores' not in e for e in report['audit']['errors']):raise
            print('历史评分格式警告已保留并排除；继续冻结混合规则。')
    finally:a.ROOT=old

def summarize(source,c):
    if len(c['units'])!=90:raise ValueError('新90题尚未完整，不读取标签')
    path=p.OLD/'01_manifests/evaluation_labels.json';lock=json.loads((path.parent/'dataset_lock.json').read_text())
    if p.sha(path)!=lock['labels_sha256']:raise ValueError('标签来源漂移')
    labels=json.loads(path.read_text());rows=source['rows']
    import sys
    sys.path.insert(0,str(p.PROJECT/'VLM/mage_vl/practical'))
    from practical_analyze import comparison
    arms=h.ARMS
    preds={arm:{r['index']:(c['units'][r['index']] if arm=='hybrid32' else source['units'][f'{r["index"]}:{arm}'])['pred'] for r in rows} for arm in arms}
    def group(rr):
        if not rr:return dict(N=0)
        return dict(N=len(rr),videos=len({r['video_id'] for r in rr}),methods={arm:dict(correct=sum(preds[arm][r['index']]==labels[r['index']] for r in rr),accuracy=sum(preds[arm][r['index']]==labels[r['index']] for r in rr)/len(rr)) for arm in arms},versus_uniform=comparison(rr,preds['hybrid32'],preds['uniform32'],labels),versus_air=comparison(rr,preds['hybrid32'],preds['air_cap32'],labels))
    uu=list(c['units'].values());invalid=sum(u['pred'] not in list('ABCD') for u in uu)
    report=dict(stage='additional90',signature=c['signature'],audit=dict(passed=not invalid,invalid_answers=invalid,excluded_source_score_formats=c['protocol']['inherited_invalid_scores']),overall=group(rows),by_duration={d:group([r for r in rows if r['duration']==d]) for d in ['medium','long']},cost=dict(new_QA_calls=sum(u['new_calls'] for u in uu),reused_answers=sum(not u['new_calls'] for u in uu),inference_s=sum(u['inference_s'] for u in uu),preprocess_s=sum(u['preprocess_s'] for u in uu),visual_tokens=sum(u['visual_tokens'] for u in uu)),source_cost=json.loads((source['stage_dir']/'summary.json').read_text())['cost'],note='Frozen candidate extension on remaining development90,zero question/video overlap with first90. Not official full benchmark or new independent test. No parameter changes; source selector cost included.')
    out=EXT/'additional90';details=[dict(index=r['index'],video_id=r['video_id'],gt=labels[r['index']],predictions={a:preds[a][r['index']] for a in arms}) for r in rows]
    p.write_once(out/'predictions.json',details);p.write_once(out/'summary.json',report);p.write_once(out/'units_snapshot.json',uu)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('新增90题汇总路径:',out/'summary.json')
    if invalid:raise ValueError('答案审计失败，保留结果')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','preflight','run','summarize']);args=parser.parse_args()
    print('SPACE',p.disk_guard());source=source_context(args.action=='init')
    print('新增90题/30视频，旧阶段零重叠；source答案',len(source['units']),'/180')
    if args.action in ['init','preflight']:return
    if args.action=='run':
        if os.environ.get('PRACTICAL_STAGE_CONFIRMED')!='1':raise RuntimeError('用户确认才启动')
        with (p.OLD/'cache/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);source=source_context()
            if not source['expected'].issubset(source['units']):
                p.write_once(source['store']/'protocol.json',source['protocol'])
                import air_runtime as runtime
                original_root=runtime.ROOT;runtime.ROOT=EXT
                try:runtime.run_gpu(SimpleNamespace(stage='additional90 source',min_free=10,budget=1),source)
                finally:runtime.ROOT=original_root
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            source_summary(source)
            with hybrid_scope(source):
                print('开始新增90题混合候选；冻结模块进度条沿用dev30字样，实际题号以扩样manifest为准。',flush=True)
                c=h.context(init=True);h.run(c)
        print('两阶段完成，请summarize；不会进入全量。')
    else:
        if not source['expected'].issubset(source['units']):raise ValueError('source90尚未完整')
        with hybrid_scope(source):c=h.context()
        summarize(source,c)

if __name__=='__main__':main()
