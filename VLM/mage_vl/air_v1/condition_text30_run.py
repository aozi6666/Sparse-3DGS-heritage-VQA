"""先验证固定30题；本入口只有parse30，无定位、视频读取或最终QA动作。"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess

import air_protocol as p
import condition_text_parser as base
import condition_text30 as k
import condition_text_run as parent
import condition_text_api as api

OUT = p.ROOT / k.VERSION
CODE = ['condition_text30.py','condition_text30_run.py','run_condition_text30.sh','condition_text_parser.py','condition_text_api.py','condition_text_run.py','run_condition_text_parser.sh','air_protocol.py']


def context(initialize=False):
    branch = subprocess.check_output(['git', '-C', str(p.PROJECT), 'branch', '--show-current'], text=True).strip()
    if branch != p.BRANCH:
        raise ValueError('分支不符；保留现有现场')
    source_path = p.ROOT / 'condition_clips_noobs_qa30/candidate.json'
    source = json.loads(source_path.read_text())
    if len(source['indices']) != 30 or not set(base.PARSE5) <= set(source['indices']):
        raise ValueError('固定30题不在旧开发30题中')
    pool, _ = p.load_rows()
    lookup = {row['index']: row for row in pool}
    # 只把题面和完整四选项带入新上下文，不转交路径、旧预测、人工位置或任务标签。
    ids = source['indices']
    previous = parent.context()
    if len(previous['units']) != 5 or any(u['parsed'] is None for u in previous['units'].values()):
        raise ValueError('先完成旧parse5')
    rows = {i: {name: lookup[i][name] for name in ['index', 'question', 'bodies']} for i in ids}
    fixed = {i: previous['fixed'][i] if i in base.PARSE5 else k.contract(row) for i, row in rows.items()}
    folder = Path(__file__).parent
    protocol = dict(version=k.VERSION, indices=ids, parent_signature=previous['signature'], parent_records={i:u['record_sha256'] for i,u in previous['units'].items()}, rows=rows, contracts=fixed,
                    source=dict(frozen30_candidate_sha256=p.sha(source_path),
                                inputs_sha256=p.sha(p.OLD / '01_manifests/inputs.json'),
                                splits_sha256=p.sha(p.OLD / '01_manifests/splits.json')),
                    code={name: p.sha(folder / name) for name in CODE},
                    api=dict(model=k.MODEL, endpoint=k.ENDPOINT, thinking='disabled', temperature=0,
                             max_tokens=k.MAX_TOKENS, response_format='json_object', max_attempts=k.MAX_ATTEMPTS),
                    limits=dict(max_calls=50, reported_token_stop=60000, min_free_gib=10, air_root_max_gib=1),
                    policy='Text query compilation only; frozen30 extension, original5 reused. Program owns event IDs/source order/ranges/relations. '
                           'No video/model loading, no answers, no GT, no semantic certification; no automatic next stage.')
    if initialize:
        p.write_once(OUT / 'candidate.json', protocol)
    if not (OUT / 'candidate.json').exists() or json.loads((OUT / 'candidate.json').read_text()) != protocol:
        raise ValueError('先init；代码/来源变化不覆盖旧协议，需另立版本')
    c = dict(signature=p.digest(protocol), protocol=protocol, rows=rows, fixed=fixed, out=OUT, units={}, parent=previous)
    for i in c['protocol']['indices']:
        path = OUT / 'parses' / (i + '.json')
        if not path.exists():
            continue
        unit = json.loads(path.read_text()); p.unseal(unit)
        expected = compile_record(c, i, require_complete=True)
        if unit != expected:
            raise ValueError('解析单元回放/来源漂移：' + i)
        c['units'][i] = unit
    api.events(p, c)
    return c


def compile_record(c, index, require_complete=False):
    if index in base.PARSE5:
        old = c['parent']['units'][index]
        return p.seal(dict(signature=c['signature'], index=index, contract_sha256=p.digest(c['fixed'][index]), parsed=old['parsed'], error=None, raw_text=old['raw_text'], attempts=[], origin='reused_parse5', source_record_sha256=old['record_sha256'], semantic_review='prior_text_review_only', video_observed=False, answer_produced=False))
    if c['fixed'][index].get('contract_status') == 'unsupported':
        return p.seal(dict(signature=c['signature'], index=index, contract_sha256=p.digest(c['fixed'][index]), parsed=None, error=None, raw_text='', attempts=[], origin='unsupported_no_API', semantic_review='unsupported_not_success', video_observed=False, answer_produced=False))
    attempts = []
    last = None
    error = None
    for n in range(k.MAX_ATTEMPTS):
        body = k.payload(c['rows'][index], c['fixed'][index], repair=error)
        slot = index + '_' + str(n)
        path = OUT / 'responses' / (slot + '.json')
        if not path.exists():
            if require_complete:
                raise ValueError('解析单元缺少原响应：' + slot)
            return None
        cached = api.validate_cached(p, c, path, slot, body)
        last, error, raw = api.parse_response(cached['response'], c['fixed'][index])
        attempts.append(dict(slot=slot, response_sha256=p.sha(path), error=error))
        if last is not None:
            break
    return p.seal(dict(signature=c['signature'], index=index, contract_sha256=p.digest(c['fixed'][index]),
                       parsed=last, error=error, raw_text=raw, attempts=attempts,
                       semantic_review='pending_not_certified', video_observed=False, answer_produced=False))


def parse30(c, retry_uncertain=False):
    for i in c['protocol']['indices']:
        if i not in c['units'] and (i in base.PARSE5 or c['fixed'][i].get('contract_status') == 'unsupported'):
            unit = compile_record(c, i)
            p.write_once(OUT / 'parses' / (i + '.json'), unit)
            c['units'][i] = unit
    # 阶段授权由用户的终端命令显式表达；不在测试/预检中调用网络。
    if os.environ.get('DEEPSEEK_API_CONFIRMED') != '1':
        raise RuntimeError('运行前设置DEEPSEEK_API_CONFIRMED=1；仅发送这30题题面和选项')
    if len(c['units']) == len(c['protocol']['indices']):
        print('30题已落盘，复用结果；没有新API请求。')
        return
    secret = None
    from tqdm import tqdm
    with tqdm(total=30, initial=len(c['units']), desc='A.I.R. 文本条件编译 parse30', unit='题', dynamic_ncols=True) as progress:
        for i in c['protocol']['indices']:
            if i in c['units']:
                continue
            p.disk_guard()
            error = None
            for n in range(k.MAX_ATTEMPTS):
                body = k.payload(c['rows'][i], c['fixed'][i], repair=error)
                slot = i + '_' + str(n)
                path = OUT / 'responses' / (slot + '.json')
                if path.exists():
                    cached = api.validate_cached(p, c, path, slot, body)
                else:
                    if secret is None:
                        secret = api.load_secret(p.PROJECT)
                    cached = api.call(p, c, slot, body, secret, retry_uncertain=retry_uncertain)
                parsed, error, _ = api.parse_response(cached['response'], c['fixed'][i])
                if parsed is not None:
                    break
            unit = compile_record(c, i, require_complete=True)
            p.write_once(OUT / 'parses' / (i + '.json'), unit)
            c['units'][i] = unit
            progress.update(1)
            progress.set_postfix(index=i, valid=unit['parsed'] is not None)
    print('30题记录已保存（旧5题复用，无法拆分题单列）；请summarize30审阅，不自动定位。')


def summarize(c):
    if set(c['units']) != set(c['protocol']['indices']):
        raise ValueError('30题未完成；原命令可继续未完成题')
    details = []
    calls = []
    for i in c['protocol']['indices']:
        u = c['units'][i]
        details.append(dict(index=i, question=c['rows'][i]['question'], all_options=c['rows'][i]['bodies'],
                            fixed_contract=c['fixed'][i], compiled_queries=u['parsed'], error=u['error'],
                            semantic_review='pending_not_certified'))
        for attempt in u['attempts']:
            cached = json.loads((OUT / 'responses' / (attempt['slot'] + '.json')).read_text())
            response = cached['response']; usage = response.get('usage') or {}
            calls.append(dict(slot=attempt['slot'], response_model=response.get('model'), usage=usage,
                              elapsed_s=cached['elapsed_s'], error=attempt['error']))
    invalid = [i for i, u in c['units'].items() if u['parsed'] is None and u.get('origin') != 'unsupported_no_API']
    report = dict(signature=c['signature'], N=30, reused_questions=5, unsupported=[i for i,u in c['units'].items() if u.get('origin')=='unsupported_no_API'], structurally_valid=sum(u['parsed'] is not None for u in c['units'].values()), invalid=invalid,
                  relations=dict(Counter(d['fixed_contract']['relation'] for d in details)),
                  semantic_pass=None, new_video_decodes=0, new_visual_inference=0, final_QA=0,
                  historical_parse5_cost=json.loads((parent.OUT / 'parse_review/parse5/summary.json').read_text())['cost'],
                  cost=dict(persisted_requests=len([x for x in api.events(p, c) if x['event'] == 'start']),
                            saved_responses=len(calls), reported_total_tokens=sum(x['usage'].get('total_tokens', 0) for x in calls),
                            request_wall_s=sum(x['elapsed_s'] for x in calls), calls=calls),
                  note='Cost is new stage requests; historical_parse5_cost retains original cost. Unsupported cases are not successful parses. Program structural checks and lexical preservation only; review meaning manually. '
                       'No answer prediction, no video observation, no accuracy-gain result. '
                       'Hosted model alias/temperature0 do not freeze provider weights or guarantee identical outputs.')
    p.write_once(OUT / 'parse_review/parse30/summary.json', report)
    p.write_once(OUT / 'parse_review/parse30/review_packet.json', details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('30题语义审阅：', OUT / 'parse_review/parse30/review_packet.json')
    if invalid:
        raise ValueError('编译失败已保留，先诊断，不扩样：' + ','.join(invalid))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['init', 'preflight', 'parse30', 'summarize30'])
    ap.add_argument('--retry-uncertain', action='store_true', help='人工核查后才显式重发无缓存请求；仍计入累计10次上限')
    args = ap.parse_args()
    print('SPACE', p.disk_guard(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = context(initialize=args.action == 'init')
        if args.action in ['init', 'preflight']:
            path = api.env_path(p.PROJECT)
            print(f'纯CPU清单固定30题；已完成{len(c["units"])}/30；env文件存在={path.is_file()}。未读取密钥、未联网、未加载模型。')
            print('签名：', c['signature'])
            return
        if args.action == 'parse30':
            parse30(c, args.retry_uncertain)
        else:
            summarize(c)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        # API异常在客户端已脱敏，不打印堆栈中的请求对象。
        import sys
        print('停止：' + str(exc), file=sys.stderr)
        sys.exit(1)
