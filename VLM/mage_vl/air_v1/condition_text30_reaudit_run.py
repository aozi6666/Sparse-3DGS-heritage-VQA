"""离线重审固定30题最后一次缓存；无API、env加载、视频或QA入口。"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import air_protocol as p
import condition_text30_run as source
import condition_text_run as source5
import condition_text30_reaudit as k

OUT = p.ROOT / k.VERSION
CODE = ['condition_text30_reaudit.py', 'condition_text30_reaudit_run.py', 'run_condition_text30_reaudit.sh',
        'condition_text30.py', 'condition_text30_run.py', 'condition_text_parser.py', 'condition_text_api.py',
        'condition_text_run.py', 'air_protocol.py']


def offline_guard():
    """main时启用：即使未来误接网络/密钥/视觉路径，也直接停止。"""
    explicit = os.environ.get('DEEPSEEK_ENV_FILE')
    secret_path = Path(explicit).expanduser().resolve() if explicit else None
    def hook(event, args):
        if event in ['socket.connect', 'socket.getaddrinfo']:
            raise RuntimeError('离线重审禁止网络')
        if event == 'import' and args[0].split('.')[0] in ['torch', 'transformers', 'decord', 'cv2']:
            raise RuntimeError('离线重审禁止加载视觉模型或解码器')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if (path.name in ['.env', 'deepseek.env', 'evaluation_labels.json', 'predictions.json', 'units.jsonl']
                or path.name.startswith('.env.') or (secret_path and path.resolve() == secret_path)
                or path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.safetensors', '.parquet', '.pt', '.pth']
                or 'oracle_evidence_diag_v1' in str(path)):
                raise RuntimeError('离线重审禁止读取密钥、真值、旧QA预测、人工证据或视频/权重')
    sys.addaudithook(hook)


def last_response(c, index):
    unit = c['source']['units'][index]
    if unit.get('origin') == 'unsupported_no_API':
        return None
    if unit.get('origin') == 'reused_parse5':
        old = c['source']['parent']['units'][index]
        path = source5.OUT / 'responses' / (old['attempts'][-1]['slot'] + '.json')
    else:
        path = source.OUT / 'responses' / (unit['attempts'][-1]['slot'] + '.json')
    cached = json.loads(path.read_text()); p.unseal(cached)
    choice = cached['response']['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('缓存最终响应未正常结束，不能靠新规则当成功')
    if choice['message']['content'] != unit['raw_text']:
        raise ValueError('逐题记录与最终原响应不一致')
    return dict(path=str(path), sha256=p.sha(path), raw_text=unit['raw_text'])


def context(initialize=False):
    if subprocess.check_output(['git', '-C', str(p.PROJECT), 'branch', '--show-current'], text=True).strip() != p.BRANCH:
        raise ValueError('分支不符，保留现场')
    old = source.context()
    ids = old['protocol']['indices']
    if len(old['units']) != 30 or len(ids) != 30:
        raise ValueError('先完成旧30题记录（含失败和unsupported），不补跑API')
    summary_path = source.OUT / 'parse_review/parse30/summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['signature'] != old['signature'] or summary['N'] != 30:
        raise ValueError('旧30题汇总来源不符')
    files = [source.OUT / 'candidate.json', summary_path, source.OUT / 'parse_review/parse30/review_packet.json',
             source.OUT / 'attempts.jsonl', source5.OUT / 'candidate.json', source5.OUT / 'parse_review/parse5/summary.json',
             p.OLD / '01_manifests/inputs.json', p.OLD / '01_manifests/splits.json',
             p.ROOT / 'condition_clips_noobs_qa30/candidate.json']
    for root in [source.OUT, source5.OUT]:
        files.extend(sorted((root / 'parses').glob('*.json')))
        files.extend(sorted((root / 'responses').glob('*.json')))
    parts = {i: k.partition(old['rows'][i], old['fixed'][i]) for i in ids}
    folder = Path(__file__).parent
    protocol = dict(version=k.VERSION, source_signature=old['signature'], indices=ids,
                    source_files={str(path): p.sha(path) for path in files},
                    source_records={i: old['units'][i]['record_sha256'] for i in ids},
                    partitions=parts, code={name: p.sha(folder / name) for name in CODE},
                    policy='Offline re-audit of FINAL cached responses only. No rewrite, new API, earlier-attempt selection, '
                           'GT or video. Source conditions/identities retained; no localization capability upgrades.')
    if initialize:
        p.write_once(OUT / 'candidate.json', protocol)
    if not (OUT / 'candidate.json').exists() or json.loads((OUT / 'candidate.json').read_text()) != protocol:
        raise ValueError('先init；来源/代码变化拒绝覆盖，请另立版本')
    c = dict(signature=p.digest(protocol), protocol=protocol, source=old, out=OUT, units={})
    for i in ids:
        path = OUT / 'records' / (i + '.json')
        if path.exists():
            unit = json.loads(path.read_text()); p.unseal(unit)
            if unit != audit_unit(c, i):
                raise ValueError('离线记录回放漂移：' + i)
            c['units'][i] = unit
    return c


def audit_unit(c, index):
    old = c['source']; original = old['units'][index]
    part = c['protocol']['partitions'][index]
    final = last_response(c, index)
    parsed, errors, status = None, [], 'unsupported'
    if final:
        parsed, errors = k.validate(final['raw_text'], old['rows'][index], old['fixed'][index], part)
        if parsed is not None and original['parsed'] is not None and parsed != original['parsed']:
            raise ValueError('原有效查询在离线阶段被改写')
        status = 'unchanged_valid' if parsed is not None and original['parsed'] is not None else 'recovered_cached_final' if parsed is not None else 'still_invalid'
    return p.seal(dict(signature=c['signature'], index=index, source_record_sha256=original['record_sha256'],
                       final_response=final, partition_sha256=p.digest(part), status=status, parsed=parsed,
                       errors=errors, original_error=original['error'], new_API_calls=0,
                       semantic_review='pending_text_meaning_review', localization_certified=False))


def run(c):
    from tqdm import tqdm
    ids = c['protocol']['indices']
    with tqdm(total=len(ids), initial=len(c['units']), desc='30题缓存离线重审', unit='题', dynamic_ncols=True) as bar:
        for i in ids:
            if i in c['units']:
                continue
            unit = audit_unit(c, i)
            p.write_once(OUT / 'records' / (i + '.json'), unit)
            c['units'][i] = unit
            bar.update(1)


def summarize(c):
    if len(c['units']) != 30:
        raise ValueError('先run完成30题缓存重审')
    old = c['source']; details = []
    for i in c['protocol']['indices']:
        u = c['units'][i]
        details.append(dict(index=i, question=old['rows'][i]['question'], options=old['rows'][i]['bodies'],
                            original_contract=old['fixed'][i], partition=c['protocol']['partitions'][i],
                            cached_final_queries=u['parsed'], status=u['status'], errors=u['errors'],
                            original_error=u['original_error'], source_final_response=u['final_response'],
                            localization_status='not_certified; original_capability_limits_remain'))
    states = Counter(u['status'] for u in c['units'].values())
    report = dict(signature=c['signature'], N=30, original_valid=sum(u['parsed'] is not None for u in old['units'].values()),
                  revised_text_checks_valid=sum(u['parsed'] is not None for u in c['units'].values()),
                  statuses=dict(states), recovered=[i for i,u in c['units'].items() if u['status']=='recovered_cached_final'],
                  unsupported=[i for i,u in c['units'].items() if u['status']=='unsupported'],
                  still_invalid=[i for i,u in c['units'].items() if u['status']=='still_invalid'],
                  new_API_calls=0, new_model_generations=0, new_video_decodes=0, QA_calls=0,
                  historical_cost=json.loads((source.OUT/'parse_review/parse30/summary.json').read_text())['cost'],
                  historical_parse5_cost=json.loads((source5.OUT/'parse_review/parse5/summary.json').read_text())['cost'],
                  semantic_pass=None, accuracy_gain_evidence=False,
                  note='Posthoc engineering correction of mandatory-word partition. FINAL cached response is unchanged; '
                       'no earlier-attempt choice. Text checks are not visual grounding/QA gain. '
                       'Future/after operators retained structurally; global, speech, first/count/absence capability limits unchanged.')
    p.write_once(OUT / 'results/summary.json', report)
    p.write_once(OUT / 'results/review_packet.json', details)
    # 防止离线重审过程中旧数据被修改；不依赖“重审成功”来忽略来源漂移。
    if any(p.sha(path) != sha for path, sha in c['protocol']['source_files'].items()):
        raise ValueError('源文件在运行中变化，保留记录并停止')
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('historical_')}, ensure_ascii=False, indent=2))
    print('离线审阅：', OUT / 'results/review_packet.json')
    if states['still_invalid']:
        raise ValueError('仍有不通过记录，保持失败，不自动放宽或请求API')


def main():
    offline_guard()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['init', 'preflight', 'run', 'summarize'])
    args = ap.parse_args()
    print('SPACE', p.disk_guard(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = context(args.action == 'init')
        if args.action in ['init','preflight']:
            print(f'离线清单30题；已有{len(c["units"])}/30；不读取env、不访问API。签名：{c["signature"]}')
        elif args.action == 'run':
            run(c)
        else:
            summarize(c)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('停止：'+str(exc), file=sys.stderr)
        sys.exit(1)
