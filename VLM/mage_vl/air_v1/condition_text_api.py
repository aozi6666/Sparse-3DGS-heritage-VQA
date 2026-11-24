"""标准库DeepSeek客户端；凭据只在发送时驻留内存，不写请求/响应记录。"""
import json
import os
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.request

import condition_text_parser as k


def env_path(project):
    explicit = os.environ.get('DEEPSEEK_ENV_FILE')
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else project / path
    choices = [project / name for name in ['.env', 'deepseek.env'] if (project / name).is_file()]
    if len(choices) > 1:
        raise RuntimeError('检测到两个env文件，请用DEEPSEEK_ENV_FILE明确选择；不自动混用')
    return choices[0] if choices else project / '.env'


def load_secret(project):
    """不source shell，不展开变量/命令；只接受DEEPSEEK_API_KEY这一行。"""
    secret = os.environ.get('DEEPSEEK_API_KEY')
    if not secret:
        path = env_path(project)
        if not path.is_file():
            raise RuntimeError('未找到env或DEEPSEEK_API_KEY；不要把密钥发到聊天')
        found = []
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            key, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and key.strip() == 'DEEPSEEK_API_KEY':
                try:
                    tokens = shlex.split(value, comments=True, posix=True)
                except ValueError:
                    raise RuntimeError('DEEPSEEK_API_KEY引号格式错误；不显示原文') from None
                if len(tokens) != 1:
                    raise RuntimeError('DEEPSEEK_API_KEY格式错误；不显示原文')
                found.append(tokens[0])
        if len(found) != 1:
            raise RuntimeError('env必须有且仅有一个DEEPSEEK_API_KEY；不显示内容')
        secret = found[0]
    if (len(secret) < 16 or any(x.isspace() for x in secret) or 'REPLACE' in secret.upper()
            or any(x in secret for x in ['$', '`', '\x00'])):
        raise RuntimeError('DEEPSEEK_API_KEY为空或仍为占位/表达式；不显示内容')
    return secret


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def transport(payload, secret):
    """只访问固定官方端点；HTTP异常不输出服务端正文或Authorization。"""
    request = urllib.request.Request(k.ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
                                    headers={'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json'})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=120) as response:
        body = response.read(2 * 1024**2 + 1)
    if len(body) > 2 * 1024**2:
        raise RuntimeError('response_too_large')
    obj = json.loads(body)
    if not isinstance(obj, dict):
        raise RuntimeError('response_not_object')
    return obj


def validate_cached(p, c, path, slot, payload):
    cached = json.loads(path.read_text()); p.unseal(cached)
    if cached['signature'] != c['signature'] or cached['slot'] != slot or cached['request_sha256'] != p.digest(payload):
        raise ValueError('API响应来源/请求身份漂移，保留旧文件')
    if cached['request'] != payload:
        raise ValueError('已缓存请求正文漂移')
    return cached


def events(p, c):
    rows = p.read_journal(c['out'] / 'attempts.jsonl')
    starts = {}
    ended = set()
    for row in rows:
        if row.get('signature') != c['signature'] or row.get('event') not in ['start', 'http_error', 'response']:
            raise ValueError('API调用日志身份或事件非法')
        n = row.get('attempt')
        if type(n) is not int or n < 1:
            raise ValueError('API请求编号非法')
        if row['event'] == 'start':
            if n != len(starts) + 1 or n in starts:
                raise ValueError('API请求计数不连续')
            starts[n] = row
        elif n not in starts or n in ended or row['slot'] != starts[n]['slot']:
            raise ValueError('API日志完成事件来源非法')
        else:
            ended.add(n)
    return rows


def call(p, c, slot, payload, secret, retry_uncertain=False, send=transport):
    if not re.fullmatch(r'\d+_[01]', slot):
        raise ValueError('非法API缓存槽位')
    path = c['out'] / 'responses' / (slot + '.json')
    if path.exists():
        return validate_cached(p, c, path, slot, payload)
    journal = c['out'] / 'attempts.jsonl'
    history = events(p, c)
    starts = [e for e in history if e['event'] == 'start']
    prior = [e for e in starts if e['slot'] == slot]
    if any(e['request_sha256'] != p.digest(payload) for e in prior):
        raise ValueError('历史请求正文漂移，不重发')
    if prior and not retry_uncertain:
        raise RuntimeError('此请求曾发出但无可复用响应，可能已计费；核查后显式加 --retry-uncertain，不自动重发')
    if len(starts) >= c['protocol']['limits']['max_calls']:
        raise RuntimeError('已达本实验累计API请求上限，不自动增加')
    used = 0
    for old_path in (c['out'] / 'responses').glob('*.json'):
        item = json.loads(old_path.read_text()); p.unseal(item)
        if item['signature'] != c['signature']:
            raise ValueError('响应用量来源漂移')
        total = (item['response'].get('usage') or {}).get('total_tokens', 0)
        if type(total) is not int or total < 0:
            raise ValueError('API用量字段非法')
        used += total
    if used >= c['protocol']['limits']['reported_token_stop']:
        raise RuntimeError('已达记录token停止线；不是金额硬上限，不自动增加')
    p.disk_guard()
    attempt = len(starts) + 1
    p.append(journal, p.seal(dict(signature=c['signature'], event='start', attempt=attempt, slot=slot,
                                  request_sha256=p.digest(payload), unix_s=time.time())))
    start = time.perf_counter()
    try:
        response = send(payload, secret)
    except urllib.error.HTTPError as exc:
        status = exc.code; exc.close()
        p.append(journal, p.seal(dict(signature=c['signature'], event='http_error', attempt=attempt,
                                      slot=slot, http_status=status)))
        raise RuntimeError(f'DeepSeek HTTP {status}；已保存状态，不显示响应正文，不自动重试') from None
    except Exception:
        raise RuntimeError('API网络/响应异常，保留待核查请求，可能已计费；不自动重试') from None
    if not isinstance(response, dict):
        raise RuntimeError('API响应非对象，保留待核查请求，不自动重试')
    cached = p.seal(dict(signature=c['signature'], slot=slot, request_sha256=p.digest(payload),
                         request=payload, response=response, elapsed_s=time.perf_counter() - start,
                         unix_s=time.time(), attempt=attempt))
    # 先保存原响应，再检查结构。解析失败也复用原响应，不无声多花一次钱。
    p.write_once(path, cached)
    p.append(journal, p.seal(dict(signature=c['signature'], event='response', attempt=attempt, slot=slot,
                                  response_sha256=p.sha(path))))
    return cached


def parse_response(response, fixed):
    try:
        choices = response['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            return None, 'single_choice_required', ''
        choice = choices[0]
        raw = choice['message']['content']
        if not isinstance(raw, str):
            return None, 'text_content_required', ''
        if choice.get('finish_reason') != 'stop':
            return None, 'response_not_finished', raw
        parsed, error = k.validate(raw, fixed)
        return parsed, error, raw
    except (KeyError, TypeError, IndexError):
        return None, 'invalid_response_envelope', ''
