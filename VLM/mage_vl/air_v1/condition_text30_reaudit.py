"""按原文位置拆分检索内容/题目操作符；只重审缓存，不改写查询或模型输出。"""
from copy import deepcopy
import json
import re

import condition_text_parser as base

VERSION = 'condition_text30_offline_reaudit_v1'


def span_record(question, start, end, kind):
    return dict(document='question', start=start, end=end, quote=question[start:end], kind=kind)


def partition(row, fixed):
    """规则不接收题号/响应/GT；不能按模型恰好漏掉哪些词来删条件。"""
    q = row['question']
    if fixed['question_text'] != q:
        raise ValueError('question_identity_changed')
    masks, operators = [], []
    # 只匹配完整、位于句首的评测套话。对象动作中的based/following/video不豁免。
    for pattern in [r'^\s*According to the content of the video,\s*',
                    r'^\s*Based on (?:the information provided by )?the video,\s*']:
        match = re.match(pattern, q, re.I)
        if match:
            masks.append(span_record(q, match.start(), match.end(), 'question_preface'))
    # 仅句首或已识别套话之后的疑问操作符；不能豁免标题/引文里的同名短语。
    lead = max((m['end'] for m in masks), default=0)
    match = re.match(r'\s*which of the following\b', q[lead:], re.I)
    if match:
        masks.append(span_record(q, lead + match.start(), lead + match.end(), 'choice_question_operator'))
    # “What happened after X”只给定X；after约束问答目标，不是X的可见动作。
    match = re.match(r'^\s*(What happened)\s+(after|before)\s+(.+?)\s*[?]?$', q, re.I)
    if match and fixed['relation'] == 'unknown_successor_or_predecessor':
        masks.append(span_record(q, match.start(1), match.end(1), 'unknown_outcome_question_operator'))
        cue = span_record(q, match.start(2), match.end(2), 'temporal_target_operator')
        masks.append(cue)
        operators.append(dict(kind='unknown_successor' if match[2].lower() == 'after' else 'unknown_predecessor',
                              cue=cue, anchor_quote=q[match.start(3):match.end(3)].rstrip('?').rstrip(),
                              target_event_not_given=True, known_event_order_edges=[]))
    # future语气不丢弃：保存在问题操作符里；查询里的will/we不强制保留。
    match = re.match(r'^\s*When will we\b', q, re.I)
    if match and fixed['relation'] == 'narrated_time_or_schedule':
        cue = span_record(q, match.start(), match.end(), 'requested_future_time_operator')
        masks.append(cue)
        operators.append(dict(kind='requested_future_time', cue=cue, requested_time_is_unknown=True))
    masks.sort(key=lambda x: (x['start'], x['end']))
    if any(a['end'] > b['start'] for a, b in zip(masks, masks[1:])):
        raise ValueError('overlapping_partition_spans')
    events = []
    for event in fixed['events']:
        refs = event['source_refs']
        if not refs:
            raise ValueError('missing_event_source')
        positions = []
        # 使用第一处来源重建词位置；作品的其余选项引用也逐个验证，绝不借别的事件的词。
        for n, ref in enumerate(refs):
            if ref['document'] == 'question':
                doc = q
            elif re.fullmatch(r'option:[ABCD]', ref['document']):
                doc = row['bodies']['ABCD'.index(ref['document'][-1])]
            else:
                raise ValueError('unknown_source_document')
            start, end = ref['start'], ref['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc):
                raise ValueError('invalid_source_span')
            if base.clean(doc[start:end]) != base.clean(event['source_text']):
                raise ValueError('source_text_changed')
            if n:
                continue
            for token in base.TOKEN.finditer(doc, start, end):
                covering = [m for m in masks if ref['document'] == 'question' and m['start'] <= token.start() and token.end() <= m['end']]
                positions.append(dict(word=token[0], start=token.start(), end=token.end(), document=ref['document'],
                                      role=covering[0]['kind'] if covering else 'retrieval_content'))
        allowed = list(dict.fromkeys(base.content_words(event['source_text'])))
        if event['required_content_words'] != allowed:
            raise ValueError('original_required_words_changed')
        required, exempt = [], []
        for word in allowed:
            occurrences = [t for t in positions if t['word'].lower() == word.lower()]
            if not occurrences:
                raise ValueError('word_position_missing')
            # 同一个词在其他位置还是动作/对象时，不能因套话里也出现就全局放行。
            (exempt if all(t['role'] != 'retrieval_content' for t in occurrences) else required).append(word)
        if not required:
            raise ValueError('no_retrieval_content_after_partition')
        events.append(dict(event_id=event['event_id'], source_text=event['source_text'], source_refs=deepcopy(refs),
                           allowed_source_words=allowed, required_retrieval_words=required,
                           query_optional_words=exempt, token_roles=positions))
    return dict(question_text=q, original_contract=deepcopy(fixed), events=events,
                question_operator_spans=masks, explicit_operators=operators,
                # 原结构全部复制，不改给定边/未知关系/范围/否定能力标记；不增加定位能力。
                retained_structure={name: deepcopy(fixed.get(name)) for name in
                                    ['relation', 'given_edges', 'event_order_kind', 'time_range', 'literal_qualifiers',
                                     'unresolved', 'grounding_policy', 'contract_status']},
                semantic_certified=False, localization_capability_upgraded=False)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def validate(raw, row, fixed, part):
    """来源词可出现，但只有明确套话/已另存的操作符允许不进入检索短句。"""
    if part != partition(row, fixed):
        raise ValueError('partition_or_retained_constraints_changed')
    try:
        obj = json.loads(raw, object_pairs_hook=strict_object)
    except (ValueError, TypeError):
        return None, ['invalid_or_duplicate_key_JSON']
    if not isinstance(obj, dict) or set(obj) != {'queries'} or not isinstance(obj['queries'], list):
        return None, ['exact_queries_schema_required']
    if len(obj['queries']) != len(part['events']):
        return None, ['event_count_changed']
    errors = []
    for query, event in zip(obj['queries'], part['events']):
        if not isinstance(query, dict) or set(query) != {'event_id', 'text'}:
            errors.append('exact_event_id_text_keys_required'); continue
        if query['event_id'] != event['event_id']:
            errors.append('event_identity_or_source_order_changed'); continue
        text = query['text']
        if not isinstance(text, str) or text != base.clean(text) or not 1 <= len(text) <= 260 or not 1 <= len(base.TOKEN.findall(text)) <= 48:
            errors.append('invalid_query_length_or_whitespace'); continue
        words = base.content_words(text)
        foreign = [w for w in words if not any(base.forms(w) & base.forms(s) for s in event['allowed_source_words'])]
        missing = [s for s in event['required_retrieval_words'] if not any(base.forms(s) & base.forms(w) for w in words)]
        if not words:
            errors.append('empty_retrieval_content')
        if foreign:
            errors.append(dict(event_id=event['event_id'], foreign_words=foreign))
        if missing:
            errors.append(dict(event_id=event['event_id'], missing_required_retrieval_words=missing))
    return (None, errors) if errors else (obj, [])
