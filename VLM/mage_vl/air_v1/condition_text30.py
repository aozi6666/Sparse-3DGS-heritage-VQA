"""固定开发30题的结构扩展；旧5题原样复用，不把未知事件或选项原因当事实。"""
import re
import condition_text_parser as base
from condition_text_parser import PARSE5, MODEL, ENDPOINT, MAX_TOKENS, MAX_ATTEMPTS, payload

VERSION = 'condition_grounding_deepseek_v2_parse30'


def contract(row):
    q = row['question']; lower = q.lower()
    # 旧5题必须逐字保持，API正文/结果才可复用；并非按题号修正语义。
    try:
        fixed = base.contract(row)
    except ValueError as exc:
        if str(exc) != 'event_list_not_deterministically_extractable':
            raise
        # 选项中的walking/taking a walk等不能无依据合并，更不能选一组作为真顺序。
        return dict(question_text=q, events=[], relation='unknown_temporal_order', given_edges=[],
                    event_order_kind='no_event_list_in_question', time_range=base.numeric_range(q),
                    unresolved=['option_only_event_lists_need_review; no_guessing_or_option_order_inference'],
                    semantic_certified=False, video_observed=False,
                    contract_status='unsupported', grounding_policy='exact_noobs_fallback',
                    fallback_reason='no_deterministic_question_event_list')
    # 只在旧规则尚未识别类型时补充通用题面规则，不改已有range/edge/显式事件。
    if fixed['relation'] == 'independent':
        relation = 'independent'
        if re.search(r'\b(main idea|main focus|main content|what style|comparative strengths|in common)\b', lower):
            relation = 'global_summary_or_comparison'
        elif re.search(r"\bisn't mentioned\b|\bnot mentioned\b|\bnot appear\b", lower):
            relation = 'absence_unimplemented'
        elif re.search(r'\b(after|before)\b', lower):
            relation = 'unknown_successor_or_predecessor'
        elif re.search(r'\bwhat happens when\b', lower):
            relation = 'trigger_then_unknown_outcome'
            match = re.search(r'\bwhat happens when\s+(.+)', q, re.I)
            fixed['events'] = [base.question_event(q, 'q1', match.start(1), match.end(1))]
        elif re.search(r'\bwhen the score is\b', lower):
            relation = 'score_conditioned_measurement'
        elif re.search(r'\bwhich days|\bwhen will\b', lower):
            relation = 'narrated_time_or_schedule'
        fixed['relation'] = relation
    # 限定条件保持原词，不擅自转成秒数。多个条件可同时成立。
    qualifiers = []
    for pattern in [r'\bat the beginning\b', r'\bat first\b', r'\bopening and closing\b',
                    r'\b(?:first|second|third|last) (?:case|magic)\b', r'\bwhen the score is\s+\d+[-–]\d+']:
        qualifiers.extend(m[0] for m in re.finditer(pattern, q, re.I))
    fixed['literal_qualifiers'] = qualifiers
    fixed['contract_status'] = 'compiled_candidate'
    unsupported = fixed['relation'] not in ['independent', 'unknown_temporal_order', 'spatial_order', 'known_before_after', 'relative_position', 'trigger_then_unknown_outcome']
    fixed['grounding_policy'] = 'review_before_grounding' if not unsupported else 'exact_noobs_fallback_until_capability_implemented'
    # 这些是能力限制，不是已观察到的证据类别/答案。
    for pattern, flag in [
        (r'\b(why|reason|formation)\b', 'cause_or_mechanism_requires_more_than_object_similarity'),
        (r'\b(percentage|score|days of the week|argument|mentioned|anticipated)\b', 'possible_ocr_speech_or_narrative_dependency'),
        (r'\b(main idea|main focus|style|in common|comparative strengths)\b', 'global_coverage_not_guaranteed_by_local_peaks')]:
        if re.search(pattern, lower):
            fixed['unresolved'].append(flag)
    return fixed


def annotate_old(fixed):
    """旧5题不改API契约；扩展解释在报告外层标注，不污染复用身份。"""
    return dict(contract_status='reused_reviewed5', grounding_policy='review_before_grounding',
                note='Text preservation passed earlier; visual localization not certified.')
