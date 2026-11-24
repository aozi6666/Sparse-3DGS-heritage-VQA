"""Pure-CPU A.I.R. frame selection with explicit Mage adaptation boundaries.

Method reference: A.I.R., arXiv:2510.04428v3, Sec. 3 / Algorithm 1.
Reference implementation (MIT): UCF-AIR/A.I.R., commit
400fb52f1cfbf64debc6408cd215cfa375fd9797. This module is a small, corrected
reimplementation of selection logic, not the original model wrapper.

No model, video decoder, filesystem or label is accessed here. The runner owns
callbacks and their immutable per-frame caches. Replaying cached callbacks after
interruption produces the same selection and trace. One deliberate adaptation:
we retain only positive VLM scores (>3); if none are positive we return the exact
uniform baseline indices instead of presenting neutral/negative frames as valid
evidence. This fallback is explicit in the result, never counted as AIR success.
"""
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
import re
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class AIRConfig:
    final_budget: int = 32
    candidates_per_round: int = 12
    max_iterations: int = 6
    gmm_gamma: float = 0.7
    positive_threshold: int = 3
    min_event_seconds: float = 3.0
    merge_gap_seconds: float = 2.0
    length_weight: float = 0.2
    local_min_stride: int = 15
    local_growth: float = 1.5

    def __post_init__(self):
        for name in ('final_budget', 'candidates_per_round', 'max_iterations', 'local_min_stride'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.positive_threshold) is not int or not 1 <= self.positive_threshold <= 4:
            raise ValueError('positive_threshold must be in 1..4')
        for name in ('gmm_gamma', 'min_event_seconds', 'merge_gap_seconds', 'length_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.local_growth) or self.local_growth <= 1:
            raise ValueError('local_growth must be finite and greater than one')


Config = AIRConfig


def uniform_indices(total_frames: int, budget: int = 32) -> list[int]:
    if type(total_frames) is not int or total_frames < 1:
        raise ValueError('total_frames must be a positive integer')
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if total_frames <= budget:
        return list(range(total_frames))
    # 必须与Mage原生select_frame_indices逐项相同：round而非NumPy整数截断。
    # 延迟导入，只在CPU计算索引，不加载模型；避免半整数及float32舍入差异。
    import torch
    return torch.linspace(0, total_frames - 1, budget, device='cpu').round().long().tolist()


def parse_score(raw_text: str) -> int | None:
    """只认明确且唯一的Score字段；解析失败保留None，绝不默认为3。"""
    if not isinstance(raw_text, str):
        return None
    hits = re.findall(r'(?im)^\s*Score\s*:\s*([1-5])\s*(?:/\s*5\s*)?$', raw_text)
    # A malformed/duplicate score field must not become an apparently valid score.
    fields = re.findall(r'(?im)^\s*Score\s*:', raw_text)
    if len(hits) != 1 or len(fields) != 1:
        return None
    return int(hits[0])


def _normalize(values: Mapping[int, float]) -> dict[int, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo <= 1e-12:
        return {i: 0.5 for i in values}
    return {i: (float(value) - lo) / (hi - lo) for i, value in values.items()}


def _checked_indices(indices, total_frames: int, name: str) -> list[int]:
    result = []
    for i in indices:
        if not isinstance(i, Integral) or isinstance(i, (bool, np.bool_)):
            raise ValueError(f'{name}: indices must be integers')
        i = int(i)
        if not 0 <= i < total_frames:
            raise ValueError(f'{name}: frame index outside video')
        result.append(i)
    if not result or result != sorted(set(result)):
        raise ValueError(f'{name}: expected nonempty, sorted, unique frame indices')
    return result


def _gmm_threshold(values: list[float], config: AIRConfig) -> tuple[float, dict]:
    # 单样本/常数信号没有可识别的两簇，不拟合虚假的相关/不相关GMM。
    if len(values) < 2 or max(values) - min(values) <= 1e-12:
        return float(values[0]), dict(mode='degenerate_constant', means=None, stds=None)
    from sklearn.mixture import GaussianMixture
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    fit = GaussianMixture(n_components=2, random_state=0, n_init=1,
                          max_iter=200, reg_covar=1e-6).fit(x)
    if not fit.converged_:
        raise ValueError('GMM failed to converge; stop rather than use an unverified threshold')
    means = fit.means_.ravel().tolist()
    stds = np.sqrt(fit.covariances_.reshape(-1)).tolist()
    threshold = max(means) - config.gmm_gamma * max(stds)  # paper Eq. (1)
    return float(threshold), dict(mode='two_component_gmm', means=means, stds=stds)


def coarse_events(indices: list[int], values: list[float], fps: float,
                  config: AIRConfig) -> tuple[list[dict], dict]:
    threshold, diagnostics = _gmm_threshold(values, config)
    runs = []
    active = []
    for i, value in zip(indices, values):
        if value >= threshold:
            active.append(i)
        elif active:
            runs.append(active)
            active = []
    if active:
        runs.append(active)
    merged = []
    for run in runs:
        if merged and (run[0] - merged[-1][-1]) / fps <= config.merge_gap_seconds:
            merged[-1].extend(run)
        else:
            merged.append(run[:])
    kept = [run for run in merged if (run[-1] - run[0]) / fps >= config.min_event_seconds]
    events = [dict(start=int(run[0]), end=int(run[-1])) for run in kept]
    diagnostics.update(threshold=threshold, raw_events=len(runs), merged_events=len(merged),
                       kept_events=len(events), fallback=None)
    if not events:
        # 无有效事件不是无视频证据：以整个已扫描时间范围作可审计候选退路。
        events = [dict(start=indices[0], end=indices[-1])]
        diagnostics['fallback'] = 'no_event_survived_use_full_coarse_range'
    return events, diagnostics


def _initial_candidates(events: list[dict], raw: dict[int, float],
                        target: int) -> tuple[list[int], list[dict], str | None]:
    available = sorted(raw)
    target = min(target, len(available))
    data = []
    for e in events:
        items = [i for i in available if e['start'] <= i <= e['end']]
        if items:
            data.append(dict(start=e['start'], end=e['end'], items=items,
                             length=max(1, e['end'] - e['start']), allocation=1))
    if not data:
        raise ValueError('event set contains no coarse observations')
    note = None
    if len(data) > target:
        # The paper's one-frame-per-event and fixed K cannot both hold here.
        data.sort(key=lambda e: (-np.mean([raw[i] for i in e['items']]), e['start']))
        data = data[:target]
        note = 'event_count_exceeds_K_select_highest_mean_events'
    capacity = min(target, sum(len(e['items']) for e in data))
    total_length = sum(e['length'] for e in data)
    desired = [1 + max(0, capacity - len(data)) * e['length'] / total_length for e in data]
    # Eq. (9) floor-and-add-one requires normalization to exactly K. A
    # deterministic largest-deficit allocation implements that normalization.
    for _ in range(capacity - len(data)):
        feasible = [j for j, e in enumerate(data) if e['allocation'] < len(e['items'])]
        if not feasible:
            break
        j = min(feasible, key=lambda j: (-(desired[j] - data[j]['allocation']), data[j]['start']))
        data[j]['allocation'] += 1
    selected = []
    logged = []
    for e in data:
        ranking = sorted(e['items'], key=lambda i: (-raw[i], i))
        choices = ranking[:e['allocation']]
        selected.extend(choices)
        logged.append(dict(start=e['start'], end=e['end'], allocation=e['allocation'], selected=choices))
    return sorted(set(selected)), logged, note


def rank_interval_candidates(probed_indices: list[int], similarities: Mapping[int, float],
                             analyzed_indices, total_frames: int, limit: int,
                             length_weight: float = 0.2) -> tuple[list[int], list[dict]]:
    """按区间潜力保留排名顺序；禁止sorted(set(...))后截取造成前部偏置。"""
    boundaries = sorted({0, total_frames - 1, *probed_indices})
    raw = {int(i): float(v) for i, v in similarities.items()}
    norm = _normalize(raw)
    known_indices = np.asarray(sorted(norm), dtype=np.int64)
    known_values = np.asarray([norm[int(i)] for i in known_indices], dtype=np.float64)
    intervals = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        mask = (known_indices >= start) & (known_indices <= end)
        values = known_values[mask]
        relevance = float(values.mean()) if len(values) else 0.0
        complexity = float(np.abs(np.diff(values)).sum()) / max(1, end - start) if len(values) > 1 else 0.0
        potential = relevance * (1.0 + complexity) * (1.0 + length_weight * math.log10(max(1, end - start)))
        intervals.append(dict(start=int(start), end=int(end), potential=float(potential),
                              observations=int(len(values))))
    intervals.sort(key=lambda x: (-x['potential'], x['start'], x['end']))
    seen = set(analyzed_indices)
    selected = []
    for interval in intervals:
        # Each interval proposes its endpoints; already-analyzed endpoints do
        # not consume the round's C, otherwise the same early frames can stall.
        for idx in (interval['start'], interval['end']):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= limit:
                    return selected, intervals
    return selected, intervals


def localized_candidates(anchors: list[int], probed_indices: list[int],
                         total_frames: int, remaining_budget: int,
                         config: AIRConfig) -> list[int]:
    """论文Eq.(6)：围绕已验证真实帧号作指数步长搜索；新帧必须下轮再评分。"""
    if not anchors or remaining_budget <= 0:
        return []
    known = set(probed_indices)
    bounds = sorted({0, total_frames - 1, *probed_indices})
    per_side = max(1, math.ceil(remaining_budget / (2 * len(anchors))))
    generated = []
    for anchor in sorted(set(anchors)):
        position = bounds.index(anchor)
        for sign in (-1, 1):
            endpoint = bounds[max(0, position - 1)] if sign < 0 else bounds[min(len(bounds) - 1, position + 1)]
            span = abs(endpoint - anchor)
            stride = max(config.local_min_stride, 0.05 * span)
            for m in range(per_side):
                offset = stride * config.local_growth ** m
                if offset > span:
                    break
                idx = int(round(anchor + sign * offset))
                if 0 <= idx < total_frames and idx not in known:
                    generated.append(idx)
                    known.add(idx)
    return sorted(generated)


def _callback_map(callback: Callable, indices: list[int], *, scores: bool) -> dict:
    if not indices:
        return {}
    result = callback(indices[:])
    if not isinstance(result, Mapping):
        raise ValueError('callback must return a mapping keyed by requested frame index')
    converted = {}
    for k, value in result.items():
        try:
            valid_key = not isinstance(k, bool) and str(k) == str(int(k))
        except (ValueError, TypeError, OverflowError):
            valid_key = False
        if not valid_key:
            raise ValueError('callback frame index is not an integer identity')
        i = int(k)
        if i in converted:
            raise ValueError('callback contains duplicate frame identities')
        if scores:
            if not isinstance(value, Mapping) or 'score' not in value or not isinstance(value.get('raw_text'), str):
                raise ValueError('score callback requires {score, raw_text} records')
            value = dict(value)
            score = value['score']
            if score is not None:
                if not isinstance(score, Integral) or isinstance(score, (bool, np.bool_)) or not 1 <= score <= 5:
                    raise ValueError('score must be integer1..5 or explicit None')
                value['score'] = int(score)
            if parse_score(value['raw_text']) != value['score']:
                raise ValueError('score callback score does not match its raw Score field')
            # No tensor/nan fields may silently leak into the persisted trace.
            json.dumps(value, allow_nan=False)
            converted[i] = value
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('similarity callback returned nonfinite value')
            converted[i] = value
    if set(converted) != set(indices):
        raise ValueError('callback result identities differ from requested frame indices')
    return converted


def select_frames(total_frames: int, fps: float, coarse_indices,
                  coarse_similarities, score_frames: Callable,
                  similarity_frames: Callable, config: AIRConfig | None = None,
                  on_round: Callable | None = None) -> dict:
    """Select <=B frame indices; callbacks own expensive work/checkpointing.

    `coarse_similarities` and `similarity_frames(indices)` are raw CLIP cosine
    similarities. `score_frames(indices)` returns an index-keyed mapping of
    JSON records containing `score` (integer1..5 or None) and `raw_text`.
    A completed `on_round` trace is deterministic and independent of labels.
    """
    config = config or AIRConfig()
    uniform = uniform_indices(total_frames, config.final_budget)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    indices = _checked_indices(coarse_indices, total_frames, 'coarse_indices')
    values = [float(v) for v in coarse_similarities]
    if len(values) != len(indices) or not all(math.isfinite(v) for v in values):
        raise ValueError('coarse similarities must be aligned finite values')
    raw = dict(zip(indices, values))
    budget = len(uniform)
    events, event_diagnostics = coarse_events(indices, values, fps, config)
    initial, allocation, allocation_note = _initial_candidates(
        events, raw, 2 * max(budget, config.candidates_per_round))
    pool = initial[:]
    analyzed = {}
    positives = set()
    rounds = []
    stop = 'max_iterations'
    for number in range(1, config.max_iterations + 1):
        candidates, ranking = rank_interval_candidates(pool, raw, analyzed, total_frames,
            config.candidates_per_round, config.length_weight)
        if not candidates:
            stop = 'candidate_pool_exhausted'
            break
        missing = [i for i in candidates if i not in raw]
        raw.update(_callback_map(similarity_frames, missing, scores=False))
        scores = _callback_map(score_frames, candidates, scores=True)
        analyzed.update(scores)
        fresh_positive = [i for i in candidates if scores[i]['score'] is not None
                          and scores[i]['score'] > config.positive_threshold]
        positives.update(fresh_positive)
        # 使用累计已验证帧数早停，不能检查单轮C是否大于2B。
        reached = len(positives) >= budget
        next_indices = []
        if not reached and number < config.max_iterations:
            next_indices = localized_candidates(fresh_positive, sorted(set(pool + candidates)),
                total_frames, budget - len(positives), config)
            raw.update(_callback_map(similarity_frames, [i for i in next_indices if i not in raw], scores=False))
            pool = sorted(set(pool + candidates + next_indices))
        trace = dict(iteration=number, candidates=candidates, interval_ranking=ranking,
            score_records={str(i): scores[i] for i in candidates}, positive_indices=fresh_positive,
            cumulative_positive=sorted(positives), new_neighbor_indices=next_indices,
            pool_after=pool[:], budget_reached=reached)
        json.dumps(trace, allow_nan=False)
        rounds.append(trace)
        if on_round is not None:
            on_round(trace)
        if reached:
            stop = 'cumulative_budget_reached'
            break
    fallback = None
    if positives:
        # Paper Algorithm1 truncates an overfull set by CLIP similarity; stable
        # index tie-breaking makes replay exact. Chronology is restored only
        # after truncation, never before candidate potential ranking.
        chosen = sorted(positives, key=lambda i: (-raw[i], i))[:budget]
        final = sorted(chosen)
    else:
        final = uniform
        fallback = 'no_positive_vlm_evidence_use_exact_uniform_baseline'
    invalid = [i for i, record in analyzed.items() if record['score'] is None]
    result = dict(version='mage_air_selection_v1',config=asdict(config),total_frames=total_frames,
        fps=float(fps),budget=budget,coarse_indices=indices,coarse_event_diagnostics=event_diagnostics,
        coarse_events=events,initial_indices=initial,initial_allocation=allocation,
        initial_allocation_note=allocation_note,rounds=rounds,final_indices=final,
        final_seconds=[i / fps for i in final],fallback_reason=fallback,stop_reason=stop,
        analyzed_indices=sorted(analyzed),invalid_score_indices=sorted(invalid),
        observed_similarity={str(i):raw[i] for i in sorted(raw)},
        adaptation='Positive scores only; no positive evidence falls back to exact uniform baseline; final <=32 cap, not original duration-adaptive budget.')
    if len(final) != len(set(final)) or not 1 <= len(final) <= budget or final != sorted(final):
        raise AssertionError('selection produced invalid final indices')
    json.dumps(result, allow_nan=False)
    return result
