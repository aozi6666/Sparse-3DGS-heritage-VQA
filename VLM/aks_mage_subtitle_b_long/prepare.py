"""Prepare paired subtitle inputs from the frozen B subtitles and fixed-20 visual audit, entirely on CPU."""
import collections
import gzip
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys

from tokenizers import Tokenizer
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
V = HERE.parent
P = V / "mage_vl/practical"
sys.path.insert(0, str(P))
from practical_common import MODEL, sha, signature
from subtitle_common import subtitle_source

BASE = V / "eval_results/practical_mage_v1/24_codec_full714_32768_v1"
BC = V / "aks_mage_subtitle_bc/results/d62405efc78461c4/checkpoint.sqlite"
LIMIT = 2048
STOP = set("a an the is are was were be been it this that these those at of to in on and or for as by from with does do did what which who when where why how video following according shown".split())


def words(text):
    return [w for w in re.findall(r"[^\W_]+", text.casefold()) if w not in STOP]


def stamp(ms):
    sec, ms = divmod(ms, 1000)
    minute, sec = divmod(sec, 60)
    hour, minute = divmod(minute, 60)
    return f"{hour:02}:{minute:02}:{sec:02}.{ms:03}"


def segments(cues, ids, max_size=None):
    runs = []
    run = []
    for cue in cues:
        if not cue['text']:
            continue
        if cue['cue_id'] not in ids:
            run = []
            continue
        if not run or (max_size and len(run) >= max_size) or cue['start_ms'] > run[-1]['end_ms'] + 2000:
            run = []
            runs.append(run)
        run.append(cue)
    return runs


def render(runs):
    lines = []
    for run in runs:
        start = stamp(min(c['start_ms'] for c in run))
        end = stamp(max(c['end_ms'] for c in run))
        body = ' '.join(c['text'] for c in run)
        lines.append(f"S{run[0]['cue_id']} [{start} --> {end}] {body}")
    return '\n'.join(lines)


def spread(runs):
    queue = collections.deque([(0, len(runs))])
    while queue:
        lo, hi = queue.popleft()
        if lo < hi:
            mid = (lo + hi) // 2
            yield runs[mid]
            queue.extend([(lo, mid), (mid + 1, hi)])


def ranked(question, runs):
    query = set(words(question))
    docs = [collections.Counter(words(' '.join(c['text'] for c in run))) for run in runs]
    lengths = [sum(doc.values()) for doc in docs]
    avg = sum(lengths) / max(1, len(docs))
    if not avg:
        return []
    df = collections.Counter(t for doc in docs for t in query.intersection(doc))
    scores = []
    for i, doc in enumerate(docs):
        norm = 1.2 * (0.25 + 0.75 * lengths[i] / avg)
        score = sum(math.log1p((len(docs)-df[t]+0.5)/(df[t]+0.5))
                    * doc[t]*2.2/(doc[t]+norm) for t in sorted(query.intersection(doc)))
        if score > 0:
            scores.append((-score, i))
    return [runs[i] for _, i in sorted(scores)]


def make_pair(cues, local, question, original, count):
    def output(ids):
        text = render(segments(cues, ids))
        return dict(ids=sorted(ids), text=text, tokens=count(text))
    ref = output(local)
    assert ref['text'] == original, 'C reconstruction differs'
    if local and ref['tokens'] <= LIMIT:
        return ref, ref
    available = {c['cue_id'] for c in cues if c['text']}
    time_order = list(spread(segments(cues, local, 3)))
    remote_order = ranked(question, segments(cues, available, 3))

    def pack(ids, order, cap):
        # Reject entire segments that would exceed the actual rendered token budget.
        misses = 0
        for run in order:
            new = {c['cue_id'] for c in run} - ids
            if not new:
                continue
            if count(render(segments(cues, ids | new))) <= cap:
                ids.update(new)
                misses = 0
            else:
                misses += 1
                if misses == 4:
                    break
        return ids

    time = output(pack(set(), time_order, LIMIT))
    hybrid = pack(set(), time_order, 1536)
    # Rank across the video; only wholly nonlocal chunks can enter the retrieval quota.
    remote = [run for run in remote_order if all(c["cue_id"] not in local for c in run)]
    hybrid = pack(hybrid, remote, LIMIT)
    return time, output(pack(hybrid, time_order, LIMIT))


def main():
    from functools import lru_cache
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    count = lru_cache(maxsize=256)(lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids))
    qs = {q['id']: q for q in json.loads((V/'aks_mage/questions.json').read_text())}
    plan = json.loads((BASE/'plan.json').read_text())
    # A single short read transaction; never acquire the B/C experiment's run lock.
    db = sqlite3.connect(BC.as_uri()+'?mode=ro', uri=True, timeout=2)
    try:
        audit = [json.loads(s) for (s,) in db.execute('SELECT data FROM audit ORDER BY id').fetchall()]
    finally:
        db.close()
    assert len(audit) == len(qs) == 714 and {a['id'] for a in audit} == qs.keys()
    audit = [a for a in audit if a['duration']=='long']
    assert len(audit)==237 and len({a['video'] for a in audit})==79
    version = signature([sha(__file__), sha(P/'subtitle_common.py'), sha(MODEL/'tokenizer.json'), sha(BASE/'plan.json'), sha(V/'aks_mage/questions.json')])[:16]
    out = HERE/'results'/version
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.defaultdict(list)
    last = None
    for a in tqdm(audit, desc='CPU subtitles', unit='q'):
        vid, key = a['video'], a['id']
        q = qs[key]
        assert q['video'] == vid
        if last != vid:
            src, cues = subtitle_source(vid)
            assert src['sha256'] == plan['srt_sources'][vid]['sha256']
            record = json.loads((BASE/'video_records'/f'{vid}.json').read_text())
            b = record['video']['evidence']['codec_complete32768']['continuous']
            base_ids = {c['cue_id'] for c in b['original']['selected_cues']} | set(b['added_cue_ids'])
            last = vid
        ids = base_ids
        assert signature(b['text'])==a['b_hash'] and count(b['text'])==a['b_tokens']
        fingerprint = signature([a['ids_hash'], b['text'], sorted(ids), src['sha256'], q['question']])
        file = out/f'{key}.json.gz'
        if file.exists():
            with gzip.open(file, 'rt') as f:
                row = json.load(f)
            assert row['source_hash'] == fingerprint
        else:
            t, h = make_pair(cues, ids, q['question'], b['text'], count)
            row = dict(id=key, video=vid, source_hash=fingerprint, srt_sha256=src['sha256'],
                       duration=a['duration'], b_tokens=a['b_tokens'], b_status=a['b_status'],
                       srt_status=src['status'], b_input_ref=a['index'], visual_hash=a['ids_hash'],
                       time=t, hybrid=h, new_cues=sorted(set(h['ids'])-ids))
            tmp = file.with_suffix(f'.{os.getpid()}.tmp')
            with gzip.open(tmp, 'wt') as f:
                json.dump(row, f, ensure_ascii=False)
            tmp.replace(file)
        assert max(row[k]['tokens'] for k in ('time','hybrid')) <= LIMIT
        stats[a['duration']].append(row)
    report = {}
    for d, rows in stats.items():
        report[d] = dict(n=len(rows), over=sum(x['b_tokens']>LIMIT for x in rows),
                         changed=sum(x['time']['text']!=x['hybrid']['text'] for x in rows),
                         with_nonlocal=sum(bool(x['new_cues']) for x in rows),
                         empty_B=dict(collections.Counter(x['b_status'] for x in rows if not x['b_tokens'])),
                         median_tokens={k:statistics.median([x['b_tokens'] if k=='B' else x[k]['tokens'] for x in rows]) for k in ('B','time','hybrid')})
    temp = out/'summary.tmp'
    temp.write_text(json.dumps(report,indent=2))
    temp.replace(out/'summary.json')
    print(out, report)


if __name__ == '__main__':
    main()
