"""实用涨点实验的纯 CPU 协议；不导入 Torch，不执行第三方参考代码。"""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

PROJECT = Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA')
ROOT = PROJECT/'VQA/eval_results/practical_mage_v1'
MODEL = PROJECT/'VQA/mage_vl/models'
DATA = PROJECT/'VQA/mage_vl/data/VideoMME'
OLD_CACHE = PROJECT/'VQA/mage_vl/outputs/r2v_codec_cache'
LETTERS = 'ABCD'
SLOT = '__PRACTICAL_MAGE_QUESTION_SLOT_V1__'
PROMPT = ('These are the frames of a video. Select the best answer to the following '
          'multiple-choice question based on the video. Respond with only the letter '
          '(A, B, C, or D) of the correct option.\n{question}\n{options}')
STAGES = dict(smoke='02_smoke', dev30='03_dev30', dev60='03_dev60',
              val='04_val', test='05_test', full='06_full714')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write_once(path,obj):
    """一致时幂等；冲突或残缺文件拒绝覆盖。"""
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise ValueError(f'文件内容冲突，不覆盖: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def read_lines(path):
    if not Path(path).exists():
        return []
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError(f'残缺尾行，保留现场，不自动截断: {path}')
            rows.append(json.loads(line))
    return rows


def append_line(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
    f.flush(); os.fsync(f.fileno())


def permutation(k):
    if k not in range(4):
        raise ValueError('置换 ID 只能为 0/1/2/3')
    return [(j+k)%4 for j in range(4)]


def prompt(row,k):
    # 保留选项正文的原始空白，仅替换标签；恒等排列必须等于旧 Prompt。
    order=permutation(k)
    return PROMPT.format(question=row['question'],options='\n'.join(
        LETTERS[j]+'.'+row['bodies'][original] for j,original in enumerate(order)))


def position_sensitive(row):
    text=row['question']+'\n'+'\n'.join(row['bodies'])
    patterns=[r'\b(?:all|none|both|any)\s+(?:(?:of\s+)?(?:the\s+)?)?(?:above|below|following|options|answers)\b',
              r'\b(?i:option|answer|choice)s?\s*[\(\[]?[A-D]\b',
              r'\b[A-D]\s*(?:and|or|&|,)\s*[A-D]\b',
              r'\b(?:both|either|neither)\s+[A-D]\b',
              r'以上|上述|以下.*(?:都|均)|[A-D]\s*[和与及、]\s*[A-D]']
    # 英文单字母仅在大写语境识别，避免把普通冠词 a 当作选项名。
    return any(re.search(p,text,re.I if i==0 else 0) for i,p in enumerate(patterns))


def remap(values,k):
    if len(values)!=4:
        raise ValueError('概率必须为四维')
    mapped=[0.0]*4
    for new,original in enumerate(permutation(k)):
        mapped[original]=values[new]
    return mapped


def softmax(values):
    if len(values)!=4 or not all(math.isfinite(x) for x in values):
        raise ValueError('非法四字母 logits')
    ex=[math.exp(x-max(values)) for x in values]
    return [v/sum(ex) for v in ex]


def parse_answer(text):
    # 与旧配对脚本相同：自由生成中提取首个独立字母；不把单词内字母当答案。
    match=re.search(r'(?<![A-Z])([ABCD])(?![A-Z])',text.upper())
    return match.group(1) if match else ''


def ballot_key(index,backend,k):
    return f'{index}:{backend}:p{k}'


def aggregate(ballots,rule):
    """只消费预测/分数，无真值入口。非法 ballot 不伪造为合法投票。"""
    if not ballots or ballots[0]['permutation_id']!=0:
        raise ValueError('聚合需要包含恒等排列')
    original=ballots[0]['semantic_pred']
    valid=[b for b in ballots if b['semantic_pred'] in list(LETTERS)]
    if rule=='identity' or not valid:
        return original
    # 前缀为“答案是...”时，第一位置 logits 不是答案字母分数，不能参与概率平均。
    scored=[b for b in valid if b.get('score_eligible',True)]
    avg=[sum(b['semantic_p'][i] for b in scored)/len(scored) if scored else 0.0 for i in range(4)]
    if rule=='mean':
        if not scored:
            return original
        return LETTERS[max(range(4),key=lambda i:avg[i])]
    if rule!='vote':
        raise ValueError('未知聚合器')
    count=Counter(b['semantic_pred'] for b in valid)
    tied=[c for c in LETTERS if count[c]==max(count.values())]
    if original in tied:
        return original
    return max(tied,key=lambda c:(avg[LETTERS.index(c)],-LETTERS.index(c)))


def load_dataset():
    inputs,labels=[],{}
    with (DATA/'Video-MME.tsv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            options=ast.literal_eval(row['candidates'])
            if len(options)!=4 or any(not o.startswith(c+'.') for c,o in zip(LETTERS,options)):
                raise ValueError(f'选项格式错误: {row["index"]}')
            video=(DATA/row['video_path']).resolve()
            if not video.is_relative_to(DATA.resolve()) or not video.is_file():
                raise FileNotFoundError(video)
            item={k:row[k] for k in ['index','question','duration','domain','sub_category','task_type']}
            item.update(video_id=row['video'],video_path=str(video),bodies=[o[2:] for o in options])
            if SLOT in json.dumps(item) or any('<|' in s for s in [item['question'],*item['bodies']]):
                raise ValueError('题目含保留占位符/特殊 token，需人工检查')
            if row['index'] in labels or row['answer'].strip().upper() not in list(LETTERS):
                raise ValueError('重复 index 或非法真值')
            item['no_permute']=position_sensitive(item)
            inputs.append(item)
            labels[row['index']]=row['answer'].strip().upper()
    return inputs,labels


def make_splits(rows):
    sets={k:[] for k in ['short_dev','dev','val','test','dev30']}
    rng=random.Random(0)
    all_videos={}
    for r in rows:
        previous=all_videos.setdefault(r['video_id'],r['duration'])
        if previous!=r['duration']:
            raise ValueError('同一视频跨时长标签')
    for duration,sizes in [('medium',(32,16,38)),('long',(28,14,37))]:
        videos=sorted(v for v,d in all_videos.items() if d==duration)
        if len(videos)!=sum(sizes):
            raise ValueError(f'{duration} 视频数量变化，请重新审核划分')
        rng.shuffle(videos)
        a,b,_=sizes
        sets['dev']+=videos[:a]; sets['val']+=videos[a:a+b]; sets['test']+=videos[a+b:]
        sets['dev30']+=videos[:a//2]
    sets['short_dev']=sorted(v for v,d in all_videos.items() if d=='short')
    return sets


def stage_rows(root,stage):
    bundle=json.loads((root/'01_manifests/inputs.json').read_text())
    # 推理加载的输入清单不包含答案字段；标签另存且只由离线汇总读取。
    rows=bundle['rows']
    if any('answer' in r or 'gt' in r for r in rows):
        raise ValueError('推理输入清单混入真值')
    splits=json.loads((root/'01_manifests/splits.json').read_text())
    if stage=='smoke':
        return [r for r in rows if r['duration']=='short'][:6]
    if stage=='full':
        return rows
    pool='dev' if stage=='dev60' else stage
    videos=set(splits[pool])
    return [r for r in rows if r['video_id'] in videos]


def jobs_for(row,codec_vote=False,chosen=None):
    jobs=[]
    for backend in ['codec','frames']:
        need_vote=(backend=='frames' or codec_vote)
        if chosen:
            need_vote=chosen.startswith(backend+'_') and not chosen.endswith('_identity')
        for k in range(4 if need_vote and not row['no_permute'] else 1):
            jobs.append((backend,k))
    return jobs


def load_ballots(path,sig,inputs):
    lookup={r['index']:r for r in inputs}
    found={}
    for r in read_lines(path):
        idx,b,k=r['index'],r['backend'],r['permutation_id']
        key=ballot_key(idx,b,k)
        if idx not in lookup or b not in ['codec','frames'] or key in found or r['signature']!=sig:
            raise ValueError('ballot 重复、来源不符或 index 越界')
        order=permutation(k)
        if r['prompt_sha256']!=signature(prompt(lookup[idx],k)):
            raise ValueError('ballot 提示词签名不一致')
        display=r['display_pred']
        if 'raw_text' in r and parse_answer(r['raw_text'])!=display:
            raise ValueError('ballot 的原始输出与答案解析不一致')
        if 'score_eligible' in r and r['score_eligible']!=bool(r['first_letter'] and r['first_letter']==display):
            raise ValueError('ballot 的字母打分资格不一致')
        expected=LETTERS[order[LETTERS.index(display)]] if display in list(LETTERS) else ''
        if r['semantic_pred']!=expected or r['semantic_p']!=remap(softmax(r['plain_logits']),k):
            raise ValueError('ballot 语义回排/概率不一致')
        if k and lookup[idx]['no_permute']:
            raise ValueError('位置依赖题不应置换')
        found[key]=r
    return found
