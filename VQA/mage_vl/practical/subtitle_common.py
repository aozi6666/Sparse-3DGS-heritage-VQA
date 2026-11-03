"""对齐字幕的纯 CPU 协议；不导入 Torch、不抽帧、不读取答案标签。"""
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

from practical_common import (ROOT, DATA, MODEL, SLOT, LETTERS, PROMPT,
    signature, sha, prompt, parse_answer, read_lines)

BASE_SIG = '86f337f035388253a22e9dcb7d7a6977005aae4ab1273fa75313735ba5ab2c16'
TAG = 'f32_px150000_aligned_sub_v1'
RESULTS = ROOT/'09_subtitles'
STAGES = ('smoke', 'dev30', 'dev60')
MAX_SUB_TOKENS = 2048
SUB_HEADER = "This video's subtitles are listed below:\n"
TIME = r'(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})'
TIMING = re.compile(r'^\s*'+TIME+r'\s*-->\s*'+TIME+r'(?:\s+.*)?$')


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[]
    def handle_data(self, data):
        self.parts.append(data)
    def handle_starttag(self, tag, attrs):
        if tag.lower()=='br':
            self.parts.append(' ')


def clean_text(text):
    # 只清除字幕样式；不要用“必须含某一种 font 标签”的正则过滤整个字幕。
    if '<|' in text or SLOT in text or '\x00' in text:
        raise ValueError('字幕含保留 token/占位符/NUL，停止人工检查')
    text=re.sub(r'\{\\[^}]*\}', '', text).replace('\\N', ' ').replace('\\n', ' ')
    parser=PlainText()
    parser.feed(text); parser.close()
    result=' '.join(html.unescape(''.join(parser.parts)).split())
    if '<|' in result or SLOT in result:
        raise ValueError('字幕转义后含保留 token')
    return result


def milliseconds(parts):
    h,m,s,ms=parts
    if int(m)>=60 or int(s)>=60:
        raise ValueError('非法 SRT 分/秒')
    return ((int(h)*60+int(m))*60+int(s))*1000+int(ms.ljust(3,'0'))


def parse_srt(text):
    """严格解析：缺失/空文件可回退，存在但损坏的文件不静默当成无字幕。"""
    text=text.lstrip('\ufeff').replace('\r\n','\n').replace('\r','\n')
    if not text.strip():
        return []
    # 某些真实SRT在<font>正文内也有空行，不能按空行分块，否则会丢台词。
    # 以时间轴行和其数字序号作为边界；正文的空行保留给清理器处理。
    lines=text.strip().splitlines()
    times=[(i,TIMING.fullmatch(line)) for i,line in enumerate(lines) if '-->' in line]
    if not times or any(match is None for _,match in times):
        raise ValueError('SRT 时间轴损坏或不存在')
    preamble=[line.strip() for line in lines[:times[0][0]] if line.strip()]
    if preamble and (len(preamble)!=1 or not preamble[0].isdigit()):
        raise ValueError('SRT 前缀格式异常')
    indexed=bool(preamble)
    starts=[]
    for offset,_ in times:
        header=offset-1
        while header>=0 and not lines[header].strip():
            header-=1
        if indexed and (header<0 or not lines[header].strip().isdigit()):
            raise ValueError('SRT 缺少字幕序号')
        starts.append(header if indexed else offset)
    cues=[]
    for number,(offset,match) in enumerate(times,1):
        start=milliseconds(match.groups()[:4]); end=milliseconds(match.groups()[4:])
        if start>=end:
            raise ValueError(f'SRT 第 {number} 段结束时间不大于开始时间')
        end_offset=starts[number] if number<len(times) else len(lines)
        body=clean_text('\n'.join(lines[offset+1:end_offset]))
        cues.append(dict(cue_id=number,start_ms=start,end_ms=end,text=body))
    return sorted(cues,key=lambda c:(c['start_ms'],c['end_ms'],c['cue_id']))


def subtitle_source(video_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',video_id):
        raise ValueError('非法视频 ID')
    path=DATA/'subtitle'/f'{video_id}.srt'
    if path.is_symlink():
        raise ValueError(f'字幕不能是符号链接: {path}')
    if not path.exists():
        return dict(path=str(path),status='missing',bytes=0,sha256=None,cues=0,nonempty_cues=0),[]
    raw=path.read_bytes()
    try:
        text=raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError(f'字幕不是 UTF-8，保留原文件，不自动猜编码: {path}') from error
    try:
        cues=parse_srt(text)
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error
    useful=sum(bool(c['text']) for c in cues)
    return dict(path=str(path),status='available' if useful else 'empty',bytes=len(raw),
                sha256=sha(path),cues=len(cues),nonempty_cues=useful),cues


def subtitle_prompt(row, text):
    original=prompt(row,0)
    if not text:
        return original  # 无证据时必须逐字回退，不能加入空字幕模板改变 Prompt。
    anchor='Select the best answer'
    before,after=original.split(anchor,1)
    return before+SUB_HEADER+text+'\n'+anchor+after


def make_alignment(video_id, source, cues, frame_indices, frame_seconds, count_tokens):
    """按真实秒数做 [start,end) 命中；重复采样同一 cue 仅加入一次。"""
    if len(frame_indices)!=32 or len(frame_seconds)!=32:
        raise ValueError('实际采样不是32帧')
    if (any(type(i) is not int or i<0 for i in frame_indices)
            or any(a>=b for a,b in zip(frame_indices,frame_indices[1:]))
            or any(not math.isfinite(t) or t<0 for t in frame_seconds)
            or any(a>=b for a,b in zip(frame_seconds,frame_seconds[1:]))):
        raise ValueError('帧号或秒数非有限、非递增')
    hits=[]; chosen={}
    for seconds in frame_seconds:
        microseconds=round(seconds*1_000_000)
        matches=[c for c in cues if c['text'] and c['start_ms']*1000<=microseconds<c['end_ms']*1000]
        hits.append([c['cue_id'] for c in matches])
        for cue in matches:
            chosen[cue['cue_id']]=cue
    selected=sorted(chosen.values(),key=lambda c:(c['start_ms'],c['end_ms'],c['cue_id']))
    text='\n'.join(c['text'] for c in selected)
    tokens=count_tokens(text) if text else 0
    if tokens>MAX_SUB_TOKENS:
        raise ValueError(f'命中字幕 {tokens} token 超过 {MAX_SUB_TOKENS}，停止，不截断/扩大时间窗')
    reason='used' if selected else ('no_hit' if source['status']=='available' else source['status'])
    return dict(video_id=video_id,source=source,frame_indices=frame_indices,frame_seconds=frame_seconds,
                frame_cue_ids=hits,selected_cues=selected,text=text,text_tokens=tokens,status=reason,
                matching='start <= actual frame time < end; microsecond comparison; unique cue ID; time order')


class CaptureVideoProcessor:
    """捕获原生 frame_timestamps；属性读写都转交，确保 num_frames 临时覆盖真的生效。"""
    def __init__(self, base):
        object.__setattr__(self,'base',base)
        object.__setattr__(self,'timestamps',None)
    def __getattr__(self,name):
        return getattr(self.base,name)
    def __setattr__(self,name,value):
        if name in ['base','timestamps']:
            object.__setattr__(self,name,value)
        else:
            setattr(self.base,name,value)
    def __call__(self,*args,**kwargs):
        result=self.base(*args,**kwargs)
        self.timestamps=[[float(t) for t in seq] for seq in result['frame_timestamps']]
        return result


def sealed(record):
    if 'record_sha256' in record:
        raise ValueError('重复封装记录')
    return dict(record,record_sha256=signature(record))


def load_units(path, sig, rows, baseline, evidence_loader):
    """以每题为断点；残行、重复、错配、篡改均拒绝自动修复。"""
    lookup={r['index']:r for r in rows}
    found={}
    for record in read_lines(path):
        item=dict(record); digest=item.pop('record_sha256',None)
        if digest!=signature(item):
            raise ValueError('字幕单元校验失败')
        idx=item['index']
        if idx not in lookup or idx in found or item['signature']!=sig:
            raise ValueError('字幕单元重复/index越界/配置不符')
        row=lookup[idx]; old=baseline[f'{idx}:frames:p0']
        evidence=evidence_loader(row['video_id'])
        expected_origin='generated_with_subtitle' if evidence['status']=='used' else 'reused_frames32_'+evidence['status']
        if (item['video_id']!=row['video_id'] or item['evidence_sha256']!=signature(evidence)
                or item['prompt_sha256']!=signature(subtitle_prompt(row,evidence['text']))
                or item['subtitle_status']!=evidence['status'] or item['origin']!=expected_origin
                or item['pred']!=parse_answer(item['raw_text'])
                or item['pred'] not in ['',*LETTERS]):
            raise ValueError('字幕单元与证据/Prompt/输出不一致')
        if evidence['status']!='used' and (item['pred']!=old['semantic_pred']
                or item['raw_text']!=old['raw_text'] or item['cost']['new_answer_calls']!=0):
            raise ValueError('无字幕回退不是原始基线')
        for name in ['grid_thw','pv_shape','patch_time_range','n_tok']:
            if item['visual'][name]!=old[name]:
                raise ValueError(f'字幕路线改变了原视觉布局: {name}')
        if item['checks']['base_prefix_sha256']!=old['prefix_sha256']:
            raise ValueError('无字幕输入签名不一致')
        if not item['checks']['visual_unchanged']:
            raise ValueError('视觉审计未通过')
        if item['origin_stage']=='smoke' and not all(item['checks'].get(k) is True for k in
                ['native_input_equal','no_sub_answer_equal']):
            raise ValueError('冒烟原生输入/基线答案审计失败')
        cost=item['cost']
        if cost['new_answer_calls']!=int(evidence['status']=='used') or any(
                not math.isfinite(cost[k]) or cost[k]<0 for k in
                ['answer_s','preprocess_s','control_s','peak_allocated_gib']):
            raise ValueError('字幕单元成本字段异常')
        found[idx]=record
    return found
