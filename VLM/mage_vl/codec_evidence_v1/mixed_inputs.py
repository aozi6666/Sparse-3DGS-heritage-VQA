"""Read-only native Codec + images, audited against separately processed slots."""
import contextlib
import importlib
import re
import sys
import time
from pathlib import Path
import protocol as p

VISUAL = ('pixel_values', 'image_grid_thw', 'patch_positions')
VS, VE, IP, VP = '<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>'
VIDEO_RE = re.compile(re.escape(VS) + r'\s*' + re.escape(VP) + r'\s*' + re.escape(VE) + r'\n?')


def video_expansion(raw, expanded):
    """Recover the WHOLE insertion, including the leading timestamp, independently."""
    matches = list(VIDEO_RE.finditer(raw))
    p.require(len(matches) == 1, '必须恰好一个视频槽')
    match = matches[0]
    before, after = raw[:match.start()], raw[match.end():]
    p.require(expanded.startswith(before) and expanded.endswith(after), '纯视频文本边界改变')
    return expanded[len(before):len(expanded)-len(after)] if after else expanded[len(before):]


def tensor_digest(values):
    import hashlib
    import torch
    h = hashlib.sha256()
    for n, t in sorted(values.items()):
        h.update(n.encode()); h.update(str(t.dtype).encode()); h.update(str(list(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def equal(a, b):
    import torch
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                  and torch.equal(a[k], b[k]) for k in a)


class Capture:
    def __init__(self, base):
        self.base, self.text = base, None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, text, *args, **kwargs):
        self.text = list(text) if not isinstance(text, str) else [text]
        return self.base(text, *args, **kwargs)


class Engine:
    def __init__(self, m):
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(1)
        self.m = m
        self.proc = AutoProcessor.from_pretrained(str(p.MODEL), trust_remote_code=True, local_files_only=True)
        self.proc.image_processor.max_pixels = 150000
        self.proc.image_processor.min_pixels = 3136
        self.module = importlib.import_module(type(self.proc).__module__.rsplit('.', 1)[0] + '.codec_video_processing_mage_vl')
        files = {}
        for name, mod in [('processing_mage_vl.py', sys.modules[type(self.proc).__module__]),
                          ('codec_video_processing_mage_vl.py', self.module)]:
            source = Path(mod.__file__)
            p.require(p.sha(source) == p.sha(p.MODEL / name), '实际动态模块与冻结源码不同：' + name)
            files[name] = dict(path=str(source), sha256=p.sha(source))
        self.loaded_modules = files
        self.loaded, self.loaded_vid = None, None

    @contextlib.contextmanager
    def readonly(self, video):
        original = self.module.process_codec_video

        def load(path, cfg):
            p.require(path == video['video_path'], '禁止未知视频自动编码')
            p.require(str(self.module._cache_dir_for(path, cfg)) == video['cache_dir'], '缓存键不符')
            for k, value in p.CODEC.items():
                p.require(getattr(cfg, k) == value, 'Codec配置不符：' + k)
            if self.loaded_vid != path:
                for f, h in video['cache_files_sha256'].items():
                    p.require(p.sha(f) == h, '旧缓存改变：' + f)
                self.loaded = self.module._load_codec_result(Path(video['cache_dir']))
                self.loaded_vid = path
            p.require(self.loaded['fps'] == video['fps'], '缓存FPS不符')
            return self.loaded

        self.module.process_codec_video = load
        try:
            yield
        finally:
            self.module.process_codec_video = original

    def text(self, row):
        # Import only pure formatting helpers, never old context/init/run.
        sys.path.insert(0, str(p.PROJECT / 'VLM/mage_vl/practical'))
        from subtitle_common import subtitle_prompt
        return subtitle_prompt(row, self.m['videos'][row['video_id']]['subtitle_text'])

    def call(self, row, images=(), ids=(), order='video-first', answer_text=None):
        v = self.m['videos'][row['video_id']]
        text = self.text(row) if answer_text is None else answer_text
        image_content = []
        for i, fid in enumerate(ids):
            image_content.extend([{'type': 'text', 'text': f'\nSupplementary full frame {i+1} from the SAME video at {fid/v["fps"]:.6f} seconds:\n'}, {'type': 'image'}])
        p.require(len(images) == len(ids), '图片与来源帧号不一一对应')
        content = ([{'type': 'video'}] + image_content if order == 'video-first'
                   else image_content + [{'type': 'video'}]) + [{'type': 'text', 'text': text}]
        chat = self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        tok = self.proc.tokenizer
        capture = Capture(tok); self.proc.tokenizer = capture
        try:
            with self.readonly(v):
                x = self.proc(text=[chat], images=list(images) or None, videos=[row['video_path']],
                    video_backend='codec', max_pixels=150000,
                    codec_config=dict(p.CODEC, cache_root=Path(v['cache_dir']).parent), return_tensors='pt', padding=True)
        finally:
            self.proc.tokenizer = tok
        p.require(capture.text is not None and len(capture.text) == 1, '捕获文本失败')
        return dict(x), chat, capture.text[0]

    def make(self, row, images=(), ids=(), order='video-first', verify_old=True):
        import torch
        start = time.perf_counter()
        base, base_chat, expanded_base = self.call(row)
        old = p.baseline(self.m, row)
        if verify_old:
            p.require(tensor_digest(base) == old['tensor_sha256'], '纯Codec与旧A完整Tensor不同')
            p.require(p.digest(base['input_ids'].tolist()) == old['input_sha256'], '纯Codec输入token不同')
            p.require(p.digest(self.text(row)) == old['prompt_sha256'], '旧字幕/问答文本不同')
        if not images:
            return base, dict(passed=True, tensor_sha256=tensor_digest(base), input_tokens=int(base['input_ids'].shape[1]),
                visual_tokens=int(base['image_grid_thw'].prod(-1).sum()) // self.proc.spatial_merge_size**2,
                extra_visual_tokens=0, image_grids=[], preprocess_s=time.perf_counter()-start)
        actual, chat, expanded = self.call(row, images, ids, order)
        image_chat = self.proc.apply_chat_template([{'role': 'user', 'content':[{'type':'image'} for _ in images]}], tokenize=False, add_generation_prompt=True)
        separate = self.proc(text=[image_chat], images=list(images), return_tensors='pt', padding=True)
        counts = (separate['image_grid_thw'].prod(-1) // self.proc.spatial_merge_size**2).tolist()
        it = iter(counts)
        expected_text = re.sub(re.escape(IP), lambda _: IP * next(it), chat)
        codec_span = video_expansion(base_chat, expanded_base)
        p.require(len(VIDEO_RE.findall(expected_text)) == 1, '必须恰有一个Codec媒体槽')
        expected_text = VIDEO_RE.sub(lambda _: codec_span, expected_text, count=1)
        p.require(expanded == expected_text, '逐媒体文本展开错位（非仅总token计数）')
        expected = dict(self.proc.tokenizer([expected_text], padding=True, return_tensors='pt'))
        for k in VISUAL:
            slots = [base[k], separate[k]] if order == 'video-first' else [separate[k], base[k]]
            expected[k] = torch.cat(slots, dim=0)
        p.require(equal(actual, expected), '媒体槽Tensor/顺序/位置不符')
        n_pad = int((actual['input_ids'] == self.proc.tokenizer.convert_tokens_to_ids(IP)).sum())
        n_visual = int(actual['image_grid_thw'].prod(-1).sum()) // self.proc.spatial_merge_size**2
        p.require(n_pad == n_visual, '视觉占位总数不符')
        limit = p.read(p.MODEL / 'config.json')['text_config']['max_position_embeddings']
        p.require(int(actual['input_ids'].shape[1]) + 32 <= limit, '超过模型上下文，停止不截断')
        for im in images:
            p.require(im.mode == 'RGB', '补图必须RGB')
        return actual, dict(passed=True, tensor_sha256=tensor_digest(actual), input_tokens=int(actual['input_ids'].shape[1]),
            visual_tokens=n_visual, extra_visual_tokens=sum(counts), image_grids=separate['image_grid_thw'].tolist(),
            base_tensor_sha256=old['tensor_sha256'], subtitle_sha256=self.m['videos'][row['video_id']]['subtitle_sha256'],
            order=order, source_indices=list(ids), preprocess_s=time.perf_counter()-start)


def audit_cpu(m):
    import torch
    from PIL import Image
    from tqdm import tqdm
    saved = p.OUT / 'audits/cpu_mixed.json'
    if saved.exists():
        old = p.unseal(p.read(saved))
        p.require(old['signature'] == m['signature'] and old['passed'], 'CPU审计断点不符')
        print('CPU联合审计完整，跳过；GPU验收仍需用户运行。')
        return
    p.require(not torch.cuda.is_initialized(), 'CPU审计不应初始化CUDA')
    engine = Engine(m)
    row = p.scope_rows(m, 'pilot60')[0]
    # Distinct contents AND grids catch order bugs even when totals coincide.
    red = Image.new('RGB', (192, 128), (250, 10, 20))
    blue = Image.new('RGB', (128, 256), (10, 20, 250))
    green = Image.new('RGB', (320, 192), (5, 240, 20))
    cases = [('empty', [], [], 'video-first'), ('one', [red], [0], 'video-first'),
        ('two', [red, blue], [0, 1], 'video-first'), ('swapped', [blue, red], [1, 0], 'video-first'),
        ('image-first', [red, blue], [0, 1], 'image-first'),
        ('four', [red, blue, green, red], [0, 1, 2, 3], 'video-first')]
    reports = {}
    for name, images, ids, order in tqdm(cases, desc='联合输入逐槽CPU审计', unit='例', dynamic_ncols=True):
        _, audit = engine.make(row, images, ids, order)
        reports[name] = audit
    p.require(reports['two']['tensor_sha256'] != reports['swapped']['tensor_sha256'], '交换哨兵图后输入未改变')
    p.require(not torch.cuda.is_initialized(), 'CPU审计意外初始化CUDA')
    report = p.seal(dict(signature=m['signature'], passed=True, cases=reports,
        loaded_modules=engine.loaded_modules, cuda_initialized=False, QA_calls=0, video_decodes=0,
        note='Sentinel pictures only; original Codec caches read-only. GPU semantic acceptance still pending.'))
    p.atomic(p.OUT / 'audits/cpu_mixed.json', report)
    print('联合输入CPU审计通过：空/1/2/4图片、换序、图片在前；零GPU、零视频解码。')


@contextlib.contextmanager
def frames(video, ids):
    """Exact index requests through one decoder; no persistent RGB/JPEG cache."""
    import cv2
    from PIL import Image
    import hashlib
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video['video_path'])
    try:
        p.require(cap.isOpened(), '无法打开来源视频')
        p.require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == video['total_frames'], '解码器总帧数与元数据不同')
        p.require(abs(cap.get(cv2.CAP_PROP_FPS) - video['fps']) < .01, '解码器FPS与冻结元数据不符')
        images, hashes = [], []
        for fid in ids:
            p.require(type(fid) is int and 0 <= fid < video['total_frames'], '解码帧号非法')
            p.require(cap.set(cv2.CAP_PROP_POS_FRAMES, fid), '定位失败')
            ok, arr = cap.read()
            p.require(ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (fid+1)) < .5, '解码失败/帧号漂移')
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        yield images, hashes
    finally:
        cap.release()

