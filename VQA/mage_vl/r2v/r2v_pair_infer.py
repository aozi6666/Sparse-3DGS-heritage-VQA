#!/usr/bin/env python3
"""R2V 阶段 0/1；preflight 仅用标准库，infer 才加载模型。"""
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

PROMPT = ('These are the frames of a video. Select the best answer to the following '
          'multiple-choice question based on the video. Respond with only the letter '
          '(A, B, C, or D) of the correct option.\n{question}\n{options}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, obj):
    # 独占创建：绝不覆盖已有配置或结果。
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def disk_check(min_gb):
    free = shutil.disk_usage('/root/autodl-tmp').free / 1024**3
    if free < min_gb:
        raise RuntimeError(f'可用磁盘 {free:.2f} GiB < {min_gb} GiB；停止并保留文件')
    return round(free, 2)


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def prepare(args):
    if git('branch', '--show-current') != 'vlm':
        raise RuntimeError('必须在 vlm 分支；本脚本不自动切换分支')
    disk_check(args.min_free_gb)
    root, model = args.data_root.resolve(), args.model.resolve()
    required = ['config.json', 'processing_mage_vl.py', 'modeling_mage_vl.py',
                'video_processing_mage_vl.py', 'codec_video_processing_mage_vl.py',
                'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']
    for name in required:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    index = model / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    for name in shards:
        if not (model / name).is_file() or not (model / name).stat().st_size:
            raise FileNotFoundError(model / name)
    for executable in [os.environ.get('CV_PREINFER_BIN', 'cv-preinfer'), 'ffmpeg', 'ffprobe']:
        if not shutil.which(executable):
            raise FileNotFoundError(f'找不到 {executable}；不自动安装')
    tsv = root / 'Video-MME.tsv'
    with tsv.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['duration'] == 'short']
    if not rows or len({r['index'] for r in rows}) != len(rows):
        raise ValueError('short 集为空或 index 重复')
    # 冒烟固定取 TSV 前 N 条，不做隐式随机抽样；seed 仍锁定为 0。
    if args.phase == 'smoke':
        rows = rows[:args.smoke_count]
    for row in rows:
        options = ast.literal_eval(row['candidates'])
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f'选项格式错误: {row["index"]}')
        for letter, option in zip('ABCD', options):
            if not isinstance(option, str) or not option.startswith(letter + '.'):
                raise ValueError(f'选项标签错误: {row["index"]}')
        if row['answer'].strip().upper() not in list('ABCD'):
            raise ValueError(f'真值错误: {row["index"]}')
        video = (root / row['video_path']).resolve()
        if not video.is_relative_to(root) or not video.is_file():
            raise FileNotFoundError(video)
        row['_video'] = str(video)
        row['_prompt'] = PROMPT.format(question=row['question'], options='\n'.join(options))
    fingerprint = {name: digest(model / name) for name in required if not name.startswith('tokenizer')}
    for name in ['video_preprocessor_config.json', 'generation_config.json', 'model.safetensors.index.json']:
        if (model / name).exists():
            fingerprint[name] = digest(model / name)
    lock = dict(protocol=1, phase=args.phase, model=str(model), data_root=str(root),
                tsv_sha256=digest(tsv), indices=[r['index'] for r in rows],
                prompt=PROMPT, prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                max_pixels=args.max_pixels, min_pixels=3136, num_frames=args.frames,
                codec=dict(engine='hevc', target_canvas=32, patch=16, group_size=32,
                           images_per_group=4, spatial_mask_mode='off'),
                do_sample=False, max_new_tokens=32, use_subtitle=False, seed=0,
                sampling='TSV first N short rows' if args.phase == 'smoke' else 'all short rows',
                git_commit=git('rev-parse', 'HEAD'), script_sha256=digest(__file__),
                summary_script_sha256=digest(Path(__file__).with_name('r2v_pair_summarize.py')),
                model_sources_sha256=fingerprint,
                weights={n: {'bytes': (model/n).stat().st_size, 'mtime_ns': (model/n).stat().st_mtime_ns} for n in shards},
                videos={r['_video']: {'bytes': Path(r['_video']).stat().st_size,
                                     'mtime_ns': Path(r['_video']).stat().st_mtime_ns} for r in rows},
                cache_root=str(args.cache_root.resolve()), min_free_gb=args.min_free_gb)
    print(json.dumps(dict(phase=args.phase, questions=len(rows), videos=len(lock['videos']),
                         free_gib=disk_check(args.min_free_gb), max_pixels=args.max_pixels,
                         frames=args.frames, output=str(args.output.resolve())), ensure_ascii=False), flush=True)
    return rows, lock


def load_completed(path, rows):
    done, expected = {}, {r['index']: r for r in rows}
    if not path.exists():
        return done
    # 残行不自动截断/删除：保留现场，避免静默丢题或改变统计分母。
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError('jsonl 尾行不完整；停止，不自动修复')
            obj = json.loads(line)
            idx = obj['index']
            if idx in done or idx not in expected:
                raise ValueError(f'重复或未知 index: {idx}')
            for backend in ['codec', 'frames']:
                if obj[f'{backend}_ok'] != (obj[f'{backend}_pred'] == obj['gt']):
                    raise ValueError(f'已有结果正确性字段不一致: {idx}')
            if obj['gt'] != expected[idx]['answer'].strip().upper():
                raise ValueError(f'已有结果真值不一致: {idx}')
            done[idx] = obj
    return done


def infer(args, rows, lock):
    output = args.output.resolve()
    if args.phase == 'short':
        smoke_dir = output.parent / 'smoke'
        smoke_lock = json.loads((smoke_dir / 'config_lock.json').read_text())
        smoke_summary = json.loads((smoke_dir / 'summary_short.json').read_text())
        differing = {'phase', 'indices', 'videos', 'sampling'}
        if ({k: v for k, v in smoke_lock.items() if k not in differing} !=
                {k: v for k, v in lock.items() if k not in differing}):
            raise ValueError('冒烟配置与 short 不一致；先对当前配置重新冒烟')
        smoke_rows = load_completed(smoke_dir / 'pair_short.jsonl', rows)
        if set(smoke_rows) != set(smoke_lock['indices']) or smoke_summary['metrics']['N'] != len(smoke_rows):
            raise ValueError('冒烟配对或汇总不完整')
    output.mkdir(parents=True, exist_ok=True)
    # 同一输出目录只允许一个进程，防止并发重复追加。
    with (output / '.run.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_path = output / 'config_lock.json'
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != lock:
                raise ValueError('配置锁不同；请使用新的实验标签，不混跑')
        else:
            if (output / 'pair_short.jsonl').exists():
                raise ValueError('发现无配置锁的结果文件；拒绝接管')
            dump_new(lock_path, lock)
        result_path = output / 'pair_short.jsonl'
        done = load_completed(result_path, rows)
        if len(done) == len(rows):
            print('全部题已完成；无需加载模型', flush=True)
            return
        # 缓存按完整 codec 配置隔离；不访问历史实验缓存。
        codec = dict(lock['codec'], max_pixels=args.max_pixels)
        cache_tag = hashlib.sha256(json.dumps(codec, sort_keys=True).encode()).hexdigest()[:16]
        cache = args.cache_root.resolve() / cache_tag
        cache.mkdir(parents=True, exist_ok=True)
        # 底层遇到残缺缓存可能重建目录；此处提前拒绝，保护已有缓存。
        for entry in cache.iterdir():
            if entry.is_dir() and (not (entry/'meta.json').is_file() or
                                   not (entry/'src_patch_position.npy').is_file()):
                raise RuntimeError(f'缓存目录残缺，停止且不删除: {entry}')
        os.environ['ONLINE_CODEC_CACHE_DIR'] = str(cache)
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('需要 CUDA GPU')
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        processor = AutoProcessor.from_pretrained(str(args.model.resolve()), trust_remote_code=True, local_files_only=True)
        # 关键：Frames 不读取顶层 max_pixels，必须修改 video_processor。
        processor.video_processor.max_pixels = args.max_pixels
        processor.video_processor.min_pixels = 3136
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = 3136
        sms = int(processor.spatial_merge_size)
        print(f'生效预算: frames.max_pixels={processor.video_processor.max_pixels}, '
              f'min_pixels={processor.video_processor.min_pixels}, spatial_merge_size={sms}', flush=True)
        model = AutoModelForCausalLM.from_pretrained(str(args.model.resolve()), trust_remote_code=True,
                    local_files_only=True, torch_dtype='auto', device_map={'': 0}).eval()
        import transformers
        print(f'torch={torch.__version__}, transformers={transformers.__version__}, '
              f'GPU={torch.cuda.get_device_name(0)}', flush=True)

        def run(row, backend):
            disk_check(args.min_free_gb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            text = processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'video'}, {'type': 'text', 'text': row['_prompt']}]}],
                tokenize=False, add_generation_prompt=True)
            # 双路都传 mp4 路径，保留处理器原生时间戳；不预抽 PIL。
            inputs = processor(text=[text], videos=[row['_video']], video_backend=backend,
                num_frames=args.frames, max_pixels=args.max_pixels, codec_config=codec,
                return_tensors='pt', padding=True)
            grid = inputs['image_grid_thw'].tolist()
            shape = list(inputs['pixel_values'].shape)
            tok_numerator = sum(t*h*w for t, h, w in grid)
            if tok_numerator % (sms*sms):
                raise ValueError('grid 无法整除 spatial_merge_size²')
            positions = inputs['patch_positions']
            position_shape = list(positions.shape)
            position_time_range = [float(positions[..., 0].min()), float(positions[..., 0].max())]
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            raw = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # 只接受独立字母，不能把英文单词中的 A/B/C/D 当答案。
            match = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', raw.upper())
            pred = match.group(1) if match else ''
            return dict(pred=pred, raw=raw, n_tok=tok_numerator//(sms*sms), t=elapsed,
                        pv_shape=shape, grid_thw=grid, patch_positions_shape=position_shape,
                        patch_time_range=position_time_range,
                        peak_mem=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved=torch.cuda.max_memory_reserved()/1024**3)

        with result_path.open('a', encoding='utf-8') as f:
            for row in rows:
                if row['index'] in done:
                    continue
                result = {k: row[k] for k in ['index', 'duration', 'domain', 'sub_category', 'task_type', 'question']}
                result.update(video_id=row['video'], video_path=row['_video'], gt=row['answer'].strip().upper())
                result.update(effective_frames_max_pixels=processor.video_processor.max_pixels,
                              effective_frames_min_pixels=processor.video_processor.min_pixels,
                              spatial_merge_size=sms, torch_version=torch.__version__,
                              transformers_version=transformers.__version__, gpu=torch.cuda.get_device_name(0))
                # 一题两路都成功才落盘；异常立即退出，已落盘完整配对保留。
                for backend, suffix in [('codec', 'c'), ('frames', 'f')]:
                    values = run(row, backend)
                    result[f'{backend}_pred'] = values.pop('pred')
                    result[f'{backend}_raw'] = values.pop('raw')
                    result[f'{backend}_ok'] = result[f'{backend}_pred'] == result['gt']
                    result.update({f'{key}_{suffix}': value for key, value in values.items()})
                    print(row['index'], backend, result[f'{backend}_pred'] or 'INVALID',
                          f'tokens={result[f"n_tok_{suffix}"]} latency={result[f"t_{suffix}"]:.2f}s', flush=True)
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'infer'])
    p.add_argument('--phase', choices=['smoke', 'short'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--max-pixels', type=int, default=150000)
    p.add_argument('--frames', type=int, choices=[16, 32], default=32)
    p.add_argument('--smoke-count', type=int, choices=range(5, 11), default=6)
    p.add_argument('--min-free-gb', type=float, default=10)
    args = p.parse_args()
    if args.max_pixels < 3136 or args.min_free_gb < 1:
        p.error('像素预算至少 3136，磁盘保留至少 1 GiB')
    rows, lock = prepare(args)
    if args.action == 'infer':
        infer(args, rows, lock)


if __name__ == '__main__':
    main()
