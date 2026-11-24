#!/usr/bin/env python3
"""阶段 2：固定阶段 1 答案，只提取完整视觉前缀后的下一 token 分数。"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random

from r2v_pair_infer import digest, disk_check, dump_new, git, load_completed, prepare

LETTERS = 'ABCD'


def normalize(logits):
    """四字母条件分布；这是选择信号，绝不是答案正确概率。"""
    if len(logits) != 4 or not all(math.isfinite(x) for x in logits):
        raise ValueError('必须是四个有限 logit')
    maximum = max(logits)
    values = [math.exp(x - maximum) for x in logits]
    return [x/sum(values) for x in values]


def logadd(a, b):
    maximum = max(a, b)
    return maximum + math.log(math.exp(a-maximum) + math.exp(b-maximum))


def features(logits, answer):
    p = normalize(logits)
    ordered = sorted(p, reverse=True)
    chosen = LETTERS.index(answer)
    return dict(p=p, top1=LETTERS[max(range(4), key=lambda i: p[i])],
                margin=ordered[0]-ordered[1],
                answer_margin=p[chosen]-max(v for i, v in enumerate(p) if i != chosen),
                answer_p=p[chosen])


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError(f'残缺尾行，停止且不截断: {path}')
            rows.append(json.loads(line))
    return rows


def source_data(args):
    # 复用阶段 1 的只读准备函数，避免悄悄改变 Prompt、视觉预算或数据筛选。
    old = json.loads((args.source/'config_lock.json').read_text())
    if old['phase'] != 'short' or old['min_pixels'] != 3136 or old['use_subtitle']:
        raise ValueError('仅接受完整无字幕阶段 1 short 配对')
    args.phase, args.smoke_count = 'short', 6
    args.max_pixels, args.frames = old['max_pixels'], old['num_frames']
    args.model, args.data_root, args.cache_root = [Path(old[k]) for k in ['model', 'data_root', 'cache_root']]
    rows, current = prepare(args)
    # 源脚本/权重/视频变动必须停止；git 提交号变化单独记录，不等于模型协议变化。
    for key in old:
        if key not in {'git_commit', 'min_free_gb'} and old[key] != current.get(key):
            raise ValueError(f'阶段 1 来源漂移: {key}')
    pairs = load_completed(args.source/'pair_short.jsonl', rows)
    if set(pairs) != {r['index'] for r in rows}:
        raise ValueError('阶段 1 配对没有全部完成')
    for row in rows:
        pair = pairs[row['index']]
        if pair['video_id'] != row['video'] or pair['video_path'] != row['_video'] or pair['question'] != row['question']:
            raise ValueError('阶段 1 题目或视频与 TSV 不一致')
    return old, rows, pairs


def cache_location(old):
    codec = dict(old['codec'], max_pixels=old['max_pixels'])
    tag = hashlib.sha256(json.dumps(codec, sort_keys=True).encode()).hexdigest()[:16]
    return Path(old['cache_root'])/tag, codec


def check_existing_caches(old, rows):
    cache, codec = cache_location(old)
    # 阶段 2 只复用这次配对的缓存，缺失则报错，不重新预处理、不生成新视频缓存。
    for video in {r['_video'] for r in rows}:
        raw = (f'{video}|eng={codec["engine"]}|tc={codec["target_canvas"]}|gs={codec["group_size"]}'
               f'|ipg={codec["images_per_group"]}|patch={codec["patch"]}'
               f'|mp={codec["max_pixels"]}|mask={codec["spatial_mask_mode"]}')
        target = cache/f'{Path(video).stem}_{hashlib.md5(raw.encode()).hexdigest()}'
        if not all((target/n).is_file() for n in ['meta.json', 'src_patch_position.npy']):
            raise FileNotFoundError(f'缺少完整已有缓存，不自动重建: {target}')
    return cache, codec


def make_lock(args, old, rows):
    folder = Path(__file__).parent
    model = args.model
    token_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                   'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja']
    return dict(protocol='r2v_signal_v1', scope=args.scope,
                interpretation='posthoc_short_diagnostic_only; no calibration or threshold fitting',
                scoring='generate(max_new_tokens=1).logits[0]: raw unprocessed next-token logits',
                primary='plain A/B/C/D conditional softmax',
                sensitivity='logsumexp over plain and leading-space single-token variants',
                answer_policy='keep phase1 answers; never silently replace by scored top1',
                seed=0, indices=[r['index'] for r in rows],
                source=str(args.source.resolve()), source_lock_sha256=digest(args.source/'config_lock.json'),
                source_pairs_sha256=digest(args.source/'pair_short.jsonl'), source_config=old,
                tokenizer_files_sha256={n:digest(model/n) for n in token_files if (model/n).is_file()},
                scripts_sha256={n:digest(folder/n) for n in ['r2v_signal_score.py', 'r2v_signal_analyze.py']},
                min_free_gb=args.min_free_gb, git_commit=git('rev-parse', 'HEAD'))


def verify_smoke(args, lock):
    folder = args.output.parent/'smoke'
    smoke = json.loads((folder/'signal_config_lock.json').read_text())
    for key in lock:
        if key not in {'scope', 'indices'} and smoke.get(key) != lock[key]:
            raise ValueError(f'阶段 2 冒烟与全量配置不一致: {key}')
    rows = read_jsonl(folder/'signal_short.jsonl')
    summary = json.loads((folder/'signal_summary.json').read_text())
    if {r['index'] for r in rows} != set(smoke['indices']) or len(rows) != len(smoke['indices']):
        raise ValueError('阶段 2 冒烟结果不完整')
    if not summary['audit']['smoke_interface_pass']:
        raise ValueError('阶段 2 冒烟接口检查未通过；先检查汇总')


def token_probe(tokenizer):
    result, used = {}, set()
    for c in LETTERS:
        group = {}
        for name, text in [('plain', c), ('space', ' '+c)]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
                raise ValueError(f'{text!r} 不是可精确回解的单 token，禁止猜测 token id')
            if ids[0] in used:
                raise ValueError('候选 token 重叠')
            used.add(ids[0])
            group[name] = ids[0]
        group['newline_probe'] = tokenizer.encode('\n'+c, add_special_tokens=False)
        result[c] = group
    return result


def score(args, old, rows, pairs, lock):
    from tqdm import tqdm
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.run.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = args.output/'signal_config_lock.json'
        target = args.output/'signal_short.jsonl'
        if config.exists():
            if json.loads(config.read_text()) != lock:
                raise ValueError('阶段 2 配置锁不同，请用新标签；拒绝混跑')
        else:
            if target.exists():
                raise ValueError('发现无配置锁的分数文件')
            dump_new(config, lock)
        existing = read_jsonl(target)
        expected = {r['index'] for r in rows}
        done = set()
        for r in existing:
            if r['index'] not in expected or r['index'] in done or r['phase1'] != pairs[r['index']]:
                raise ValueError('续跑结果重复、来源不符或越界')
            for b in ['codec', 'frames']:
                if not all(math.isfinite(v) for v in r[b]['plain_logits']+r[b]['space_logits']):
                    raise ValueError('续跑结果含非法分数')
            done.add(r['index'])
        if done == expected:
            print(f'已完成 {len(done)}/{len(expected)} 题；无需加载模型')
            return
        cache, codec = check_existing_caches(old, rows)
        os.environ['ONLINE_CODEC_CACHE_DIR'] = str(cache)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用')
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
        processor.video_processor.max_pixels = old['max_pixels']
        processor.video_processor.min_pixels = old['min_pixels']
        processor.image_processor.max_pixels = old['max_pixels']
        processor.image_processor.min_pixels = old['min_pixels']
        probes = token_probe(processor.tokenizer)
        probe_path = args.output/'token_probe.json'
        if probe_path.exists():
            if json.loads(probe_path.read_text()) != probes:
                raise ValueError('token id 探测结果漂移')
        else:
            dump_new(probe_path, probes)
        model = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                    local_files_only=True, torch_dtype='auto', device_map={'':0}).eval()
        if {r['transformers_version'] for r in pairs.values()} != {transformers.__version__} or {r['torch_version'] for r in pairs.values()} != {torch.__version__}:
            raise ValueError('Torch/Transformers 版本与阶段 1 不一致')
        sms = int(processor.spatial_merge_size)

        def one(row, backend, suffix):
            import time
            disk_check(args.min_free_gb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            text = processor.apply_chat_template([{'role':'user', 'content':[
                {'type':'video'}, {'type':'text','text':row['_prompt']}]}],
                tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[row['_video']], video_backend=backend,
                num_frames=old['num_frames'], max_pixels=old['max_pixels'], codec_config=codec,
                return_tensors='pt', padding=True)
            prior = pairs[row['index']]
            grid, shape = inputs['image_grid_thw'].tolist(), list(inputs['pixel_values'].shape)
            if grid != prior[f'grid_thw_{suffix}'] or shape != prior[f'pv_shape_{suffix}']:
                raise ValueError(f'视觉输入形状与阶段 1 不一致: {row["index"]}/{backend}')
            positions = inputs['patch_positions']
            time_range = [float(positions[:,0].min()),float(positions[:,0].max())]
            if time_range != prior[f'patch_time_range_{suffix}'] or sms != prior['spatial_merge_size']:
                raise ValueError('时间位置/merge size 与阶段 1 不一致')
            prefix_hash = hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
            inputs = {k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
            inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
            # 使用原 generate 的 prefill 路径，避免远程模型 generation 准备逻辑不同。
            # 只生成一个 token 并取原始 logits，不保存全词表或大型 tensor。
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                    return_dict_in_generate=True, output_logits=True)
            if len(generated.logits) != 1:
                raise ValueError('没有得到恰好一个位置的原始 logits')
            raw = generated.logits[0][0].float()
            token_id = int(generated.sequences[0,-1])
            if token_id != int(raw.argmax()):
                raise ValueError('greedy token 与原始 logits argmax 不一致，需检查生成后处理')
            ids = [probes[c][v] for v in ['plain','space'] for c in LETTERS]
            selected = raw[ids].cpu().tolist()
            plain, space = selected[:4], selected[4:]
            answer = prior[f'{backend}_pred']
            if answer not in LETTERS or len(answer) != 1:
                raise ValueError('阶段 1 答案不是合法单字母')
            # 冒烟额外验证 forward 最后位置和 generate 的分数，防止取错 token 位置。
            check = None
            if args.scope == 'smoke':
                with torch.inference_mode():
                    direct = model(**inputs, logits_to_keep=1, use_cache=False, return_dict=True)
                check_logits = direct.logits[0,-1,ids].float().cpu()
                reference = torch.tensor(selected)
                check = dict(max_abs_error=float((check_logits-reference).abs().max()),
                    passed=bool(torch.allclose(check_logits, reference, atol=0.05, rtol=0.005)))
                del direct
                if not check['passed']:
                    raise ValueError(f'forward/generate 分数核对失败: {check}')
            chosen = next((c for c in LETTERS if token_id in [probes[c]['plain'],probes[c]['space']]), None)
            partition = float(torch.logsumexp(raw, dim=0))
            mass_plain = sum(math.exp(x-partition) for x in plain)
            mass_union = sum(math.exp(x-partition) for x in selected)
            torch.cuda.synchronize()
            return dict(plain_logits=plain, space_logits=space,
                plain=features(plain,answer),
                union=features([logadd(a,b) for a,b in zip(plain,space)],answer),
                letter_mass_plain=mass_plain, letter_mass_union=mass_union,
                generated_token_id=token_id,
                generated_token_text=processor.tokenizer.decode([token_id],clean_up_tokenization_spaces=False),
                first_letter=chosen, matches_phase1=(chosen==answer),
                grid_thw=grid, pv_shape=shape, patch_time_range=time_range,
                prefix_sha256=prefix_hash, forward_check=check,
                latency_s=time.perf_counter()-start,
                peak_mem_gib=torch.cuda.max_memory_allocated()/1024**3)

        with target.open('a',encoding='utf-8') as out, tqdm(total=len(rows),initial=len(done),
                desc=f'R2V 字母分数 {args.scope}',unit='题',dynamic_ncols=True) as bar:
            for row in rows:
                if row['index'] in done:
                    continue
                record = dict(index=row['index'],video_id=row['video'],phase1=pairs[row['index']])
                for backend,suffix in [('codec','c'),('frames','f')]:
                    record[backend] = one(row,backend,suffix)
                record['runtime'] = dict(torch=torch.__version__,transformers=transformers.__version__,
                                         gpu=torch.cuda.get_device_name(0))
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                out.flush()
                os.fsync(out.fileno())
                bar.update(1)
                bar.set_postfix(index=row['index'])
        print(f'完成：{target}；下一步先运行 summarize 检查接口审计。')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','score'])
    p.add_argument('--scope',choices=['smoke','short'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-free-gb',type=float,default=10)
    args = p.parse_args()
    if args.min_free_gb < 1:
        p.error('至少保留 1 GiB 磁盘')
    args.source,args.output = args.source.resolve(),args.output.resolve()
    if args.output == args.source or args.source in args.output.parents:
        p.error('阶段 2 必须使用独立输出目录')
    old,rows,pairs = source_data(args)
    if args.scope == 'smoke':
        rows = rows[:6]
    check_existing_caches(old,rows)
    lock = make_lock(args,old,rows)
    if args.action == 'preflight':
        print(f'只读预检通过：{len(rows)} 题，复用已有缓存；未加载模型。')
        return
    if args.scope == 'short':
        verify_smoke(args,lock)
    score(args,old,rows,pairs,lock)


if __name__ == '__main__':
    main()
