#!/usr/bin/env python3
"""显示完整配对题的 tqdm 进度；实际推理仍交给原入口，保持配置锁兼容。"""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading

from tqdm import tqdm


def completed(path, expected):
    if not path.exists():
        return 0
    done = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            # 写入中的最后一行尚不算完成；原推理脚本负责严格校验续跑文件。
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            if row['index'] in expected and 'codec_ok' in row and 'frames_ok' in row:
                done.add(row['index'])
    return len(done)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['smoke', 'short'])
    args = p.parse_args()
    project = Path(__file__).resolve().parents[3]
    # 与原 .sh 的路径规则完全一致；不修改原配置或实验脚本哈希。
    tag = os.environ.get('R2V_TAG') or (
        f"px{os.environ.get('R2V_MAX_PIXELS', '150000')}_f{os.environ.get('R2V_FRAMES', '32')}")
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        p.error('实验标签只允许字母数字下划线横线')
    data = Path(os.environ.get('MAGE_VIDEOMME_ROOT', str(project/'VQA/mage_vl/data/VideoMME')))
    with (data/'Video-MME.tsv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['duration'] == 'short']
    if args.phase == 'smoke':
        rows = rows[:6]
    expected = {r['index'] for r in rows}
    if not expected or len(expected) != len(rows):
        raise ValueError('数据集为空或 index 重复')
    result = project/'VQA/eval_results/r2v_mage'/tag/args.phase/'pair_short.jsonl'
    initial = completed(result, expected)
    print(f'已完成完整配对 {initial}/{len(expected)} 题；重跑此命令会跳过已完成题。', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    messages = queue.Queue()
    # 独立进程组：Ctrl+C 同时通知推理及它启动的子进程，避免遗留任务。
    child = subprocess.Popen(['bash', str(Path(__file__).with_name('run_r2v_pair.sh')), args.phase],
        cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', start_new_session=True)

    def read_output():
        try:
            for line in child.stdout:
                messages.put(line.rstrip('\n'))
        finally:
            child.stdout.close()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    interrupted = False
    with tqdm(total=len(expected), initial=initial, desc=f'R2V {args.phase} 配对',
              unit='题', dynamic_ncols=True, mininterval=0.5) as bar:
        try:
            while child.poll() is None or reader.is_alive() or not messages.empty():
                try:
                    tqdm.write(messages.get(timeout=0.2))
                except queue.Empty:
                    pass
                current = completed(result, expected)
                if current > bar.n:
                    bar.update(current - bar.n)
                bar.refresh()
        except KeyboardInterrupt:
            interrupted = True
            tqdm.write('正在停止；完整配对结果保留。再次运行同一命令即可续跑。')
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
        finally:
            # 显示进度发生异常时也停止子进程，不能让推理静默留在后台。
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            current = completed(result, expected)
            if current > bar.n:
                bar.update(current - bar.n)
    if interrupted:
        return 130
    code = child.returncode
    if code == 0 and current != len(expected):
        raise RuntimeError('子进程退出但配对数量不完整')
    if code == 0:
        print('配对完成；请运行原入口 summarize 汇总结果。', flush=True)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
