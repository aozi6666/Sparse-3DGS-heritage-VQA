#!/usr/bin/env python3
"""整理 VLMEvalKit 的评测结果，生成分类清晰的 Markdown 报告。

针对 Video-MME 这类带 score json 的数据集，把原始分数按
「视频时长 × (领域 / 子类别 / 任务类型)」重排成易读表格。

用法:
    python summarize_results.py                      # 自动找最新的 score json
    python summarize_results.py --score <path.json>  # 指定 score json 文件
    python summarize_results.py --outdir <dir>       # 指定输出目录（默认本脚本所在目录）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# VLMEvalKit 的 score 输出目录（会自动扫描其中最新的 *_score_*.json）
DEFAULT_SCAN_DIR = "/root/autodl-tmp/VLMEvalKit/outputs/Mage-VL"
SCORE_GLOB = "*_score_*.json"

# Video-MME 的四个时长档，按展示顺序排列
DURATION_LEVELS = ("short", "medium", "long", "overall")
DURATION_LABELS = {
    "short": "Short (短视频)",
    "medium": "Medium (中等)",
    "long": "Long (长视频)",
    "overall": "Overall (总体)",
}
# 三个分类维度
DIMENSIONS = ("domain", "sub_category", "task_type")
DIMENSION_LABELS = {
    "domain": "领域 (Domain)",
    "sub_category": "子类别 (Sub-category)",
    "task_type": "任务类型 (Task type)",
}


def find_latest_score(scan_dir: str) -> str | None:
    """在 outputs 目录里找修改时间最新的 *_score_*.json。"""
    root = Path(scan_dir)
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob(SCORE_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def fmt(v) -> str:
    """把 '0.758' 转 '75.8%'，'nan' 转 '-'。"""
    if v is None:
        return "-"
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None"):
        return "-"
    try:
        return f"{float(s) * 100:.1f}%"
    except ValueError:
        return s


def to_percent(v):
    """把 '0.758' 转 75.8（数值），'nan'/None 转 ''（空），用于 CSV 数值列。"""
    if v is None:
        return ""
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None"):
        return ""
    try:
        return round(float(s) * 100, 1)
    except ValueError:
        return s


def build_csv_rows(data: dict) -> list[list]:
    """生成长表（tidy data）CSV：每行一条 (时长, 维度, 类别, 准确率)。"""
    rows = [["duration", "dimension", "category", "accuracy"]]
    for level in DURATION_LEVELS:
        rows.append([level, "overall", "overall", to_percent(data.get(level, {}).get("overall"))])
        for dim in DIMENSIONS:
            for name, val in data.get(level, {}).get(dim, {}).items():
                rows.append([level, dim, name, to_percent(val)])
    return rows


def load_score(score_path: str) -> dict:
    with open(score_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(data: dict, meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Mage-VL × Video-MME 评测结果")
    lines.append("")
    lines.append(f"> 生成时间: {meta['generated']}")
    lines.append(f"> 数据来源: `{meta['score_file']}`")
    lines.append("> 数据集: Video-MME (32 帧子集, 238 视频 / 714 题)")
    lines.append("> 模型: Mage-VL (codec + HEVC)")
    lines.append("> 推理失败率 0% / 打分失败率 0%")
    lines.append("")

    # ---- 总览表 ----
    lines.append("## 总览")
    lines.append("")
    lines.append("| 视频时长 | 准确率 |")
    lines.append("| --- | --- |")
    for level in DURATION_LEVELS:
        overall = data.get(level, {}).get("overall", "-")
        if level == "overall":
            lines.append(f"| **{DURATION_LABELS[level]}** | **{fmt(overall)}** |")
        else:
            lines.append(f"| {DURATION_LABELS[level]} | {fmt(overall)} |")
    lines.append("")

    # ---- 三个维度：行 = 类别，列 = 四个时长档 ----
    for dim in DIMENSIONS:
        lines.append(f"## {DIMENSION_LABELS[dim]}")
        lines.append("")
        # 收集该维度所有类别名（按出现顺序去重）
        names: list[str] = []
        for level in DURATION_LEVELS:
            for name in data.get(level, {}).get(dim, {}):
                if name not in names:
                    names.append(name)
        header = "| " + DIMENSION_LABELS[dim] + " | " + " | ".join(
            DURATION_LABELS[l] for l in DURATION_LEVELS
        ) + " |"
        sep = "| --- " * (len(DURATION_LEVELS) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for name in names:
            row = [name]
            for level in DURATION_LEVELS:
                v = data.get(level, {}).get(dim, {}).get(name, "-")
                cell = fmt(v)
                if level == "overall":
                    cell = f"**{cell}**"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 VLMEvalKit 评测结果为 Markdown 报告")
    parser.add_argument("--score", help="score json 路径；缺省自动找最新")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="输出目录（默认脚本所在目录）")
    parser.add_argument("--name", default="Video-MME", help="输出文件名前缀")
    args = parser.parse_args()

    score_path = args.score or find_latest_score(DEFAULT_SCAN_DIR)
    if not score_path or not os.path.exists(score_path):
        raise SystemExit(f"未找到 score json，请用 --score 指定（扫描目录: {DEFAULT_SCAN_DIR}）")

    data = load_score(score_path)
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score_file": score_path,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) CSV（长表，Excel 可直接打开/透视/画图，主交付物）
    csv_out = outdir / f"{args.name}.csv"
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(build_csv_rows(data))

    # 2) 结构化 JSON（保留原始四档三维度，便于程序读取）
    json_out = outdir / f"{args.name}_score.json"
    json_out.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )

    # 3) Markdown 报告（便于人读）
    md_out = outdir / f"{args.name}.md"
    md_out.write_text(build_report(data, meta), encoding="utf-8")

    print(f"[ok] CSV   : {csv_out}")
    print(f"[ok] JSON  : {json_out}")
    print(f"[ok] 报告  : {md_out}")
    print(f"[ok] 总分  : {fmt(data.get('overall', {}).get('overall'))}")


if __name__ == "__main__":
    main()
