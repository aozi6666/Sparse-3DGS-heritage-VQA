# ocrbench_v2_image_textseat · 图像 QueryIP（OCR 座位）

> 状态：**代码已就绪 · Acc 待你跑**（Agent 不代跑）  
> 命令：[`COMMANDS.md` §10](../../../Explore-Then-Select/COMMANDS.md)

## 协议

| 项 | 值 |
|----|-----|
| 数据 | OCRBench_v2（先 `--n 200` 冒烟，过闸再全量 `--n 0`） |
| 预算 | `keep_ratio=0.75` · 同图同 B（`full_image` 除外，为满图参考） |
| modes | `full_image` · `random_tiles` · `queryip_image_textseat` · `queryip_image_textseat_shuf` |
| 选区 | 图像 grid tile + EasyOCR 框映射座位 + 纹理填满；**禁止 CLIP** |
| 续跑 | `--resume`；jsonl 按 `index` |
| 产物 | 仅本目录 jsonl/summary（图仍在 `$LMUData`） |

## 闸门（相对同 B）

1. `Acc(textseat) ≥ Acc(random_tiles)`
2. `Acc(textseat) − Acc(textseat_shuf) ≥ +1.0 pp`（至少为正）
3. 相对 `full_image`：允许略降，但勿崩盘（建议 Δ ≥ −3 pp，否则报告并停包装）

## 结果表（跑完填）

| mode | Acc / 指标 | n |
|------|------------|---|
| full_image | — | — |
| random_tiles | — | — |
| queryip_image_textseat | — | — |
| queryip_image_textseat_shuf | — | — |
