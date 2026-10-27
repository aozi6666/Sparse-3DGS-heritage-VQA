# wide_textseat_n200 · **负协议归档（已停）**

> **不要按 `max_pixels=300000` 重跑。** 视频同 K 若再做，锁 **`max_pixels=150000`**。  
> 文字证据主战场已迁到 **OCRBench_v2 图像 textseat**（见 [`../ocrbench_v2_image_textseat/`](../ocrbench_v2_image_textseat/)）。

## 为何失败

| 现象 | 含义 |
|------|------|
| `n_wide ≈ 3.4万–4.2万` | 中宽 `300000` 把池撑爆；`B=0.75N` 仍极大 |
| 早期 Acc：proxy ≤ random | 高 keep 区（对照旧 ksweep@0.9）proxy 无优势 |
| `need_text=0`，三路 pred 相同 | VMME 上 textseat **未触发**，测不到问题分支 |

部分 jsonl / summary / nohup 日志可留作负结果证据；**不作贡献**。

## 历史协议（仅存档）

Video-MME short-200 · `max_pixels=300000` · `keep_ratio=0.75` · modes=`random,queryip,queryip_textseat,queryip_textseat_shuf`
