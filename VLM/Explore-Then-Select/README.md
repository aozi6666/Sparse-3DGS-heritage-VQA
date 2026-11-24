> 本分支已移除旧 mage_query_ip 包、运行脚本及 VLMEvalKit 注册。以下为历史记录，命令不再适用于本分支。完整旧实现保留于 main 的提交 2e30b06；历史评测结果原位保留。A.I.R. 将独立实现，只作用于 Frames 的选帧，Codec 与冻结字幕成果不变。

# Explore-Then-Select（本仓切片）

本目录**只保留** Mage Query-IP MVP：

- [`mage_query_ip/`](mage_query_ip/) — 问题类型重加权 + 固定预算选区
- [`scripts/smoke_soccer.sh`](scripts/smoke_soccer.sh) — soccer 冒烟
- [`COMMANDS.md`](COMMANDS.md) — 本地命令
- [`README_MAGE_QUERY_IP.md`](README_MAGE_QUERY_IP.md) — 说明

完整上游 Explore-Then-Select（LLaVA/Qwen `model/` `test/` 等）在仓外：

`/root/autodl-tmp/Explore-Then-Select`
