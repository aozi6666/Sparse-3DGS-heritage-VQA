# 工作 B · 视频问答与评测

本目录为系统仓中的 **VQA / 评测** 工作区（GitHub：`Sparse-3DGS-heritage-VQA`）。

## 目录

| 路径 | 内容 |
|------|------|
| `mage_vl/` | 基于 [microsoft/Mage](https://github.com/microsoft/Mage) 的 `mage_vl`；另含 streaming 相关增量 |
| `VLMEvalKit/` | 基于 [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 的 **完整可运行代码树（含本仓改动）** |
| `patches/vlmevalkit/` | 相对上游的 diff（便于以后同步，不是唯一运行方式） |
| `eval_results/` | **实验指标**（Video-MME 等），请保留 |
| `scripts/` | 复现辅助脚本 |

**不包含**：`mage_flow`（文生图，留在仓外）、`mage_vl/data`、`mage_vl/models`、venv、ffmpeg 大包。

## 上游与许可

- Mage-VL：遵循 Mage 仓库中 Mage-VL 许可（见上游 README / LICENSE）  
- VLMEvalKit：见 `VLMEvalKit/LICENSE`；来源说明见 `VLMEvalKit/UPSTREAM.md`

## 权重与环境（不进 Git）

- Mage-VL 权重：放到 `mage_vl/models/`（已 gitignore）或任意本地路径并改配置  
- 评测环境：可用仓外 `../VLMEvalKit/vlmeval_env`，或按 `envs/requirements-vlmevalkit.txt` 重建，并把 editable 指到 **`VQA/VLMEvalKit`**

## 查看已有结果

```bash
ls eval_results/
cat eval_results/Video-MME.md
# 或 Video-MME_score.json
```
