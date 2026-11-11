# Mage Video Studio

工程：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/Project`  
当前分支：`feat/project-video-deme`（按用户指定，保留 deme 拼写）。

React＋TypeScript＋FastAPI，上传录制视频后按8秒分段生成短解说，SSE显示文字，浏览器SpeechSynthesis按句播报，也可暂停提问。视频理解使用AutoDL已有本地Mage-VL；第一版采用Frames＋SDPA，Codec和事件门控留待后续。没有ASR或外部LLM调用。

**当前默认是无GPU预览模式**：上传、播放、页面、音色选择和“试听”可用；模型相关按钮禁用。启动不会导入torch/transformers/cv2或加载权重。真实Mage生成、浏览器声音效果和整段讲解同步需要分别验收，接口替身测试不代表推理效果。

## 进入虚拟环境

在AutoDL终端：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source Project/.venv/bin/activate
python --version
```

退出用 `deactivate`。启动脚本会直接使用该环境，不要求提前激活。

## CPU预览

前端已构建，无需再安装依赖。AutoDL终端运行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
DEMO_ENABLE_MODEL=0 bash Project/bootstrap/serve.sh
```

保持该终端打开。在自己电脑另开终端，使用现有SSH配置：

```bash
ssh -N -L 7860:127.0.0.1:7860 autodl-thesis
```

浏览器打开 `http://127.0.0.1:7860`。没有配置此SSH别名的电脑用自己原来的SSH主机和端口参数。服务只监听服务器回环地址；默认一个worker，不使用reload。停止自己启动的服务按Ctrl+C。

**现在不要启用模型。** GPU恢复后再单独进行模型加载、短视频生成、显存和取消测试；届时显式设置 `DEMO_ENABLE_MODEL=1` 才会在首次生成请求时加载模型。无需填写外部API Key，也不下载模型。

## 检查与重建

轻量环境检查，替代之前会导入大型运行库而被Killed的脚本：

```bash
bash Project/bootstrap/setup_env.sh check
```

该检查仅核对Python和包元数据、依赖兼容性、模型文件头/索引/长度、Node和ffmpeg版本。不会执行处理器或模型；报告明确记录运行时与GPU未测试。

只在修改代码后按需执行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA/Project
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest backend/tests -q
export PATH="$PWD/.node/bin:$PATH"
cd frontend
NODE_OPTIONS=--max-old-space-size=256 node --test tests/speech.test.cjs
NODE_OPTIONS=--max-old-space-size=256 npm run build
```

当前无需重跑backend/frontend安装，也无需升级npm。后端已有本地torch复用逻辑：uv sync跳过torch后，再恢复本地硬链接，最后检查依赖，避免被uv再次卸掉。

## 数据、恢复与资源

- 原视频：`Project/storage/uploads/`，UUID命名，不保留逐帧JPEG，不另存全部片段视频。
- 任务及解说：`Project/storage/demo.sqlite3`。服务重启后未完成任务变为暂停；网页保留最近的视频/任务ID，可点击继续。浏览器存储被清除时，后端数据仍在。
- 完成片段不重复生成；中断片段重新生成。视频、模型配置/源码指纹或流水线参数不一致时拒绝混用旧结果。
- 缓冲：当前片段就绪后播放，预取最多下一段；模型慢时暂停等待。它是录制视频的分段处理，不承诺实时速度。
- 播报按完整句子排队，等片段完成后入队；每次只读一句，等待队列最多12句。拖动、换视频、新提问会清除旧播报。浏览器可能使用本机或网络音色，可选择列表中标为“本机”的音色。
- 提问读取当前画面及此前最多12秒，暂停自动讲解调度；已开始的片段先结束，再串行回答。客户端断开通过停止标记请求结束生成，不强杀进程。
- FastAPI接收大文件时使用的临时上传缓冲放在`Project/storage/tmp/`，请求结束由框架释放；写入最终上传文件期间存在短暂双份占用，不永久保存重复副本。
- 上传上限2GiB，保留至少5GiB空间；解析失败或中断上传已经接收的内容不自动删除。重复主动上传会产生独立原文件，当前不自动去重。
- 安装/检查日志：`storage/logs/`；检查报告与被替换的初始化脚本备份：`storage/audits/`。

## 模型和迁移边界

`Project/models/mage-vl`仍软链接到 `VLM/mage_vl/models`。权重结构核对通过，但没有全载荷checksum或GPU运行验收。ffmpeg/ffprobe通过硬链接复用已有工具。`Mage_ViT`只读参考，应用不import其源码。

**当前不可删除旧 `VLM/mage_vl`**，模型实体仍在那里。最终迁移需先排查其他实验引用、保存数据，再同盘移动权重并验收。历史字幕/评测/3DGS数据均未改动。

原始阶段规划见 [PLAN_20261110.md](docs/PLAN_20261110.md)，最新进展及验收边界见 [STATUS_20261111.md](docs/STATUS_20261111.md)。
