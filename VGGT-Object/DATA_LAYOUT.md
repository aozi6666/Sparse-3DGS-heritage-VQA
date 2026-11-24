# 当前上传数据的目录整理与接入（2026-11-19）

## 1. 真实位置

```text
VGGT-Object/
  configs/data_sources.yaml        # 当前服务器的实际来源映射
  data/
    omni3d/                       # 20件真实物体，COLMAP + ZoeDepth伪深度
    VGGT_finetune_v1/              # 与omni3d同级，保留上传包原样
      smithsonian/                # 10件文物的RGB、浮点Z深度、mask、c2w
      co3dv2/                     # 10条序列，部分带估计深度
      manifests/                  # 原包40对象划分，不改写
      scripts/                    # 原包附带脚本，不自动执行
    raw/                          # 旧版官方原始数据转换入口，当前这套包无需搬入
    processed/bundle_index_<id>/   # 新生成的统一相机/路径元数据，只有小型JSON
    manifests/
      objects.json                # 通过接入检查后生成的训练清单
      source_catalog.json         # 全部40对象的服务器路径和监督用途
  reports/upload_status.json      # 上传情况快照
  reports/bundle_import.json       # 接入结果与统计
```

你提到的 `data/omni3d/VGGT_finetune_v1` 当前不存在；实际是 `data/VGGT_finetune_v1`。不要在上传途中改名或移动目录，也不用重复复制一份到raw。

## 2. 上传状态

运行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
bash VGGT-Object/run.sh data-status
```

这只是只读快照：文件数、空文件数、已到达字节数、最近内容变化时间。ctime用于判断服务器是否有写入，不能依赖复制时保留的原mtime。没有源文件校验和时，不声称已完整传输或给百分比。

本次初查 omni3d 有32,112文件，其中31,934个为空，实际文件内容约9.1 MiB，最近约一小时无写入；这是未完成的上传。VGGT_finetune_v1则仍在增长。数据改变后以新快照为准。

上传端需要补传/覆盖空文件；不能使用“已有文件全部跳过”来续传，否则占位空文件会永远保留。未确认本地源路径和传输工具前，这里不自动启动第二路复制，也不删除远程占位文件。

## 3. 上传完成后的接入

当前这套是整理过的数据包，不再走 `prepare --dataset omniobject3d` 的官方Blender目录解析。

第一轮先接入 Smithsonian 的渲染Z深度，自动保留原包8/1/1对象划分；**不会把omni3d的ZoeDepth当GT**。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
bash VGGT-Object/run.sh data-status
bash VGGT-Object/run.sh import-bundle --datasets smithsonian
bash VGGT-Object/run.sh validate --limit 100
```

接入只生成JSON索引，直接引用已上传RGB/深度/mask，**不复制媒体、不再生成一份深度缓存、不下载、不跑GPU**。所有选中对象的引用文件必须存在且非零；缺失/空文件会阻止发布训练objects.json。validate进一步实际解码、检查几何并输出少量质量图。

原包manifest里omni3d是Mac绝对路径，接入工具按对象ID映射到服务器data/omni3d；不去修改上传中的原始manifest。train/val/test保持原包划分，检测重复与跨集合泄漏。不要再运行原包中可能写死Mac路径的build_manifests脚本。

## 4. 三种监督用途

| 数据 | 用途 | 当前是否进入默认GT训练清单 |
|---|---|---|
| Smithsonian网格渲染深度 | 文物深度适配和独立对象验证 | 是，引用原始float32 NPY，c2w转w2c |
| omni3d ZoeDepth | 后续伪监督探索、真实图像与重建泛化 | 否，不能当GT绝对深度 |
| CO3Dv2有深度的5条序列 | 质量确认后的低权重真实域适配 | 默认否，明确review后才启用 |
| CO3Dv2无深度的5条序列 | RGB/相机相关评测，不能参与当前有监督深度训练 | 否 |

CO3D原包某些清单把is_ground_truth写成true，但它仍是重建估计深度，不会因此改成传感器GT。桥接代码强制记录为estimated。依据原包明确depth_note读取float16位模式及scale_adjustment；声明为ray_distance时，加载时按内参转换为Z，不另落地缓存。坐标/尺度仍需真实质量检查。

质量确认后可显式接入CO3D：

```bash
bash VGGT-Object/run.sh import-bundle --datasets smithsonian co3dv2 --review-co3d-depth
bash VGGT-Object/run.sh validate --limit 100
```

此标志表示你确认几何质量，不会自动证明深度正确。无深度序列记录为rgb_only并排除。当前实现不把RGB-only数据偷偷塞进有监督loss，也不声称已完成无深度自监督训练。

## 5. 训练报错原因与后续命令

你粘贴的FileNotFoundError来自objects.json还没生成，发生在模型训练前。CPU check通过只证明代码/接口检查通过，不代表数据已经上传或准备好；tail -f会一直等待，Ctrl+C只退出日志查看。

接入与质量检查通过、有GPU以后再执行：

```bash
nohup bash VGGT-Object/run.sh train --phase smoke --experiment E2 --run depthlora_smoke_v1 \
  > VGGT-Object/logs/depthlora_smoke_v1.log 2>&1 < /dev/null &
tail -f VGGT-Object/logs/depthlora_smoke_v1.log
```

如果该run目录已有半成品，换v2，或确认存在latest/trainer.pt后用--resume。数据没准备好时新入口给出明确步骤，不再只报一个底层文件路径错误。

第一轮只有10件Smithsonian文物，独立val/test各1件，是小规模先导实验，不能把单件结果写成广泛泛化结论。混合数据权重会按实际可用训练集显示，不冒充原方案上千对象规模。

## 6. 首轮搭建验收记录（后续训练以REVIEW_DESIGN_V3.md为准）

- 代码与桥接相关24项CPU测试通过，未运行GPU。
- 已成功为10件Smithsonian建立零复制元数据索引，并生成data/manifests/objects.json，保持8 train / 1 val / 1 test。
- 全部10件完成CPU实际解码与几何检查；每件4个抽样视角有效像素35,679–89,537，GT跨视角有效投影支持2,529–5,780像素，没有完全无投影支持的对象。
- reports/data_quality包含10张RGB/GT/mask质量图与quality.json，合计约0.84 MiB。这不是训练结果，也不是微调效果验证。
- data/manifests/source_catalog.json记录40个对象的服务器路径与监督用途。omni3d及CO3D没有混入本次10对象GT训练清单。
- 上传原文件、原清单未移动、未改写；没有复制RGB/深度/GLB；索引以绝对路径引用原文件。未启动训练。
- 初始objects.json缺失错误已解决；之后训练会校验当前manifest是否与CPU质量检查对应，防止跳过数据准备。
