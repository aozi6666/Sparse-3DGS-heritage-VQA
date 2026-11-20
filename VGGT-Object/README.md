# VGGT-Object：冻结 Aggregator 的深度领域适配

当前默认方案：**整个 Aggregator 与 Camera/Point/Track 分支冻结，Depth Head 全参数微调**。Conv-LoRA保留为轻量对照，不再凭参数少就认定更合理。选择依据、损失与小规模实测见 [REVIEW_DESIGN_V3.md](REVIEW_DESIGN_V3.md)。

工程：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VGGT-Object`，分支`feat/sparse-vcar-seg`。复用原`.venv-ddgs-vggt`、`3DGS/vggt`、`3DGS/vggt/models/model.pt`；没有下载数据/模型，也没有改上游VGGT源码。

## 当前数据

- `data/omni3d/`：现有真实物体COLMAP/ZoeDepth数据，上传仍有大量零字节文件；不是GT监督集。
- `data/VGGT_finetune_v1/`：整理包，与omni3d同级；Smithsonian 10件文物有渲染Z深度，CO3D 10序列只有部分带估计深度。
- `data/manifests/objects.json`：保留原包8 train / 1 val / 1 test。
- `data/manifests/objects_review_v2.json`：当前研究清单，7 train / 2 val / 1 test；原测试文物不变，希腊雕像从train留作第二个val对象。原媒体/原清单未移动修改。
- 训练默认读取`configs/base.yaml`中声明的研究清单；原上传包通过零复制元数据桥接。看[DATA_LAYOUT.md](DATA_LAYOUT.md)了解路径与上传问题。
- 当前监督混合比例明确为Smithsonian 100%；不把缺失数据重新归一化后还称为三数据集混合。严格混合检查开启。

## 当前配置

| 项目 | 默认/预算 |
|---|---|
| 模式 | F0：全Depth Head；Aggregator冻结 |
| 数据 | 7件训练，2件独立验证，1件保留测试 |
| 输入 | 392×392，4视角，1 scene，梯度累积2 |
| 学习率 | AdamW 2e-6，weight decay 0.01，warmup 5%，cosine到1e-6 |
| 损失 | L1回归 + 0.05置信度 + 0.1多尺度log-depth梯度 |
| 几何一致性 | 默认0，F1对照启用最高0.05；GT固定可见对应 |
| 训练 | smoke20步；review120；pilot最多400；后续Stage A最多800 |
| 验证 | 每50步；每件验证对象6组固定输入，先对象内平均再对象间平均 |
| 早停 | 4次验证无0.001有效改善；best保存最低合格误差，与早停阈值分开 |
| 选模型 | 共享场景尺度对齐AbsRel；canonical未对齐误差超过原版1.25倍不选 |
| 增强 | pilot开启轻量亮度/对比度/噪声；不改变图像几何、K、E、深度与mask |

只用两个验证对象仍是先导研究；重复6组输入不会变成12个独立对象。

## 命令

以下从仓库根目录执行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
CUDA_VISIBLE_DEVICES='' bash VGGT-Object/run.sh check
bash VGGT-Object/run.sh data-status
CUDA_VISIBLE_DEVICES='' bash VGGT-Object/run.sh validate --limit 100
```

当前已启动的试验不要重复创建。查看日志：

```bash
tail -f VGGT-Object/logs/depth_full_pilot_v3.log
```

新开一轮（已存在目录会拒绝覆盖）：

```bash
nohup bash VGGT-Object/run.sh train --phase pilot --experiment F0 --run depth_full_pilot_v4 \
  > VGGT-Object/logs/depth_full_pilot_v4.log 2>&1 < /dev/null &
# 恢复同一轮，配置、清单、增强策略须一致：
bash VGGT-Object/run.sh train --phase pilot --experiment F0 --run depth_full_pilot_v4 --resume
```

E2是Conv-LoRA基础对照，E4是LoRA+MV；F0是全Depth Head基础方案，F1是全Depth Head+MV。E0只评测。LoRA默认r8/alpha16/dropout0.05/LR2e-5，注入projects和scratch中的卷积；新配置排除最后2输出通道的小卷积，避免适配器参数反而更多。旧检查点按各自model_info中配置加载。

## 数据接入边界

现在训练清单已生成并通过CPU解码/几何检查。后续添加新数据时，先`data-status`与`import-bundle`，再明确更新版本化研究清单、运行validate；不能把改过的数据清单直接用于旧run续训。

```bash
# 仍保留原包划分的基础接入；不自动改变当前冻结的研究清单。
bash VGGT-Object/run.sh import-bundle --datasets smithsonian
# CO3D须先人工/几何审查：
bash VGGT-Object/run.sh import-bundle --datasets smithsonian co3dv2 --review-co3d-depth
```

上传目录只读取，不移动原文件。Omni3d ZoeDepth不冒充GT。CO3D无深度序列不进当前有监督loss；有估计深度的序列也不自动充当可靠GT。Stage B要求确实存在已审查CO3D数据，缺少就报错，不冒充真实域适配。

## 评测

```bash
bash VGGT-Object/run.sh eval --split val --views 4 --resolution 392 --limit 20 \
  --output VGGT-Object/results/baseline_val_new
bash VGGT-Object/run.sh eval --split val --views 4 --resolution 392 --limit 20 \
  --adapter VGGT-Object/experiments/depth_full_pilot_v3/best \
  --output VGGT-Object/results/adapted_val_new
```

指标：canonical未预测尺度拟合的AbsRel/RMSE/RMSE Log/δ1–δ3、每场景共享median尺度对齐的同类指标、深度边界F1、固定GT可见对应下的相对Z一致性、相对位姿误差、AUC30、资源开销。canonical使用GT规范化，不是无尺度RGB恢复米制深度；不每帧单独对齐。Camera/密集特征冻结，其位姿不应因Depth Head训练而改变。

test不用于挑学习率、loss、步数或模型。最终模型选择完后才评测保留测试对象；只有1件，不能泛化为大数据集结论。

## 下游3DGS

```bash
bash VGGT-Object/run.sh export-points --split test --object-id OBJECT_ID --views 4 --resolution 392 \
  --adapter VGGT-Object/experiments/depth_full_pilot_v3/best \
  --output VGGT-Object/results/object_4_adapted_data
bash VGGT-Object/run.sh gs --views 4 --steps 10000 \
  --data VGGT-Object/results/object_4_adapted_data \
  --output VGGT-Object/results/object_4_adapted_gs
```

不传adapter是原版VGGT。4/6/9使用同一留出集合和按已知相机中心最远点采样的嵌套输入集合：4⊂6⊂9。深度尺度仅用输入相机中心基线估计，不拟合GT深度。该分支是已知相机的初始化对照，不能宣称未经标定的4张照片独立完成全流程。无repair、无Drop、非完整SynGS。默认不保存最终高斯模型，只输出点云/少量测试渲染/指标。

## 省盘与复现

原5GB模型从现有位置只读加载。全Depth Head约124.6 MiB，不保存整个VGGT；一轮best+latest及最新优化器约0.5 GiB。LoRA仍只需约2.4 MiB适配器。只保留best/latest，原子替换后清理本run已被替代的旧检查点，不删除数据或其他实验。

模型信息保存实验/清单哈希/模式/目标层/版本/步数/种子。恢复校验清单、协议、loss、增强等。训练记录逐项loss及低频逐项梯度范数、验证曲线、有效MV对应数、耗时和显存。best支持零步原始模型，训练没有改善时不伪造“最佳已微调”结果。

当前实际GPU是RTX 4090 D（约24GB），Torch 2.5.1+cu124的推理与训练已跑通。5090 D属于另一硬件条件，不能用此次验证替代其兼容性验证。

参考：[VGGT官方训练配置](https://github.com/facebookresearch/vggt/blob/main/training/config/default.yaml)、[DPT论文](https://arxiv.org/abs/2103.13413)、[LoRA原论文](https://arxiv.org/abs/2106.09685)。
