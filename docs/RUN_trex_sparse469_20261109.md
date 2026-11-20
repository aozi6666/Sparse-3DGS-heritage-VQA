# trex 4 / 6 / 9 视角：冻结 VGGT + DropGaussian 粗训练

日期：2026-11-09。用户要求仅提供可执行终端命令，不代为启动训练。已完成 CPU 数据划分、训练视角三角化和命令检查；尚未运行本实验的 VGGT 推理、训练或评测，因此当前没有新指标或渲染结果。

## 实验范围

- 数据：`3DGS/GaussianObject/data/LLFF/trex_fullscene`，沿用已去畸变照片和 PINHOLE 标定；保持博物馆真实背景，不使用前景抠图或白背景。
- 只做：预训练 VGGT 推理与点云补充 → 3DGS 粗训练并启用 DropGaussian → 固定测试集指标与渲染。
- 不做：VGGT 微调、depth-head 适配、LOO、LoRA、ControlNet 或 repair。
- VGGT 权重固定为 `3DGS/vggt/models/model.pt`；`eval()`、`requires_grad_(False)`、`inference_mode()`；记录权重 SHA-256。
- VGGT 使用现有流程的单帧深度接口，各选中训练视角独立推理，随后做尺度对齐和跨训练视角一致性过滤；不能称为 VGGT 联合多帧推理实验。
- VGGT 用于初始化增强，本入口显式设置 `--mono_depth_weight 0`，避免额外引入单目深度训练损失。
- DropGaussian：最大概率 0.2、线性调度，训练完成后正常渲染，不在评测时随机丢弃高斯。
- 统一 10,000 步，分辨率 1008×756（原图的 1/4），位置学习率调度 10,000 步，增密至 6,000 步；高斯增密停止阈值 500,000（原训练器的软阈值，一轮增密可能越过它）。
- 最终固定步数模型用于比较，不根据测试指标挑选训练步数。

这是保留背景的稀疏视角工程验证，不是发表版 SynGS 的 Visual Hull 协议，也不是 VGGT/Drop 的独立消融。三组只改变训练视角数量，不能据此单独判断哪个模块有效或失效。

## 划分与初始化，避免沿用55视角场景几何

以下索引从0开始，按相机图像名（stem）排序：

| 组 | 训练索引 |
|---|---|
| 4 | 1, 21, 34, 54 |
| 6 | 1, 14, 21, 34, 41, 54 |
| 9 | 1, 7, 14, 21, 28, 34, 41, 47, 54 |

三组共享测试索引：**0, 8, 16, 24, 32, 40, 48**。其余图像不参加训练或评测。训练集合满足 4 ⊂ 6 ⊂ 9，与测试集合不相交。

不复用55视角的 `scene_sfm_vggt.ply`、深度文件、SfM三维坐标或点云颜色。每组从自身训练相机的二维观测重新三角化（至少2个观测、最大重投影误差4像素、最小夹角1°），颜色也只采样训练图像。VGGT 尺度锚点、置信度筛选、跨视角深度一致性均只使用该组训练视角。

**保留的已知条件**：相机标定/位姿与特征轨迹身份来自已有55视角 COLMAP 重建，而非仅凭4/6/9张图重新完成特征匹配与位姿估计。因此应描述为“给定相机标定与特征轨迹条件下”的稀疏输入重建，不能宣称整个 SfM 过程没有使用全量图像。测试 RGB 不用于高斯参数优化；训练器会加载测试图像供后续评估，但本入口关闭训练过程中的测试评测，仅在最终模型完成后统一评测。

CPU 实测三角化结果：

| 训练视角数 | 训练观测重算的初始三维点数 | 每帧尺度锚点候选数 |
|---:|---:|---|
| 4 | 4435 | 2105, 3619, 2832, 3306 |
| 6 | 6416 | 2844, 2801, 4026, 3205, 3933, 3513 |
| 9 | 9447 | 3048, 3396, 3004, 4839, 4728, 3639, 4395, 4594, 4267 |

这些不是训练质量指标。最终补点数必须看各组 `scene_init_report.json`。若 VGGT 没有留下任何可靠补点，程序会报错，不会静默退化后仍标为“VGGT增强”。

## 在远程终端启动

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
mkdir -p 3DGS/GaussianObject/output
nohup bash 3DGS/GaussianObject/sh/full_scene/run_trex_sparse469.sh > 3DGS/GaussianObject/output/trex_sparse469_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/GaussianObject/output/trex_sparse469_v1.console.log
```

`Ctrl+C` 退出 tail 不会终止后台训练。三组顺序运行，不并发占用 GPU。SSH 断开不会因终端挂断而结束 nohup 后台任务；实例关机等仍会终止它。运行日志末尾出现 `COMPLETE:` 且结果根目录存在 `COMPLETE.json`，才表示三组全部完成。

同一命令可重跑：已完整结束的阶段跳过，VGGT 深度按帧复用；中断的粗训练从该组起点重新训练，因为没有保存优化器快照。并发启动相同输出目录会被文件锁阻止。参数、代码、使用的图像或标定发生改变时，会要求使用新的 `--run-tag`，避免混用实验。

改变训练步数的例子（新实验必须用新目录名）：

```bash
bash /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/sh/full_scene/run_trex_sparse469.sh --iterations 15000 --run-tag trex_sparse469_15k
```

只读计划：

```bash
python /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/tools/run_sparse_469.py --plan
```

## 输出和省盘策略

结果根目录：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/output/trex_sparse469_v1/`

```text
trex_sparse469_v1/
├── run_config.json                 # 参数、数据/代码指纹、确切划分
├── vggt_weight.json                # 预训练权重 SHA-256
├── v4/                             # v6、v9 结构相同
│   ├── cfg_args
│   ├── logs/{depth_init,train,evaluate}.log
│   ├── point_cloud/iteration_10000/point_cloud.ply
│   └── test/ours_10000/
│       ├── results.json            # 7张固定测试视角的算术平均指标
│       ├── camera_names.json       # PNG序号到相机图像名的映射
│       ├── renders/00000.png ... 00006.png
│       └── gt/00000.png ... 00006.png
├── v6/
├── v9/
├── metrics_comparison.csv          # 三组PSNR/SSIM/LPIPS
├── metrics_comparison.md
├── metrics_comparison.png          # 3个指标随视角数的变化曲线
├── render_comparison.png           # 各测试视角：GT / 4视角 / 6视角 / 9视角
└── COMPLETE.json                   # 三组成功完成的标记
```

汇总文件会在每完成一组后更新；只有 `COMPLETE.json` 出现后才是完整三组。PSNR、SSIM 越大越好，LPIPS（原始尺度，不乘100）越小越好。LPIPS 用 VGG，按 full_scene 协议将 [0,1] 图像 normalize 到 [-1,1]；指标通过现有 `render.py` 对浮点渲染计算，PNG 为保存时裁剪后的图。与以前单个训练视角的分数不属于相同评测协议，不能直接据其高低判断改进。

额外数据在 `3DGS/GaussianObject/data/LLFF/trex_sparse469_v1/{v4,v6,v9}/`，记录划分、训练几何、VGGT 深度/置信度、补点报告。图像文件夹是指向已有去畸变图像的符号链接，避免复制三份照片。仅保存最终高斯模型，不保存LOO、修复模型、优化器快照或视频；完成粗训后删除本次输出内可再生成的重复 `input.ply`，不删除旧实验或源文件。磁盘增量取决于实际高斯数量，应查看 `du -sh`，不以预估值替代实测。

查看汇总：

```bash
cat /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/output/trex_sparse469_v1/metrics_comparison.md
```

仅重新汇总已存在结果（不训练）：

```bash
python /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/tools/run_sparse_469.py --summarize-only
```

## 已完成检查及限制

- 两个 Python 文件编译检查、Bash 语法检查、现有 train_gs/render CLI参数检查通过。
- CPU实际完成4/6/9划分和各自三角化；重复 `--prepare-only` 验证通过，不重复生成已有几何。
- 检查集合嵌套、测试隔离，以及合成多相机的 DLT 坐标恢复和错误对应拒绝。
- 已确认 CUDA 可用、matplotlib 可用；部署时磁盘可用约20GB。
- 本次未启动 VGGT GPU 推理、训练、渲染或指标计算；尚不能保证质量优于55视角旧实验或VGGT能在该场景补充多少可靠点。
- 如果后续需要证明 VGGT / Drop 的独立收益，需要在同一划分下另做模块消融；这次三组视角数量对比不承担该结论。

## 本次新增入口

- `3DGS/GaussianObject/tools/run_sparse_469.py`
- `3DGS/GaussianObject/tools/sparse_scene_init.py`
- `3DGS/GaussianObject/sh/full_scene/run_trex_sparse469.sh`

现有55视角流程、目标物体去背景流程及其代码均未修改。

最终核对：实际 `readColmapSceneInfo` 对三个目录均成功加载，训练相机分别4/6/9、测试相机均7，图像尺寸1008×756；初始点云数与各组CPU报告一致。验证日志与脚本SHA归档于 `docs/reproduction/trex_sparse469_20261109/`。
