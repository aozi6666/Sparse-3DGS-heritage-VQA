# feat/sparse-vcar-seg：已完成任务归档（2026-11-20）

本页是完成状态与复现入口索引；不把计划、演示或小样本结果写成正式论文结论。归档快照见本目录source_manifest.json。代码提交前做过只读核对；本次没有启动GPU训练。

## 1. VGGT-Object：深度头全参数＋后六层Global Attention LoRA

默认DPT-only全参数适配保留；A16增加global_blocks18–23的qkv/proj rank16/alpha32/dropout0.05 LoRA，原Aggregator/DINO/Camera权重冻结。DPT32,654,562＋LoRA589,824＝33,244,386可训练参数。
实际A16数据是10件Smithsonian、240张渲染图，7/2/1物体划分；Omni和CO3D未参与。原始Omni images仍有零字节占位，不能当已完成上传。
20步、392×392、4视角、累积2，在RTX4090D上通过反传/冻结/保存重载检查。allocated峰值6.9429GiB。val尺度对齐AbsRel 0.02212352→0.02049880；仅2个验证物体，不能证明LoRA优于等预算DPT-only。旧400步基线保留。
入口：VGGT-Object/run.sh、AGGREGATOR_LORA_SMOKE.md。Blender Depth pass被代码标为相机Z，但没有显式ray-distance→Z换算，深度几何契约仍需独立核验。

## 2. 3DGS / DropGaussian 距离梯度实验

隔离入口3DGS/GaussianObject/tools/gradient_depth_drop/start.sh。真实屏幕二维中心梯度norm，用训练相机loss反传，不用测试监督。3-view为原9-view相机首/中/末子集，seed0、10000步；固定每相机共同初始深度范围，10个0.1箱。
v3/horns、v3/fortress已完成两方法及绘图；瞬时超阈使用>，实际累计增密条件使用>=阈值0.0002。空箱NaN，有样本且真零梯度保留0。CSV保留全范围，图仅horns0–0.7、fortress0–0.3。
平均梯度更大不等于质量更好；各箱的实际升降及阈值比例见快照interpretation.md/bin_comparison.csv。旧9-view没有新增阈值计数，不伪造补算；补充9-view新协议未见完成证据，不标完成。
最终PNG/PDF和CSV留在服务器output/gradient_depth_drop_v1/v3，并拷贝轻量证据到本归档。

## 3. 已完成的重建可视化与工程流程

完成trex全背景流程、4/6/9视角基础与VGGT+Drop对照、horns补点对照、LLFF horns/fortress阶段图、Mip360阶段图和前景图。各完成标志/指标/配置按真实文件归档；训练阶段图不冒充不同方法，训练视角展示指标不冒充未见视角性能。
保留最终图、GT、metrics、模型与输入；本次仅清理已完成repair的调试图/TensorBoard和冗长日志，未删除LOO训练输入或任何数据集/权重。Visual Hull交互掩膜相关准备不视为18组消融已完成。

## 4. Mage-VL字幕与语义选帧

Frames连续字幕2048：实际来源时间命中cue，补前后各最多一句（间隔≤2秒），合并并标时间，保留锚点、预算不足均匀删整邻句，缺字幕精确回退。未修改ViT，Qwen3是Mage内置语言骨干，没有ASR/Agent/LoRA参与该字幕收益。
原225题：140→163正确，62.22%→72.44%，+10.22pp；相对原对齐154题多9题，额外差异区间跨0。714为历史开发/验证的本地覆盖统计，非官方完整或新独立测试。
Codec全来源字幕：原225题131→158，+12.00pp；不是与Frames等文本预算胜负证明。语义时间选帧候选30题16→18，区间跨0，不能说稳定有效；字幕与选帧联合未验证。
详见docs/REPORT_mage_subtitle_method_results_20261119.md及本归档原始summary。

## 5. VCAR和第三方依赖

VCAR作为上游子模块登记，保持原远程和提交，不上传SAM权重或嵌套环境。其gaussiansplatting是本地另行取得的依赖，未混入本仓库源码；从已有本地源码记录版本，安装见VCAR/Install.md。

## 6. 路径整理、清理和Git范围

当前主目录VLM/，VQA是指向VLM的兼容符号链接，保留历史绝对/相对路径访问，零复制。字幕归档中冻结源文件与哈希依赖不做批量路径替换。
权重、数据集、虚拟环境、重建模型、EXR、视频、逐题大缓存留服务器且由.gitignore排除。VLM/eval_results完整原始记录留本机，Git仅上传本归档选定的最终数值/图表和必要历史文档。旧历史commit中的文件不会因.gitignore自动从Git历史消失，本次不擅自重写主分支历史。
完整旧日志压缩存服务器.git/maintenance_20261120/logs；Git保存末尾与SHA清单。清理明细cleanup_manifest.json。删除调试图可由已有训练脚本重建，不删除最终论文图。

## 7. 复现与下一步

首先按各README准备环境和已有本地数据；Git克隆不包含权重/数据。Git本次只做CPU语法/单元检查，未重新验证所有历史GPU实验。下一步是等预算F0/A16对照、独立test、新文旅评测，以及确切的深度几何语义验证。

## 8. 本次整理验证

VGGT-Object CPU 34项通过；梯度分箱5项通过，3-view原图/相机/权重存在性预检查通过。未启动GPU。最终提交使用scripts/check_staged_assets.py检查暂存blob大小、权重/视频/数据扩展和显式凭据模式。子模块VCAR的.git已正规吸收到主仓库.git/modules/VCAR；原上游来源与提交保留。原VGGT示例视频只从Git索引移除，服务器文件未删。

补充验收：字幕practical流程163项、full-scene接口6项也通过；合计208项CPU单元测试（34＋5＋163＋6），另有3-view数据/相机预检查。API相关测试使用测试替身，不作为新真实推理实验。保留少量冻结源文件尾空白与CSV原CRLF格式；不为消除diff --check格式提示而破坏来源哈希。
