# main整理与分割支线退出（2026-11-20）

用户确认：仅保留VLM；更新当前入口；旧冻结记录只读留档。取消VCAR分割。

## VGGT-Object保全

DPT全参数、Aggregator后6层rank16 LoRA、损失、数据桥接、优化器/调度、保存加载、测试和全部配置均保留。关键代码及原始VGGT/DPT/LoRA权重在整理前后用SHA256核对；权重继续只在服务器，不上传Git。唯一计划内VGGT变更是CLI允许在main执行CPU检查，训练数学逻辑不变。

## 只使用VLM

原VQA是VLM软链接，没有第二份数据，已删除链接。当前源码、shell入口和文档改用VLM，仓库名Sparse-3DGS-heritage-VQA及数据集CiQi-VQA名称不改。

VLM/eval_results中的历史逐题记录、cue/相机选择、冻结协议和锁定哈希不改；docs/archive中的旧快照也保留原样。它们含历史VQA路径属于追溯资料，不是当前可直接执行的新入口。新版源码与旧锁定源哈希不再相同，因此不能在旧结果目录强行init/run或绕过身份检查。将来新实验必须新输出目录、新签名、重新核对输入；本次未宣称所有冻结历史命令可原样重跑。

## VCAR取消

移除VCAR子模块及本地仓库、SAM权重、Seg Python3.12专用虚拟环境和.uv-cache-seg专用缓存；不删除3DGS/DropGaussian_release，不删除3DGS/vggt，不删除.venv-ddgs-vggt或Mage依赖。

Project是独立视频解说演示工程，不属于VCAR，处理以用户单独确认记录为准。其他未能确认用途的环境、素材与历史仓库不批量删除。

## 合并与删除顺序

完成核验与CPU测试后，将本分支整理提交变基至最新origin/main。切到main并fast-forward包含全部整理提交，先推送并核对远端main。确认关键代码在main可读取且远端提交一致后，才删除远程和本地feat/sparse-vcar-seg。main不会回退到旧版本，也不丢弃未提交改动。

## 本次核验结果

- VGGT保全清单37项逐项核对：除CLI允许main的两行分支守卫外，其余36项字节一致，涵盖原始模型、DPT/LoRA权重及实际数据划分；完整SHA见archive/main_migration_20261120/vggt_preservation.json。
- 低内存CPU实例中，VGGT32项分项通过；完整套件中的原始模型CPU加载和真实DPT反传2项因资源限制被系统终止，本次不宣称通过。这两项的实现和权重哈希均未变，之前归档有34项通过记录。
- 字幕practical 163项通过；AIR排除真实原生processor重型套件后336项中334项在3DGS环境通过，另外2项需要decord，已在保留的Mage环境单独通过。未绕过或伪造测试；不把缺依赖当路径迁移错误。原生processor套件因2GiB资源限制未在本次重跑完成。
- 所有当前VGGT/VLM/Project已跟踪Python源码完成AST语法检查，当前Python入口无残留VQA目录路径。
- 用户确认Project及其环境、缓存、素材全部保留；ddgs、.venv-ddgs-vggt及/root/autodl-tmp/Mage_ViT/mage均保留。
- VCAR、Seg、.uv-cache-seg和本地VCAR子模块Git存储已删除；3DGS/DropGaussian_release子模块保留。未清理原始数据、最终渲染、3DGS权重或VGGT实验权重。

最终补充：在实际Mage专用环境中，AIR轻量CPU套件（排除真实原生processor重型文件）336项整套通过，两个媒体检查不再缺decord。目录替换仅针对路径，VQABench/通用VQA任务类型及仓库名称保持原语义，不批量改成VLM。Project FastAPI环境已通过导入检查。
