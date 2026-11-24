#!/usr/bin/env python3
"""Read completed experiments; no model inference or parameter selection here."""
import hashlib
import json
from pathlib import Path
import tarfile

P=Path(__file__).resolve().parents[1]
OUT=P/'reports/review_v3'
OUT.mkdir(parents=True,exist_ok=True)


def read(path):return json.loads(Path(path).read_text())


names=['review_lora_e2_v2','review_full_f0_v2','review_full_mv_f1_v3']
rows=[];reference=None
for name in names:
    root=P/'experiments'/name
    complete=read(root/'complete.json');baseline=read(root/'baseline_validation.json')
    latest=read(root/'validation_latest.json');params=read(root/'parameters.json')
    cases=[(r['id'],r['case'],r['indices']) for r in baseline['rows']]
    if reference is None:reference=cases
    assert reference==cases,'View protocols differ; cannot report matched comparison'
    rows.append(dict(run=name,steps=complete['steps'],objects=latest['objects'],cases=latest['cases'],
         aligned_AbsRel=latest['summary']['scene_scale_aligned']['AbsRel'],
         canonical_AbsRel=latest['summary']['canonical_unaligned']['AbsRel'],
         trainable=params['trainable'],peak_gpu_GiB=complete['peak_gpu_GiB'],
         duration_seconds=complete['duration_seconds']))
main=P/'experiments/depth_full_pilot_v3'
completion=read(main/'complete.json');best=read(main/'best/model_info.json')
baseline=read(main/'baseline_validation.json');latest=read(main/'validation_latest.json')
original_test=read(P/'results/review_test_original_v3/metrics.json')
adapted_test=read(P/'results/review_test_adapted_v3/metrics.json')
assert [(r['id'],r['case'],r['indices']) for r in original_test['rows']]==[(r['id'],r['case'],r['indices']) for r in adapted_test['rows']]
cfg=read(main/'config.json')
dataset=read(cfg['manifest'])
summary=dict(protocol='7 independent train objects / 2 val / 1 test; six fixed subsets per eval object',
             paired_endpoints=rows,paired_steps=120,
             validation_baseline=baseline['summary'],pilot=completion,best_step=best['step'],
             pilot_last_validation=latest['summary'],test_original=original_test['summary'],test_adapted=adapted_test['summary'],
             test_objects=original_test['objects'],test_cases=original_test['cases'],
             data_split={s:[o['id'] for o in dataset['objects'] if o['split']==s] for s in ['train','val','test']},
             interpretation='Validation improves; held-out test aligned AbsRel improves about 1%, canonical AbsRel regresses. No stable real-world or downstream 3DGS improvement established.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')

# Small, post-run provenance archive. No data, checkpoints, weights or caches.
with tarfile.open(main/'postrun_source_snapshot.tar.gz','w:gz') as archive:
    for folder,relative in [(P/'vggt_object','vggt_object'),(P/'configs','configs'),(P.parent/'3DGS/reconstruction/third_party/vggt/vggt','upstream_vggt')]:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.suffix in ('.py','.yaml') and '__pycache__' not in f.parts:
                archive.add(f,arcname=str(Path(relative)/f.relative_to(folder)))
    archive.add(P/'run.sh',arcname='run.sh')
(main/'dataset_manifest_snapshot.json').write_bytes(Path(cfg['manifest']).read_bytes())
fingerprints=[]
for obj in dataset['objects']:
    f=Path(obj['path'])/'cameras.json'
    fingerprints.append(dict(id=obj['id'],metadata_path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(main/'postrun_metadata_hashes.json').write_text(json.dumps(dict(note='Post-run capture, not a preregistered hash of original media.',objects=fingerprints),indent=2)+'\n')

lines=['# VGGT-Object 训练设计审查 V3','',
'## 结论','',
'当前可用数据只有10件Smithsonian渲染文物。默认采用冻结整个Aggregator、全参数微调Depth Head；保留Conv-LoRA作为轻量对照。这个选择有相同120步的先导对照支持，不是认定LoRA在所有DPT任务上都不好。主训练400步已完成。',
'',
'验证改善明显，但保留测试对象的收益有限且部分原始尺度指标退化；不能宣称真实场景泛化或4/6/9视角3DGS已稳定提升。',
'',
'## 数据组织与实验划分','',
'- data/omni3d 与 data/VGGT_finetune_v1 保持原上传位置；只建JSON索引，未移动或复制RGB/深度/GLB。大量零字节的omni3d不可训练，ZoeDepth不当GT。',
'- 原清单objects.json保留8/1/1；当前objects_review_v2.json另存7/2/1。希腊雕像从原训练集留作第二个验证对象，原测试方彝不变。',
'- 7个训练对象、2个验证对象、1个测试对象；视角是同一个物体的重复观测，不能把192张图或12组验证输入写成独立对象数。',
'- CO3D估计深度和RGB-only序列保持登记，但未混入当前GT监督。严格混合检查和Stage B门槛防止缺数据时冒充三数据集训练。',
'',
'## DPT使用LoRA是否合理','',
'VGGT的DPTHead解码部分包含卷积投影、多尺度融合和上采样。可对卷积展开矩阵做BA低秩增量，但这不是把注意力qkv的LoRA结论直接搬过来就能保证更优。',
'',
'全Depth Head微调仅32,654,562参数，Aggregator仍冻结在no_grad中；不是更新整个约1B的VGGT。在本轮同预算对照中，全深度头的验证误差更低、训练也更快。它还能适配归一化与上采样通路，可能比只改卷积低秩增量更有容量；这个机制解释仍需模块消融验证。',
'',
'LoRA适合适配器极小、多个领域切换的需求。本轮对照r8/alpha16/dropout0.05，约63万参数。后续可选LoRA配置排除最后仅2输出通道的小卷积，以免低秩增量比该层原权重还多。该小调整没有冒充已重跑的对照。',
'',
'官方VGGT训练配置本身提供冻结Aggregator的微调例子；LoRA原论文主要验证语言模型，不能直接证明DPT卷积解码器一定优于全头微调。',
'',
'## 相同120步的实测对照','',
'各组都从同一原始VGGT出发；相同7个训练对象、输入采样、392分辨率、4视角、累积2、基础损失。各模式采用保守LR：LoRA 2e-5，全头2e-6。下表使用统一120步终点，不混用旧版80步best与120步latest。验证为2对象×6固定输入，先对象内平均，再对象宏平均。',
'',
'| 模型 | 尺度对齐AbsRel↓ | canonical未对齐AbsRel↓ | 可训练参数 | 峰值显存GiB | 用时s |',
'|---|---:|---:|---:|---:|---:|',
f'| 原始VGGT | {baseline["summary"]["scene_scale_aligned"]["AbsRel"]:.6f} | {baseline["summary"]["canonical_unaligned"]["AbsRel"]:.6f} | 0 | — | — |']
for label,row in zip(['Depth Head Conv-LoRA','全Depth Head','全Depth Head＋MV'],rows):
    lines.append(f'| {label} | {row["aligned_AbsRel"]:.6f} | {row["canonical_AbsRel"]:.6f} | {row["trainable"]:,} | {row["peak_gpu_GiB"]:.2f} | {row["duration_seconds"]:.1f} |')
lines+=['',
'MV从0.016170到0.016168几乎相同，当前没有额外收益证据，因此默认关闭。LoRA更省参数/检查点，但本实现的额外卷积分支没有让训练峰值显存更小；不能泛化为所有实现都如此。',
'',
'## 损失函数与修正','',
'默认总损失：L = L1_depth + 0.05 L_conf + 0.1 L_grad_log。',
'',
'- L1_depth：有效前景内相机Z深度与GT的标量绝对误差，与官方向量范数回归在单通道情形一致。',
'- L_conf：e*C − 0.2*log(C)，保留官方深度/置信度激活。权重从1降到0.05，避免它主导小样本优化。该项可为负，因此总loss不是准确率，也不要求一直为正。',
'- L_grad_log：log-depth在1/2/4三个采样尺度的梯度差，严格要求梯度两端都有效；权重0.1，不跨背景计算。',
'- MV仅作F1/E4对照：GT确定的可见对应、前景双线性有效支持、投影越界/负GT深度/遮挡筛除；用预测源深度变换后的目标Z与GT对应位置的目标预测Z作相对误差，SmoothL1 beta=0.05。3% GT相对深度阈值，双向全对。',
'- 修复旧版预测依赖过滤：预测变差不再让对应自动消失，从而假装MV误差变好。权重从50步后逐渐加到0.05；记录真实有效对应数。',
'- GT深度、相机平移共享场景尺度，先变到首相机坐标，按有效点平均距离统一规范化；不逐帧归一化。canonical不是米制恢复，另报一个全场景共享比例的尺度对齐指标，不能只报对齐后结果。',
'',
'## 实际训练配置','',
'全Depth Head，AdamW LR2e-6、weight decay0.01、5% warmup、cosine到1e-6、clip norm1.0。392×392、4视角、1 scene、累积2。70%输入结合相邻已知相机与全局帧，30%随机覆盖，按对象打乱队列平衡采样。',
'',
'训练仅增加共享场景亮度/对比度0.9–1.1和sigma0.002噪声，不改变K/E/depth/mask。val/test不增强。当前7件训练对象不适合直接套8,000+2,000步的大数据预算；pilot最多400步，每50步验证，4次没有0.001有效改善则早停。best始终保存最低合格误差，早停min_delta单独控制；未对齐误差超过原始基线1.25倍的候选不选。零步原始模型也可成为best，失败适配不会自动替换它。',
'',
'## 400步先导训练与保留测试','',
f'训练实际{completion["steps"]}步，耗时{completion["duration_seconds"]:.1f}秒，峰值{completion["peak_gpu_GiB"]:.2f} GiB，最佳检查点第{best["step"]}步。先固定验证集best，再进行保留测试，没有根据test选择回退步数或模式。',
'',
'| 集合/指标 | 原始VGGT | 400步候选 |',
'|---|---:|---:|',
f'| val 尺度对齐AbsRel↓ | {baseline["summary"]["scene_scale_aligned"]["AbsRel"]:.6f} | {latest["summary"]["scene_scale_aligned"]["AbsRel"]:.6f} |',
f'| val canonical AbsRel↓ | {baseline["summary"]["canonical_unaligned"]["AbsRel"]:.6f} | {latest["summary"]["canonical_unaligned"]["AbsRel"]:.6f} |',
f'| test 尺度对齐AbsRel↓ | {original_test["summary"]["scene_scale_aligned"]["AbsRel"]:.6f} | {adapted_test["summary"]["scene_scale_aligned"]["AbsRel"]:.6f} |',
f'| test canonical AbsRel↓ | {original_test["summary"]["canonical_unaligned"]["AbsRel"]:.6f} | {adapted_test["summary"]["canonical_unaligned"]["AbsRel"]:.6f} |',
f'| test 尺度对齐RMSE↓ | {original_test["summary"]["scene_scale_aligned"]["RMSE"]:.6f} | {adapted_test["summary"]["scene_scale_aligned"]["RMSE"]:.6f} |',
'',
'保留test只有1件独立文物、6组固定输入：尺度对齐AbsRel仅约1%改善，canonical AbsRel约9.8%变差，而RMSE改善。说明部分较大误差减少，但尺度/小误差分布仍有问题，不能只挑RMSE或验证集的漂亮数字。这个test已使用，后续若据此改设计，必须说明是探索，另保留新测试对象验证。',
'',
'Camera Head与Aggregator都冻结，测试中相对位姿指标保持一致。当前没有真实手机GT深度，也没有完成4/6/9视角3DGS收益验证；不能声称“文物深度更准/重建更好”已经充分成立。',
'',
'## 省盘、复现与后续','',
'基础5GB VGGT权重只读复用。每个全深度头约124.6 MiB，一轮best+latest+最新Adam状态约0.5 GiB；不保存整个VGGT。只保留best/latest，不按每个验证步累积大模型。元数据与源码事后快照很小，不复制媒体。',
'',
'- 最佳候选：experiments/depth_full_pilot_v3/best；完整模型信息、配置、清单哈希保存在其中。',
'- 训练日志：logs/depth_full_pilot_v3.log；逐步loss/梯度/显存与验证曲线在experiments/depth_full_pilot_v3。',
'- 测试图：results/review_test_original_v3与review_test_adapted_v3；列为RGB / GT / 预测深度 / 置信度，共用GT色标。',
'- 机器可读对照：reports/review_v3/summary.json。',
'- 事后源码快照、研究清单与元数据哈希保存在主run中；这是追溯归档，不冒充训练前预注册的完整原始媒体校验。',
'',
'下一步优先补独立物体/真实拍摄验证、固定相机下4/6/9的原版VGGT+3DGS与微调VGGT+3DGS对照，以及多随机种子/对象折叠；不要继续针对这1件test加步数调参。下游4⊂6⊂9输入共享同一测试帧集合，点云尺度不拟合GT深度。CO3D/omni上传与监督质量未就绪前，不开启伪标签混合，也不自动执行更长Stage A/B。',
'',
'参考：[VGGT官方训练配置](https://github.com/facebookresearch/vggt/blob/main/training/config/default.yaml)、[DPT论文](https://arxiv.org/abs/2103.13413)、[LoRA原论文](https://arxiv.org/abs/2106.09685)。']
(P/'REVIEW_DESIGN_V3.md').write_text('\n'.join(lines)+'\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,label in [(n,n.replace('review_','').replace('_v2','').replace('_v3','')) for n in names]+[('depth_full_pilot_v3','full pilot + RGB augmentation')]:
        root=P/'experiments'/name;base=read(root/'baseline_validation.json')['summary']
        history=[json.loads(x) for x in (root/'validation_history.jsonl').read_text().splitlines()]
        for ax,key,title in [(axes[0],'score','Scene-scale-aligned AbsRel'),(axes[1],'canonical_AbsRel','Canonical unaligned AbsRel')]:
            mode='scene_scale_aligned' if key=='score' else 'canonical_unaligned'
            ax.plot([0]+[r['step'] for r in history],[base[mode]['AbsRel']]+[r[key] for r in history],marker='o',markersize=3,label=label)
            ax.set_title(title);ax.set_xlabel('Optimizer steps');ax.grid(alpha=.25)
    axes[0].set_ylabel('Lower is better');axes[1].legend(fontsize=7)
    fig.suptitle('Validation: 2 held-out objects, 6 fixed input subsets each (not 12 objects)')
    fig.tight_layout();fig.savefig(OUT/'validation_curve.png',dpi=180);plt.close(fig)
except ImportError:
    print('Matplotlib unavailable; JSON/Markdown report complete, no dependency downloaded.')
print(P/'REVIEW_DESIGN_V3.md')
print(OUT/'summary.json')
