"""使用已审计汇总更新说明；逐文件保留修改前原文和哈希，不动推理代码。"""
from pathlib import Path
import json,hashlib,csv
P=Path('/root/autodl-tmp/Sparse-3DGS-heritage-VQA');O=P/'VQA/eval_results/practical_mage_v1/17_subtitle_final_report/20261103_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
M=json.loads((O/'stage_metrics.json').read_text());PA=json.loads((O/'paired_comparisons.json').read_text());C=json.loads((O/'test225_token_cost.json').read_text())
manifest=json.loads((O/'source_manifest.json').read_text())
for path,m in manifest['sources'].items():assert sha(P/path)==m['sha256']
if (O/'document_changes.json').exists():
    recorded=json.loads((O/'document_changes.json').read_text())
    for path,m in recorded.items():assert sha(P/path)==m['after_sha256']
    print('Document update already complete; hashes verified, no duplicate headers.');raise SystemExit(0)
title='SUMMARY_mage_subtitle_final_results_20261103.md'
lines=['# Mage连续字幕：冻结配置的最终结果与完整成果索引',
'更新：2026-11-03（Asia/Shanghai），分支vlm。本页为当前结论；此前实验及负结果原位保留。',
'## 1. 当前结论',
'固定2048连续字幕在test225上由无字幕140/225（62.22%）提高至163/225（72.44%），救回30、误伤7，净增23题（+10.22个百分点），视频聚类95%区间[5.33,15.11]。相对原对齐字幕154/225（68.44%）净增9题（+4.00个百分点），区间[-0.44,8.46]仍跨零，不能宣称补句模块独立显著优于原对齐字幕。',
'测试中等114题83→91；长视频111题57→72（救回18、误伤3）。开发和补充验证的长视频均持平，必须同时披露，不能写成所有划分均稳定提升。置信区间来自预存视频聚类bootstrap，问题级McNemar是辅助统计；不以小题型的事后结果作为确定性结论。',
'保留连续字幕2048为当前系统问答配置；原对齐、无字幕均保留为必要基线。3072开发预算对照119→117，保留为负向消融；不因测试中4个视频省略邻句而再调预算。换序、加帧、置信度选边的负结果继续保留。DeepSeek pilot没有显示叠加价值，Agent/API路线暂停。',
'## 2. 分阶段报告，禁止混成新测试集',
'| 阶段与角色 | 题/视频 | 无字幕Frames-32 | 原对齐字幕 | 连续字幕2048 |',
'|---|---:|---:|---:|---:|']
roles={'short219':'short219：历史开发覆盖','dev180':'dev180：开发','val90':'val90：已诊断的补充验证','test225':'test225：冻结最终评测（披露历史使用）'}
for stage in roles:
    mm=[next(m for m in M if m['stage']==stage and m['duration']=='overall' and m['method']==a) for a in ['no_sub','aligned','continuous']]
    lines.append('| '+roles[stage]+f" | {mm[0]['N']}/{mm[0]['videos']} | "+' | '.join(f"{m['correct']}/{m['N']}（{m['accuracy_pct']:.2f}%）" for m in mm)+' |')
lines+=['四个划分在问题索引和视频ID上互斥，219+180+90+225=714题，73+60+30+75=238视频，恰好覆盖当前本地输入清单。test225只有中长视频；没有新增独立短视频测试。旧smoke包含在short219、dev30包含在dev180、API pilot来自开发池，均不重复计入。',
'本地全覆盖只是覆盖性统计，不等于官方完整Video-MME，也不等于714题都是未见测试题。历史基准使用必须披露。测试结果出来后不再调预算、按时长选答案或修改规则；未来新方案需要新的评测安排。',
'## 3. 核心方法和贡献边界',
'实际采样32帧 → 用真实秒数命中[start,end)字幕cue → cue ID去重、时间排序 → 为命中cue最多补前后各一个非空邻句（相隔不超过2秒）→ 合并连续文本块并标时间 → 固定2048字幕文本token → 原Mage一次贪心回答；缺字幕/空白/无命中则精确回退同题无字幕输入和回答。',
'补句与时间标记联合改变，本实验没有将二者分离，不能分别归因。输入没有问题相关检索、外部裁判、多轮Agent、ASR或新训练。32帧和视觉像素预算固定，新增的是文本证据，总输入预算不固定。原对齐字幕不是“截取完整字幕前2048 token”；它按采样时刻命中，在本批数据中未因文本前缀截断而变短。',
'定位为免训练的系统集成与实证增强，不包装成新字幕对齐算法。原对齐思想来自已有视频字幕评测方式；本轮没有复制A.I.R./VCD/VideoHV-Agent核心算法，也不能将这些参考仓库列为本次提升的实现来源。重建贡献独立成立；公开视频问答结果没有证明重建质量导致了问答提升，也没有证明文旅资产定位和交互闭环已完成。',
'可用于论文的陈述：在固定视觉配置下，集成时间对齐、局部连续且带时间标记的字幕证据，并支持缺字幕时的视觉回退。在固定的225题中长视频子集上，相对无字幕Mage准确率提高10.22个百分点；相对原对齐字幕的额外提升为4.00个百分点，但其区间跨零。',
'## 4. 输入成本与限制',
'| test225配置 | 视觉token | 字幕token | 总输入token | 每题总输入token |',
'|---|---:|---:|---:|---:|']
for r in C:lines.append(f"| {r['method']} | {r['visual_tokens']:,} | {r['subtitle_tokens']:,} | {r['input_tokens']:,} | {r['input_tokens_per_question']:.2f} |")
lines+=['视觉token三路相同；连续字幕总输入相对无字幕增加约26.93%，相对原对齐增加约18.49%。2048是字幕文本预算，不是每题总输入上限。图中Other input包含问题、选项和格式token。共享预处理及额外原生输入检查使本次计时不适合直接当作公平部署速度基准。',
'## 5. 图、数据和案例',
'成果目录：[完整离线成果](../VQA/eval_results/practical_mage_v1/17_subtitle_final_report/20261103_v1/)。原始summary/预测/JSONL原位保留；新source_manifest记录来源大小与SHA，summary快照另存，不复制模型和视频。',
'- stage_duration_metrics.csv / .json：四阶段、分时长三路准确率。',
'- paired_comparisons.csv / .json：救回、误伤、净增、区间和辅助统计。',
'- all_predictions.csv / .json：714题去重明细及阶段角色。',
'- test225_token_cost.csv / .json：逻辑输入及新增生成成本。',
'- qualitative_cases.md / .json：按固定规则及索引排序选取的10个成功/失败示例，带问题选项、真值、原始输出、源视频路径、实际采样秒数、字幕原文和记录哈希。它们是看过GT后的定性示例，不是随机效果估计；未重新检查视频画面，不能仅凭字幕和回答断言纠错机制。',
'- figures/：5组PNG、PDF、SVG，英文坐标可用于论文；PNG预览、PDF/SVG可编辑。',
'\n## 6. 图表预览']
for name,caption in [('01_accuracy_by_stage','四阶段准确率，保留角色区分'),('02_accuracy_by_duration','各划分内部的分时长准确率'),('03_rescue_harm','纠错与误伤，含净增'),('04_test225_token_cost','固定视觉配置下的输入token成本'),('05_paired_cluster_intervals','配对准确率差及视频聚类区间')]:
    lines+=[f'### {caption}',f'![{caption}](../VQA/eval_results/practical_mage_v1/17_subtitle_final_report/20261103_v1/figures/{name}.png)']
lines+=['## 7. 收尾与后续','当前不需要再次运行本地full：所有714题已有三路结果，重复全量只是重做已有推理；先离线去重合并并保留来源和评测角色即可。若旧输入、模型、提示词或配置身份不一致，则只调查具体冲突，不盲目复用；本轮逐题和分组统计已通过核验。',
'仍需补的论文工作：人工核查成功/失败案例中的真实画面证据；按相同字幕条件做外部模型对比时另立协议；文旅场景独立评测与重建—问答联系验证。以上均为待做，不是本次已完成能力。当前不启动新实验，Agent继续暂停。']
content='\n\n'.join(lines)+'\n';new=P/'docs'/title
changes={}
def update(path,text):
    path=P/path;rel=path.relative_to(P);before=path.read_text() if path.exists() else None
    if before==text:return
    if before is not None:
        snapshot=O/'docs_before_update'/rel;snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists():assert snapshot.read_text()==before
        else:snapshot.write_text(before)
    oldsha=sha(path) if before is not None else None
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    changes[str(rel)]=dict(before_sha256=oldsha,after_sha256=sha(path),snapshot=str((O/'docs_before_update'/rel).relative_to(P)) if before is not None else None)
update('docs/'+title,content)
block='''## 当前结论（2026-11-03，字幕路线已完成冻结评测）

保留Frames-32＋连续补句＋时间标记，固定2048字幕token；无训练、无Agent/API。test225三路正确数：无字幕140、原对齐154、连续163（72.44%）。相对无字幕净增23题／10.22个百分点，聚类95%区间[5.33,15.11]；相对原对齐净增9题／4个百分点，区间[-0.44,8.46]，尚不能证明补句独立显著优于原对齐。

short219/dev180/val90/test225连续字幕分别175/119/54/163题正确，四阶段互斥覆盖本地714题／238视频，不能混称714题独立测试或官方完整基准。原对齐val90负结果、3072预算负结果及API pilot负结果全部保留。当前不再调字幕或重复跑本地full；Agent暂停，文旅闭环和重建对问答的贡献尚待验证。

完整三路数据、分时长统计、5组图、成本和10个案例见[最新最终结果总结](LINK)。历史内容若与本节不符，以本节及最新总结为准；历史实验不删除。
'''
for name in ['SUMMARY_heritage_mage_aligned_subtitles.md','SUMMARY_mage_subtitles_deepseek_review_20261103.md','PLAN_r2v_practical_accuracy_gain.md','PLAN_continuous_deepseek_pilot.md','PLAN_subtitle_budget_short_completion.md','PLAN_continuous_val90_fixed2048.md','PLAN_continuous_test225_fixed2048.md']:
    path=P/'docs'/name
    if not path.exists():continue
    text=path.read_text();first,rest=text.split('\n',1)
    update('docs/'+name,first+'\n\n'+block.replace('LINK',title)+'\n---\n\n## 以下为此前阶段记录／原实施说明（历史保留）\n'+rest)
path=P/'README.md';text=path.read_text()
text=text.replace('| **B · 视频问答 / 评测** | Mage-VL + **Query-IP（同 K 选区定稿）** + VLMEvalKit | **主表已定** [`VQA/`](VQA/) · [`eval_results/`](VQA/eval_results/) |','| **B · 视频问答 / 评测** | Mage Frames-32＋连续字幕2048；Query-IP历史路线保留 | **冻结test225已完成**：163/225；详见最新结果总结 |')
needle='## 工作 B：Mage-VL + Query-IP 选区 + VLMEvalKit'
assert needle in text
text=text.replace(needle,'## 工作 B：Mage字幕增强（当前）\n\n'+block.replace('LINK','docs/'+title)+'\n\n### Query-IP历史路线（独立协议，不能与字幕结果直接混比）')
text=text.replace('### 工作 B · Query-IP / Mage','### 工作 B · Query-IP / Mage（下列旧TODO为历史路线；当前字幕状态见上文）')
update('README.md',text)
path=P/'VQA/README.md';text=path.read_text();first,rest=text.split('\n',1)
update('VQA/README.md',first+'\n\n'+block.replace('LINK','../docs/'+title)+'\n---\n\n## 以下为历史Query-IP研究与应用设计（不代表本轮字幕能力已实现）\n'+rest)
(O/'document_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(updated=list(changes),source_files_verified=len(manifest['sources'])),ensure_ascii=False))
