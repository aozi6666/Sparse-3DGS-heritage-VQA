# 方案：VGGT-DPT 深度头域适应 + 置信度耦合点云加密

> 状态：**设计已锁定，待实现 / 待人工跑实验**。  
> 范围：步骤 **1（VGGT Visual Hull）** 的深度先验质量；可选轻量改 `vggt_visual_hull_enhanced.py`；**不改** CADR / LoRA / repair 主线（步骤 7–8）。  
> 环境：继续 `.venv-ddgs-vggt`；VGGT 权重 `3DGS/vggt/models/model.pt`。  
> 相关：根 [`README.md`](README.md) 工作 A · CADR；[`3DGS/eval_results/README.md`](3DGS/eval_results/README.md)。  
> **约束：训练 / 下载 / 评测一律由人工在终端执行；agent 只给命令与改代码，不代跑模型。**

---

## 0. 一句话决策（锁定）

| 事项 | 决策 |
| --- | --- |
| 总策略 | **A+B 捆绑叙事**；实现上 **A 必做，B 作可选消融** |
| A | **Decoder-only**：冻 `aggregator`（含 DINOv2），**只训 `depth_head`（DPT）** |
| B | **不强制**替换现有质量门限；`depth_conf` 仅作可选增强 / 消融 |
| 现有 Zoe–VGGT 门限 | **保留**（已有效果，勿先动） |
| 数据集 | **现成 mip360**（kitchen / bonsai / garden 等）+ 论文要报的遗产/物体场景；**不以 Co3D 作主线** |
| 监督 | **COLMAP `points3D` 投影稀疏深度**（`depth>0` 才算 loss）；Zoe **不当主监督** |
| 架构叙事 | 勿写成「发明 ViT 解码器」；写成 **领域适配 + 与稀疏 3DGS 初始化耦合** |
| 不做（主线） | 全量微调、重写 Fusion、MuSA 级 adapter、DepthSplat 整系统重做 |
| 与 CADR | **正交**：CADR=修复监督；本方案=几何初始化；可同论文两章/两贡献 |

---

## 1. 背景与现状（对齐代码）

线 B：

1. **VGGT hull** → 2. 粗 GS(±Drop) → 3. 评测 → 5–6. LOO → 7. LoRA → 8. repair(CADR) → 9. 评测

现状（`vggt_visual_hull_enhanced.py` + `1_vggt_hull.sh`）：

- 加载**原版** `VGGT` 权重推理，**未训练** depth head  
- 图中「Frozen Alternating Attention + Learnable ViT/DPT decoder」对应官方结构：`aggregator` 冻、`depth_head=DPTHead` 可训  
- 质量门限：

\[
\text{score}_i=\frac{1}{1+\operatorname{mean}|D^{\text{Zoe}}_i-D^{\text{VGGT}}_i|}
\]

  低于 `--vggt_quality_threshold` 的视角不参与增强  
- VGGT 输出的 **`depth_conf` 未使用**（见 §3.2：先不动）

已验证（`3DGS/eval_results/lineB_kitchen_mip360.csv`）：

- 初始化：`visual_hull_vggt_enhanced_4`  
- 粗训 Drop test PSNR ≈ **26.14**；LoRA repair ≈ **+0.8 PSNR**  
- **VGGT vs classic hull 仍为 `delta_pending`** → 几何侧消融尚未归档

**本方案目标：**

> 在少改网络的前提下，用 COLMAP 稀疏深度对 VGGT 的 DPT depth head 做域适应，提升稀疏视角下深度先验与初始点云质量；质量门限默认保持 Zoe–VGGT 一致性；`depth_conf` 仅作可选消融，证明「多视角几何先验 + 置信度」相对单目 depth prior 的优势。

成功标准（kitchen / 现有协议，相对**同一** `visual_hull_vggt_enhanced_4` 原权重基线）：

- 粗训（步骤 3，`ours_10000`）test **PSNR 稳定 ≥ +0.15**，或主观几何明显更干净且 PSNR 不降；  
- 若只改善初始化点云、下游持平：仍可写方法 + 消融，但须如实报告；  
- **不要求** CADR / repair 同步再涨（那是另一贡献）。

---

## 2. 相关工作 → 本方案映射

| 方法 | 核心思想 | 本仓库轻量对应 |
| --- | --- | --- |
| **VGGT** (CVPR’25) | Aggregator + DPT 出 depth/point + aleatoric conf；官方建议冻 aggregator、训 camera/depth head | A：只训 `depth_head` |
| **DPT** (Ranftl) | Reassemble + Fusion 多尺度深度头 | 即图中「ViT decoder」；不重造 |
| **Depth Anything / 遗产 UAV ISPRS’26** | 冻 backbone、只调 decoder 做域适应 | A 的叙事锚点 |
| **FSGS / DN-Splatter / CDGS** | 深度先验进 3DGS；CDGS 强调 confidence-aware | B：可选 `depth_conf` 门控；你比单目更贴 **多视角 VGGT** |
| **DepthSplat** | 更好深度 ↔ 更好 3DGS | 只借鉴结论；不重做整套 feed-forward GS |

设计原则：

1. **少改网络**：只动 depth head 参数与 hull 脚本门控逻辑  
2. **监督宁缺毋滥**：稀疏 COLMAP 真点 > 稠密 Zoe 伪深度  
3. **已有效果的门限先不动**（Zoe–VGGT）  
4. **与 CADR 解耦**：步骤 1 的改动不影响步骤 8 默认行为  

---

## 3. 方法设计

### 3.1 A：Decoder-only 域适应（必做）

**可训 / 冻结**

| 模块 | 训练？ |
| --- | --- |
| `aggregator`（DINOv2 + Alternating Attention） | **冻** |
| `camera_head` / `point_head` / `track_head` | **冻**（主线） |
| `depth_head`（DPT） | **训** |

可选弱增强（默认关，消融用）：解冻 aggregator **最后 1–2 个 block**，LR = decoder 的 0.1×（对齐 finetune_omega 类做法）。主线论文表以「仅 depth_head」为准。

**数据（适配你当前工作）**

- 根目录：`3DGS/GaussianObject/data/mip360/{kitchen,bonsai,garden,...}`  
- 每场景已有：`images/`、`masks/`、`sparse/0/`（含 `points3D`）、`sparse_*.txt`、`zoe_depth/`  
- **训练视角**：该场景 COLMAP 覆盖到的图像（可用较密视角集合；**不要用论文测试协议的稀疏子集去训**）  
- **测试 / 下游**：仍用 `SPARSE_ID=4`、`SPARSE_VIEW_NUM=9` 等线 B 协议  
- **不主用 Co3D**；时间够再作「泛化附录」，非方法成立前提  

**监督怎么得到（关键）**

对训练图像 \(I_i\)（已知 \(K_i,T_i\)）：

1. 读 COLMAP `points3D`  
2. 投影到图像平面 → 稀疏深度图 \(D^{\text{gt}}_i\)（有点处为深度，其余 0）  
3. 可选：与 `masks` 求交，背景不计 loss  
4. Loss（示意）：

\[
\mathcal{L}_{\text{depth}}=\frac{1}{|\Omega|}\sum_{p\in\Omega}
\big|\,f_{\text{depth}}(I)_p - D^{\text{gt}}_{i,p}\,\big|
\quad \Omega=\{p\mid D^{\text{gt}}_{i,p}>0\}
\]

尺度：若 VGGT 深度与 COLMAP 尺度不一致，用**每图中位数尺度对齐**后再算 L1（或官方 training loss 中的尺度处理）；实现时优先复用 `3DGS/vggt/training/` 已有 depth loss。

**禁止当主监督**：ZoeDepth 稠密图（噪声教师；Zoe 已用于一致性门限，角色不混）。

**权重产物**

- 保存：`3DGS/vggt/models/model_depthft_<tag>.pt`（或只存 `depth_head` state_dict）  
- `1_vggt_hull.sh` 通过 `VGGT_MODEL=...` 切换，**不覆盖**原 `model.pt`

### 3.2 B：置信度耦合加密（可选消融，非默认替换）

**默认：保持现有 Zoe–VGGT `quality_score` 门限不变。**

原因：线 B 已用该门限跑通并归档；改门限**不保证**涨点；仓库尚无「VGGT vs classic hull」对照。

可选增强（实验 ID 见 §7，默认关）：

| 模式 | 行为 |
| --- | --- |
| `GATE_ZOE`（默认） | 现状：Zoe vs VGGT 一致性 |
| `GATE_CONF` | 仅用 `depth_conf` 均值 / 分位数过阈 |
| `GATE_ZOE_AND_CONF` | 现有 score 过阈 **且** 像素级 `depth_conf` 再滤反投影点 |

推荐消融顺序：先做完 A（换微调权重 + 默认门限）→ 再开 `GATE_ZOE_AND_CONF`。  
**不要**在 A 未完成前单独大改门限。

### 3.3 与论文 Figure（VGGT-Depth prior）的对齐写法

- Snowflake：`aggregator`（预训练、冻结）  
- Fire：`depth_head`（本工作适配）  
- 输出：depth（+ 可选 conf）→ Visual Hull 增强 → 3DGS 初始化  
- 贡献句建议：

> 在冻结 VGGT 几何 backbone 的前提下，以 COLMAP 稀疏深度监督适配 DPT 深度头，并将其用于稀疏视角下的视觉外壳点云增强；默认保留双深度一致性门控，并消融 VGGT 置信度的作用。

---

## 4. 工程范围（改什么 / 不改什么）

### 4.1 预期改动文件

| 文件 / 目录 | 改动 |
| --- | --- |
| `3DGS/vggt/training/` 或新建轻量 `finetune_depth_head_colmap.py` | COLMAP→稀疏深度 dataloader + 只训 depth_head |
| `3DGS/GaussianObject/sh/lineB_gaussianobject/1_vggt_hull.sh` / `_common.sh` | `VGGT_MODEL` 指向微调权重 |
| `vggt_visual_hull_enhanced.py` | **可选**：`GATE_*` 与 `depth_conf` 像素过滤（默认关） |
| `3DGS/eval_results/lineB_kitchen_mip360.csv` | 人工跑完后追加行 |

### 4.2 明确不改

- CADR / `gaussian_object_system.py` repair 逻辑（除非联调实验）  
- LoRA / ControlNet / AbsoluteReality 流程  
- 粗训 Drop 超参默认值  
- 覆盖原 `models/model.pt`

### 4.3 Agent / 人工分工

| 谁 | 做什么 |
| --- | --- |
| Agent | 写/改代码与本 PLAN；给出终端命令；解读你贴回的指标 |
| **你（人工）** | 下载（若需要）、微调、hull、粗训、评测；把 `results.json` / CSV 片段贴回 |

---

## 5. 实验协议（固定不变项）

与 `3DGS/eval_results/README.md` 对齐，**几何侧对比只改步骤 1 的权重/门控**：

| 项 | 值 |
| --- | --- |
| 场景（主表） | mip360 / kitchen |
| 扩展场景（附录） | bonsai / garden；遗产场景若有则补 |
| resolution | 4 |
| sparse_view_num | 9 |
| sparse_id（hull） | 4 → `visual_hull_vggt_enhanced_4` |
| sh_degree | 2 |
| 粗训 | `USE_DROP=1`，`iterations=10000` |
| 初始化 tag | 原权重 vs 微调权重用**不同** `INIT_PCD` / 输出 ply 名，避免覆盖 |
| 指标 | 步骤 3：test PSNR / SSIM / LPIPS；可选点云可视化 |
| repair | 主表**可不重跑**；若时间够，对最优 init 复用已有 LoRA 跑一步 8/9 看传递增益 |

---

## 6. 必须完成的对比实验（主表）

| ID | 名称 | 设定 | 目的 |
| --- | --- | --- | --- |
| G0 | Classic hull | `visual_hull`（无 VGGT 增强），同粗训协议 | 下界；**补 `delta_pending`** |
| G1 | VGGT-orig + GATE_ZOE | 现线 B：`model.pt` + Zoe–VGGT 门限 | **直接基线**（CSV 已有粗训行） |
| G2 | **VGGT-DPT-ft + GATE_ZOE（Ours-A）** | 微调 depth head 权重 + **同一门限** | 主方法 |
| G3 | VGGT-DPT-ft + GATE_ZOE_AND_CONF（Ours-A+B） | G2 + conf 像素/视角门控 | 证明 B 是否加分 |
| G4 | VGGT-orig + GATE_CONF | 原权重 + 仅 conf 门限 | 隔离「换门限」效应（可选） |

主表至少：**G0 / G1 / G2**。  
G3 推荐；G4 时间不够可砍。

指标落点：步骤 3 `ours_10000` 的 test 集；写入 CSV 新行，`notes` 标明 `VGGT_MODEL` 与 `GATE_*`。

---

## 7. 必须完成的消融实验

均在 **同一粗训协议** 上，只改步骤 1：

| ID | 设定 | 验证什么 |
| --- | --- | --- |
| D0 | G1（原权重 + GATE_ZOE） | 基线 |
| D1 | 仅换微调权重，GATE_ZOE（= G2） | A 是否有效 |
| D2 | G2 + GATE_ZOE_AND_CONF（= G3） | B 叠加是否有效 |
| D3 | 微调时 **Zoe 当主监督**（反例，可选） | 证明「稀疏 COLMAP > Zoe 教师」 |
| D4 | 解冻 aggregator 末层（可选） | 是否值得多训参数 |

报告建议：**D0 / D1 必做**；D2 强烈建议；D3/D4 作补充。

若 D1 ≤ D0：先查尺度对齐与过拟合（是否用了测试稀疏视角训练），**不要**先上 G3 换门限找补。

---

## 8. 你后续应做的实验类型（执行清单）

> 下列命令**请你本地执行**；跑完把关键 `results.json` 路径或指标贴回即可。Agent **不代跑**。

### 阶段 0：补基线空洞（优先、便宜）

目的：填上 `VGGT vs classic hull`。

```bash
# 若尚无 classic hull ply，按 docs/reproduction 生成 visual_hull_4.ply 后：
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/sh/lineB_gaussianobject
# 用 INIT_PCD=visual_hull_4（或你仓库里 classic 名）跑步骤 2→3，USE_DROP=1
# 具体 flag 以你现有 2_train_gs.sh 的 --init_pcd_name 为准
```

产出：G0 的 test PSNR/SSIM/LPIPS → 写入 CSV。

### 阶段 1：准备微调数据清单（无训练）

目的：确认每场景能投影出稀疏深度。

建议你跑完后检查：每张训练图有效深度像素数（应 > 几百；过少则该图跳过）。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
# 待实现脚本示例名（实现后由 agent 给出精确路径）：
# python 3DGS/GaussianObject/tools/export_colmap_sparse_depth_stats.py \
#   --data_dir 3DGS/GaussianObject/data/mip360/kitchen
```

### 阶段 2：Decoder-only 微调（人工）

目的：得到 `model_depthft_*.pt`。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
# 实现就绪后，典型形态（具体入口以落地脚本为准）：
# python <finetune_script>.py \
#   --resume 3DGS/vggt/models/model.pt \
#   --freeze aggregator \
#   --train_modules depth_head \
#   --data_root 3DGS/GaussianObject/data/mip360 \
#   --scenes kitchen,bonsai,garden \
#   --out 3DGS/vggt/models/model_depthft_mip360.pt
```

注意：

- **hold-out**：若 kitchen 是主测场景，微调时应 **留出 kitchen 全场景** 或至少不用 sparse 测试视角；更干净的做法是 **在 bonsai+garden 上训，kitchen 零样本测**（论文更漂亮）。二选一写进实验设置，不要混。  
- **推荐默认**：**Train = bonsai + garden（及非主测遗产场景）；Test 主表 = kitchen**（避免同一场景泄漏）。若只有 kitchen，则用「非 sparse_9 列表中的其余视角训 / sparse 协议测」，并在论文写明。

### 阶段 3：换权重重跑 hull → 粗训 → 评测（主实验 G2）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/sh/lineB_gaussianobject
source ./_common.sh && lineb_print_paths

# 指向微调权重；INIT_PCD 用新名字避免覆盖基线 ply
VGGT_MODEL=/root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/vggt/models/model_depthft_mip360.pt \
INIT_PCD=visual_hull_vggt_depthft_${SPARSE_ID} \
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 1

# 粗训与评测（2→3）；GS_DIR 建议另开目录如 kitchen_drop_depthft
# （具体 env 名以实现时 _common.sh 扩展为准，避免覆盖 kitchen_drop）
```

对比：G1（已有 `kitchen_drop` / `ours_10000`）vs G2。

### 阶段 4：门控消融（G3 / G4，可选）

仅在 G2 ≥ G1 之后做。改 `GATE_*` 后重跑步骤 1→3（可复用同权重）。

### 阶段 5：传递到 repair（可选）

若 G2 粗训明显涨点，对 G2 的 `GS_DIR` 复用已有 LoRA 跑 5–9，看 CADR 是否进一步受益。  
**非本方案成立前提。**

### 阶段 6：多场景泛化（附录）

同一协议在 bonsai / garden / 遗产场景重复 G1 vs G2。

---

## 9. 实施顺序（给后续改代码用）

1. 实现 COLMAP→稀疏深度导出 + 统计脚本（人工跑 stats）。  
2. 实现 depth_head-only 微调入口（冻 aggregator）；保存独立 ckpt。  
3. `1_vggt_hull.sh` / `_common.sh` 支持 `VGGT_MODEL` + 可配置 `INIT_PCD` / `GS_DIR` 后缀。  
4. 人工跑阶段 2–3 → 填 CSV：**G1 vs G2**。  
5. （可选）`GATE_ZOE_AND_CONF` → G3。  
6. （可选）G0 classic hull 补档。  
7. 论文：主表 G0–G2；消融 D0–D2；定性：深度图 + 初始 ply + 新视角渲染。

---

## 10. 风险与回退

| 风险 | 处理 |
| --- | --- |
| 尺度不一致导致 loss 假收敛 | 每图 median scale align；可视化叠加 COLMAP 点 |
| 在 kitchen 上训又在 kitchen 稀疏测 → 泄漏 | 优先跨场景训；否则禁用 sparse 列表视角 |
| 微调后反而不如原权重 | 回退 `model.pt`；检查 LR（建议从 `1e-5`～`5e-5` 扫） |
| 一改门限指标乱跳 | 默认门限不动；B 只作消融 |
| 与 CADR 抢工期 | 本方案只动步骤 1；CADR 产物保持不动 |
| 想靠 Zoe 稠密监督省事 | 禁止作主监督；D3 反例可证伪 |

---

## 11. 检查清单

- [ ] 冻结策略：仅 `depth_head` 可训（主线）  
- [ ] 监督：COLMAP 稀疏深度；Zoe 非主监督  
- [ ] 数据：mip360 现成场景；Co3D 非主线  
- [ ] 默认门限：仍为 Zoe–VGGT（`GATE_ZOE`）  
- [ ] `depth_conf` 仅可选；未替换默认门限  
- [ ] 微调权重与原 `model.pt` 分文件保存  
- [ ] hull / GS 输出目录不覆盖 `kitchen_drop` 基线  
- [ ] CSV：G0（pending）/ G1（已有）/ G2（待跑）  
- [ ] 论文贡献不写成「发明 DPT / ViT decoder」  
- [ ] 训练与下载由人工执行，agent 不代跑  

---

## 12. 与 CADR 方案的关系

| | 本方案（本文件） | CADR（根 README 工作 A / eval_results） |
| --- | --- | --- |
| 步骤 | 1（几何初始化） | 8（扩散修复监督） |
| 置信度 | VGGT `depth_conf`（可选）+ Zoe 一致性（默认） | `w_need × w_trust`（视角级） |
| 是否互斥 | 否，正交 | 否 |
| 推荐论文结构 | 贡献 1：VGGT-DPT 适配初始化 | 贡献 2：CADR repair |

---

## 13. 待你确认后即可开工的实现项

实现前只需确认一件事（若与上文推荐一致可直接说「按推荐」）：

> **微调数据划分**：跨场景（bonsai+garden 训 / kitchen 测）还是单场景（kitchen 非测试视角训）？

确认后 agent **只改代码与补脚本**；微调与评测命令仍由你执行，结果贴回再解读。
